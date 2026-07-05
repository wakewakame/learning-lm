#!/usr/bin/env python3
"""Chrome の DevTools プロトコル (CDP) で HTML を PDF に印刷する。

使い方: python3 scripts/print_pdf.py 入力.html 出力.pdf

Chrome の --print-to-pdf は CLI からフッターを指定できないため、
CDP の Page.printToPDF を直接呼び、「中央にページ番号だけ」のフッターを付ける。
依存は Python 標準ライブラリと Chrome のみ (WebSocket クライアントも自前実装)。
"""

import base64
import hashlib
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

FOOTER = (
    '<div style="width:100%; text-align:center; font-size:9px;'
    ' font-family:\'Hiragino Sans\',sans-serif;">'
    '<span class="pageNumber"></span></div>'
)
HEADER = "<div></div>"  # ヘッダーは無し


class WebSocket:
    """RFC 6455 の最小限のクライアント実装 (テキストフレームのみ)"""

    def __init__(self, url):
        # ws://host:port/path を分解
        rest = url.removeprefix("ws://")
        hostport, _, path = rest.partition("/")
        host, _, port = hostport.partition(":")
        self.sock = socket.create_connection((host, int(port)))
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        req = (
            f"GET /{path} HTTP/1.1\r\n"
            f"Host: {hostport}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.sock.recv(4096)
        accept = base64.b64encode(
            hashlib.sha1(
                (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
            ).digest()
        ).decode()
        assert f"Sec-WebSocket-Accept: {accept}".encode() in resp, "handshake 失敗"

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("接続が閉じられた")
            buf += chunk
        return buf

    def send(self, text):
        payload = text.encode()
        mask = secrets.token_bytes(4)
        header = b"\x81"  # FIN + text frame
        n = len(payload)
        if n < 126:
            header += bytes([0x80 | n])
        elif n < 65536:
            header += bytes([0x80 | 126]) + n.to_bytes(2, "big")
        else:
            header += bytes([0x80 | 127]) + n.to_bytes(8, "big")
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(header + mask + masked)

    def recv(self):
        # 継続フレームを連結して 1 メッセージを返す
        message = b""
        while True:
            b1, b2 = self._recv_exact(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            n = b2 & 0x7F
            if n == 126:
                n = int.from_bytes(self._recv_exact(2), "big")
            elif n == 127:
                n = int.from_bytes(self._recv_exact(8), "big")
            payload = self._recv_exact(n)
            if opcode == 9:  # ping には pong を返す
                self.sock.sendall(b"\x8a\x80" + secrets.token_bytes(4))
                continue
            if opcode == 8:
                raise ConnectionError("サーバーが close を送信")
            message += payload
            if fin:
                return message.decode()


def cdp_call(ws, msg_id, method, params=None):
    ws.send(json.dumps({"id": msg_id, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == msg_id:
            if "error" in msg:
                raise RuntimeError(f"{method}: {msg['error']}")
            return msg.get("result", {})
        # イベント通知は読み捨てる


def main():
    html_path, pdf_path = os.path.abspath(sys.argv[1]), sys.argv[2]
    with tempfile.TemporaryDirectory() as profile:
        chrome = subprocess.Popen(
            [
                CHROME,
                "--headless",
                "--disable-gpu",
                "--remote-debugging-port=0",
                f"--user-data-dir={profile}",
                f"file://{html_path}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            # 選ばれたポートは DevToolsActivePort に書き出される
            port_file = os.path.join(profile, "DevToolsActivePort")
            for _ in range(300):
                if os.path.exists(port_file):
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("Chrome が起動しない")
            port = open(port_file).read().splitlines()[0]

            # 対象ページの WebSocket URL を取得
            for _ in range(300):
                targets = json.load(
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list")
                )
                pages = [t for t in targets if t.get("type") == "page"]
                if pages:
                    break
                time.sleep(0.1)
            else:
                raise TimeoutError("ページが見つからない")
            ws = WebSocket(pages[0]["webSocketDebuggerUrl"])

            # 読み込みと Web フォント (KaTeX) の完了を待ってから印刷する
            cdp_call(ws, 1, "Runtime.enable")
            cdp_call(
                ws,
                2,
                "Runtime.evaluate",
                {
                    "expression": (
                        "(document.readyState === 'complete'"
                        " ? Promise.resolve()"
                        " : new Promise(res =>"
                        "     window.addEventListener('load', res, {once: true}))"
                        ").then(() => document.fonts.ready).then(() =>"
                        " new Promise(res => requestAnimationFrame(() =>"
                        "   requestAnimationFrame(res))))"
                    ),
                    "awaitPromise": True,
                    "timeout": 300000,
                },
            )
            result = cdp_call(
                ws,
                3,
                "Page.printToPDF",
                {
                    "displayHeaderFooter": True,
                    "headerTemplate": HEADER,
                    "footerTemplate": FOOTER,
                    "printBackground": True,
                    "preferCSSPageSize": True,
                },
            )
            with open(pdf_path, "wb") as f:
                f.write(base64.b64decode(result["data"]))
        finally:
            chrome.terminate()
            chrome.wait()


if __name__ == "__main__":
    main()
