#!/bin/bash
# docs/ai/*.md を PDF に変換する。
#
#   使い方:
#     ./scripts/md2pdf.sh              # 全文書を個別 PDF + 結合版 all.pdf に変換
#     ./scripts/md2pdf.sh docs/ai/1_least_squares_method.md   # 指定ファイルのみ
#
# 仕組み: pandoc で Markdown → HTML (数式は KaTeX、CDN から取得して HTML に埋め込む)、
# Chrome で HTML → PDF (scripts/print_pdf.py が CDP 経由で印刷し、
# フッター中央にページ番号を入れる)。出力先は build/pdf/。
# KaTeX を埋め込みにするのは、CDN 読み込みの完了前に印刷が走って数式が欠けるのを防ぐため。
set -eu

cd "$(dirname "$0")/.."
OUT=build/pdf
mkdir -p "$OUT"

md2pdf() {
  # $1: 出力名 (拡張子なし)、$2...: 入力 md ファイル (複数なら結合)
  local name=$1
  shift
  pandoc "$@" \
    -f gfm+tex_math_dollars \
    -t html -s --katex --embed-resources \
    --metadata pagetitle="$name" \
    --css scripts/pdf.css \
    -o "$OUT/$name.html"
  python3 scripts/print_pdf.py "$OUT/$name.html" "$OUT/$name.pdf"
  rm "$OUT/$name.html"
  echo "$OUT/$name.pdf"
}

if [ $# -gt 0 ]; then
  for f in "$@"; do
    md2pdf "$(basename "$f" .md)" "$f"
  done
else
  files=(docs/ai/[0-9]_*.md)
  for f in "${files[@]}"; do
    md2pdf "$(basename "$f" .md)" "$f"
  done
  md2pdf all "${files[@]}"
fi
