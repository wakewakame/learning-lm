# QR 分解 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

## 1. 復習と本文書のゴール

[1. 線形最小二乗法](./1_least_squares_method.md) では、データ $\mathbf{y} \in \mathbb{R}^n$ を計画行列 $\Phi \in \mathbb{R}^{n \times m}$ とパラメータ $\boldsymbol{\beta} \in \mathbb{R}^m$ のモデル $\Phi \boldsymbol{\beta}$ で近似する問題

$$
\min_{\boldsymbol{\beta}} \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2
$$

を考え、その解が**正規方程式**

$$
\Phi^\top \Phi \, \boldsymbol{\beta} = \Phi^\top \mathbf{y}
$$

で与えられることを学んだ。理論上はこれで話が閉じている。ところが同じ文書の第 6 節で「数値計算では $\Phi^\top \Phi$ を明示的に作ってはいけない。QR 分解を使うのが標準」と予告だけして、理由の説明を先送りにしていた。本文書はその答え合わせである。具体的には次の 3 つを説明する。

1. **直交行列**とは何か。なぜ数値計算で特別扱いされるのか (第 2 節)
2. **QR 分解**とは何か。どうやって計算するのか (第 3〜5 節)。そして QR 分解を使うと最小二乗問題がどう解けるのか (第 6 節)
3. **条件数**とは何か。なぜ正規方程式は危険で、QR 分解は安全なのか (第 7 節)。その差は実際のコードでどれほど現れるのか (第 8 節)

先に結論を一行で述べておく。

> 正規方程式は $\Phi^\top \Phi$ を作る時点で「誤差の増幅率 (条件数)」を**二乗**に悪化させてしまう。QR 分解は誤差を一切増幅しない直交行列だけで問題を変形するため、増幅率を悪化させずに最小二乗問題を解ける。

## 2. 直交行列 — 長さと角度を変えない変換

### 2.1 回転と鏡映

平面上でベクトルを角度 $\theta$ だけ回転させる操作は、行列

$$
Q_{\text{回転}} =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

を掛けることで表せる。例えば $\theta = 90^\circ$ なら $Q = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ で、$\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ は $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ に写る。確かに $x$ 軸方向の矢印が $y$ 軸方向へ $90^\circ$ 回っている。

もう 1 つ、$x$ 軸に関する**鏡映** (折り返し) は

$$
Q_{\text{鏡映}} =
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
$$

で表せる。$\begin{bmatrix} 3 \\ 4 \end{bmatrix}$ は $\begin{bmatrix} 3 \\ -4 \end{bmatrix}$ に写る。$y$ 成分の符号だけが反転し、$x$ 軸を鏡として折り返した形になっている。

回転と鏡映に共通するのは、**図形の形を一切歪めない**ことである。矢印の長さは変わらないし、2 本の矢印のなす角度も変わらない。位置と向きだけが変わる。このような「長さと角度を保つ線形変換」を表す行列を、これから**直交行列**と呼ぶことにする。まずはこの幾何的な性質を数式の条件に翻訳しよう。

### 2.2 「長さと角度を変えない」を式で書くと $Q^\top Q = I$ になる

長さも角度も、内積で書ける。ベクトル $\mathbf{u}$ の長さは $\| \mathbf{u} \| = \sqrt{\mathbf{u} \cdot \mathbf{u}}$ であり、$\mathbf{u}$ と $\mathbf{v}$ のなす角 $\theta$ は $\cos\theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\| \mathbf{u} \| \| \mathbf{v} \|}$ で決まる ([0. 数学の準備](./0_math_preliminaries.md) の内積の節を参照)。したがって「長さと角度を変えない」は、次の 1 つの条件にまとめられる。

$$
(Q \mathbf{u}) \cdot (Q \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}
\qquad (\text{すべての } \mathbf{u}, \mathbf{v} \text{ に対して})
$$

つまり「$Q$ で写しても内積が変わらない」である。ここで [0. 数学の準備](./0_math_preliminaries.md) で導いた公式

$$
(A \mathbf{u}) \cdot \mathbf{v} = \mathbf{u} \cdot (A^\top \mathbf{v})
$$

を思い出す (行列を内積の反対側へ移すと転置になる、という公式である)。これを $A = Q$、$\mathbf{v}$ の側を $Q \mathbf{v}$ として使うと

$$
(Q \mathbf{u}) \cdot (Q \mathbf{v}) = \mathbf{u} \cdot \bigl( Q^\top (Q \mathbf{v}) \bigr) = \mathbf{u} \cdot \bigl( (Q^\top Q) \mathbf{v} \bigr)
$$

と変形できる。よって内積保存の条件は

$$
\mathbf{u} \cdot \bigl( (Q^\top Q) \mathbf{v} \bigr) = \mathbf{u} \cdot \mathbf{v}
\qquad (\text{すべての } \mathbf{u}, \mathbf{v})
$$

となる。この条件が $Q^\top Q = I$ と同じであることを確かめよう。$\mathbf{u}$ として第 $i$ 成分だけが 1 の基本ベクトル $\mathbf{e}_i$、$\mathbf{v}$ として $\mathbf{e}_j$ を選ぶと、左辺は行列 $Q^\top Q$ の第 $(i, j)$ 成分、右辺は $\mathbf{e}_i \cdot \mathbf{e}_j$、すなわち $i = j$ のとき 1、それ以外は 0 になる。これは単位行列 $I$ の第 $(i, j)$ 成分そのものである。すべての $i, j$ で成分が一致するのだから

$$
Q^\top Q = I
$$

を得る。逆に $Q^\top Q = I$ なら、上の変形を逆にたどれば内積が保存されることも直ちに分かる。そこで次のように定義する。

> **定義 (直交行列)**: 正方行列 $Q$ が $Q^\top Q = I$ を満たすとき、$Q$ を**直交行列** (orthogonal matrix) と呼ぶ。

冒頭の回転行列で検算しておく。

$$
Q_{\text{回転}}^\top Q_{\text{回転}} =
\begin{bmatrix}
\cos\theta & \sin\theta \\
-\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix} =
\begin{bmatrix}
\cos^2\theta + \sin^2\theta & 0 \\
0 & \sin^2\theta + \cos^2\theta
\end{bmatrix}
= I
$$

確かに直交行列である ($\cos^2\theta + \sin^2\theta = 1$ を使った)。鏡映の行列 $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ も、自分自身との積が $I$ になることがすぐ確かめられる。

### 2.3 直交行列の性質

$Q^\top Q = I$ から、実用上重要な性質が芋づる式に出てくる。

**性質 1: 逆行列が転置である。** $Q^\top Q = I$ は、逆行列の定義そのものにより $Q^{-1} = Q^\top$ を意味する。普通の行列では逆行列を求めるのに掃き出し法などの計算が必要だが、直交行列では**行と列を入れ替えるだけ**でよい。「$Q$ を消したい」ときに $Q^\top$ を掛ければ済む、という手軽さが後で効いてくる。

**性質 2: どんなベクトルの長さも変えない。** 任意の $\mathbf{x}$ に対して

$$
\| Q \mathbf{x} \|^2
= (Q \mathbf{x}) \cdot (Q \mathbf{x})
= \mathbf{x} \cdot (Q^\top Q \mathbf{x})
= \mathbf{x} \cdot \mathbf{x}
= \| \mathbf{x} \|^2
$$

となる。これは定義 (内積保存) の特別な場合 ($\mathbf{u} = \mathbf{v} = \mathbf{x}$) にすぎないが、この後何度も使うので独立に確認した。

**性質 3: 列ベクトルが正規直交である。** $Q$ の列ベクトルを $\mathbf{q}_1, \dots, \mathbf{q}_n$ とすると、$Q^\top Q$ の第 $(i, j)$ 成分は $\mathbf{q}_i \cdot \mathbf{q}_j$ である ($Q^\top$ の第 $i$ 行は $\mathbf{q}_i$ を横に寝かせたものだから)。したがって $Q^\top Q = I$ は

$$
\mathbf{q}_i \cdot \mathbf{q}_j =
\begin{cases}
1 & (i = j) \\
0 & (i \neq j)
\end{cases}
$$

つまり「各列の長さが 1 で、どの 2 列も互いに直交している」ことと同じである。このようなベクトルの組を**正規直交** (orthonormal) と呼ぶ。直交行列とは「正規直交なベクトルを列として並べた正方行列」だと言い換えられる。

**性質 4: 誤差を増幅しない。** 数値計算では、データにも計算の途中結果にも小さな誤差が必ず混入する。ベクトル $\mathbf{x}$ に誤差 $\Delta \mathbf{x}$ が乗った状態で $Q$ を掛けると、結果は $Q \mathbf{x} + Q \Delta \mathbf{x}$ となるが、性質 2 より誤差部分の大きさは $\| Q \Delta \mathbf{x} \| = \| \Delta \mathbf{x} \|$ で**全く増えない**。第 7 節の言葉を先取りすれば「直交行列の条件数は 1」である。直交行列は、数値計算において最も安全に掛けられる行列なのである。

## 3. QR 分解とは何か

### 3.1 定義

> **定理 (QR 分解)**: 任意の行列 $A \in \mathbb{R}^{n \times m}$ ($n \geq m$ とする) は
>
> $$
> A = Q R
> $$
>
> と分解できる。ここで $Q \in \mathbb{R}^{n \times n}$ は直交行列、$R \in \mathbb{R}^{n \times m}$ は**上三角行列** (対角より下の成分がすべて 0 の行列) である。

$n > m$ (行数の方が多い、縦長の行列) のとき、$R$ は上三角なので下側 $n - m$ 行はすべて 0 である。そこで

$$
A = Q
\begin{bmatrix}
R_1 \\ O
\end{bmatrix}
= Q_1 R_1
$$

と書ける。$R_1 \in \mathbb{R}^{m \times m}$ は正方の上三角行列、$Q_1 \in \mathbb{R}^{n \times m}$ は $Q$ の左 $m$ 列だけを取り出した行列である ($Q$ の右側の列は $R$ の 0 の行としか掛からないので、消えてしまうからである)。$Q_1$ は正方ではないが、その $m$ 本の列は性質 3 の意味で正規直交であり、$Q_1^\top Q_1 = I$ ($m \times m$ の単位行列) を満たす。

- $A = QR$ ($Q$ が正方) を**フル QR 分解**
- $A = Q_1 R_1$ を**薄い QR 分解** (thin QR / economy QR)

と呼ぶ。理論の説明にはフルが、実装には薄い方が便利なことが多い。

### 3.2 上三角行列はなぜ嬉しいか — 後退代入

QR 分解は行列を「直交行列」と「上三角行列」の積に分ける。直交行列の嬉しさは前節で見たので、次は上三角行列の嬉しさである。それは一言で言えば「**連立方程式が代入だけで解ける**」ことである。

具体例を解いてみる。

$$
\begin{bmatrix}
2 & 1 & 1 \\
0 & 3 & 2 \\
0 & 0 & 4
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix} =
\begin{bmatrix}
5 \\ 8 \\ 4
\end{bmatrix}
$$

最後の行は $4 x_3 = 4$ という 1 変数の方程式なので、即座に $x_3 = 1$。これを 2 行目 $3 x_2 + 2 x_3 = 8$ に代入すれば $3 x_2 = 8 - 2 = 6$ で $x_2 = 2$。さらに 1 行目 $2 x_1 + x_2 + x_3 = 5$ に代入して $2 x_1 = 5 - 2 - 1 = 2$、$x_1 = 1$。

このように**下の行から順に、既に求めた値を代入していくだけ**で解が求まる。この手続きを**後退代入** (back substitution) と呼ぶ。掃き出し法のような消去の操作が一切不要で、速くて単純である。

### 3.3 QR 分解の設計思想

以上をまとめると、QR 分解の思想はこう表現できる。

> どんな行列も「空間を一切歪めない部分 ($Q$)」と「代入だけで処理できる単純な部分 ($R$)」に分けられる。難しい計算はすべて $R$ に押し付け、$Q$ は誤差を増やさずに掛けたり消したり ($Q^\top$ を掛けるだけ) できる。

では、この分解はどうやって計算するのか。代表的な構成法を 2 つ紹介する。1 つ目 (グラム・シュミット法) は分解の**存在と意味**を理解するのに最適で、2 つ目 (ハウスホルダー変換) は**実際の数値計算**で使われる方法である。

## 4. 構成法 1: グラム・シュミット法

### 4.1 まず 2 本のベクトルで — 手計算の例

グラム・シュミット法の目標は、与えられたベクトルの組から**正規直交なベクトルの組**を作ることである。2 次元の 2 本のベクトル

$$
\mathbf{a}_1 = \begin{bmatrix} 3 \\ 4 \end{bmatrix},
\qquad
\mathbf{a}_2 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}
$$

でやってみる。頭の中に座標平面を描いてほしい。$\mathbf{a}_1$ は右上へ長く伸びた矢印、$\mathbf{a}_2$ はそれより少し上向きの短い矢印で、2 本は少し違う方向を向いている。

**手順 1: 1 本目の方向を最初の軸にする。** $\mathbf{a}_1$ の向きはそのまま採用し、長さだけを 1 に揃える。$\| \mathbf{a}_1 \| = \sqrt{3^2 + 4^2} = 5$ なので

$$
\mathbf{q}_1 = \frac{\mathbf{a}_1}{\| \mathbf{a}_1 \|} = \begin{bmatrix} 3/5 \\ 4/5 \end{bmatrix}
$$

**手順 2: 2 本目から「1 本目方向の成分」を引き算する。** $\mathbf{a}_2$ は $\mathbf{q}_1$ と直交していないので、そのままでは使えない。そこで $\mathbf{a}_2$ を「$\mathbf{q}_1$ 方向の成分」と「$\mathbf{q}_1$ に垂直な成分」に分け、前者を捨てる。$\mathbf{q}_1$ 方向の成分の大きさは、太陽が真上から照らしたときに $\mathbf{a}_2$ が $\mathbf{q}_1$ の方向に落とす**影の長さ**であり、内積で計算できる ($\mathbf{q}_1$ の長さが 1 だから)。

$$
\mathbf{q}_1 \cdot \mathbf{a}_2 = \frac{3}{5} \cdot 1 + \frac{4}{5} \cdot 2 = \frac{3 + 8}{5} = \frac{11}{5}
$$

影の分を $\mathbf{a}_2$ から引くと、垂直な成分だけが残る。

$$
\tilde{\mathbf{q}}_2
= \mathbf{a}_2 - \frac{11}{5} \, \mathbf{q}_1
= \begin{bmatrix} 1 \\ 2 \end{bmatrix} - \begin{bmatrix} 33/25 \\ 44/25 \end{bmatrix}
= \begin{bmatrix} -8/25 \\ 6/25 \end{bmatrix}
$$

本当に垂直になったか検算する。

$$
\mathbf{q}_1 \cdot \tilde{\mathbf{q}}_2 = \frac{3}{5} \cdot \left( -\frac{8}{25} \right) + \frac{4}{5} \cdot \frac{6}{25} = \frac{-24 + 24}{125} = 0
$$

確かに直交している。最後に長さを 1 に揃える。$\| \tilde{\mathbf{q}}_2 \| = \frac{\sqrt{(-8)^2 + 6^2}}{25} = \frac{10}{25} = \frac{2}{5}$ なので

$$
\mathbf{q}_2 = \frac{\tilde{\mathbf{q}}_2}{\| \tilde{\mathbf{q}}_2 \|} = \begin{bmatrix} -4/5 \\ 3/5 \end{bmatrix}
$$

**手順 3: 元のベクトルを $\mathbf{q}$ たちで書き直す。** ここまでの計算を逆向きに読むと

$$
\mathbf{a}_1 = 5 \, \mathbf{q}_1,
\qquad
\mathbf{a}_2 = \frac{11}{5} \, \mathbf{q}_1 + \frac{2}{5} \, \mathbf{q}_2
$$

である ($\mathbf{a}_2$ の式は $\tilde{\mathbf{q}}_2 = \mathbf{a}_2 - \frac{11}{5} \mathbf{q}_1$ と $\tilde{\mathbf{q}}_2 = \frac{2}{5} \mathbf{q}_2$ を合わせたもの)。この 2 本の式を、列を並べた行列の形にまとめると

$$
\underbrace{
\begin{bmatrix}
3 & 1 \\
4 & 2
\end{bmatrix}
}_{A} =
\underbrace{
\begin{bmatrix}
3/5 & -4/5 \\
4/5 & 3/5
\end{bmatrix}
}_{Q}
\underbrace{
\begin{bmatrix}
5 & 11/5 \\
0 & 2/5
\end{bmatrix}
}_{R}
$$

となる。これが QR 分解である。$Q$ の列は正規直交 (作り方から明らか) で、$R$ は上三角になっている。念のため右辺の第 2 列を検算すると、$\frac{3}{5} \cdot \frac{11}{5} + \left( -\frac{4}{5} \right) \cdot \frac{2}{5} = \frac{33 - 8}{25} = 1$、$\frac{4}{5} \cdot \frac{11}{5} + \frac{3}{5} \cdot \frac{2}{5} = \frac{44 + 6}{25} = 2$ となり、確かに $\mathbf{a}_2 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$ が再現される。

ちなみにこの $Q$ は $\cos\theta = 3/5, \sin\theta = 4/5$ の回転行列そのものである。「$A$ = 回転 × 上三角」という分解が実際に手で作れた。

**なぜ $R$ が上三角になるのか**をここで言葉にしておく。作り方から、$\mathbf{q}_1$ は $\mathbf{a}_1$ だけから、$\mathbf{q}_2$ は $\mathbf{a}_1, \mathbf{a}_2$ から作られる。逆に言えば $\mathbf{a}_1$ は $\mathbf{q}_1$ だけで、$\mathbf{a}_2$ は $\mathbf{q}_1, \mathbf{q}_2$ で書ける。$\mathbf{a}_k$ を書くのに $\mathbf{q}_{k+1}$ 以降が**決して要らない**。「$k$ 列目は $k$ 番目までの $\mathbf{q}$ で書ける」という構造が、係数行列 $R$ の下半分が 0、すなわち上三角ということの正体である。

### 4.2 一般の場合

$A \in \mathbb{R}^{n \times m}$ の列ベクトルを $\mathbf{a}_1, \dots, \mathbf{a}_m$ とし、線形独立 ([0. 数学の準備](./0_math_preliminaries.md) 参照。どの列も他の列の組み合わせで作れない、という条件) と仮定する。2 次元の例と全く同じ発想で、$k$ 本目からは「既に作った $\mathbf{q}_1, \dots, \mathbf{q}_{k-1}$ 方向の影」を全部引き算すればよい。

$$
\begin{aligned}
\mathbf{q}_1 &= \frac{\mathbf{a}_1}{\| \mathbf{a}_1 \|} \\
\tilde{\mathbf{q}}_k &= \mathbf{a}_k - \sum_{j=1}^{k-1} ( \mathbf{q}_j \cdot \mathbf{a}_k ) \, \mathbf{q}_j,
\qquad
\mathbf{q}_k = \frac{\tilde{\mathbf{q}}_k}{\| \tilde{\mathbf{q}}_k \|}
\qquad (k = 2, \dots, m)
\end{aligned}
$$

Σ の中身は「$\mathbf{q}_j$ 方向の影の長さ × $\mathbf{q}_j$」であり、2 次元の例の手順 2 を $k - 1$ 回分まとめて書いただけである。列が線形独立なら $\tilde{\mathbf{q}}_k$ が $\mathbf{0}$ になることはなく (もし $\mathbf{0}$ なら $\mathbf{a}_k$ が前の列の組み合わせで書けてしまい、独立性に反する)、手続きは最後まで実行できる。

この式を $\mathbf{a}_k$ について解き直すと

$$
\mathbf{a}_k
= \sum_{j=1}^{k-1} \underbrace{( \mathbf{q}_j \cdot \mathbf{a}_k )}_{r_{jk}} \, \mathbf{q}_j + \underbrace{\| \tilde{\mathbf{q}}_k \|}_{r_{kk}} \, \mathbf{q}_k
$$

となり、$\mathbf{a}_k$ は $\mathbf{q}_1, \dots, \mathbf{q}_k$ **まで**の線形結合で書ける。係数 $r_{jk}$ を $(j, k)$ 成分に並べた行列 $R_1$ は上三角になり、列ごとの式をまとめれば

$$
A = Q_1 R_1
$$

すなわち薄い QR 分解が得られる。**QR 分解が必ず存在すること**は、このグラム・シュミットの手続きがいつでも最後まで実行できることから従う (列が線形従属の場合は第 9 節で扱う)。

### 4.3 浮動小数点での弱点 — 桁落ち

理論上は完璧なこの手続きが、コンピュータの浮動小数点演算 (有効数字約 16 桁の近似計算) では問題を起こす。原因は**桁落ち**である。

桁落ちとは、ほぼ等しい 2 つの数の引き算で有効数字が激減する現象である。例えば有効数字 8 桁で $1.0000001 - 1.0000000 = 0.0000001$ を計算すると、答えの有効数字は 1 桁しかない。8 桁あった情報のうち 7 桁が引き算で「相殺」されて消えてしまった。

グラム・シュミット法の中核は $\mathbf{a}_k - (\text{影の合計})$ という引き算である。もし $\mathbf{a}_k$ が既存の $\mathbf{q}_1, \dots, \mathbf{q}_{k-1}$ の張る空間に**ほぼ入っている** (ほぼ平行な列を持つ行列ではこれが起きる) と、$\mathbf{a}_k$ と影の合計はほぼ等しいベクトルになり、その差 $\tilde{\mathbf{q}}_k$ は桁落ちだらけの「誤差の塊」になる。それを正規化して $\mathbf{q}_k$ にすると、誤差が長さ 1 に引き伸ばされ、$\mathbf{q}_k$ は他の $\mathbf{q}_j$ と正確には直交しなくなる。誤差は後続のステップに引き継がれ、最終的に $Q$ の直交性 ($Q^\top Q = I$) が目に見えて崩れることがある。

計算の順序を工夫した**修正グラム・シュミット法** (影を 1 方向ずつ、その都度引く) でかなり改善するが、実用の数値計算ライブラリは次に述べるハウスホルダー変換を採用している。

## 5. 構成法 2: ハウスホルダー変換

### 5.1 鏡映を行列で書く

ハウスホルダー変換のアイデアは「**鏡でベクトルを狙った方向に折り返す**」ことである。まず、鏡映を行列で書く方法を導く。

平面 (一般には空間) に鏡を置く。鏡の面は原点を通るとし、鏡の面に**垂直な**ベクトルを $\mathbf{v} \neq \mathbf{0}$ とする (2 次元なら鏡は直線で、$\mathbf{v}$ はその法線ベクトル)。任意のベクトル $\mathbf{x}$ を鏡で映すとどうなるか。

$\mathbf{x}$ を「$\mathbf{v}$ 方向の成分」と「鏡の面に平行な成分」に分ける。$\mathbf{v}$ 方向の成分は、グラム・シュミット法でも使った影の計算で

$$
\left( \frac{\mathbf{v} \cdot \mathbf{x}}{\mathbf{v} \cdot \mathbf{v}} \right) \mathbf{v}
$$

である ($\mathbf{v}$ の長さが 1 とは限らないので $\mathbf{v} \cdot \mathbf{v}$ で割って調整している)。鏡映では、鏡の面に平行な成分はそのまま残り、面に垂直な成分 ($\mathbf{v}$ 方向の成分) だけが符号反転する。つまり $\mathbf{x}$ から $\mathbf{v}$ 方向の成分を**2 回**引けばよい (1 回引くと面上に落ち、もう 1 回引くと反対側に出る)。

$$
H \mathbf{x} = \mathbf{x} - 2 \left( \frac{\mathbf{v} \cdot \mathbf{x}}{\mathbf{v} \cdot \mathbf{v}} \right) \mathbf{v}
$$

これを行列の形に整理する。$\mathbf{v} \cdot \mathbf{x} = \mathbf{v}^\top \mathbf{x}$ はスカラーなので、$(\mathbf{v}^\top \mathbf{x}) \mathbf{v} = \mathbf{v} (\mathbf{v}^\top \mathbf{x}) = (\mathbf{v} \mathbf{v}^\top) \mathbf{x}$ と書き換えられる。ここで $\mathbf{v} \mathbf{v}^\top$ は「縦ベクトル × 横ベクトル」の積で、$n \times n$ 行列になることに注意する (成分で書けば第 $(i, j)$ 成分が $v_i v_j$ の行列である)。よって

$$
H = I - \frac{2}{\mathbf{v}^\top \mathbf{v}} \, \mathbf{v} \mathbf{v}^\top
$$

これが**ハウスホルダー変換** (Householder 変換、鏡映変換) である。$H$ が直交行列であることを確認しておく。まず $(\mathbf{v} \mathbf{v}^\top)^\top = \mathbf{v} \mathbf{v}^\top$ だから $H^\top = H$ (対称行列)。さらに

$$
H^2
= \left( I - \frac{2 \, \mathbf{v} \mathbf{v}^\top}{\mathbf{v}^\top \mathbf{v}} \right)^2
= I - \frac{4 \, \mathbf{v} \mathbf{v}^\top}{\mathbf{v}^\top \mathbf{v}} + \frac{4 \, \mathbf{v} (\mathbf{v}^\top \mathbf{v}) \mathbf{v}^\top}{(\mathbf{v}^\top \mathbf{v})^2}
= I - \frac{4 \, \mathbf{v} \mathbf{v}^\top}{\mathbf{v}^\top \mathbf{v}} + \frac{4 \, \mathbf{v} \mathbf{v}^\top}{\mathbf{v}^\top \mathbf{v}}
= I
$$

(途中、$\mathbf{v} \mathbf{v}^\top \mathbf{v} \mathbf{v}^\top = \mathbf{v} (\mathbf{v}^\top \mathbf{v}) \mathbf{v}^\top$ で真ん中の $\mathbf{v}^\top \mathbf{v}$ がスカラーとして外に出せることを使った)。「2 回映すと元に戻る」という鏡の当然の性質である。$H^\top H = H H = I$ となるので、$H$ は直交行列である。

### 5.2 手計算の例 — $(3, 4)$ を $(5, 0)$ に折り返す

ハウスホルダー変換を QR 分解に使う鍵は、次の事実である。

> 鏡をうまく選べば、任意のベクトル $\mathbf{x}$ を「第 1 成分以外がすべて 0」のベクトルに写せる。

鏡映は長さを変えないから、行き先は長さが同じ $\| \mathbf{x} \| \mathbf{e}_1$ (ただし $\mathbf{e}_1 = (1, 0, \dots, 0)^\top$) しかありえない。では鏡をどこに置くか。$\mathbf{x}$ と行き先 $\| \mathbf{x} \| \mathbf{e}_1$ を対称に入れ替える鏡は、幾何的に考えて 2 つのベクトルの**ちょうど中間**を通る。そのような鏡の法線ベクトルは、2 つのベクトルの差

$$
\mathbf{v} = \mathbf{x} - \| \mathbf{x} \| \mathbf{e}_1
$$

である (差のベクトルは 2 点を結ぶ方向を向き、鏡はそれと垂直だから)。

2 次元で手計算してみる。$\mathbf{x} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ とすると $\| \mathbf{x} \| = 5$ なので、目標は $\begin{bmatrix} 5 \\ 0 \end{bmatrix}$ である。

$$
\mathbf{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix} - \begin{bmatrix} 5 \\ 0 \end{bmatrix} = \begin{bmatrix} -2 \\ 4 \end{bmatrix},
\qquad
\mathbf{v}^\top \mathbf{v} = 4 + 16 = 20
$$

$$
\mathbf{v} \mathbf{v}^\top =
\begin{bmatrix} -2 \\ 4 \end{bmatrix}
\begin{bmatrix} -2 & 4 \end{bmatrix} =
\begin{bmatrix}
4 & -8 \\
-8 & 16
\end{bmatrix}
$$

$$
H = I - \frac{2}{20}
\begin{bmatrix}
4 & -8 \\
-8 & 16
\end{bmatrix} =
\begin{bmatrix}
1 - 2/5 & 4/5 \\
4/5 & 1 - 8/5
\end{bmatrix} =
\begin{bmatrix}
3/5 & 4/5 \\
4/5 & -3/5
\end{bmatrix}
$$

実際に $\mathbf{x}$ に掛けてみる。

$$
H \mathbf{x} =
\begin{bmatrix}
3/5 & 4/5 \\
4/5 & -3/5
\end{bmatrix}
\begin{bmatrix} 3 \\ 4 \end{bmatrix} =
\begin{bmatrix}
9/5 + 16/5 \\
12/5 - 12/5
\end{bmatrix} =
\begin{bmatrix} 5 \\ 0 \end{bmatrix}
$$

狙いどおり、第 2 成分が 0 になった。この $H$ が対称であること、$H^2 = I$ となること (各自検算されたい) も確認できる。図形的には、$H$ は「原点を通り、$\mathbf{x} = (3,4)$ と $(5,0)$ の中間方向 $(4, 2)$ に沿って伸びる直線」を鏡とする折り返しである。

なお実装上は符号に一工夫あり、$\mathbf{v} = \mathbf{x} + \mathrm{sign}(x_1) \| \mathbf{x} \| \mathbf{e}_1$ ($x_1$ と**同符号**を足す) を使う。$\mathbf{x}$ がもともと $\mathbf{e}_1$ に近い方向のとき、引き算版では $\mathbf{v}$ がほぼゼロベクトルになって桁落ちするからである (このとき行き先は $-\| \mathbf{x} \| \mathbf{e}_1$ になるが、支障はない)。

### 5.3 一般化 — 鏡映で 1 列ずつ潰して上三角にする

準備が整った。$A \in \mathbb{R}^{n \times m}$ を上三角にするには、鏡映を次のように繰り返せばよい。

**ステップ 1**: $A$ の第 1 列を $\mathbf{x}$ として前項の鏡映 $H_1$ を作り、$A$ 全体に掛ける。すると $H_1 A$ の第 1 列は $(r_{11}, 0, \dots, 0)^\top$ になる。第 1 列の対角より下が全部 0 になった。

**ステップ 2**: $H_1 A$ の第 1 行と第 1 列を除いた右下の $(n-1) \times (m-1)$ ブロックに注目し、そのブロックの第 1 列を潰す鏡映 $\hat{H}_2$ ($(n-1)$ 次元の鏡映) を作る。これを

$$
H_2 =
\begin{bmatrix}
1 & \mathbf{0}^\top \\
\mathbf{0} & \hat{H}_2
\end{bmatrix}
$$

と単位行列で「かさ上げ」して $n \times n$ の直交行列にし、$H_1 A$ に掛ける。$H_2$ の第 1 行・第 1 列は単位行列と同じなので、**ステップ 1 で作った第 1 列の 0 は壊れない**。こうして第 2 列も対角より下が 0 になる。

**ステップ 3 以降**: 同様に、右下の残りブロックの第 1 列を潰す操作を全部で $m$ 回繰り返す。絵で描くと ($*$ は 0 とは限らない成分)

$$
A =
\begin{bmatrix} * & * & * \\ * & * & * \\ * & * & * \\ * & * & *
\end{bmatrix}
\ \xrightarrow{H_1}\
\begin{bmatrix} * & * & * \\
0 & * & * \\
0 & * & * \\
0 & * & *
\end{bmatrix}
\ \xrightarrow{H_2}\
\begin{bmatrix} * & * & * \\
0 & * & * \\
0 & 0 & * \\
0 & 0 & *
\end{bmatrix}
\ \xrightarrow{H_3}\
\begin{bmatrix} * & * & * \\
0 & * & * \\
0 & 0 & * \\
0 & 0 & 0
\end{bmatrix}
= R
$$

のように、左の列から順に「対角より下」が消えていく。結果として

$$
H_m \cdots H_2 H_1 A = R \quad (\text{上三角})
$$

となる。両辺の左から $H_1 H_2 \cdots H_m$ を掛けると、$H_k^2 = I$ より左辺は $A$ に戻り

$$
A = \underbrace{H_1 H_2 \cdots H_m}_{Q} R
$$

を得る。直交行列の積はまた直交行列なので ($(H_1 H_2)^\top (H_1 H_2) = H_2^\top H_1^\top H_1 H_2 = I$)、$Q$ は直交行列であり、これでフル QR 分解が構成できた。

### 5.4 なぜグラム・シュミットより安定なのか

グラム・シュミットとハウスホルダーの決定的な違いは、後者が**直交行列を掛ける操作しかしていない**ことである。

グラム・シュミット法は「引き算で直交なベクトルを**作り出そう**」とするため、桁落ちが起きると直交性そのものが壊れる。一方ハウスホルダー法は、1 ステップごとに厳密な鏡映 (直交行列) を掛けているだけである。性質 2 (長さ保存) により、各ステップで途中結果に含まれる誤差は**増幅されない**。また $Q$ は「鏡映の積」という形で持つので、構造上ほぼ完璧な直交性が保証される。$Q$ の直交性は計算機の丸め誤差の水準 (機械イプシロン $\varepsilon \approx 2.2 \times 10^{-16}$、倍精度で表現できる相対誤差の最小単位) のオーダーで保たれる。

実際、本リポジトリの実装 (`src/lib.rs` の `qr_thin`、ハウスホルダー方式) をランダムな $5 \times 3$ 行列に適用すると、`src/bin/2_qr_decomposition.rs` の出力は

```
‖QᵀQ - I‖ = 7.645e-16 (直交性)
‖QR - A‖  = 9.372e-16 (再構成)
```

となり、直交性も再構成精度も機械イプシロンの数倍に収まっている。LAPACK (`geqrf`) をはじめ、NumPy や MATLAB など主要な数値計算ライブラリの QR 分解もこのハウスホルダー方式である。

## 6. QR 分解で最小二乗法を解く

いよいよ本題、最小二乗問題

$$
\min_{\boldsymbol{\beta}} \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2
\qquad (\Phi \in \mathbb{R}^{n \times m}, \ n > m)
$$

を QR 分解で解く。$\Phi$ のフル QR 分解 $\Phi = Q R$ を代入し、直交行列の性質を使って式を変形していく。

**ステップ 1: 残差を $Q$ でくくる。** $Q Q^\top = I$ (直交行列は $Q^\top Q = I$ かつ正方なので $Q Q^\top = I$ も成り立つ) を使って $\mathbf{y} = Q Q^\top \mathbf{y}$ と書き直すと

$$
\Phi \boldsymbol{\beta} - \mathbf{y}
= Q R \boldsymbol{\beta} - Q Q^\top \mathbf{y}
= Q \left( R \boldsymbol{\beta} - Q^\top \mathbf{y} \right)
$$

**ステップ 2: $Q$ を消す。** 性質 2 (直交行列は長さを変えない) より

$$
\| \Phi \boldsymbol{\beta} - \mathbf{y} \|^2
= \bigl\| Q \left( R \boldsymbol{\beta} - Q^\top \mathbf{y} \right) \bigr\|^2
= \| R \boldsymbol{\beta} - Q^\top \mathbf{y} \|^2
$$

最小化したい量が「上三角行列 $R$ だけの問題」に化けた。ここが QR 分解の見せ場である。**この変形で使ったのは直交行列だけなので、問題の性質 (誤差の増幅されやすさ) は一切悪化していない**。

**ステップ 3: 上下に分割する。** $R = \begin{bmatrix} R_1 \\ O \end{bmatrix}$ ($R_1$ は $m \times m$ 上三角) だったことを思い出し、$Q^\top \mathbf{y}$ も同じ高さで

$$
Q^\top \mathbf{y} =
\begin{bmatrix}
\mathbf{c}_1 \\ \mathbf{c}_2
\end{bmatrix}
\qquad (\mathbf{c}_1 \in \mathbb{R}^m, \ \mathbf{c}_2 \in \mathbb{R}^{n-m})
$$

と分割する。すると

$$
R \boldsymbol{\beta} - Q^\top \mathbf{y} =
\begin{bmatrix}
R_1 \boldsymbol{\beta} - \mathbf{c}_1 \\ - \mathbf{c}_2
\end{bmatrix}
$$

なので、ノルムの二乗は上下のブロックごとの和になり ([0. 数学の準備](./0_math_preliminaries.md) のノルムの定義から、成分の二乗和は分けて足せる)

$$
\| \Phi \boldsymbol{\beta} - \mathbf{y} \|^2
= \| R_1 \boldsymbol{\beta} - \mathbf{c}_1 \|^2 + \| \mathbf{c}_2 \|^2
$$

**ステップ 4: 最小化する。** 第 2 項 $\| \mathbf{c}_2 \|^2$ は $\boldsymbol{\beta}$ を含まない定数である。第 1 項は 0 以上で、$R_1$ が可逆 ($\Phi$ の列が線形独立なら対角成分がすべて非零) なら

$$
R_1 \boldsymbol{\beta} = \mathbf{c}_1
$$

を満たす $\boldsymbol{\beta}$ でちょうど 0 にできる。これは上三角の連立方程式なので、第 3.2 節の**後退代入で即座に解ける**。しかも、最小化してもなお残る誤差 (残差のノルム) が $\| \mathbf{c}_2 \|$ として**計算の副産物で手に入る**。

まとめると、QR 分解による最小二乗法の手順は

1. $\Phi = QR$ と分解する (ハウスホルダー変換で)
2. $Q^\top \mathbf{y}$ を計算し、上 $m$ 成分 $\mathbf{c}_1$ を取り出す (実装では $\mathbf{y}$ に鏡映 $H_m \cdots H_1$ を順に掛けるだけでよく、$Q$ を行列として組み立てる必要すらない。`src/lib.rs` の `lstsq_qr` はそう実装されている)
3. $R_1 \boldsymbol{\beta} = \mathbf{c}_1$ を後退代入で解く

である。**どこにも $\Phi^\top \Phi$ が現れない**ことに注目してほしい。

最後に、この解が正規方程式の解と同じものであることを確認しておく。薄い QR 分解 $\Phi = Q_1 R_1$ を正規方程式に代入すると、$Q_1^\top Q_1 = I$ より

$$
\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}
\ \Longleftrightarrow\
R_1^\top Q_1^\top Q_1 R_1 \boldsymbol{\beta} = R_1^\top Q_1^\top \mathbf{y}
\ \Longleftrightarrow\
R_1^\top R_1 \boldsymbol{\beta} = R_1^\top \mathbf{c}_1
\ \Longleftrightarrow\
R_1 \boldsymbol{\beta} = \mathbf{c}_1
$$

(最後の同値変形で $R_1^\top$ が可逆であることを使った)。つまり $R_1 \boldsymbol{\beta} = \mathbf{c}_1$ は正規方程式と数学的に**同じ方程式**である。では何が違うのか。違いは「計算機上での誤差の増幅のされ方」であり、それを測る道具が次節の条件数である。

## 7. 条件数 — 誤差はどれだけ増幅されるか

### 7.1 まず体感する — 2×2 の悪条件な連立方程式

次の連立方程式を解いてみてほしい。

$$
\begin{cases}
x_1 + x_2 = 2 \\
x_1 + 1.0001 \, x_2 = 2.0001
\end{cases}
$$

2 本目から 1 本目を引くと $0.0001 \, x_2 = 0.0001$ なので $x_2 = 1$、よって $x_1 = 1$。解は $(x_1, x_2) = (1, 1)$ である。

ここで、右辺の 2.0001 がほんの少しずれて 2.0002 になったとしよう (測定誤差や丸め誤差を想定した、わずか **0.005 %** の変化である)。

$$
\begin{cases}
x_1 + x_2 = 2 \\
x_1 + 1.0001 \, x_2 = 2.0002
\end{cases}
$$

同じように引き算すると $0.0001 \, x_2 = 0.0002$ で $x_2 = 2$、$x_1 = 0$。解は $(0, 2)$ に**激変**した。入力の 0.005 % のずれが、出力では 100 % のずれ (解が原形をとどめないほどの変化) になった。増幅率はおよそ 2 万倍である。

なぜこんなことが起きるのか。この連立方程式の 2 本の式は、グラフに描くと**ほとんど平行な 2 本の直線**である。ほとんど平行な直線同士の交点は、片方の直線がわずかに平行移動しただけで大きく滑る。これが「悪条件」の幾何的な正体である。

観測データにも計算途中の数にも誤差は必ず含まれるから、「入力の誤差が出力でどれだけ増幅されるか」は、解の信頼性を左右する死活問題である。この増幅率の上限を表す指標が条件数である。

### 7.2 条件数の定義

正方行列 $A$ (可逆とする) は、ベクトルを方向によって異なる倍率で伸縮させる。倍率 $\frac{\| A \mathbf{x} \|}{\| \mathbf{x} \|}$ の最大値と最小値を

$$
\sigma_{\max} = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\| A \mathbf{x} \|}{\| \mathbf{x} \|},
\qquad
\sigma_{\min} = \min_{\mathbf{x} \neq \mathbf{0}} \frac{\| A \mathbf{x} \|}{\| \mathbf{x} \|}
$$

とおく (最も引き伸ばされる方向の倍率と、最も潰される方向の倍率である)。このとき

> **定義 (条件数)**:
>
> $$
> \kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
> $$
>
> を $A$ の**条件数** (condition number) と呼ぶ。$\kappa$ が大きい行列を**悪条件** (ill-conditioned)、小さい ($1$ に近い) 行列を**良条件** (well-conditioned) という。

条件数が「誤差の増幅率」であることを導出する。連立方程式 $A \mathbf{x} = \mathbf{b}$ を考え、右辺に誤差 $\Delta \mathbf{b}$ が乗ったときの解のずれを $\Delta \mathbf{x}$ とする。すなわち $A (\mathbf{x} + \Delta \mathbf{x}) = \mathbf{b} + \Delta \mathbf{b}$ である。元の式を引くと

$$
A \, \Delta \mathbf{x} = \Delta \mathbf{b}
$$

ここで 2 つの不等式を作る。

1. $\| \mathbf{b} \| = \| A \mathbf{x} \| \leq \sigma_{\max} \| \mathbf{x} \|$。移項すると $\dfrac{1}{\| \mathbf{x} \|} \leq \dfrac{\sigma_{\max}}{\| \mathbf{b} \|}$
2. $\| \Delta \mathbf{b} \| = \| A \, \Delta \mathbf{x} \| \geq \sigma_{\min} \| \Delta \mathbf{x} \|$。移項すると $\| \Delta \mathbf{x} \| \leq \dfrac{\| \Delta \mathbf{b} \|}{\sigma_{\min}}$

2 つを掛け合わせると

$$
\frac{\| \Delta \mathbf{x} \|}{\| \mathbf{x} \|}
\ \leq\
\frac{\sigma_{\max}}{\sigma_{\min}} \cdot \frac{\| \Delta \mathbf{b} \|}{\| \mathbf{b} \|}
= \kappa(A) \, \frac{\| \Delta \mathbf{b} \|}{\| \mathbf{b} \|}
$$

すなわち

> **出力 (解) の相対誤差は、入力 (右辺) の相対誤差の高々 $\kappa(A)$ 倍である。**

しかもこの上限は絵空事ではない。$\mathbf{b}$ が最も伸ばされる方向、$\Delta \mathbf{b}$ が最も潰される方向を向いたときには等号が成立する。つまり $\kappa(A)$ は「最悪ケースの増幅率そのもの」である。

前項の例で確かめよう。$A = \begin{bmatrix} 1 & 1 \\ 1 & 1.0001 \end{bmatrix}$ の条件数を計算すると $\kappa(A) \approx 4.0 \times 10^4$ である。実際の増幅率は、入力の相対誤差が $\frac{\| \Delta \mathbf{b} \|}{\| \mathbf{b} \|} = \frac{0.0001}{\sqrt{2^2 + 2.0001^2}} \approx 3.5 \times 10^{-5}$、出力の相対誤差が $\frac{\| \Delta \mathbf{x} \|}{\| \mathbf{x} \|} = \frac{\| (-1, 1) \|}{\| (1, 1) \|} = 1$ なので、約 $2.8 \times 10^4$ 倍。確かに $\kappa(A)$ 倍の少し内側に収まっており、上限の見積もりが実態とよく合っている。

条件数は丸め誤差にも同じように効く。倍精度浮動小数点数の相対誤差は約 $10^{-16}$ (有効数字約 16 桁) なので、粗い目安として

> $\kappa(A) = 10^k$ の問題を解くと、答えの有効数字は約 $16 - k$ 桁に減る

と考えてよい。$\kappa = 10^{16}$ に達すると有効数字は 0 桁、つまり**答えのどの桁も信用できない**。

### 7.3 直交行列の条件数は 1

直交行列 $Q$ は性質 2 よりすべてのベクトルの長さを変えない。つまりすべての方向で倍率が 1 であり、$\sigma_{\max} = \sigma_{\min} = 1$。よって

$$
\kappa(Q) = 1
$$

これは条件数の理論上の最小値である ($\sigma_{\max} \geq \sigma_{\min}$ より $\kappa \geq 1$)。直交行列は「誤差を 1 ミリも増幅しない、数値計算上最も安全な行列」だという第 2 節の主張が、条件数の言葉で正当化された。

### 7.4 正規方程式は条件数を二乗にする

さて、正規方程式の何が問題なのか。正規方程式では係数行列として $\Phi^\top \Phi$ を使う。この行列の条件数を調べる。

$\Phi$ がベクトル $\mathbf{x}$ を $\sigma$ 倍に伸ばすとき、$\| \Phi \mathbf{x} \|^2 = \sigma^2 \| \mathbf{x} \|^2$ である。一方 [0. 数学の準備](./0_math_preliminaries.md) の公式を使うと

$$
\| \Phi \mathbf{x} \|^2 = (\Phi \mathbf{x}) \cdot (\Phi \mathbf{x}) = \mathbf{x} \cdot (\Phi^\top \Phi \mathbf{x})
$$

なので、$\Phi^\top \Phi$ は $\mathbf{x}$ 方向に対して「$\sigma^2$ 倍」の働きをすることが読み取れる。実際、$\Phi$ が最も伸ばす方向で $\Phi^\top \Phi$ は $\sigma_{\max}^2$ 倍、最も潰す方向で $\sigma_{\min}^2$ 倍に作用することが示せる (きちんとした証明には [3. 特異値分解](./3_singular_value_decomposition.md) の道具を使うのが見通しよい)。したがって

$$
\kappa(\Phi^\top \Phi) = \frac{\sigma_{\max}^2}{\sigma_{\min}^2} = \kappa(\Phi)^2
$$

**条件数が二乗になる。** これがどれほど破壊的かを、前項の目安 (有効数字 $\approx 16 - k$ 桁) で見積もる。例えば $\kappa(\Phi) = 10^8$ の問題では

- QR 分解で解けば: 増幅率 $10^8$、有効数字は約 $16 - 8 = 8$ 桁残る
- 正規方程式で解けば: 増幅率 $\kappa(\Phi^\top \Phi) = 10^{16}$、有効数字は約 $16 - 16 = 0$ 桁。**全滅**

同じ問題・同じ計算機でも、解き方の選択だけで「8 桁正しい答え」と「1 桁も正しくない答え」に分かれるのである。しかも重要なのは、この悪化が丸め誤差などではなく、**$\Phi^\top \Phi$ という行列を作った瞬間に確定する構造的な劣化**だという点である。掛け算を無限精度で実行したとしても、$\Phi^\top \Phi$ を係数とする連立方程式は $\kappa^2$ の増幅率を持つ問題になってしまっている。元の $\Phi$ が持っていた情報の一部は、この時点で失われて二度と戻らない。

### 7.5 QR 分解は条件数を悪化させない

一方、第 6 節の QR 分解による解法を振り返ると、行った変形は

1. $\Phi = QR$ と分解する — $\Phi$ 自体は変更していない
2. $Q^\top$ を掛ける — 直交行列 ($\kappa = 1$) を掛けただけで、誤差は増幅されない
3. $R_1 \boldsymbol{\beta} = \mathbf{c}_1$ を解く — この方程式の条件数は $\kappa(R_1)$

であり、最後の $\kappa(R_1)$ も悪化していない。なぜなら $\Phi = Q_1 R_1$ で $Q_1$ が長さを変えないことから、任意の $\mathbf{x}$ に対して

$$
\| \Phi \mathbf{x} \| = \| Q_1 (R_1 \mathbf{x}) \| = \| R_1 \mathbf{x} \|
$$

つまり $\Phi$ と $R_1$ はすべての方向で**同じ伸縮率**を持ち、$\sigma_{\max}$ も $\sigma_{\min}$ も一致するから

$$
\kappa(R_1) = \kappa(\Phi)
$$

である。まとめると

| 解き方 | 解くべき方程式 | 誤差の増幅率 |
| --- | --- | --- |
| 正規方程式 | $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$ | $\kappa(\Phi)^2$ |
| QR 分解 | $R_1 \boldsymbol{\beta} = \mathbf{c}_1$ | $\kappa(\Phi)$ |

数学的には同じ解を指す 2 つの方程式が、数値的にはまるで別物なのである。

## 8. 実験 — 悪条件問題で正規方程式と QR を対決させる

理屈は分かった。では実際のコードでどれほど差が出るのか。`src/bin/2_qr_decomposition.rs` の実験を見ていく。

### 8.1 実験設定 — ヴァンデルモンド行列による多項式フィット

[1. 線形最小二乗法](./1_least_squares_method.md) で見たとおり、$m-1$ 次多項式によるフィッティング $\hat{y}(x) = \beta_1 + \beta_2 x + \cdots + \beta_m x^{m-1}$ は、基底関数を $\phi_k(x) = x^{k-1}$ とした線形最小二乗法である。このときの計画行列

$$
\Phi =
\begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^{m-1} \\
1 & x_2 & x_2^2 & \cdots & x_2^{m-1} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_n & x_n^2 & \cdots & x_n^{m-1}
\end{bmatrix}
$$

は**ヴァンデルモンド行列** (Vandermonde matrix) と呼ばれ、悪条件な行列の代表例として有名である。なぜ悪条件になるのか。データ点を区間 $[0, 1]$ に取ると、高次の列 $x^8, x^9, x^{10}, \dots$ はどれも「$x = 1$ 付近で急に立ち上がる、ほとんど同じ形の曲線」になる。列同士がほぼ平行、つまり**ほぼ線形従属**なのである。ほぼ平行な列を持つ行列は、その差の方向のベクトルを極端に潰す ($\sigma_{\min}$ が極小になる) ため、条件数が跳ね上がる。7.1 節の「ほぼ平行な 2 直線」の高次元版だと思えばよい。

実験は次のように仕組まれている。

- データ点: $x_i = \frac{i}{49}$ ($i = 0, 1, \dots, 49$)、つまり $[0, 1]$ の等間隔 50 点
- 真の関数: $y = 1 + x + x^2 + \cdots + x^{m-1}$ (**係数がすべて 1**)。$\mathbf{y}$ はこの式の厳密な値で作り、ノイズは加えない
- 課題: $\mathbf{y}$ と $\Phi$ から係数を推定し、真値 1 からの最大ずれ $\max_k | \beta_k - 1 |$ を測る

正解が厳密に分かっている (すべて 1) ので、誤差を直接測定できる。ノイズを入れていないから、理想の計算機なら誤差 0 で復元できるはずである。実際に出る誤差は、すべて浮動小数点演算と条件数のせいだ、という寸法である。これを

- **正規方程式**: $\Phi^\top \Phi$ と $\Phi^\top \mathbf{y}$ を作り、掃き出し法で解く
- **QR 分解**: `lstsq_qr` (ハウスホルダー QR + 後退代入) で解く

の 2 通りで解き比べる。

### 8.2 実験結果

実行結果 (表の 2 列は係数の最大誤差) に、別途計算した $\kappa(\Phi)$ の参考値を並べる。

| 次数 $m-1$ | $\kappa(\Phi)$ (参考値) | $\kappa(\Phi)^2$ | 正規方程式の誤差 | QR 分解の誤差 |
| --- | --- | --- | --- | --- |
| 3 | $1.2 \times 10^2$ | $1.4 \times 10^4$ | $1.295 \times 10^{-13}$ | $5.107 \times 10^{-15}$ |
| 7 | $1.1 \times 10^5$ | $1.2 \times 10^{10}$ | $1.452 \times 10^{-8}$ | $4.423 \times 10^{-12}$ |
| 11 | $1.2 \times 10^8$ | $1.5 \times 10^{16}$ | $2.495 \times 10^{-1}$ | $2.197 \times 10^{-9}$ |

### 8.3 結果を読み解く

7.2 節の目安「$\kappa = 10^k$ なら有効数字 $16 - k$ 桁」と照らし合わせる。

**次数 3** ($\kappa \approx 10^2$): まだ良条件で、どちらの方法もほぼ機械精度で解けている。それでも正規方程式 ($\kappa^2 \approx 10^4$) の誤差は QR の約 25 倍あり、差の芽は既に出ている。

**次数 7** ($\kappa \approx 10^5$): QR の誤差 $4.4 \times 10^{-12}$ は目安 ($16 - 5 = 11$ 桁 → 誤差 $10^{-11}$ 程度) とよく合う。正規方程式は $\kappa^2 \approx 10^{10}$ が効いて誤差 $1.5 \times 10^{-8}$、QR より **4 桁**悪い。それでもまだ「係数はほぼ 1」と分かる水準ではある。

**次数 11** ($\kappa \approx 10^8$): 勝負が決した行である。

- QR 分解: 誤差 $2.2 \times 10^{-9}$。目安どおり ($16 - 8 = 8$ 桁 → 誤差 $10^{-8}$ 程度) で、**約 8 桁の有効数字**を保って係数を復元できている。
- 正規方程式: $\kappa^2 \approx 1.5 \times 10^{16}$ が倍精度の限界 $10^{16}$ を超えた。誤差は $2.5 \times 10^{-1}$、つまり「1 のはずの係数が 0.75 や 1.25 になる」水準で、**もはや 1 桁も正しくない**。答えとして使い物にならない。

同じデータ、同じ 64 bit 浮動小数点数を使っていながら、$\Phi^\top \Phi$ を経由したかどうかだけで「8 桁正確」と「全桁デタラメ」に分かれた。これが第 7 節の理論、$\kappa$ 対 $\kappa^2$ の差の実演である。[1. 線形最小二乗法](./1_least_squares_method.md) 第 6 節で「ライブラリの `lstsq` は内部で QR や SVD を使う。自前で正規方程式を組んではいけない」と述べた理由が、この表に凝縮されている。

## 9. 一意性とランク落ち

最後に、理論面の補足を 2 点述べる。

**一意性。** $A$ の列が線形独立なら、「$R_1$ の対角成分をすべて正に取る」という規約の下で薄い QR 分解は**一意**に定まる。実際、グラム・シュミット法の各ステップに選択の余地はない ($r_{kk} = \| \tilde{\mathbf{q}}_k \| > 0$ と決めれば $\mathbf{q}_k$ が一意に決まる)。規約を外すと、各列の符号を同時に反転する分 ($\mathbf{q}_k \to -\mathbf{q}_k$, $r_{kk} \to -r_{kk}$) の自由度だけが残る。

**ランク落ち。** 列が線形従属 ($\mathrm{rank} \, A < m$。ランクについては [0. 数学の準備](./0_math_preliminaries.md) を参照) の場合、グラム・シュミット法はある $k$ で $\tilde{\mathbf{q}}_k = \mathbf{0}$ となり正規化できず破綻する。$R_1$ も可逆でなくなり、後退代入が実行できない。数値計算では、厳密なランク落ちでなくても「ほぼ従属」なら同じ困難が生じる。実用上は**列ピボット付き QR 分解** ($A \Pi = QR$、$\Pi$ は列の並び替えを表す置換行列) を使い、残りの列のうち最も「新しい方向」を持つ列を毎回先頭に選ぶ。すると $R$ の対角成分が大きい順に並び、対角成分が急激に小さくなる位置から数値的なランクを推定できる。さらに信頼性の高いランク判定と、ランク落ち問題の解 (ノルム最小解) が必要な場合は [3. 特異値分解](./3_singular_value_decomposition.md) の出番である。

## 10. まとめ

> QR 分解 $A = QR$ は、行列を「空間を一切歪めない直交行列 $Q$ ($\kappa = 1$)」と「後退代入で解ける上三角行列 $R$」に分ける分解であり、$\Phi^\top \Phi$ を作らずに (=条件数を二乗に悪化させずに) 最小二乗問題を解くための標準的な道具である。

- **直交行列**は長さと角度を変えない変換 (回転・鏡映) であり、その条件を式にすると $Q^\top Q = I$。逆行列が転置で済み、誤差を増幅しない ($\kappa(Q) = 1$)。
- **グラム・シュミット法** (影を引き算して直交化) は QR 分解の存在と意味を教えてくれるが、桁落ちに弱い。**ハウスホルダー変換** (鏡映で 1 列ずつ対角より下を潰す) は直交行列を掛ける操作しかしないため数値的に安定で、実用ライブラリはこちらを使う。
- 最小二乗問題は $\| \Phi \boldsymbol{\beta} - \mathbf{y} \|^2 = \| R_1 \boldsymbol{\beta} - \mathbf{c}_1 \|^2 + \| \mathbf{c}_2 \|^2$ と分解でき、後退代入 $R_1 \boldsymbol{\beta} = \mathbf{c}_1$ に帰着する。残差ノルム $\| \mathbf{c}_2 \|$ も副産物で得られる。
- **条件数** $\kappa(A) = \sigma_{\max} / \sigma_{\min}$ は入力の相対誤差が出力で増幅される最悪倍率。正規方程式は $\kappa(\Phi)^2$、QR 経由は $\kappa(\Phi)$ で効き、$\kappa(\Phi) \approx 10^8$ の実験 (次数 11 の多項式フィット) では誤差 $2.5 \times 10^{-1}$ 対 $2.2 \times 10^{-9}$ という決定的な差になった (`src/bin/2_qr_decomposition.rs`)。
- 後続の [7. ガウス・ニュートン法](./7_gauss_newton_method.md) や [8. レーベンバーグ・マーカート法](./8_levenberg_marquardt_method.md) は、反復のたびに線形最小二乗の部分問題を解く。その部分問題を安定に解く土台として、本文書の QR 分解が使われる。
