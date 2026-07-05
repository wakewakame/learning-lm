# 数学の準備 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

この文書は、[1. 線形最小二乗法](./1_least_squares_method.md) から [8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) までを読むために必要な、大学初年級の数学をまとめたものである。前提とするのは日本の高校数学 (1 変数の微分積分、平面・空間ベクトル、$\Sigma$ 記法、二次関数) までであり、行列・$n$ 次元ベクトル・偏微分は未習として一から説明する。高校で学んだ「2 次元・3 次元のベクトル」「1 変数の微分」を、「$n$ 次元」「多変数」へ橋渡しすることがこの文書の役割である。

## 1. Σ記法の復習

**どこで使うか。** 最小二乗法の目的関数は「残差の二乗和」$E = \sum_{i=1}^{n} r_i^2$ であり、以降のすべての文書で $\Sigma$ 記法が登場する。まずここを確実にしておく。

### 1.1 定義

$\Sigma$ (シグマ) 記法は「番号付きの数を順に足し合わせる」ことを表す記号である。

$$
\sum_{i=1}^{n} a_i = a_1 + a_2 + \cdots + a_n
$$

読み方は「$i$ を $1$ から $n$ まで動かしながら $a_i$ を足す」。具体例を挙げる。

$$
\sum_{i=1}^{4} i^2 = 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
$$

添字 $i$ は「足し合わせるための番号」にすぎず、別の文字に変えても意味は変わらない (**ダミー変数**という)。

$$
\sum_{i=1}^{n} a_i = \sum_{k=1}^{n} a_k
$$

### 1.2 計算規則

高校で学んだとおり、$\Sigma$ は次の規則を満たす。$c$ は $i$ によらない定数とする。

$$
\sum_{i=1}^{n} ( a_i + b_i ) = \sum_{i=1}^{n} a_i + \sum_{i=1}^{n} b_i,
\qquad
\sum_{i=1}^{n} c \, a_i = c \sum_{i=1}^{n} a_i,
\qquad
\sum_{i=1}^{n} c = n c
$$

注意すべきは、**積は分配できない**ことである。

$$
\sum_{i=1}^{n} a_i b_i \ \neq\ \Bigl( \sum_{i=1}^{n} a_i \Bigr) \Bigl( \sum_{i=1}^{n} b_i \Bigr)
$$

例えば $n = 2, \ a_1 = a_2 = b_1 = b_2 = 1$ のとき、左辺は $1 + 1 = 2$ だが右辺は $2 \times 2 = 4$ である。

### 1.3 二重和

$\Sigma$ の中にさらに $\Sigma$ が入ることがある (行列の積の計算で使う)。

$$
\sum_{i=1}^{2} \sum_{j=1}^{2} a_i b_j
= \sum_{i=1}^{2} ( a_i b_1 + a_i b_2 )
= a_1 b_1 + a_1 b_2 + a_2 b_1 + a_2 b_2
$$

内側から順に展開すればよい。足す順序を入れ替えても総和は変わらないので、$\sum_i \sum_j$ と $\sum_j \sum_i$ は等しい。

## 2. ベクトル

**どこで使うか。** [1. 線形最小二乗法](./1_least_squares_method.md) では $n$ 個の観測値をまとめた $n$ 次元ベクトル $\mathbf{y}$ が主役になる。「残差の二乗和」はベクトルのノルム (長さ) として、「正規方程式」は直交条件として理解される。内積・ノルム・直交はロードマップ全体の土台である。

### 2.1 n 次元ベクトル

高校では平面ベクトル $(a_1, a_2)$ と空間ベクトル $(a_1, a_2, a_3)$ を学んだ。これを素直に延長し、$n$ 個の実数を縦に並べたもの

$$
\mathbf{a} =
\begin{bmatrix}
a_1 \\ a_2 \\ \vdots \\ a_n
\end{bmatrix}
$$

を **$n$ 次元ベクトル**と呼ぶ。$n$ 次元ベクトル全体の集合を $\mathbb{R}^n$ と書き、$\mathbf{a} \in \mathbb{R}^n$ で「$\mathbf{a}$ は $n$ 次元ベクトルである」ことを表す ($\mathbb{R}$ は実数全体、$\in$ は「〜に属する」の意味)。各 $a_i$ を**成分**という。

$n = 2, 3$ なら矢印として図に描けるが、$n \geq 4$ では図に描けない。それでも困らない。**「数を $n$ 個並べたもの」を 1 つの対象として扱い、2 次元・3 次元と同じ計算規則を適用する**、というのが $n$ 次元ベクトルの考え方である。例えば 100 個の観測値 $y_1, \dots, y_{100}$ は、100 次元ベクトル $\mathbf{y} \in \mathbb{R}^{100}$ という「1 本のベクトル」にまとめられる。図には描けなくても、長さや直交といった幾何の言葉が (後述の定義によって) そのまま通用する。

この文書ではベクトルを $\mathbf{a}, \mathbf{y}$ のような太字で書き、断りがなければ縦に並べた**列ベクトル**とする。すべての成分が $0$ のベクトルを**ゼロベクトル** $\mathbf{0}$ と書く。

### 2.2 和とスカラー倍

和とスカラー倍 (実数倍) は成分ごとに行う。高校の 2 次元・3 次元の場合とまったく同じ規則である。

$$
\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
+
\begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}
=
\begin{bmatrix} 5 \\ 7 \\ 9 \end{bmatrix},
\qquad
2 \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}
=
\begin{bmatrix} 2 \\ 4 \\ 6 \end{bmatrix}
$$

一般に $\mathbf{a} + \mathbf{b}$ の第 $i$ 成分は $a_i + b_i$、$c \mathbf{a}$ の第 $i$ 成分は $c a_i$ である。

ベクトルをスカラー倍して足し合わせたもの

$$
c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2 + \cdots + c_m \mathbf{a}_m
$$

を $\mathbf{a}_1, \dots, \mathbf{a}_m$ の**線形結合**と呼ぶ。この言葉は第 4 章 (線形独立) と、[1. 線形最小二乗法](./1_least_squares_method.md) の「$\Phi \boldsymbol{\beta}$ は $\Phi$ の列ベクトルの線形結合」という見方で使う。

### 2.3 内積

**成分による定義。** 2 つのベクトル $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ の**内積**を、対応する成分の積の和で定義する。

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n
$$

例えば

$$
\begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} 3 \\ 4 \end{bmatrix}
= 1 \times 3 + 2 \times 4 = 11
$$

内積の結果はベクトルではなく **1 つの実数 (スカラー)** であることに注意する。定義から次の性質が直ちに従う。

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a},
\qquad
\mathbf{a} \cdot ( \mathbf{b} + \mathbf{c} ) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c},
\qquad
( c \mathbf{a} ) \cdot \mathbf{b} = c \, ( \mathbf{a} \cdot \mathbf{b} )
$$

つまり内積は、普通の数の掛け算とよく似た感覚で展開・整理してよい。

**幾何による意味。** 高校では内積を $\mathbf{a} \cdot \mathbf{b} = \| \mathbf{a} \| \| \mathbf{b} \| \cos \theta$ ($\theta$ は 2 つのベクトルのなす角) と学んだ。この 2 つの定義が一致することを、2 次元で確かめる。余弦定理より、$\mathbf{a}$ と $\mathbf{b}$ の終点を結ぶ辺の長さ $\| \mathbf{a} - \mathbf{b} \|$ について

$$
\| \mathbf{a} - \mathbf{b} \|^2 = \| \mathbf{a} \|^2 + \| \mathbf{b} \|^2 - 2 \| \mathbf{a} \| \| \mathbf{b} \| \cos \theta
$$

が成り立つ。一方、左辺を成分で展開すると

$$
\begin{aligned}
\| \mathbf{a} - \mathbf{b} \|^2
&= ( a_1 - b_1 )^2 + ( a_2 - b_2 )^2 \\
&= a_1^2 - 2 a_1 b_1 + b_1^2 + a_2^2 - 2 a_2 b_2 + b_2^2 \\
&= \| \mathbf{a} \|^2 + \| \mathbf{b} \|^2 - 2 ( a_1 b_1 + a_2 b_2 )
\end{aligned}
$$

となる。2 つの式を見比べれば

$$
a_1 b_1 + a_2 b_2 = \| \mathbf{a} \| \| \mathbf{b} \| \cos \theta
$$

すなわち成分の定義と cos の定義は同じものである。3 次元でも同様に確かめられる。

$n \geq 4$ では「角度」を目で見ることはできないが、話を逆転させて、**成分の定義 $\sum_i a_i b_i$ を出発点とし、$\cos \theta = \dfrac{\mathbf{a} \cdot \mathbf{b}}{\| \mathbf{a} \| \| \mathbf{b} \|}$ によって角度を定義する**。こうして $n$ 次元でも「なす角」「直交」という幾何の言葉が使えるようになる。この値が必ず $-1$ 以上 $1$ 以下に収まること (コーシー・シュワルツの不等式 $| \mathbf{a} \cdot \mathbf{b} | \leq \| \mathbf{a} \| \| \mathbf{b} \|$) が知られており、この定義は矛盾なく機能する。

覚えておくべき直感は次の 1 行である。

> 内積は「2 つのベクトルがどれだけ同じ方向を向いているか」を測る量である。同じ向きなら正、直角なら $0$、逆向きなら負になる。

この直感は [5. 最急降下法](./5_steepest_descent.md) で「勾配と逆向きに進めば関数は減る」ことを理解する鍵になる。

### 2.4 ノルムと 2 点間距離

ベクトル $\mathbf{a}$ の**ノルム** (長さ) を

$$
\| \mathbf{a} \| = \sqrt{ \mathbf{a} \cdot \mathbf{a} } = \sqrt{ \sum_{i=1}^{n} a_i^2 }
$$

で定義する。$n = 2$ なら $\| \mathbf{a} \| = \sqrt{ a_1^2 + a_2^2 }$ で、三平方の定理による矢印の長さそのものである。例えば

$$
\left\| \begin{bmatrix} 3 \\ 4 \end{bmatrix} \right\| = \sqrt{ 3^2 + 4^2 } = \sqrt{25} = 5
$$

2 つのベクトル $\mathbf{a}, \mathbf{b}$ を点とみなしたときの **2 点間距離**は、差のノルム $\| \mathbf{a} - \mathbf{b} \|$ で測る。

最小二乗法との関係はここで見えてくる。残差ベクトル $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$ に対して

$$
\| \mathbf{r} \|^2 = \sum_{i=1}^{n} r_i^2
$$

すなわち**残差の二乗和とは、観測 $\mathbf{y}$ と予測 $\hat{\mathbf{y}}$ の距離の二乗**である。「二乗和を最小化する」ことは「$n$ 次元空間の中で最も近い点を探す」ことだ、という幾何的な見方が、[1. 線形最小二乗法](./1_least_squares_method.md) の導出の核心になる。

ノルムには次の基本性質がある。$\| \mathbf{a} \| \geq 0$ であり、$\| \mathbf{a} \| = 0$ となるのは $\mathbf{a} = \mathbf{0}$ のときに限る。また $\| c \mathbf{a} \| = | c | \, \| \mathbf{a} \|$ である。

### 2.5 直交

$\mathbf{a} \cdot \mathbf{b} = 0$ のとき、$\mathbf{a}$ と $\mathbf{b}$ は**直交する**という ($\cos \theta = 0$、すなわち直角)。例えば

$$
\begin{bmatrix} 1 \\ 2 \end{bmatrix} \cdot \begin{bmatrix} -2 \\ 1 \end{bmatrix}
= 1 \times (-2) + 2 \times 1 = 0
$$

直交するベクトルには**三平方の定理**が成り立つ。$\mathbf{a} \cdot \mathbf{b} = 0$ ならば

$$
\begin{aligned}
\| \mathbf{a} + \mathbf{b} \|^2
&= ( \mathbf{a} + \mathbf{b} ) \cdot ( \mathbf{a} + \mathbf{b} ) \\
&= \mathbf{a} \cdot \mathbf{a} + 2 \, \mathbf{a} \cdot \mathbf{b} + \mathbf{b} \cdot \mathbf{b} \\
&= \| \mathbf{a} \|^2 + \| \mathbf{b} \|^2
\end{aligned}
$$

この計算は [1. 線形最小二乗法](./1_least_squares_method.md) の幾何的導出 (「垂線の足が最も近い」) でそのまま使われる。また [2. QR 分解](./2_qr_decomposition.md) は「行列の列ベクトルを互いに直交するように作り直す」手法であり、直交はそこでも主役である。

## 3. 行列

**どこで使うか。** [1. 線形最小二乗法](./1_least_squares_method.md) の計画行列 $\Phi$、正規方程式 $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$、[4. 非線形最小二乗法](./4_nonlinear_least_squares.md) 以降のヤコビ行列 $J$ と、行列はすべての文書に登場する。ここでは「行列とは何か」「行列の積は何を意味するか」を具体例から積み上げる。

### 3.1 行列とは何か

**行列**とは、数を長方形に並べた表である。例えば

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
$$

は、横の並び (**行**) が 3 つ、縦の並び (**列**) が 2 つの行列で、**$3 \times 2$ 行列**と呼ぶ (常に「行 × 列」の順)。$n$ 行 $m$ 列の実数の行列全体を $\mathbb{R}^{n \times m}$ と書く。第 $i$ 行・第 $j$ 列の成分を $A_{ij}$ と書く。上の例では $A_{31} = 5$ である。

列ベクトルは「列が 1 つだけの行列」($n \times 1$ 行列) とみなせる。この見方は後で行列の積を統一的に扱うときに役立つ。

行列の和とスカラー倍は、ベクトルと同様に成分ごとに定義する (和は同じサイズの行列どうしのみ)。

### 3.2 行列 × ベクトル

**定義。** $n \times m$ 行列 $A$ と $m$ 次元ベクトル $\mathbf{x}$ の積 $A \mathbf{x}$ は $n$ 次元ベクトルで、その第 $i$ 成分は「$A$ の第 $i$ 行と $\mathbf{x}$ の内積」である。

$$
( A \mathbf{x} )_i = \sum_{j=1}^{m} A_{ij} \, x_j
$$

まず $2 \times 2$ で手を動かす。

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 5 \\ 6 \end{bmatrix}
=
\begin{bmatrix} 1 \times 5 + 2 \times 6 \\ 3 \times 5 + 4 \times 6 \end{bmatrix}
=
\begin{bmatrix} 17 \\ 39 \end{bmatrix}
$$

サイズには注意が要る。$A$ の列数と $\mathbf{x}$ の次元が一致していなければ積は定義されない。$(n \times m) \times (m \text{ 次元}) \to n \text{ 次元}$ と、内側の $m$ が消える形になる。

**見方 1: 連立一次方程式の省略記法。** 連立方程式

$$
\begin{cases}
x + 2y = 5 \\
3x + 4y = 6
\end{cases}
$$

は、上の積の定義を使うと

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} x \\ y \end{bmatrix}
=
\begin{bmatrix} 5 \\ 6 \end{bmatrix}
$$

と 1 本の式 $A \mathbf{x} = \mathbf{b}$ に書ける。未知数が何個あっても形は $A \mathbf{x} = \mathbf{b}$ のままである。正規方程式 $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$ も、LM 法の更新式 $( J^\top J + \lambda I ) \boldsymbol{\delta} = - J^\top \mathbf{r}$ も、すべてこの形の連立一次方程式である。

**見方 2: ベクトルをベクトルに移す関数。** $A$ を固定すると、$\mathbf{x} \mapsto A \mathbf{x}$ は「$m$ 次元ベクトルを受け取り $n$ 次元ベクトルを返す関数」になる。この関数は定義から次を満たす。

$$
A ( \mathbf{u} + \mathbf{v} ) = A \mathbf{u} + A \mathbf{v},
\qquad
A ( c \mathbf{u} ) = c \, A \mathbf{u}
$$

和とスカラー倍を保つこの性質を**線形性**という。「行列 = 線形な関数 (写像)」という見方は、行列どうしの積 (次節) を理解する土台になる。

**見方 3: 列ベクトルの線形結合。** $A$ の第 $j$ 列を $\mathbf{a}_j$ と書くと、積の定義を列ごとに整理して

$$
A \mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_m \mathbf{a}_m
$$

と書ける。先ほどの例で確かめると

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 5 \\ 6 \end{bmatrix}
= 5 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 6 \begin{bmatrix} 2 \\ 4 \end{bmatrix}
= \begin{bmatrix} 5 \\ 15 \end{bmatrix} + \begin{bmatrix} 12 \\ 24 \end{bmatrix}
= \begin{bmatrix} 17 \\ 39 \end{bmatrix}
$$

で一致する。「$\mathbf{x}$ を動かすと $A \mathbf{x}$ は $A$ の列ベクトルの線形結合全体を動く」という事実は、[1. 線形最小二乗法](./1_least_squares_method.md) の幾何的導出 ($\Phi \boldsymbol{\beta}$ の動く範囲 = 列の張る部分空間) の出発点である。

### 3.3 行列 × 行列

**動機: 関数の合成。** 行列 $B$ (関数 $\mathbf{x} \mapsto B \mathbf{x}$) を適用した後に行列 $A$ を適用する、という合成関数

$$
\mathbf{x} \ \mapsto\ B \mathbf{x} \ \mapsto\ A ( B \mathbf{x} )
$$

を考える。この合成もまた線形な関数であり、実は 1 つの行列で表せる。それを $AB$ と書き、**行列の積**と定義する。すなわち $( AB ) \mathbf{x} = A ( B \mathbf{x} )$ が成り立つように積を決める。

**成分による定義。** 上の要請から計算すると、$A$ が $n \times m$、$B$ が $m \times p$ のとき、$AB$ は $n \times p$ 行列で

$$
( AB )_{ij} = \sum_{k=1}^{m} A_{ik} B_{kj}
$$

となる。言葉でいえば「$AB$ の $(i, j)$ 成分は、$A$ の第 $i$ 行と $B$ の第 $j$ 列の内積」である。$2 \times 2$ の例で手を動かす。

$$
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
=
\begin{bmatrix}
1 \times 5 + 2 \times 7 & 1 \times 6 + 2 \times 8 \\
3 \times 5 + 4 \times 7 & 3 \times 6 + 4 \times 8
\end{bmatrix}
=
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

行列 × ベクトル (3.2 節) は、この定義で $p = 1$ とした特別な場合になっている。

**順序に注意。** 関数の合成は順序を入れ替えると別物になる ($f(g(x)) \neq g(f(x))$) のと同じで、行列の積は一般に**交換できない**。実際、上の例で順序を入れ替えると

$$
\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
=
\begin{bmatrix} 5 + 18 & 10 + 24 \\ 7 + 24 & 14 + 32 \end{bmatrix}
=
\begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix}
\neq
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

一方、結合法則 $( AB ) C = A ( BC )$ と分配法則 $A ( B + C ) = AB + AC$ は成り立つ。

### 3.4 転置

行列 $A$ の行と列を入れ替えたものを**転置**といい、$A^\top$ と書く。$( A^\top )_{ij} = A_{ji}$ である。

$$
A =
\begin{bmatrix}
1 & 2 \\
3 & 4 \\
5 & 6
\end{bmatrix}
\quad \Longrightarrow \quad
A^\top =
\begin{bmatrix}
1 & 3 & 5 \\
2 & 4 & 6
\end{bmatrix}
$$

$3 \times 2$ 行列の転置は $2 \times 3$ 行列になる。列ベクトル $\mathbf{a}$ の転置 $\mathbf{a}^\top$ は横に並んだ**行ベクトル**である。転置の基本性質は次のとおり。

$$
( A^\top )^\top = A,
\qquad
( A + B )^\top = A^\top + B^\top,
\qquad
( AB )^\top = B^\top A^\top
$$

3 つ目は**順序が入れ替わる**ことに注意する。成分で確かめると

$$
( ( AB )^\top )_{ij} = ( AB )_{ji} = \sum_{k} A_{jk} B_{ki} = \sum_{k} ( B^\top )_{ik} ( A^\top )_{kj} = ( B^\top A^\top )_{ij}
$$

となり、確かに一致する。

**内積との関係。** 列ベクトル $\mathbf{a}, \mathbf{b} \in \mathbb{R}^n$ に対し、$\mathbf{a}^\top$ ($1 \times n$) と $\mathbf{b}$ ($n \times 1$) の行列としての積は $1 \times 1$ 行列、すなわち 1 つの数であり、

$$
\mathbf{a}^\top \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \mathbf{a} \cdot \mathbf{b}
$$

つまり**内積は $\mathbf{a}^\top \mathbf{b}$ と行列の積の形で書ける**。以降の文書では $\mathbf{a} \cdot \mathbf{b}$ と $\mathbf{a}^\top \mathbf{b}$ を同じ意味で使う。特にノルムの二乗は $\| \mathbf{r} \|^2 = \mathbf{r}^\top \mathbf{r}$ と書ける。

### 3.5 対称行列

$A^\top = A$ を満たす正方行列 (行数 = 列数の行列) を**対称行列**という。成分でいえば $A_{ij} = A_{ji}$、つまり左上から右下への対角線に関して対称な行列である。

$$
\begin{bmatrix}
2 & 5 \\
5 & 9
\end{bmatrix}
$$

任意の行列 $A$ に対して $A^\top A$ は必ず対称行列になる。実際、

$$
( A^\top A )^\top = A^\top ( A^\top )^\top = A^\top A
$$

正規方程式の係数行列 $\Phi^\top \Phi$ や、ガウス・ニュートン法・LM 法に現れる $J^\top J$ はこの形であり、対称性は解法 (コレスキー分解など) や理論 (固有値がすべて実数になる等) の随所で効いてくる。

### 3.6 単位行列と逆行列

**単位行列** $I$ は、対角成分が $1$ で他がすべて $0$ の正方行列である。

$$
I =
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
\quad (2 \times 2 \text{ の場合})
$$

任意の $\mathbf{x}$ に対して $I \mathbf{x} = \mathbf{x}$、任意の (サイズの合う) 行列 $A$ に対して $A I = I A = A$ が成り立つ。数の $1$ に相当する行列である。

正方行列 $A$ に対して

$$
A A^{-1} = A^{-1} A = I
$$

を満たす行列 $A^{-1}$ が存在するとき、$A$ は**可逆** (正則) であるといい、$A^{-1}$ を**逆行列**と呼ぶ。数の逆数 $\frac{1}{a}$ に相当する。逆行列があれば、連立方程式 $A \mathbf{x} = \mathbf{b}$ は両辺に左から $A^{-1}$ を掛けて

$$
A^{-1} A \mathbf{x} = A^{-1} \mathbf{b}
\quad \Longrightarrow \quad
\mathbf{x} = A^{-1} \mathbf{b}
$$

と解ける。$2 \times 2$ の場合には公式がある。

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
\quad \Longrightarrow \quad
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
\qquad ( ad - bc \neq 0 \text{ のとき})
$$

検算しておく。

$$
\frac{1}{ad - bc}
\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
=
\frac{1}{ad - bc}
\begin{bmatrix} da - bc & db - bd \\ -ca + ac & -cb + ad \end{bmatrix}
=
\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

$ad - bc = 0$ のときは逆行列が**存在しない**。数の $0$ で割れないのと同じで、すべての正方行列が可逆なわけではない。可逆でない行列を**特異**であるという。いつ可逆になるかは、次章のランクの言葉で特徴づけられる。なお $( AB )^{-1} = B^{-1} A^{-1}$ と、転置と同様に順序が入れ替わる。

逆行列は理論を書き表すための道具であり、数値計算で逆行列を明示的に作ることは通常避ける。この事情は [2. QR 分解](./2_qr_decomposition.md) と [3. SVD](./3_singular_value_decomposition.md) で扱う。

### 3.7 内積と転置を結ぶ公式

次の公式はロードマップ全体で繰り返し使う。$A \in \mathbb{R}^{n \times m}$、$\mathbf{u} \in \mathbb{R}^m$、$\mathbf{v} \in \mathbb{R}^n$ に対して

$$
( A \mathbf{u} ) \cdot \mathbf{v} = \mathbf{u} \cdot ( A^\top \mathbf{v} )
$$

**「内積の中の行列は、転置して反対側へ移せる」**と読む。証明は成分計算で一行ずつ追える。

$$
\begin{aligned}
( A \mathbf{u} ) \cdot \mathbf{v}
&= \sum_{i=1}^{n} ( A \mathbf{u} )_i \, v_i \\
&= \sum_{i=1}^{n} \Bigl( \sum_{j=1}^{m} A_{ij} u_j \Bigr) v_i \\
&= \sum_{j=1}^{m} u_j \Bigl( \sum_{i=1}^{n} A_{ij} v_i \Bigr) \\
&= \sum_{j=1}^{m} u_j \, ( A^\top \mathbf{v} )_j \\
&= \mathbf{u} \cdot ( A^\top \mathbf{v} )
\end{aligned}
$$

(3 行目では二重和の順序を入れ替えた。4 行目では $\sum_i A_{ij} v_i = \sum_i ( A^\top )_{ji} v_i$ が $A^\top \mathbf{v}$ の第 $j$ 成分であることを使った。)

転置記法を使えば $( A \mathbf{u} )^\top \mathbf{v} = \mathbf{u}^\top A^\top \mathbf{v}$ となり、$( AB )^\top = B^\top A^\top$ の直接の帰結でもある。

この公式が効く場面を予告しておく。[1. 線形最小二乗法](./1_least_squares_method.md) では「残差 $\mathbf{y} - \Phi \boldsymbol{\beta}$ が $\Phi$ の列すべてと直交する」という条件を、この公式で $\Phi^\top ( \mathbf{y} - \Phi \boldsymbol{\beta} ) = \mathbf{0}$ という方程式 (正規方程式) に変換する。転置 $\Phi^\top$ や $J^\top$ が式のあちこちに現れる理由は、突き詰めればこの公式である。

## 4. 線形独立・線形従属とランク

**どこで使うか。** [1. 線形最小二乗法](./1_least_squares_method.md) の「解が一意に定まる条件」、[3. SVD](./3_singular_value_decomposition.md) のランク落ち、[8. LM 法](./8_levenberg_marquardt_method.md) の「$J^\top J$ が特異になる故障モード」を理解するために必要となる。

### 4.1 具体例からの直感

2 次元で考える。2 本のベクトル

$$
\mathbf{a}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix},
\qquad
\mathbf{a}_2 = \begin{bmatrix} 2 \\ 4 \end{bmatrix}
$$

は $\mathbf{a}_2 = 2 \mathbf{a}_1$ という関係にあり、同一直線上にある。このとき線形結合 $c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2 = ( c_1 + 2 c_2 ) \mathbf{a}_1$ はその直線の上しか動けない。つまり $\mathbf{a}_2$ は $\mathbf{a}_1$ に対して「新しい方向」を何も追加していない。

一方

$$
\mathbf{a}_1 = \begin{bmatrix} 1 \\ 2 \end{bmatrix},
\qquad
\mathbf{a}_2 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}
$$

なら 2 本は別の方向を向いており、線形結合 $c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2$ は平面全体を動ける (任意の $\begin{bmatrix} p \\ q \end{bmatrix}$ に対し $c_1 = q / 2, \ c_2 = p - q/2$ と取ればよい)。

「一方が他方の役割を代替できるか (方向が重複しているか)」が両者を分ける。これを一般化したのが線形独立・線形従属である。

### 4.2 定義

ベクトル $\mathbf{a}_1, \dots, \mathbf{a}_m$ が**線形独立**であるとは、

$$
c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2 + \cdots + c_m \mathbf{a}_m = \mathbf{0}
$$

を満たす係数が $c_1 = c_2 = \cdots = c_m = 0$ しかないことをいう。そうでない (全部は $0$ でない係数の組で $\mathbf{0}$ が作れる) とき**線形従属**という。

線形従属なら、例えば $c_1 \neq 0$ として

$$
\mathbf{a}_1 = - \frac{c_2}{c_1} \mathbf{a}_2 - \cdots - \frac{c_m}{c_1} \mathbf{a}_m
$$

と、どれか 1 本が残りの線形結合で書ける。「線形従属 = 冗長なベクトルが混ざっている」と理解すればよい。先の例では $2 \mathbf{a}_1 - \mathbf{a}_2 = \mathbf{0}$ なので線形従属である。

なお $n$ 次元空間には、線形独立なベクトルは最大 $n$ 本しか取れない (例: 平面上の 3 本のベクトルは必ず線形従属)。

### 4.3 ランク

行列 $A$ の**ランク** $\mathrm{rank} \, A$ とは、$A$ の列ベクトルのうち線形独立に取れる最大の本数である。言い換えると「$A \mathbf{x}$ が動ける範囲 (列の線形結合全体) が何次元の広がりを持つか」を表す。

$$
\mathrm{rank}
\begin{bmatrix}
1 & 2 \\
2 & 4
\end{bmatrix}
= 1,
\qquad
\mathrm{rank}
\begin{bmatrix}
1 & 1 \\
2 & 0
\end{bmatrix}
= 2
$$

左の行列は 2 列が同一直線上にあるためランク $1$、右は独立なのでランク $2$ である。$n \times m$ 行列のランクは $n$ と $m$ の小さい方を超えない。ランクが取り得る最大値に達しているとき**フルランク**であるという。

ランクと連立方程式・逆行列の関係をまとめる。$A$ を $n \times n$ の正方行列とすると、次はすべて同値である。

1. $A$ は可逆 (逆行列を持つ)
2. $\mathrm{rank} \, A = n$ ($A$ の列が線形独立)
3. $A \mathbf{x} = \mathbf{b}$ がどんな $\mathbf{b}$ に対してもただ 1 つの解を持つ
4. $A \mathbf{x} = \mathbf{0}$ の解が $\mathbf{x} = \mathbf{0}$ のみ

直感的には、列が線形従属だと $A \mathbf{x}$ の動ける範囲が「つぶれて」いて ($n$ 次元より狭い)、届かない $\mathbf{b}$ が出てくる一方、届く $\mathbf{b}$ には無数の解が対応してしまう、ということである。

最小二乗法の文脈では次の形で現れる。計画行列 $\Phi \in \mathbb{R}^{n \times m}$ の列が線形独立 ($\mathrm{rank} \, \Phi = m$) のとき、そしてそのときに限り $\Phi^\top \Phi$ が可逆になり、最小二乗解がただ 1 つに定まる。列が線形従属なケース (基底関数の選び方が冗長、データの配置が退化しているなど) を**ランク落ち**と呼び、その処方箋が [3. SVD](./3_singular_value_decomposition.md) と [8. LM 法](./8_levenberg_marquardt_method.md) のダンピングである。

## 5. 多変数関数と偏微分

**どこで使うか。** 最小化したい目的関数 $E(\boldsymbol{\beta})$ は $m$ 個のパラメータ $\beta_1, \dots, \beta_m$ を持つ多変数関数である。「$E$ を微分して $\mathbf{0}$ とおく」「勾配の逆方向に進む」という操作 ([1. 線形最小二乗法](./1_least_squares_method.md)、[5. 最急降下法](./5_steepest_descent.md)、[6. ニュートン法](./6_newton_method.md)) の意味を、ここで 1 変数の微分から橋渡しする。

### 5.1 2 変数関数のグラフと等高線

2 変数関数 $f(x, y)$ は「2 つの数を受け取り 1 つの数を返す関数」である。例として

$$
f(x, y) = x^2 + y^2
$$

を考える。グラフは $z = f(x, y)$ を満たす点 $(x, y, z)$ の集合で、3 次元空間内の**曲面**になる。この例ではお椀 (回転放物面) の形をしており、底は原点 $(0, 0)$、そこでの値は $f(0,0) = 0$ である。

曲面を直接描く代わりに、**等高線** (地図の等高線と同じもの) で可視化できる。$f(x, y) = c$ を満たす点の集合が「高さ $c$ の等高線」である。$f = x^2 + y^2$ なら等高線 $x^2 + y^2 = c$ は原点中心・半径 $\sqrt{c}$ の円であり、等高線が同心円をなす「すり鉢」だと読み取れる。

最小化問題とは「この地形の一番低い場所を探す」ことである。パラメータが $m$ 個ならば $m$ 次元の地形になり図には描けないが、2 変数の直感がそのまま指針になる。

### 5.2 偏微分の定義

1 変数の微分係数は

$$
f'(a) = \lim_{h \to 0} \frac{f(a + h) - f(a)}{h}
$$

だった。多変数では「どの変数を動かすか」を選ぶ必要がある。**他の変数をすべて定数として固定し、1 つの変数だけで微分したもの**を**偏微分**といい、$\dfrac{\partial f}{\partial x}$ と書く ($\partial$ は「ラウンド」と読む)。定義は

$$
\frac{\partial f}{\partial x}(a, b) = \lim_{h \to 0} \frac{f(a + h, b) - f(a, b)}{h},
\qquad
\frac{\partial f}{\partial y}(a, b) = \lim_{h \to 0} \frac{f(a, b + h) - f(a, b)}{h}
$$

計算は 1 変数の微分と同じ要領でできる。例えば $f(x, y) = x^2 y + 3y$ なら

- $\dfrac{\partial f}{\partial x}$: $y$ を定数と思って $x$ で微分し、$\dfrac{\partial f}{\partial x} = 2xy$
- $\dfrac{\partial f}{\partial y}$: $x$ を定数と思って $y$ で微分し、$\dfrac{\partial f}{\partial y} = x^2 + 3$

幾何的には、$\frac{\partial f}{\partial x}(a, b)$ は「曲面を $y = b$ の平面で切った断面の曲線の、$x = a$ における傾き」、つまり **$x$ 軸方向に進んだときの高さの変化率**である。

### 5.3 勾配ベクトル

各変数についての偏微分を並べたベクトルを**勾配** (グラディエント) といい、$\nabla f$ と書く ($\nabla$ は「ナブラ」と読む)。

$$
\nabla f =
\begin{bmatrix}
\dfrac{\partial f}{\partial x} \\[2ex]
\dfrac{\partial f}{\partial y}
\end{bmatrix}
$$

$m$ 変数関数 $f(\boldsymbol{\beta}) = f(\beta_1, \dots, \beta_m)$ なら $\nabla f = \bigl( \frac{\partial f}{\partial \beta_1}, \dots, \frac{\partial f}{\partial \beta_m} \bigr)^\top \in \mathbb{R}^m$ である。例えば $f(x, y) = x^2 + y^2$ の勾配は

$$
\nabla f = \begin{bmatrix} 2x \\ 2y \end{bmatrix}
$$

で、点 $(1, 2)$ では $\nabla f(1, 2) = \begin{bmatrix} 2 \\ 4 \end{bmatrix}$ となる。勾配は「点ごとに決まるベクトル」であり、地形の各地点に立てた矢印だとイメージするとよい。

### 5.4 勾配は最も急な上り方向

勾配の最重要性質を導く。点 $\mathbf{a}$ から、長さ $1$ のベクトル $\mathbf{u}$ の方向に微小距離 $h$ だけ進んだときの $f$ の変化を考える。1 変数のときの一次近似 $f(a + h) \approx f(a) + f'(a) h$ に対応して、多変数では

$$
f( \mathbf{a} + h \mathbf{u} ) \approx f( \mathbf{a} ) + h \, \nabla f( \mathbf{a} ) \cdot \mathbf{u}
$$

が成り立つ (各変数の変化 $h u_i$ が偏微分 $\frac{\partial f}{\partial \beta_i}$ の割合で高さを変え、その寄与を足し合わせたものが内積 $\nabla f \cdot \mathbf{u}$ になる。厳密には次節の連鎖律から導かれる)。つまり**方向 $\mathbf{u}$ に進んだときの変化率は $\nabla f \cdot \mathbf{u}$** である (方向微分と呼ぶ)。

ここで内積の幾何的意味 (2.3 節) を使う。$\| \mathbf{u} \| = 1$ なので、$\nabla f$ と $\mathbf{u}$ のなす角を $\theta$ とすれば

$$
\nabla f \cdot \mathbf{u} = \| \nabla f \| \cos \theta
$$

$\cos \theta$ は $-1$ 以上 $1$ 以下だから、この変化率は

- $\mathbf{u}$ が $\nabla f$ と**同じ向き** ($\theta = 0$) のとき最大値 $\| \nabla f \|$
- $\mathbf{u}$ が $\nabla f$ と**逆向き** ($\theta = 180^\circ$) のとき最小値 $- \| \nabla f \|$
- $\mathbf{u}$ が $\nabla f$ と**直交**するとき $0$ (高さが変わらない方向 = 等高線に沿う方向)

となる。まとめると次のようになる。

> 勾配 $\nabla f$ は「最も急な上り」の方向を向き、その長さは傾きの大きさを表す。したがって $- \nabla f$ の方向が「最も急な下り」であり、勾配は等高線と直交する。

「$- \nabla f$ 方向に一歩進めば関数は必ず減る」— これがそのまま [5. 最急降下法](./5_steepest_descent.md) のアルゴリズムである。$f = x^2 + y^2$ で確かめると、点 $(1, 2)$ の勾配 $\begin{bmatrix} 2 \\ 4 \end{bmatrix}$ は原点 (すり鉢の底) から遠ざかる向き、すなわち最も急な上りを向いており、その逆 $\begin{bmatrix} -2 \\ -4 \end{bmatrix}$ は底へ向かう最速の下りである。

### 5.5 極値の必要条件

1 変数では「極値なら $f'(a) = 0$」だった。多変数版は次のとおり。

> $f$ が点 $\mathbf{a}$ で極小 (または極大) ならば $\nabla f( \mathbf{a} ) = \mathbf{0}$ である。

理由は簡単で、もし $\nabla f( \mathbf{a} ) \neq \mathbf{0}$ なら $- \nabla f$ 方向に進めば $f$ が減り、$\nabla f$ 方向に進めば増えるので、$\mathbf{a}$ は極小でも極大でもあり得ない。$\nabla f = \mathbf{0}$ となる点を**停留点**と呼ぶ。

ただし 1 変数と同様、逆は成り立たない。停留点が極小とは限らない。多変数では極大でも極小でもない**鞍点**というものもある。例えば $f(x, y) = x^2 - y^2$ は原点で $\nabla f = \mathbf{0}$ だが、$x$ 軸方向には谷底、$y$ 軸方向には尾根であり、極値ではない (馬の鞍の形)。「勾配が $\mathbf{0}$ の点を見つけただけでは最小と言い切れない」— この問題に答えるのが第 6 章の凸性である。

### 5.6 連鎖律 (多変数版)

1 変数の合成関数の微分は $\{ f(g(t)) \}' = f'(g(t)) \, g'(t)$ だった。多変数版では、外側の関数が複数の変数を持ち、その各変数が $t$ に依存する。$z = f(x, y)$、$x = x(t)$、$y = y(t)$ のとき

$$
\frac{dz}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt}
$$

**「変数ごとの寄与を足し合わせる」**のが 1 変数との違いである。$t$ が動くと $x$ と $y$ の両方が動き、それぞれが $z$ を変化させるので、両方の経路の寄与を合算する、と理解すればよい。

具体例で検算する。$f(x, y) = x^2 y, \ x = t, \ y = t^3$ とすると、代入すれば $z = t^2 \cdot t^3 = t^5$ なので $\frac{dz}{dt} = 5 t^4$ のはずである。連鎖律で計算すると

$$
\frac{dz}{dt}
= \underbrace{2xy}_{\partial f / \partial x} \cdot \underbrace{1}_{dx/dt} + \underbrace{x^2}_{\partial f / \partial y} \cdot \underbrace{3t^2}_{dy/dt}
= 2 t \cdot t^3 + t^2 \cdot 3 t^2
= 2 t^4 + 3 t^4
= 5 t^4
$$

で一致する。

**最小二乗法での使われ方。** ロードマップで実際に使う形を先取りしておく。目的関数が「残差の二乗和」

$$
E(\boldsymbol{\beta}) = \sum_{i=1}^{n} r_i(\boldsymbol{\beta})^2
$$

のとき、$\beta_k$ での偏微分は、外側の「二乗」と内側の $r_i(\boldsymbol{\beta})$ の合成に連鎖律を使って

$$
\frac{\partial E}{\partial \beta_k}
= \sum_{i=1}^{n} 2 \, r_i \, \frac{\partial r_i}{\partial \beta_k}
$$

となる。これを $k = 1, \dots, m$ について並べてベクトルにしたものが、[4. 非線形最小二乗法](./4_nonlinear_least_squares.md) の基本公式

$$
\nabla E = 2 J^\top \mathbf{r}
\qquad
\Bigl( J \text{ は } J_{ik} = \frac{\partial r_i}{\partial \beta_k} \text{ を成分とする行列 (ヤコビ行列)} \Bigr)
$$

である。実際、$2 J^\top \mathbf{r}$ の第 $k$ 成分は $\sum_i 2 ( J^\top )_{ki} r_i = \sum_i 2 \frac{\partial r_i}{\partial \beta_k} r_i$ となり、上の偏微分と一致する。ヤコビ行列 $J$ とは「$n$ 個の残差それぞれの、$m$ 個のパラメータそれぞれについての偏微分を並べた $n \times m$ 行列」であり、ガウス・ニュートン法と LM 法の中心的な登場人物である。

### 5.7 2 階偏微分とヘッセ行列 (紹介)

1 変数で $f''$ が凹凸を判定したように、多変数でも 2 階の偏微分 $\frac{\partial^2 f}{\partial \beta_i \partial \beta_j}$ が地形の曲がり方を表す。これを $(i, j)$ 成分に持つ $m \times m$ の対称行列を**ヘッセ行列** $\nabla^2 f$ と呼ぶ。詳しくは [6. ニュートン法](./6_newton_method.md) で使うときに説明するので、ここでは「$f''$ の多変数版で、行列になる」とだけ覚えておけばよい。

## 6. 凸関数

**どこで使うか。** [1. 線形最小二乗法](./1_least_squares_method.md) では「勾配が $\mathbf{0}$ の点は本当に最小か」という問いに、「$E$ は凸だから停留点は大域的最小である」と答える。逆に [4. 非線形最小二乗法](./4_nonlinear_least_squares.md) では凸性が崩れることこそが難しさの根源になる。凸性は「停留点を見つければ勝ち」と言えるかどうかの分水嶺である。

### 6.1 1 変数の復習 — 下に凸

高校で学んだとおり、$f(x) = x^2$ のような「下に凸」な関数は、グラフがどこでも上に曲がっており、$f''(x) \geq 0$ で特徴づけられた。これを微分を使わずに言い換えると次のようになる。

> グラフ上のどの 2 点を結んでも、その線分 (弦) がグラフより上 (または同じ高さ) にある。

式で書くと、任意の 2 点 $x, y$ と $0 \leq t \leq 1$ に対して

$$
f( t x + (1 - t) y ) \leq t f(x) + (1 - t) f(y)
$$

左辺は「$x$ と $y$ を $t : (1-t)$ に内分した点でのグラフの高さ」、右辺は「$f(x)$ と $f(y)$ を同じ比率で内分した高さ = 弦の高さ」である。この不等式を満たす関数を**凸関数**と呼ぶ。$f(x) = x^2$ で確かめると

$$
\begin{aligned}
t f(x) + (1 - t) f(y) - f( t x + (1 - t) y )
&= t x^2 + (1 - t) y^2 - ( t x + (1 - t) y )^2 \\
&= t x^2 + (1 - t) y^2 - t^2 x^2 - 2 t (1 - t) x y - (1 - t)^2 y^2 \\
&= t (1 - t) x^2 - 2 t (1 - t) x y + t (1 - t) y^2 \\
&= t (1 - t) ( x - y )^2 \\
&\geq 0
\end{aligned}
$$

(2 行目から 3 行目は $t - t^2 = t(1-t)$、$(1-t) - (1-t)^2 = (1-t) t$ を使った。) 確かに凸である。

凸関数のありがたみは、**谷が 1 つしかない**ことである。凸なら「局所的にはくぼんで見えるが実はもっと深い谷が別にある」という状況が起こらない。対照的に $f(x) = x^4 - 2 x^2$ のような非凸関数は谷を 2 つ持ち、片方の谷底に降りてもそこが最小とは限らない。

### 6.2 多変数への拡張

定義の式は変数がベクトルになってもそのまま通用する。$f : \mathbb{R}^m \to \mathbb{R}$ が**凸**であるとは、任意の $\mathbf{x}, \mathbf{y} \in \mathbb{R}^m$ と $0 \leq t \leq 1$ に対して

$$
f( t \mathbf{x} + (1 - t) \mathbf{y} ) \leq t f( \mathbf{x} ) + (1 - t) f( \mathbf{y} )
$$

が成り立つことをいう。幾何的には「曲面上のどの 2 点を結んだ弦も、曲面より下に潜らない」、等高線の直感でいえば「地形がどの断面で切っても下に凸のすり鉢型」である。

例をいくつか挙げる。

- $f( \mathbf{x} ) = \| \mathbf{x} \|^2 = x_1^2 + \cdots + x_m^2$ は凸である (各項 $x_i^2$ が 1 変数として凸で、凸関数の和は凸だから)。
- 一次関数 $f( \mathbf{x} ) = \mathbf{a} \cdot \mathbf{x} + b$ は凸である (定義の不等式が等号で成立)。
- $f(x, y) = x^2 - y^2$ (鞍点の例) は凸ではない。$y$ 軸に沿った断面 $-y^2$ が上に凸だからである。

最小二乗法との関係で最も重要な例は次である。**線形最小二乗法の目的関数 $E(\boldsymbol{\beta}) = \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2$ は凸関数である。** 実際、$\boldsymbol{\beta} \mapsto \mathbf{y} - \Phi \boldsymbol{\beta}$ は一次式であり、一次式を凸関数 $\| \cdot \|^2$ に代入したものは凸になる。確かめる。$\mathbf{u} = \mathbf{y} - \Phi \boldsymbol{\beta}_1, \ \mathbf{v} = \mathbf{y} - \Phi \boldsymbol{\beta}_2$ とおくと、一次式の性質から

$$
\mathbf{y} - \Phi ( t \boldsymbol{\beta}_1 + (1 - t) \boldsymbol{\beta}_2 ) = t \mathbf{u} + (1 - t) \mathbf{v}
$$

であり、$\| \cdot \|^2$ の凸性 (各成分の 2 乗の凸性の和) から

$$
\| t \mathbf{u} + (1 - t) \mathbf{v} \|^2 \leq t \| \mathbf{u} \|^2 + (1 - t) \| \mathbf{v} \|^2
$$

となるので、$E( t \boldsymbol{\beta}_1 + (1 - t) \boldsymbol{\beta}_2 ) \leq t E( \boldsymbol{\beta}_1 ) + (1 - t) E( \boldsymbol{\beta}_2 )$ が成り立つ。

なお微分による判定法もある。1 変数の「$f'' \geq 0$ なら凸」に対応して、多変数では「ヘッセ行列 $\nabla^2 f$ がすべての点で半正定値 (任意の $\mathbf{u}$ に対し $\mathbf{u}^\top ( \nabla^2 f ) \mathbf{u} \geq 0$) なら凸」となる。[1. 線形最小二乗法](./1_least_squares_method.md) の第 4 節ではこの判定法が使われている。

### 6.3 凸なら停留点が大域的最小

凸関数の最重要定理を述べる。

> $f$ が微分可能な凸関数で、点 $\mathbf{a}$ で $\nabla f( \mathbf{a} ) = \mathbf{0}$ ならば、$\mathbf{a}$ は**大域的最小点**である。すなわち、すべての $\mathbf{x}$ に対して $f( \mathbf{x} ) \geq f( \mathbf{a} )$ が成り立つ。

証明の鍵は、微分可能な凸関数が持つ次の性質である。

$$
f( \mathbf{x} ) \geq f( \mathbf{a} ) + \nabla f( \mathbf{a} ) \cdot ( \mathbf{x} - \mathbf{a} )
\qquad \text{(すべての } \mathbf{x}, \mathbf{a} \text{ に対して)}
$$

右辺は点 $\mathbf{a}$ における一次近似 (グラフの接平面) であり、この不等式は**「凸関数のグラフは、どの点で引いた接線 (接平面) よりも常に上にある」**ことを意味する。1 変数の $f(x) = x^2$ で描いてみれば明らかだろう。放物線はどの接線よりも上にある。

この性質を認めれば、定理は 1 行で従う。$\nabla f( \mathbf{a} ) = \mathbf{0}$ を代入して

$$
f( \mathbf{x} ) \geq f( \mathbf{a} ) + \mathbf{0} \cdot ( \mathbf{x} - \mathbf{a} ) = f( \mathbf{a} )
$$

接平面の性質自体も、凸性の定義から導ける。定義の不等式を変形すると

$$
f( \mathbf{a} + t ( \mathbf{x} - \mathbf{a} ) ) = f( t \mathbf{x} + (1 - t) \mathbf{a} ) \leq t f( \mathbf{x} ) + (1 - t) f( \mathbf{a} )
$$

なので、両辺から $f( \mathbf{a} )$ を引いて $t > 0$ で割ると

$$
\frac{ f( \mathbf{a} + t ( \mathbf{x} - \mathbf{a} ) ) - f( \mathbf{a} ) }{ t } \leq f( \mathbf{x} ) - f( \mathbf{a} )
$$

左辺は $t \to 0$ の極限で方向微分 $\nabla f( \mathbf{a} ) \cdot ( \mathbf{x} - \mathbf{a} )$ に収束する (5.4 節)。よって $\nabla f( \mathbf{a} ) \cdot ( \mathbf{x} - \mathbf{a} ) \leq f( \mathbf{x} ) - f( \mathbf{a} )$ となり、これが接平面の性質そのものである。

この定理の意味を、ロードマップの言葉でまとめておく。

- **線形**最小二乗法では $E$ が凸なので、$\nabla E = \mathbf{0}$ を解くだけで大域的最小が手に入る。鞍点や偽の谷の心配は要らない。これが [1. 線形最小二乗法](./1_least_squares_method.md) が「一発で解ける」理由である。
- **非線形**最小二乗法では $E$ は一般に凸でなく、谷が複数あり得る。$\nabla E = \mathbf{0}$ を満たす点に到達しても、それは「初期値から降りていった先の谷底 (局所解)」にすぎない。これが [4. 非線形最小二乗法](./4_nonlinear_least_squares.md) 以降で反復法と初期値の議論が延々と続く理由である。

## 7. 記号一覧

この文書と [1](./1_least_squares_method.md)〜[8](./8_levenberg_marquardt_method.md) で使う記号をまとめる。

### 集合・ベクトル・行列

| 記号 | 意味 | 定義した場所 |
| --- | --- | --- |
| $\mathbb{R}$ | 実数全体の集合 | 2.1 節 |
| $\mathbb{R}^n$ | $n$ 次元ベクトル全体の集合 | 2.1 節 |
| $\mathbb{R}^{n \times m}$ | $n$ 行 $m$ 列の実行列全体の集合 | 3.1 節 |
| $\mathbf{a}, \mathbf{y}, \dots$ (太字小文字) | 列ベクトル。第 $i$ 成分は $a_i, y_i$ | 2.1 節 |
| $\mathbf{0}$ | ゼロベクトル | 2.1 節 |
| $A, \Phi, J, \dots$ (大文字) | 行列。$(i,j)$ 成分は $A_{ij}$ | 3.1 節 |
| $\mathbf{a} \cdot \mathbf{b}$, $\mathbf{a}^\top \mathbf{b}$ | 内積 $\sum_i a_i b_i$ (2 つの記法は同じ意味) | 2.3, 3.4 節 |
| $\| \mathbf{a} \|$ | ノルム $\sqrt{ \sum_i a_i^2 }$ | 2.4 節 |
| $A^\top$ | 転置 (行と列の入れ替え) | 3.4 節 |
| $I$ | 単位行列 | 3.6 節 |
| $A^{-1}$ | 逆行列 | 3.6 節 |
| $\mathrm{rank} \, A$ | ランク (線形独立な列の最大本数) | 4.3 節 |

### 微分

| 記号 | 意味 | 定義した場所 |
| --- | --- | --- |
| $\dfrac{\partial f}{\partial x}$ | 偏微分 ($x$ 以外を固定して微分) | 5.2 節 |
| $\nabla f$ | 勾配 (偏微分を並べたベクトル)。最急上昇方向 | 5.3, 5.4 節 |
| $\nabla^2 f$ | ヘッセ行列 (2 階偏微分を並べた対称行列) | 5.7 節 |

### 最小二乗法の文脈で決まった役割を持つ記号 (文書 1〜8)

| 記号 | 意味 |
| --- | --- |
| $(x_i, y_i)$ | $i$ 番目の観測データ ($i = 1, \dots, n$) |
| $n$ | データ数 |
| $m$ | パラメータ数 |
| $\boldsymbol{\beta} \in \mathbb{R}^m$ | 求めたいパラメータのベクトル |
| $\mathbf{y} \in \mathbb{R}^n$ | 観測値を並べたベクトル |
| $\Phi \in \mathbb{R}^{n \times m}$ | 計画行列 (線形最小二乗法) |
| $\mathbf{r} \in \mathbb{R}^n$ | 残差ベクトル (観測とモデル予測のずれ) |
| $E(\boldsymbol{\beta}) = \| \mathbf{r} \|^2$ | 目的関数 (残差の二乗和) |
| $J \in \mathbb{R}^{n \times m}$ | 残差のヤコビ行列 $J_{ik} = \partial r_i / \partial \beta_k$ (非線形最小二乗法) |
| $\boldsymbol{\delta} \in \mathbb{R}^m$ | 反復法における 1 ステップの更新量 |
| $\lambda$ | LM 法のダンピングパラメータ |

以上で準備は完了である。[1. 線形最小二乗法](./1_least_squares_method.md) から読み進めてほしい。
