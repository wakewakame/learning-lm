# 特異値分解 (SVD) (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

## 1. この文書で学ぶこと

[2. QR 分解](./2_qr_decomposition.md) では、行列 $A$ を「空間を歪めない直交行列 $Q$」と「後退代入で解ける上三角行列 $R$」に分けることで、最小二乗問題を安定に解いた。ただし QR 分解による解法には前提があった。**$A$ の列が線形独立であること**である。

現実のデータでは、この前提はしばしば崩れる。

- 説明変数どうしがほぼ同じ情報を持っていて、列が**線形従属に近い** (ランク落ち寸前)
- 完全に線形従属で、最小二乗解が**一つに決まらない**

こうした「グレーな行列」「壊れた行列」まで含めて、**どんな行列でも**使える分解が**特異値分解** (Singular Value Decomposition, **SVD**) である。この文書では次のことを学ぶ。

1. SVD の定義と幾何的な意味 —「どんな行列も、単位円を**回して・伸ばして・回した**楕円に移す」
2. 特異値から読み取れる情報 — ランク、条件数、行列の「実質的な次元」
3. **擬似逆行列** — 逆行列が存在しなくても「できる範囲で逆をやる」道具と、最小ノルム解
4. **低ランク近似** — 行列を少ない情報で近似する最適な方法 (エッカート・ヤングの定理)
5. 計算法の一つである**片側ヤコビ法**

前提知識は [0. 数学の準備](./0_math_preliminaries.md) の範囲 (ベクトル・行列・内積・ランク・公式 $(A\mathbf{u}) \cdot \mathbf{v} = \mathbf{u} \cdot (A^\top \mathbf{v})$) と、[1. 線形最小二乗法](./1_least_squares_method.md)、[2. QR 分解](./2_qr_decomposition.md) の内容である。**固有値・固有ベクトルは前提としない**。SVD の説明に最低限必要な分だけ、次節でこの文書内に導入する。

対応するサンプルコードは `src/bin/3_singular_value_decomposition.rs` (SVD の実装本体は `src/lib.rs` の `jacobi_svd` / `svd_lstsq`) である。

## 2. 準備: 固有値と固有ベクトル (2×2 対称行列で最小限)

### 2.1 定義 — 行列を掛けても方向が変わらないベクトル

正方行列 $B \in \mathbb{R}^{m \times m}$ に対して、零ベクトルでないベクトル $\mathbf{v}$ と数 $\lambda$ が

$$
B \mathbf{v} = \lambda \mathbf{v}
$$

を満たすとき、$\lambda$ を $B$ の**固有値**、$\mathbf{v}$ を ($\lambda$ に対応する) **固有ベクトル**と呼ぶ。

意味を言葉にすると、こうなる。行列 $B$ を掛けると、普通のベクトルは方向も長さも変わる。ところが固有ベクトルだけは特別で、**方向が変わらず、長さが $\lambda$ 倍になるだけ**である ($\lambda < 0$ なら向きが反転する)。つまり固有ベクトルは「その行列にとって特別な軸」を指している。

なお、固有ベクトルの長さと符号には自由度がある。$B \mathbf{v} = \lambda \mathbf{v}$ なら、両辺を 2 倍して $B (2\mathbf{v}) = \lambda (2\mathbf{v})$ も成り立つから、$2\mathbf{v}$ も $-\mathbf{v}$ も同じ固有値の固有ベクトルである。以後は長さ 1 に正規化したものを使う。

### 2.2 2×2 での求め方 — 特性方程式

$B \mathbf{v} = \lambda \mathbf{v}$ を移項すると

$$
(B - \lambda I) \mathbf{v} = \mathbf{0}
$$

となる ($I$ は単位行列)。もし行列 $B - \lambda I$ が逆行列を持つなら、両辺に左からその逆行列を掛けて $\mathbf{v} = \mathbf{0}$ となってしまい、「零ベクトルでない $\mathbf{v}$」という条件に反する。したがって、

> $\lambda$ が固有値である $\iff$ $B - \lambda I$ が逆行列を持たない

ここで 2×2 行列が逆行列を持つ条件を確認しておく。$M = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$ に対して $ad - bc \neq 0$ ならば、

$$
M^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

が逆行列になる。実際に掛けて確かめると

$$
\begin{bmatrix} a & b \\ c & d \end{bmatrix}
\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
= \begin{bmatrix} ad - bc & -ab + ba \\ cd - dc & -cb + da \end{bmatrix}
= (ad - bc) \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
$$

となり、$ad - bc$ で割れば単位行列である。逆に $ad - bc = 0$ のときは、$\mathbf{x} = (d, -c)^\top$ を $M$ に掛けると

$$
M \mathbf{x} = \begin{bmatrix} ad - bc \\ cd - dc \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

となる。$(d, -c)^\top \neq \mathbf{0}$ ならこれが $M \mathbf{x} = \mathbf{0}$ の非零解であり、$M$ は逆行列を持てない ($c = d = 0$ の場合は第 2 行が全部 0 なので、やはり逆行列は無い)。まとめると、**2×2 行列が逆行列を持たない $\iff$ $ad - bc = 0$** である (この量 $ad - bc$ を**行列式**と呼ぶ)。

したがって、固有値は方程式

$$
\det(B - \lambda I) = 0
$$

すなわち 2×2 なら「$B - \lambda I$ の $ad - bc$ が 0」という $\lambda$ の 2 次方程式を解けば求まる。この方程式を**特性方程式**と呼ぶ。

### 2.3 具体例

後で SVD の計算にそのまま使う行列で練習する。

$$
B = \begin{bmatrix} 25 & 20 \\ 20 & 25 \end{bmatrix}
$$

特性方程式は

$$
\det(B - \lambda I)
= (25 - \lambda)(25 - \lambda) - 20 \cdot 20
= (25 - \lambda)^2 - 400 = 0
$$

$(25 - \lambda)^2 = 400$ より $25 - \lambda = \pm 20$、すなわち

$$
\lambda_1 = 45, \qquad \lambda_2 = 5
$$

固有ベクトルは $(B - \lambda I) \mathbf{v} = \mathbf{0}$ を解けばよい。$\lambda_1 = 45$ のとき

$$
(B - 45 I) \mathbf{v}
= \begin{bmatrix} -20 & 20 \\ 20 & -20 \end{bmatrix}
\begin{bmatrix} v_1 \\ v_2 \end{bmatrix}
= \begin{bmatrix} 0 \\ 0 \end{bmatrix}
$$

第 1 行から $-20 v_1 + 20 v_2 = 0$、つまり $v_1 = v_2$。長さ 1 に正規化して

$$
\mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
$$

$\lambda_2 = 5$ のときは $20 v_1 + 20 v_2 = 0$、つまり $v_2 = -v_1$。符号は自由に選べる (2.1 節) ので、後で図を描きやすい向きとして

$$
\mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

を選ぶ。検算しておく。$B \mathbf{v}_1 = \frac{1}{\sqrt{2}} (25 + 20, \; 20 + 25)^\top = \frac{1}{\sqrt{2}} (45, 45)^\top = 45 \, \mathbf{v}_1$。確かに固有値 45 である。

ここで $\mathbf{v}_1 \cdot \mathbf{v}_2 = \frac{1}{2}(1 \cdot (-1) + 1 \cdot 1) = 0$、つまり**2 本の固有ベクトルが直交している**ことに注目してほしい。これは偶然ではない。

### 2.4 対称行列の固有ベクトルは直交する

$B$ が**対称行列** ($B^\top = B$) で、$\lambda_1 \neq \lambda_2$ に対応する固有ベクトルを $\mathbf{v}_1, \mathbf{v}_2$ とする。[0. 数学の準備](./0_math_preliminaries.md) の公式 $(A\mathbf{u}) \cdot \mathbf{v} = \mathbf{u} \cdot (A^\top \mathbf{v})$ を使うと

$$
\lambda_1 (\mathbf{v}_1 \cdot \mathbf{v}_2)
= (\lambda_1 \mathbf{v}_1) \cdot \mathbf{v}_2
= (B \mathbf{v}_1) \cdot \mathbf{v}_2
= \mathbf{v}_1 \cdot (B^\top \mathbf{v}_2)
= \mathbf{v}_1 \cdot (B \mathbf{v}_2)
= \lambda_2 (\mathbf{v}_1 \cdot \mathbf{v}_2)
$$

両端を移項して $(\lambda_1 - \lambda_2)(\mathbf{v}_1 \cdot \mathbf{v}_2) = 0$。$\lambda_1 \neq \lambda_2$ だから $\mathbf{v}_1 \cdot \mathbf{v}_2 = 0$ である。

つまり**対称行列の固有ベクトルたちは、互いに直交する「特別な座標軸」を成す**。この事実が SVD の土台になる。

なお、一般の (対称とは限らない) 正方行列では固有値が実数にならないこともあり、固有ベクトルも直交するとは限らない。この文書で固有値を使うのは対称行列に対してだけなので、その心配は不要である。

## 3. 特異値分解の定義

### 3.1 定義

**任意の**行列 $A \in \mathbb{R}^{n \times m}$ ($n \geq m$ とする) は、次の形に分解できる。

$$
A = U \Sigma V^\top
$$

ここで

- $U \in \mathbb{R}^{n \times n}$ は直交行列。その列 $\mathbf{u}_1, \dots, \mathbf{u}_n$ を**左特異ベクトル**と呼ぶ
- $V \in \mathbb{R}^{m \times m}$ は直交行列。その列 $\mathbf{v}_1, \dots, \mathbf{v}_m$ を**右特異ベクトル**と呼ぶ
- $\Sigma \in \mathbb{R}^{n \times m}$ は対角成分以外が 0 の行列で、対角成分

$$
\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_m \geq 0
$$

を**特異値**と呼ぶ。**非負**で**大きい順**に並べる約束である。

QR 分解のときと同様に「薄い」版もある。$n > m$ のとき $\Sigma$ の下側 $n - m$ 行はすべて 0 なので、$U$ の左 $m$ 列 $U_1 \in \mathbb{R}^{n \times m}$ と正方対角行列 $\Sigma_1 \in \mathbb{R}^{m \times m}$ だけで $A = U_1 \Sigma_1 V^\top$ と書ける。これを**薄い SVD** と呼ぶ。`src/lib.rs` の `jacobi_svd` が返すのはこの薄い SVD である。

強調すべきは適用範囲の広さである。**正方でなくてよい。対称でなくてよい。ランク落ちしていてよい。** どんな行列にも SVD は必ず存在する。

### 3.2 列ごとの読み方

$A = U \Sigma V^\top$ の両辺に右から $V$ を掛けると ($V^\top V = I$ より) $AV = U\Sigma$ となり、これを列ごとに書くと

$$
A \mathbf{v}_j = \sigma_j \mathbf{u}_j \qquad (j = 1, \dots, m)
$$

となる。言葉にすると、

> 入力側の直交する単位ベクトルの組 $\mathbf{v}_1, \dots, \mathbf{v}_m$ が、出力側の直交する単位ベクトルの組 $\mathbf{u}_1, \dots, \mathbf{u}_m$ に、それぞれ $\sigma_j$ 倍されて写る。

固有ベクトルの式 $B\mathbf{v} = \lambda \mathbf{v}$ とよく似ているが、決定的な違いがある。固有値の式は「入力と出力で**同じ**ベクトル」を要求するのに対し、SVD は「入力用の基底 $\mathbf{v}_j$ と出力用の基底 $\mathbf{u}_j$ を**別々に**選んでよい」。この自由度のおかげで、正方でない行列や対称でない行列でも必ず分解できるのである。

## 4. 2×2 の手計算例 — 回して、伸ばして、回す

定義だけでは実感が湧かないので、具体的な行列で SVD を最後まで手計算する。

$$
A = \begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}
$$

### 4.1 $A^\top A$ を作ると対称行列になる

$A$ 自身は対称でないので、2 章の道具が直接は使えない。そこで $A^\top A$ を計算する。

$$
A^\top A
= \begin{bmatrix} 3 & 4 \\ 0 & 5 \end{bmatrix}
\begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}
= \begin{bmatrix} 3 \cdot 3 + 4 \cdot 4 & 3 \cdot 0 + 4 \cdot 5 \\ 0 \cdot 3 + 5 \cdot 4 & 0 \cdot 0 + 5 \cdot 5 \end{bmatrix}
= \begin{bmatrix} 25 & 20 \\ 20 & 25 \end{bmatrix}
$$

これは 2.3 節で固有値を求めた行列そのものである ($A^\top A$ が対称になるのは偶然ではない。$(A^\top A)^\top = A^\top (A^\top)^\top = A^\top A$ が常に成り立つ)。固有値と固有ベクトルは求め済みで、

$$
\lambda_1 = 45, \quad \mathbf{v}_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix},
\qquad
\lambda_2 = 5, \quad \mathbf{v}_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} -1 \\ 1 \end{bmatrix}
$$

### 4.2 特異値は固有値の平方根

特異値は $A^\top A$ の固有値の平方根、と定める (なぜそう定めるかは 5 章で示す。ここでは手を動かす)。

$$
\sigma_1 = \sqrt{45} = 3\sqrt{5} \approx 6.708,
\qquad
\sigma_2 = \sqrt{5} \approx 2.236
$$

右特異ベクトルは $A^\top A$ の固有ベクトル $\mathbf{v}_1, \mathbf{v}_2$ をそのまま使う。左特異ベクトルは $A \mathbf{v}_j = \sigma_j \mathbf{u}_j$ を満たすように、

$$
\mathbf{u}_j = \frac{1}{\sigma_j} A \mathbf{v}_j
$$

で作る。計算すると

$$
A \mathbf{v}_1
= \frac{1}{\sqrt{2}} \begin{bmatrix} 3 \cdot 1 + 0 \cdot 1 \\ 4 \cdot 1 + 5 \cdot 1 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 3 \\ 9 \end{bmatrix}
\quad\Longrightarrow\quad
\mathbf{u}_1 = \frac{1}{3\sqrt{5}} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} 3 \\ 9 \end{bmatrix}
= \frac{1}{\sqrt{10}} \begin{bmatrix} 1 \\ 3 \end{bmatrix}
$$

(最後の等号は $3\sqrt{5} \cdot \sqrt{2} = 3\sqrt{10}$ で分母分子を 3 で約分した)。同様に

$$
A \mathbf{v}_2
= \frac{1}{\sqrt{2}} \begin{bmatrix} 3 \cdot (-1) + 0 \cdot 1 \\ 4 \cdot (-1) + 5 \cdot 1 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} -3 \\ 1 \end{bmatrix}
\quad\Longrightarrow\quad
\mathbf{u}_2 = \frac{1}{\sqrt{5}} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} -3 \\ 1 \end{bmatrix}
= \frac{1}{\sqrt{10}} \begin{bmatrix} -3 \\ 1 \end{bmatrix}
$$

ここで $\mathbf{u}_1 \cdot \mathbf{u}_2 = \frac{1}{10}(1 \cdot (-3) + 3 \cdot 1) = 0$、また $\|\mathbf{u}_1\| = \|\mathbf{u}_2\| = \sqrt{\frac{1 + 9}{10}} = 1$。**作った覚えがないのに、$\mathbf{u}_1, \mathbf{u}_2$ も勝手に正規直交になっている**。これも 5 章で種明かしする。

### 4.3 組み立てと検算

$$
U = \frac{1}{\sqrt{10}} \begin{bmatrix} 1 & -3 \\ 3 & 1 \end{bmatrix},
\qquad
\Sigma = \begin{bmatrix} 3\sqrt{5} & 0 \\ 0 & \sqrt{5} \end{bmatrix},
\qquad
V = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}
$$

本当に $U \Sigma V^\top = A$ に戻るか、一行ずつ確かめる。まず

$$
\Sigma V^\top
= \begin{bmatrix} 3\sqrt{5} & 0 \\ 0 & \sqrt{5} \end{bmatrix}
\cdot \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ -1 & 1 \end{bmatrix}
= \frac{1}{\sqrt{2}} \begin{bmatrix} 3\sqrt{5} & 3\sqrt{5} \\ -\sqrt{5} & \sqrt{5} \end{bmatrix}
$$

次に $U$ を左から掛ける。$\sqrt{10} \cdot \sqrt{2} = \sqrt{20} = 2\sqrt{5}$ に注意して、

$$
U (\Sigma V^\top)
= \frac{1}{2\sqrt{5}}
\begin{bmatrix} 1 & -3 \\ 3 & 1 \end{bmatrix}
\begin{bmatrix} 3\sqrt{5} & 3\sqrt{5} \\ -\sqrt{5} & \sqrt{5} \end{bmatrix}
= \frac{1}{2\sqrt{5}}
\begin{bmatrix} 3\sqrt{5} + 3\sqrt{5} & 3\sqrt{5} - 3\sqrt{5} \\ 9\sqrt{5} - \sqrt{5} & 9\sqrt{5} + \sqrt{5} \end{bmatrix}
= \frac{1}{2\sqrt{5}}
\begin{bmatrix} 6\sqrt{5} & 0 \\ 8\sqrt{5} & 10\sqrt{5} \end{bmatrix}
= \begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}
$$

確かに $A$ に戻った。

### 4.4 幾何的な意味 — 単位円が楕円に写る

$A\mathbf{x} = U \Sigma V^\top \mathbf{x}$ を右から順に読むと、ベクトル $\mathbf{x}$ には 3 つの変換が順に作用する。今回の例で各行列が何をするか、具体的に見る。

1. **$V^\top$ を掛ける = 回転。** $V$ は列が $\frac{1}{\sqrt{2}}(1,1)^\top$, $\frac{1}{\sqrt{2}}(-1,1)^\top$ の直交行列で、これは平面を反時計回りに 45° 回す回転行列である。その転置 $V^\top$ は逆回転、つまり**時計回りに 45° 回す**。この回転によって、$\mathbf{v}_1$ 方向が $x$ 軸に、$\mathbf{v}_2$ 方向が $y$ 軸に揃う。
2. **$\Sigma$ を掛ける = 軸に沿った伸縮。** $x$ 軸方向に $3\sqrt{5} \approx 6.71$ 倍、$y$ 軸方向に $\sqrt{5} \approx 2.24$ 倍する。回転や歪みは一切なく、**縦横に引き伸ばすだけ**である。
3. **$U$ を掛ける = 回転。** $U$ は $\cos\theta = \frac{1}{\sqrt{10}}$, $\sin\theta = \frac{3}{\sqrt{10}}$ の回転行列で、**反時計回りに約 71.6° 回す**。これで $x$ 軸だった方向が $\mathbf{u}_1$ に、$y$ 軸だった方向が $\mathbf{u}_2$ に向く。

単位円 (長さ 1 のベクトルの先端の集合) がどうなるか追いかけよう。

- 手順 1 の回転では、単位円は単位円のまま (回しても円は円)。
- 手順 2 の伸縮で、単位円は**横半径 $3\sqrt{5}$、縦半径 $\sqrt{5}$ の楕円**になる。
- 手順 3 の回転で、この楕円ごと約 71.6° 回る。楕円の長軸は $\mathbf{u}_1 = \frac{1}{\sqrt{10}}(1, 3)^\top$ の方向、短軸は $\mathbf{u}_2$ の方向を向く。

個別の点でも確かめる。単位円上の点 $\mathbf{v}_1 = \frac{1}{\sqrt{2}}(1,1)^\top$ は、手順 1 で $x$ 軸上の点 $(1, 0)^\top$ に回り ($V^\top \mathbf{v}_1$ は $V^\top V = I$ の第 1 列)、手順 2 で $(3\sqrt{5}, 0)^\top$ に伸び、手順 3 で $3\sqrt{5}\, \mathbf{u}_1$ に回る。全体として $A\mathbf{v}_1 = 3\sqrt{5}\, \mathbf{u}_1$、つまり**単位円上で最も強く引き伸ばされる点が $\mathbf{v}_1$ で、行き先が楕円の長軸の先端**である。同様に $\mathbf{v}_2$ は最も伸ばされない点で、行き先は短軸の先端 $\sqrt{5}\, \mathbf{u}_2$ である。

まとめると、

> **どんな行列も、単位円 (高次元なら単位球) を「回して・伸ばして・回した」楕円 (楕円体) に移す。楕円の各軸の半径が特異値 $\sigma_j$、軸の方向が左特異ベクトル $\mathbf{u}_j$ である。**

これが SVD の幾何的な意味であり、「任意の線形写像は回転・伸縮・回転の合成に過ぎない」という線形代数の中心的な事実である (直交行列には回転のほかに鏡映も含まれるが、絵としては「回す」で捉えてよい)。

ついでに一つ観察しておく。単位ベクトルはどれも、$A$ を掛けると長さが $\sigma_2 \approx 2.24$ 以上 $\sigma_1 \approx 6.71$ 以下になる (楕円上の点の原点からの距離は短半径と長半径の間にあるから)。たとえば $\mathbf{e}_1 = (1,0)^\top$ の行き先は $A\mathbf{e}_1 = (3,4)^\top$ で長さ 5 であり、確かに $2.24 \leq 5 \leq 6.71$ に収まっている。**特異値は「行列が各方向をどれだけ引き伸ばすか」の最大値と最小値**を与えるのである。

## 5. なぜ任意の行列で SVD が作れるのか

4 章の手順が一般の $A \in \mathbb{R}^{n \times m}$ でも通用することを確認する。手順は次の通りだった。

1. $A^\top A$ ($m \times m$ の対称行列) の固有値 $\lambda_1 \geq \dots \geq \lambda_m$ と正規直交な固有ベクトル $\mathbf{v}_1, \dots, \mathbf{v}_m$ を求める
2. $\sigma_j = \sqrt{\lambda_j}$ とする
3. $\sigma_j > 0$ に対して $\mathbf{u}_j = \frac{1}{\sigma_j} A \mathbf{v}_j$ とする

これが成立するために確認すべき点は 2 つある。

**(1) 固有値は非負か** ($\sqrt{\lambda_j}$ が取れるか)。$A^\top A \mathbf{v} = \lambda \mathbf{v}$ の両辺と $\mathbf{v}$ の内積を取り、公式 $(A^\top \mathbf{w}) \cdot \mathbf{v} = \mathbf{w} \cdot (A \mathbf{v})$ を $\mathbf{w} = A\mathbf{v}$ に使うと

$$
\lambda \|\mathbf{v}\|^2
= (\lambda \mathbf{v}) \cdot \mathbf{v}
= (A^\top A \mathbf{v}) \cdot \mathbf{v}
= (A \mathbf{v}) \cdot (A \mathbf{v})
= \|A \mathbf{v}\|^2 \geq 0
$$

$\|\mathbf{v}\|^2 > 0$ だから $\lambda \geq 0$ である。よってすべての固有値の平方根が取れる。

**(2) $\mathbf{u}_j$ は勝手に正規直交になるか** (4.2 節の「種明かし」)。まず長さは

$$
\|A \mathbf{v}_j\|^2
= (A \mathbf{v}_j) \cdot (A \mathbf{v}_j)
= \mathbf{v}_j \cdot (A^\top A \mathbf{v}_j)
= \mathbf{v}_j \cdot (\lambda_j \mathbf{v}_j)
= \lambda_j = \sigma_j^2
$$

なので $\|\mathbf{u}_j\| = \|A\mathbf{v}_j\| / \sigma_j = 1$。次に $i \neq j$ のとき

$$
\mathbf{u}_i \cdot \mathbf{u}_j
= \frac{(A \mathbf{v}_i) \cdot (A \mathbf{v}_j)}{\sigma_i \sigma_j}
= \frac{\mathbf{v}_i \cdot (A^\top A \mathbf{v}_j)}{\sigma_i \sigma_j}
= \frac{\lambda_j (\mathbf{v}_i \cdot \mathbf{v}_j)}{\sigma_i \sigma_j}
= 0
$$

($\mathbf{v}_i \cdot \mathbf{v}_j = 0$ を使った)。つまり、**$A$ を通しても右特異ベクトルたちの直交性が壊れない**。これが「入力の直交基底 → 出力の直交基底 + 伸縮」という SVD の構造が任意の行列で成り立つ理由である。

$\sigma_j = 0$ となる $j$ については $A \mathbf{v}_j = \mathbf{0}$ なので $\mathbf{u}_j$ をこの式では作れないが、既に作った $\mathbf{u}_j$ たちと直交する単位ベクトルを適当に補えば $U$ は直交行列として完成する (`jacobi_svd` の実装では、対応する $U$ の列を零ベクトルのままにしておく簡略化をしている。$\sigma_j = 0$ の項は $A = U\Sigma V^\top$ の積に寄与しないため、再構成には影響しない)。

**注意: この構成法をそのまま数値計算に使ってはいけない。** [1. 線形最小二乗法](./1_least_squares_method.md) で見たとおり、$A^\top A$ を作ると条件数が $\kappa(A)^2$ に悪化する。上の構成は「SVD が必ず存在する」ことの証明としては正しいが、実際の計算では $A^\top A$ を経由しないアルゴリズム (9 章の片側ヤコビ法や、実用ライブラリの Golub–Kahan 法) を使う。

## 6. SVD は行列の健康診断書

SVD が計算できると、行列に関する重要な量がすべて特異値から読み取れる。用語を 2 つ定義してから一覧する。

- $A$ の**値域** $\mathrm{range}(A)$: $A\mathbf{x}$ の形で書けるベクトル全体。「$A$ の出力が到達できる範囲」
- $A$ の**零空間** $\mathrm{null}(A)$: $A\mathbf{x} = \mathbf{0}$ となる $\mathbf{x}$ 全体。「$A$ に消されてしまう入力の方向」

非零の特異値の個数を $r$ とする ($\sigma_1 \geq \dots \geq \sigma_r > 0 = \sigma_{r+1} = \dots = \sigma_m$)。このとき

| 知りたい量 | 特異値・特異ベクトルでの表現 |
| --- | --- |
| ランク $\mathrm{rank}(A)$ | 非零特異値の個数 $r$ |
| 最大の伸び率 (2-ノルム) | $\sigma_1$ |
| 条件数 $\kappa_2(A)$ | $\sigma_1 / \sigma_r$ |
| 値域の正規直交基底 | $\mathbf{u}_1, \dots, \mathbf{u}_r$ |
| 零空間の正規直交基底 | $\mathbf{v}_{r+1}, \dots, \mathbf{v}_m$ |

それぞれ理由を述べる。

**ランク。** $A \mathbf{v}_j = \sigma_j \mathbf{u}_j$ より、$\sigma_j = 0$ の方向 $\mathbf{v}_j$ は $A$ に消される。任意の入力 $\mathbf{x}$ は $\mathbf{v}_1, \dots, \mathbf{v}_m$ の線形結合で書けるから、$A$ の出力は $\mathbf{u}_1, \dots, \mathbf{u}_r$ の線形結合で尽くされる。独立な出力方向は $r$ 本であり、これがランクである。同時に「零空間 = $\sigma_j = 0$ の $\mathbf{v}_j$ が張る空間」「値域 = $\sigma_j > 0$ の $\mathbf{u}_j$ が張る空間」も分かる。

**伸び率と条件数。** 4.4 節で見たとおり、単位ベクトルの行き先の長さは最大 $\sigma_1$、最小 $\sigma_r$ (零空間方向を除く) である。[2. QR 分解](./2_qr_decomposition.md) で登場した条件数は「誤差が最大何倍に増幅されるか」の指標であり、伸び率の最大/最小比 $\sigma_1 / \sigma_r$ に等しい。

**数値ランクというグレーゾーン。** 実務では「ランク落ちか否か」は白黒つかない。浮動小数点誤差やデータのノイズのせいで、特異値は「ちょうど 0」ではなく「ほとんど 0」($10^{-15}$ など) として現れるからである。そこで「最大特異値との比が閾値以下の特異値は 0 とみなす」ことで**数値ランク**を定義する。特異値の並びを見れば、行列が「どのくらいランク落ちに近いか」を定量的に診断できる。QR 分解 (列ピボット付き) でもある程度のランク推定はできるが、最も信頼できる診断書が SVD である。

サンプルコード `src/bin/3_singular_value_decomposition.rs` の実験 2 では、第 3 列 = 第 1 列 + 第 2 列 となる線形従属な 5×3 行列を作って SVD している。出力は

```
特異値: [2.1541, 1.1164, 0.0000] (最後がほぼ 0 → 数値ランク 2)
```

であり、3 列あるのに独立な方向は 2 本しかないこと (数値ランク 2) が特異値から一目で分かる。テスト `test_rank_detection` では、相対閾値 $\sigma_j > \sigma_1 \times 10^{-10}$ で数えたランクが 2 になることを確認している。

## 7. 擬似逆行列 — 逆行列が無くても「できる範囲で逆」

### 7.1 動機

連立方程式 $A\mathbf{x} = \mathbf{b}$ は、$A$ が正方で可逆なら $\mathbf{x} = A^{-1}\mathbf{b}$ と解ける。しかし最小二乗法で扱う $A$ は縦長 (方程式の数 > 未知数の数) で、逆行列は存在しない。さらにランク落ちしていると、[1. 線形最小二乗法](./1_least_squares_method.md) の正規方程式も一意に解けない。

そこで発想を変える。「完全な逆」が無理でも、**できる範囲で逆をやる**行列を作れないか。SVD がその答えを与える。

### 7.2 まず対角行列で考える

$A$ が対角行列なら話は簡単である。たとえば

$$
\Sigma = \begin{bmatrix} 2 & 0 \\ 0 & 3 \\ 0 & 0 \end{bmatrix}
\quad \text{に対して} \quad
\Sigma \mathbf{x} = \begin{bmatrix} 2 x_1 \\ 3 x_2 \\ 0 \end{bmatrix}
$$

なので、「逆をやる」には各成分を $\frac{1}{2}$ 倍、$\frac{1}{3}$ 倍すればよい。第 3 成分はどんな $\mathbf{x}$ でも 0 にしかならないから、そこは**諦める** (0 を掛けて無視する)。この「非零の対角成分だけ逆数にして転置した」行列

$$
\Sigma^{+} = \begin{bmatrix} 1/2 & 0 & 0 \\ 0 & 1/3 & 0 \end{bmatrix}
$$

を $\Sigma$ の擬似逆と呼ぶ。対角成分に 0 がある場合 (ランク落ち) も、その成分は $\frac{1}{0}$ とせずに 0 のままにする。「逆にできる方向だけ逆にして、できない方向は手を出さない」のである。

### 7.3 一般の行列へ — 擬似逆行列の定義

一般の $A = U \Sigma V^\top$ に対して、

$$
A^{+} = V \Sigma^{+} U^\top
$$

を $A$ の**擬似逆行列** (ムーア・ペンローズ逆行列) と呼ぶ。作り方を言葉にすると、「$A$ は回して ($V^\top$)・伸ばして ($\Sigma$)・回す ($U$) 変換だったから、その逆は、逆回し ($U^\top$)・逆伸ばし ($\Sigma^{+}$)・逆回し ($V$) を逆順にやればよい」となる。伸ばせない方向 ($\sigma_j = 0$) だけは諦める、というのが「擬似」の意味である。

$A$ が正方で可逆なら全部の $\sigma_j$ が正なので何も諦める必要がなく、$A^{+} = A^{-1}$ に一致することが確かめられる。

### 7.4 最小二乗解との関係

擬似逆行列の最大の使いどころは最小二乗法である。次が成り立つ。

> 最小二乗問題 $\min_{\mathbf{x}} \| A\mathbf{x} - \mathbf{b} \|^2$ の解は $\mathbf{x} = A^{+} \mathbf{b}$ で与えられる。解が一意でない (ランク落ちの) 場合は、解の中で**ノルム $\|\mathbf{x}\|$ が最小のもの**を与える。

成分で書き下すと、$A^{+} = V \Sigma^{+} U^\top$ の構造から

$$
\mathbf{x} = A^{+} \mathbf{b} = \sum_{j :\, \sigma_j > 0} \frac{\mathbf{u}_j \cdot \mathbf{b}}{\sigma_j} \, \mathbf{v}_j
$$

となる。読み方は「$\mathbf{b}$ を出力側の基底 $\mathbf{u}_j$ に分解し、各成分を $\sigma_j$ で割って ($\sigma_j$ 倍の逆)、入力側の基底 $\mathbf{v}_j$ で組み立て直す」。`src/lib.rs` の `svd_lstsq` はこの式をそのまま実装している。

なぜこれが最小二乗解になるのか、直感的に述べる。$\mathbf{b}$ のうち値域 ($\mathbf{u}_1, \dots, \mathbf{u}_r$ の張る空間) に入っている成分は、上の式で完全に再現できる。値域の外の成分はどんな $\mathbf{x}$ でも表現できないから、残差として残る。これは「$\mathbf{b}$ を値域に直交射影した点を狙う」という [1. 線形最小二乗法](./1_least_squares_method.md) の幾何そのものである。さらに、和を $\sigma_j > 0$ の項に限っているため、解 $\mathbf{x}$ は零空間方向 ($\mathbf{v}_{r+1}, \dots, \mathbf{v}_m$) の成分を一切持たない。零空間方向の成分は残差を変えずにノルムだけを増やすので、それを持たない解がノルム最小である。

### 7.5 最小ノルム解を手計算で確かめる

サンプルコードの実験 2 の設定を、そのまま手で追える形にする。5×3 行列 $A$ の列を $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2$ とする。このとき

$$
A \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}
= \mathbf{a}_1 + \mathbf{a}_2 - \mathbf{a}_3 = \mathbf{0}
$$

なので、$\mathbf{n} = (1, 1, -1)^\top$ は零空間の方向である。したがって $\mathbf{x}^\ast$ が $A\mathbf{x} = \mathbf{b}$ の解なら、$\mathbf{x}^\ast + t\,\mathbf{n}$ ($t$ は任意の実数) もすべて解であり、**解は一直線分の無数にある**。

コードでは真の解を $\mathbf{x}^\ast = (1, 2, 0)^\top$ として $\mathbf{b} = A \mathbf{x}^\ast$ を作っている。無数の解 $\mathbf{x}^\ast + t\,\mathbf{n}$ の中でノルム最小のものを求めよう。

$$
\| \mathbf{x}^\ast + t\,\mathbf{n} \|^2
= \| \mathbf{x}^\ast \|^2 + 2t \, (\mathbf{x}^\ast \cdot \mathbf{n}) + t^2 \|\mathbf{n}\|^2
$$

これは $t$ の 2 次関数で、最小になるのは微分が 0 のとき、つまり

$$
t = -\frac{\mathbf{x}^\ast \cdot \mathbf{n}}{\|\mathbf{n}\|^2}
= -\frac{1 \cdot 1 + 2 \cdot 1 + 0 \cdot (-1)}{1^2 + 1^2 + (-1)^2}
= -\frac{3}{3} = -1
$$

よって最小ノルム解は

$$
\mathbf{x}
= \begin{bmatrix} 1 \\ 2 \\ 0 \end{bmatrix} - \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}
= \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix},
\qquad
\|\mathbf{x}\| = \sqrt{2} \approx 1.414
$$

サンプルコードの出力

```
最小ノルム解: [-0.0000, 1.0000, 1.0000]
残差 = 6.497e-16, ‖x‖ = 1.4142 (元の解 [1.0, 2.0, 0.0] の ‖x‖ = 2.2361)
```

と完全に一致する。残差はほぼ 0 (方程式はちゃんと満たしている) のに、ノルムは元の解の $\sqrt{5} \approx 2.236$ より小さい。テスト `test_min_norm_solution` では、解が零空間方向 $\mathbf{n}$ と直交していること (= ノルム最小の証) も検証している。

### 7.6 小さい特異値の危険 — $1/\sigma$ の爆発

7.4 節の展開式には $\frac{1}{\sigma_j}$ が現れる。ここに悪条件問題の正体が見える。**$\sigma_j$ が小さいと、$\mathbf{b}$ に含まれる誤差が $\frac{1}{\sigma_j}$ 倍に増幅される**のである。

数値例で見る。特異値が $\sigma_1 = 10$, $\sigma_2 = 0.001$ の行列 (条件数 $\kappa = 10^4$) を考える。$\mathbf{b}$ の測定に $\mathbf{u}_2$ 方向の誤差 $0.01$ が混ざったとすると、解の変化は展開式より

$$
\Delta \mathbf{x} = \frac{0.01}{\sigma_2} \mathbf{v}_2 = \frac{0.01}{0.001} \mathbf{v}_2 = 10 \, \mathbf{v}_2
$$

**入力のわずか 0.01 の誤差が、解では 10 という巨大な変動になる**。同じ 0.01 の誤差でも $\mathbf{u}_1$ 方向なら $\frac{0.01}{10} = 0.001$ にしかならない。つまり誤差の増幅は特異値の小さい方向に集中して起きる。

対策は率直である。「$\sigma_2 = 0.001$ の方向は、データがほとんど情報を持っていない (単位円がほぼ潰される) 方向なのだから、その方向の復元は諦める」。すなわち、閾値以下の特異値を 0 とみなして展開式の和から除外する。これを**切り捨て SVD** (truncated SVD) と呼ぶ。上の例なら $\sigma_2$ の項を捨てて $\mathbf{x} = \frac{\mathbf{u}_1 \cdot \mathbf{b}}{\sigma_1} \mathbf{v}_1$ とする。真の解の $\mathbf{v}_2$ 成分は失われる (偏りが生じる) が、誤差の 10 倍増幅よりはるかにましである、という判断である。これはノイズ増幅と偏りを天秤にかける**正則化**の一種であり、`svd_lstsq` の引数 `rcond` (相対閾値。$\sigma_j \leq \sigma_1 \cdot \texttt{rcond}$ を捨てる) がこの閾値に当たる。

この「小さい特異値の方向が誤差を増幅するので、その方向を抑え込む」という視点は、[8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) でダンピングがなぜ効くのかを理解する土台になる。

## 8. 低ランク近似 — 特異値の大きい順に足す

### 8.1 SVD はランク 1 行列の和である

$A = U \Sigma V^\top$ を成分の積の和として書き直すと、

$$
A = \sum_{j=1}^{r} \sigma_j \, \mathbf{u}_j \mathbf{v}_j^\top
$$

となる。ここで $\mathbf{u}_j \mathbf{v}_j^\top$ は「列ベクトル × 行ベクトル」の積で、$n \times m$ 行列になる (成分は $(\mathbf{u}_j)_i (\mathbf{v}_j)_k$)。どの列も $\mathbf{u}_j$ の定数倍なのでランク 1 の行列である。つまり、

> **どんな行列も、ランク 1 の行列 $r$ 枚の重ね合わせであり、各層の「重み」が特異値 $\sigma_j$ である。**

4 章の例 $A = \begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}$ で確かめる。第 1 層は

$$
\sigma_1 \mathbf{u}_1 \mathbf{v}_1^\top
= 3\sqrt{5} \cdot \frac{1}{\sqrt{10}} \begin{bmatrix} 1 \\ 3 \end{bmatrix} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \end{bmatrix}
= \frac{3\sqrt{5}}{\sqrt{20}} \begin{bmatrix} 1 & 1 \\ 3 & 3 \end{bmatrix}
= \frac{3}{2} \begin{bmatrix} 1 & 1 \\ 3 & 3 \end{bmatrix}
= \begin{bmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{bmatrix}
$$

($\sqrt{20} = 2\sqrt{5}$ を使った)。第 2 層は

$$
\sigma_2 \mathbf{u}_2 \mathbf{v}_2^\top
= \sqrt{5} \cdot \frac{1}{\sqrt{10}} \begin{bmatrix} -3 \\ 1 \end{bmatrix} \cdot \frac{1}{\sqrt{2}} \begin{bmatrix} -1 & 1 \end{bmatrix}
= \frac{\sqrt{5}}{2\sqrt{5}} \begin{bmatrix} 3 & -3 \\ -1 & 1 \end{bmatrix}
= \begin{bmatrix} 1.5 & -1.5 \\ -0.5 & 0.5 \end{bmatrix}
$$

2 枚を足すと

$$
\begin{bmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{bmatrix} + \begin{bmatrix} 1.5 & -1.5 \\ -0.5 & 0.5 \end{bmatrix}
= \begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix} = A
$$

確かに戻る。

### 8.2 途中で打ち切る = 低ランク近似

この和を、重み (特異値) の大きい順に $k$ 項だけで打ち切った行列を

$$
A_k = \sum_{j=1}^{k} \sigma_j \, \mathbf{u}_j \mathbf{v}_j^\top
$$

と書く。$A_k$ のランクは $k$ 以下である。上の例で $k = 1$ なら $A_1 = \begin{bmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{bmatrix}$ であり、誤差は捨てた第 2 層そのもの、

$$
A - A_1 = \begin{bmatrix} 1.5 & -1.5 \\ -0.5 & 0.5 \end{bmatrix}
$$

である。誤差の大きさを測るために、行列の**フロベニウスノルム** (全成分の 2 乗和の平方根)

$$
\| M \|_F = \sqrt{\sum_{i} \sum_{j} M_{ij}^2}
$$

を導入する (ベクトルのノルムの行列版である)。すると

$$
\| A - A_1 \|_F = \sqrt{1.5^2 + (-1.5)^2 + (-0.5)^2 + 0.5^2} = \sqrt{2.25 + 2.25 + 0.25 + 0.25} = \sqrt{5}
$$

これは**ちょうど $\sigma_2$ に等しい**。偶然ではなく、直交性のおかげで一般に

$$
\| A - A_k \|_F = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \dots + \sigma_r^2}
$$

が成り立つ (捨てた層の重みの 2 乗和がそのまま誤差になる)。

### 8.3 エッカート・ヤングの定理 — この打ち切りが最適

驚くべきことに、この単純な打ち切りは近似として**最適**である。

> **エッカート・ヤングの定理:** ランク $k$ 以下のあらゆる行列 $B$ の中で、$\| A - B \|_F$ を最小にするのは $B = A_k$ である (最大伸び率で測る 2-ノルムでも同じく最適で、そのときの誤差は $\sigma_{k+1}$)。

つまり「行列をランク $k$ で近似したければ、SVD をとって特異値の大きい成分から $k$ 個足せばよい。それ以上うまい方法は存在しない」。証明は割愛するが、直感はこうである。$A_k$ は「$A$ が最も強く伸ばす $k$ 方向」だけを残した行列であり、捨てたのは伸び率 $\sigma_{k+1}$ 以下の弱い方向だけ。どんなランク $k$ 行列も $k$ 方向ぶんの情報しか持てないので、強い方向を優先して残すのが最善、というわけである。

この定理は「小さい特異値は行列の本質的でない成分である」という直感に理論的な保証を与える。応用は広い。

- **画像圧縮**: 画像を輝度の行列とみなして $A_k$ で置き換えると、$n \times m$ 個の数値の代わりに $k(n + m + 1)$ 個 ($\mathbf{u}_j$, $\mathbf{v}_j$, $\sigma_j$ を $k$ 組) で済む
- **主成分分析 (PCA)**: データ行列の特異ベクトルは「データが最も散らばる方向」を与える
- ノイズ除去: 小さい特異値の層はノイズ由来であることが多く、打ち切りがノイズ除去になる

### 8.4 サンプルコードとの対応

実験 3 では、ランダムな 8×5 行列で $k = 1, 2, 3, 4$ の低ランク近似を作り、誤差と理論値を比べている。

```
特異値: [2.2166, 1.6294, 1.4407, 0.9025, 0.3896]
k=1: ‖A - A_k‖_F = 2.386773 (理論値 2.386773)
k=2: ‖A - A_k‖_F = 1.744072 (理論値 1.744072)
k=3: ‖A - A_k‖_F = 0.982975 (理論値 0.982975)
k=4: ‖A - A_k‖_F = 0.389641 (理論値 0.389641)
```

たとえば $k = 4$ の誤差 $0.389641$ は最後の特異値 $\sigma_5 = 0.3896$ そのもの、$k = 3$ の誤差は $\sqrt{\sigma_4^2 + \sigma_5^2} = \sqrt{0.9025^2 + 0.3896^2} \approx 0.98298$ であり、8.2 節の公式と 6 桁一致している。

## 9. 計算法: 片側ヤコビ法

最後に、`src/lib.rs` の `jacobi_svd` が使っている**片側ヤコビ法** (Hestenes 法) を説明する。5 章で注意したとおり $A^\top A$ を丸ごと作る方法は使えないので、別の道具立てが要る。

### 9.1 発想 — 列同士を直交にできれば勝ち

薄い SVD $A = U \Sigma V^\top$ を右から $V$ を掛けた形 $AV = U\Sigma$ で眺める。右辺 $U\Sigma$ は「正規直交な列 $\mathbf{u}_j$ をそれぞれ $\sigma_j$ 倍した」行列だから、**列同士が互いに直交する行列**である。つまり、

> 直交行列 $V$ をうまく選んで $B = AV$ の**列同士をすべて直交させる**ことができれば、$\sigma_j = \| B \text{ の第 } j \text{ 列} \|$、$\mathbf{u}_j = (B \text{ の第 } j \text{ 列}) / \sigma_j$ と置くだけで SVD が完成する。

実際、このとき $B = U \Sigma$ (対角 $\Sigma$、列直交 $U$) であり、$A = B V^\top = U \Sigma V^\top$ となる。問題は「列を直交化する $V$ をどう作るか」に帰着した。

### 9.2 2 列ずつ回転で直交化する

全列を一気に直交化するのは難しいので、**2 列ずつ**片付ける。$B$ の第 $p$ 列 $\mathbf{b}_p$ と第 $q$ 列 $\mathbf{b}_q$ を選び、この 2 列を平面回転 (角度 $\theta$) で混ぜる。

$$
\mathbf{b}_p' = c\, \mathbf{b}_p - s\, \mathbf{b}_q,
\qquad
\mathbf{b}_q' = s\, \mathbf{b}_p + c\, \mathbf{b}_q
\qquad (c = \cos\theta, \; s = \sin\theta)
$$

新しい 2 列が直交する ($\mathbf{b}_p' \cdot \mathbf{b}_q' = 0$) ように $\theta$ を決めたい。$\alpha_{pp} = \|\mathbf{b}_p\|^2$, $\alpha_{qq} = \|\mathbf{b}_q\|^2$, $\alpha_{pq} = \mathbf{b}_p \cdot \mathbf{b}_q$ と置いて内積を展開すると

$$
\mathbf{b}_p' \cdot \mathbf{b}_q'
= (c\, \mathbf{b}_p - s\, \mathbf{b}_q) \cdot (s\, \mathbf{b}_p + c\, \mathbf{b}_q)
= cs\, \alpha_{pp} + c^2 \alpha_{pq} - s^2 \alpha_{pq} - cs\, \alpha_{qq}
$$

これを 0 と置くと

$$
(c^2 - s^2)\, \alpha_{pq} = cs\, (\alpha_{qq} - \alpha_{pp})
$$

$t = \tan\theta = s / c$ を使って両辺を $c^2 \alpha_{pq}$ で割り、$\zeta = \dfrac{\alpha_{qq} - \alpha_{pp}}{2 \alpha_{pq}}$ と置くと

$$
1 - t^2 = 2 \zeta t
\quad\Longleftrightarrow\quad
t^2 + 2\zeta t - 1 = 0
\quad\Longrightarrow\quad
t = -\zeta \pm \sqrt{\zeta^2 + 1}
$$

2 解のうち絶対値が小さい方 (回転角が 45° 以下になる方) を選ぶと数値的に安定であり、分子を有理化した形

$$
t = \frac{\mathrm{sign}(\zeta)}{|\zeta| + \sqrt{1 + \zeta^2}},
\qquad
c = \frac{1}{\sqrt{1 + t^2}},
\qquad
s = c\, t
$$

が実装 (`jacobi_svd` 内) に現れる式である。$\alpha_{pq}$ は $A^\top A$ の $(p, q)$ 成分に他ならないが、**必要な 2 列分の内積をその場で計算するだけで、$A^\top A$ 全体を作って保持することはない**。これが「片側」(行列の右側からしか触らない) の名の由来であり、この方式は精度が良いことでも知られる。

### 9.3 スイープを繰り返す

回転で列 $p, q$ を直交にすると、以前直交させたはずの他のペアの直交性が少し崩れる。そこで「全ペア $(p, q)$ を一巡する」処理 (**スイープ**) を、すべてのペアの正規化内積 $|\alpha_{pq}| / \sqrt{\alpha_{pp} \alpha_{qq}}$ が許容誤差以下になるまで繰り返す。崩れ方は回を追うごとに小さくなることが知られており、実用上は数スイープ〜十数スイープで収束する (`jacobi_svd` は上限 60 スイープ)。

その間、掛けた回転をすべて単位行列に積み上げていったものが $V$ になる。収束したら 9.1 節のとおり列ノルムから $\sigma_j$ を取り出し、大きい順に並べ替えて完成である。

なお、実用ライブラリ (LAPACK など) の標準は、$A$ を直交変換で二重対角行列に変形してから特異値を求める Golub–Kahan 法系のアルゴリズムで、大きい行列では片側ヤコビ法より速い。本リポジトリで片側ヤコビ法を採っているのは、実装が短く、仕組みが上のように初等的に導出でき、精度も高いからである。

## 10. サンプルコードで確かめる

`src/bin/3_singular_value_decomposition.rs` は 3 つの実験からなる。`cargo run --bin 3_singular_value_decomposition` で実行できる。

**実験 1: SVD の性質検証 (ランダムな 6×4 行列)。** `jacobi_svd` の出力が定義どおりか、3 つの量で確かめる。

```
特異値: [2.0385, 1.6211, 1.0628, 0.2856]
‖UΣVᵀ - A‖ = 2.474e-15 (再構成)
‖UᵀU - I‖  = 6.813e-16, ‖VᵀV - I‖ = 1.427e-15 (直交性)
条件数 κ = σ_max/σ_min = 7.137
```

再構成誤差 $\| U\Sigma V^\top - A \|_F$ と直交性の崩れ $\| U^\top U - I \|_F$, $\| V^\top V - I \|_F$ がいずれも $10^{-15}$ 前後、つまり倍精度浮動小数点の精度いっぱいで満たされている。条件数 $\kappa = \sigma_1 / \sigma_4 \approx 7$ は健康な (悪条件でない) 行列であることを示す (6 章)。

**実験 2: ランク落ちの検出と最小ノルム解。** 6 章 (数値ランク 2 の検出) と 7.5 節 (最小ノルム解 $(0, 1, 1)^\top$) で見たとおり、理論の予想と数値結果が一致する。

**実験 3: 低ランク近似。** 8.4 節で見たとおり、打ち切り誤差がエッカート・ヤングの定理の理論値 $\sqrt{\sigma_{k+1}^2 + \dots + \sigma_m^2}$ と 6 桁一致する。

またテストには、特異値の定義の分かりやすい例が入っている (`test_known_singular_values`)。対角成分が $3, -4$ の 3×2 行列の特異値は $4, 3$ である。$-4$ が $4$ になるのは特異値が**非負**と約束されているから ($\mathbf{u}$ か $\mathbf{v}$ の符号で吸収する)、$4$ が先に来るのは**降順**と約束されているからである。ほかに、さまざまなサイズでの性質検証 (`test_svd_properties`)、相対閾値によるランク検出 (`test_rank_detection`)、最小ノルム解が零空間と直交すること (`test_min_norm_solution`) を自動テストしている。

## 11. まとめ

> SVD は「任意の線形写像 = 回転 → 軸ごとの伸縮 → 回転」という標準形 $A = U \Sigma V^\top$ を与える分解である。単位円は半径 $\sigma_j$・軸 $\mathbf{u}_j$ の楕円に写り、特異値の並びが行列の健康状態 (ランク・条件数・実質的な次元) をすべて教えてくれる。

- 固有値・固有ベクトルは「行列を掛けても方向が変わらないベクトル」であり、対称行列では直交する。SVD は $A^\top A$ の固有値の平方根 (特異値) と固有ベクトル (右特異ベクトル) から構成できるが、数値計算では条件数が 2 乗に悪化するため $A^\top A$ を経由しない (この文書では片側ヤコビ法を使った)。
- 擬似逆行列 $A^{+} = V \Sigma^{+} U^\top$ は「逆にできる方向だけ逆をやり、$\sigma_j = 0$ の方向は諦める」行列で、ランク落ちした最小二乗問題でも**最小ノルム解**を与える。QR 分解が要求した「列が線形独立」という前提が不要になる。
- 解の展開式に現れる $1/\sigma_j$ は、小さい特異値の方向で誤差を爆発的に増幅する。閾値以下の特異値を捨てる切り捨て SVD は、この増幅を抑える正則化である。
- 低ランク近似は「特異値の大きい層から順に足す」だけで作れ、エッカート・ヤングの定理によりそれが最適であることが保証される。
- 最小二乗法の解法としては SVD が最も頑健であり、QR 分解でも扱えないランク落ち・悪条件のケースの最後の砦になる。そのぶん計算コストは QR より高い。
- 「小さい特異値が誤差を増幅する」という本章の視点は、[8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) でダンピングが悪条件なヤコビ行列に効く理由を理解する土台になる。
