# 線形最小二乗法 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

著者による学習メモ: [`docs/1_least_squares_method.md`](../1_least_squares_method.md)

この文書は [0. 数学の準備](./0_math_preliminaries.md) の内容 (Σ記法、ベクトルと内積、行列、線形独立、偏微分と勾配、凸関数) を前提とする。忘れた箇所があればそのつど戻って確認してほしい。

## 1. 問題設定 — 「解けない連立方程式」をどうにかしたい

### 1.1 データに関数を当てはめる

実験や測定で $n$ 個のデータ点

$$
(x_1, y_1), \ (x_2, y_2), \ \dots, \ (x_n, y_n)
$$

が得られたとする。例えば $x_i$ が時刻で $y_i$ がそのときの温度、といった具合である。このデータの背後にある傾向を、なめらかな関数 $\hat{y}(x)$ で表したい。「データに関数を**フィッティング**する」とはこのことである。

どんな関数を当てはめるかは、あらかじめ形を決めておく。ここでは

$$
\hat{y}(x) = \sum_{k=1}^{m} \beta_k \, \phi_k(x)
= \beta_1 \phi_1(x) + \beta_2 \phi_2(x) + \cdots + \beta_m \phi_m(x)
$$

という形を考える ($\sum$ は $k = 1$ から $m$ までの和を表す。[0. 数学の準備](./0_math_preliminaries.md) の Σ記法を参照)。ここで

- $\phi_1(x), \dots, \phi_m(x)$ は**基底関数** (basis function) と呼ばれる、**自分で選んで固定する**関数
- $\beta_1, \dots, \beta_m$ が、データから決めたい**未知の係数**

である。つまり「関数の部品 $\phi_k$ は決めておき、その混ぜ合わせ方 $\beta_k$ をデータに合うように決める」というのがこの問題である。

基底関数の選び方の例を挙げる。

- $\phi_1(x) = 1, \ \phi_2(x) = x$ と選べば $\hat{y} = \beta_1 + \beta_2 x$、つまり**直線**
- $\phi_k(x) = x^{k-1}$ ($k = 1, \dots, m$) と選べば $\hat{y} = \beta_1 + \beta_2 x + \cdots + \beta_m x^{m-1}$、つまり**多項式**
- $\phi_1(x) = \sin x, \ \phi_2(x) = \cos x$ と選べば $\hat{y} = \beta_1 \sin x + \beta_2 \cos x$ のような**三角関数の重ね合わせ**

ここで用語について 1 つ注意しておく。この手法は**線形**最小二乗法と呼ばれるが、「線形」とは**未知数 $\beta_k$ について 1 次式である**という意味であって、「直線しか扱えない」という意味ではない。上の例のとおり、$\phi_k(x) = x^2$ や $\sin x$ のような曲がった関数を部品にしてもよい。$\hat{y}(x)$ を「$\beta_k$ の式」と見たときに 1 次式 (各 $\beta_k$ が単独で、掛け算されずに現れる) であればよいのである。逆に $\hat{y} = e^{\beta x}$ のように未知数 $\beta$ が指数の肩に乗っているモデルは「$\beta$ について非線形」であり、この文書の手法では直接扱えない (第 8 節で触れる)。

### 1.2 連立方程式として書いてみる

「データに合うように $\beta_k$ を決める」を素直に式にすると、各データ点 $i = 1, \dots, n$ について

$$
y_i = \beta_1 \phi_1(x_i) + \beta_2 \phi_2(x_i) + \cdots + \beta_m \phi_m(x_i)
$$

が成り立ってほしい、ということになる。ここで注意してほしいのは、$\phi_k(x_i)$ は「決まった関数に決まった数値を代入したもの」なので**ただの数**だという点である。つまり上の式は、未知数 $\beta_1, \dots, \beta_m$ についての **1 次の連立方程式**である。未知数が $m$ 個、式が $n$ 本ある。

これをベクトルと行列でまとめて書く ([0. 数学の準備](./0_math_preliminaries.md) の「行列×ベクトル」を参照)。

$$
\mathbf{y} =
\begin{bmatrix}
y_1 \\ y_2 \\ \vdots \\ y_n
\end{bmatrix}
\in \mathbb{R}^n,
\qquad
\Phi =
\begin{bmatrix}
\phi_1(x_1) & \phi_2(x_1) & \cdots & \phi_m(x_1) \\
\phi_1(x_2) & \phi_2(x_2) & \cdots & \phi_m(x_2) \\
\vdots & \vdots & \ddots & \vdots \\
\phi_1(x_n) & \phi_2(x_n) & \cdots & \phi_m(x_n)
\end{bmatrix}
\in \mathbb{R}^{n \times m},
\qquad
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_1 \\ \beta_2 \\ \vdots \\ \beta_m
\end{bmatrix}
\in \mathbb{R}^m
$$

とおくと、$n$ 本の方程式はまとめて

$$
\mathbf{y} = \Phi \boldsymbol{\beta}
$$

と書ける。$\Phi$ の第 $i$ 行は「$i$ 番目のデータ点における各基底関数の値」、第 $k$ 列は「$k$ 番目の基底関数を全データ点で評価した値の並び」である。この $\Phi$ を**計画行列** (design matrix) と呼ぶ。

### 1.3 なぜ厳密には解けないのか

もしデータ数と未知数の個数が等しく ($n = m$)、しかも運良く方程式が独立なら、この連立方程式は普通に解ける。しかし現実のフィッティングでは通常

- **データ数の方が多い** ($n > m$)。例えば 100 点のデータに直線 (未知数 2 個) を当てはめる。
- **観測には誤差が乗る**。$y_i$ は真の値からわずかにずれている。

という状況にある。未知数 2 個に対して式が 100 本あれば、誤差のせいで全部を同時に満たすことはまず不可能である。式の本数が未知数より多い連立方程式を**過剰決定系** (overdetermined system) と呼ぶ。

そこで発想を切り替える。「全部の式をぴったり満たす $\boldsymbol{\beta}$」は諦め、「全部の式を**できるだけ惜しく**満たす $\boldsymbol{\beta}$」を探す。これが最小二乗法の出発点である。次節では「できるだけ惜しく」を数式でどう表すかを考える。

## 2. 目的関数 — なぜ「二乗和」なのか

### 2.1 残差とその二乗和

$\boldsymbol{\beta}$ を 1 つ決めると、各データ点でのモデルの予測値 $\hat{y}(x_i)$ が決まり、実際の観測値 $y_i$ とのずれ

$$
r_i = y_i - \hat{y}(x_i) = y_i - \sum_{k=1}^{m} \beta_k \phi_k(x_i)
$$

が決まる。この $r_i$ を**残差** (residual) と呼ぶ。$n$ 個まとめてベクトルで書けば

$$
\mathbf{r} = \mathbf{y} - \Phi \boldsymbol{\beta}
$$

である。残差が全部 0 なら方程式は完璧に解けている。そうはいかないので、「残差の全体的な大きさ」を 1 つの数値にまとめて、それを最小化することにする。

最小二乗法では、残差の**二乗和**

$$
E(\boldsymbol{\beta})
= \sum_{i=1}^{n} r_i^2
= \sum_{i=1}^{n} \Bigl( y_i - \sum_{k=1}^{m} \beta_k \phi_k(x_i) \Bigr)^2
$$

を採用する。ここでベクトルのノルム ([0. 数学の準備](./0_math_preliminaries.md) 参照) を思い出すと、$\|\mathbf{r}\|^2 = r_1^2 + r_2^2 + \cdots + r_n^2$ だったから、$E$ は

$$
E(\boldsymbol{\beta}) = \| \mathbf{r} \|^2 = \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2
$$

とも書ける。つまり $E$ は「$n$ 次元空間の中での、観測ベクトル $\mathbf{y}$ とモデルの予測ベクトル $\Phi \boldsymbol{\beta}$ の距離の 2 乗」である。この見方が第 3 節の主役になる。この $E$ を**目的関数** (最小化したい関数) と呼ぶ。

### 2.2 なぜ二乗なのか — 他の選択肢との比較

ずれの大きさのまとめ方は二乗和だけではない。例えば

- 絶対値の和 $\sum_i |r_i|$
- 最大値 $\max_i |r_i|$

でもよさそうに見える。実際どちらも使われることがある。それでも二乗和が標準になっているのには、はっきりした理由が 3 つある。

**理由 1: 微分できて、解が式で書ける。**
$E(\boldsymbol{\beta})$ は各 $\beta_k$ の 2 次関数である (残差が $\beta_k$ の 1 次式で、それを 2 乗して足すから)。高校で学んだとおり、2 次関数の最小化は「微分して 0 とおく」だけで解ける。実際、第 4 節で見るように最小二乗法の解は連立 1 次方程式を解くだけで**閉じた式**として得られる。一方 $|r_i|$ は $r_i = 0$ で微分できない (グラフが折れ曲がっている) ため、絶対値の和の最小化は同じようには進められず、反復的な数値計算が必要になる。

**理由 2: 幾何的にきれい。**
$\sqrt{E} = \|\mathbf{y} - \Phi\boldsymbol{\beta}\|$ はユークリッド距離そのものである。だから「$E$ を最小化する」は「距離を最小化する」であり、最小化問題が「点から平面 (部分空間) に垂線を下ろす」というよく知る幾何の問題に化ける。これが第 3 節の内容である。

**理由 3: 統計的な裏付けがある。**
観測誤差がお互いに独立で、どのデータ点でも同じ程度のばらつきを持つなら、最小二乗解はある意味で最良の推定になることが知られている (ガウス・マルコフの定理)。さらに誤差が正規分布に従うなら、最小二乗解は「観測されたデータが最も起こりやすくなるパラメータ」(最尤推定) と一致する。この文書では深入りしないが、「二乗和は単なる計算の都合ではなく統計的にも筋が良い」ことは覚えておいてよい。

一方で弱点もある。二乗は大きなずれをさらに大きく強調する ($r_i = 10$ の寄与は $r_i = 1$ の 100 倍) ため、**外れ値** (測定ミスなどで大きく外れた点) に解が強く引きずられる。これは第 8 節で再訪する。

## 3. 導出 (幾何) — 直交射影としての最小二乗法

最小二乗法の一番美しい理解は幾何にある。いきなり $n$ 次元で考えると目がくらむので、まず 2 次元・3 次元の絵から始めて、それをそのまま $n$ 次元に持ち上げる。

### 3.1 まず 2 次元で — 点から直線への最短距離

平面上に、原点を通る直線 $\ell$ と、直線上にない点 $\mathrm{P}$ があるとする。直線 $\ell$ 上の点 $\mathrm{Q}$ を動かして、距離 $\mathrm{PQ}$ を最小にしたい。

答えは当然「$\mathrm{P}$ から $\ell$ に下ろした**垂線の足**」である。これは中学以来の常識だが、なぜかを三平方の定理で確かめておく。垂線の足を $\mathrm{H}$ とすると、$\ell$ 上の任意の点 $\mathrm{Q}$ に対して三角形 $\mathrm{PHQ}$ は $\mathrm{H}$ で直角なので

$$
\mathrm{PQ}^2 = \mathrm{PH}^2 + \mathrm{HQ}^2 \ \geq\ \mathrm{PH}^2
$$

となり、等号は $\mathrm{Q} = \mathrm{H}$ のとき ($\mathrm{HQ} = 0$ のとき) に限る。つまり**最短距離を与える点は垂線の足であり、そのとき「点と足を結ぶベクトル」は直線に垂直**である。

これをベクトルで言い直す。直線 $\ell$ を「ベクトル $\mathbf{a}$ の実数倍全体 $\{ t \mathbf{a} \mid t \in \mathbb{R} \}$」と書くと、$\mathrm{P}$ の位置ベクトル $\mathbf{p}$ に最も近い点 $t^* \mathbf{a}$ は

$$
(\mathbf{p} - t^* \mathbf{a}) \cdot \mathbf{a} = 0
$$

を満たす点である (残ったずれ $\mathbf{p} - t^*\mathbf{a}$ が直線の方向 $\mathbf{a}$ と直交する)。内積の条件 1 本で最短点が決まる。

### 3.2 次に 3 次元で — 点から平面への最短距離

今度は空間内に、原点を通る平面 $S$ と、平面上にない点 $\mathrm{P}$ があるとする。平面上で $\mathrm{P}$ に最も近い点は、やはり **$\mathrm{P}$ から平面に下ろした垂線の足** $\mathrm{H}$ である。理由も同じで、平面上の任意の点 $\mathrm{Q}$ に対して $\mathrm{PH} \perp \mathrm{HQ}$ だから三平方の定理がそのまま使える。

ベクトルで書くとどうなるか。原点を通る平面は、平行でない 2 本のベクトル $\mathbf{a}_1, \mathbf{a}_2$ を使って

$$
S = \{ t_1 \mathbf{a}_1 + t_2 \mathbf{a}_2 \mid t_1, t_2 \in \mathbb{R} \}
$$

と表せる ($\mathbf{a}_1, \mathbf{a}_2$ の「張る」平面)。最短点 $\mathbf{h} = t_1^* \mathbf{a}_1 + t_2^* \mathbf{a}_2$ の条件は「ずれ $\mathbf{p} - \mathbf{h}$ が平面 $S$ に垂直」だが、平面内のすべての方向と垂直であることは、**平面を張る 2 本のベクトルそれぞれと垂直**であることと同じである (平面内の任意のベクトルは $\mathbf{a}_1, \mathbf{a}_2$ の組み合わせで書けるから)。よって条件は

$$
(\mathbf{p} - \mathbf{h}) \cdot \mathbf{a}_1 = 0,
\qquad
(\mathbf{p} - \mathbf{h}) \cdot \mathbf{a}_2 = 0
$$

という**内積の条件 2 本**になる。未知数 $t_1^*, t_2^*$ が 2 個、条件が 2 本なので、ちょうど解ける連立 1 次方程式である。

ここまでをまとめると、次の図式が見えてくる。

> 「部分空間の中で点に最も近い点」= 垂線の足 = **直交射影** (orthogonal projection)。
> その条件は「残ったずれが、部分空間を張る各ベクトルと直交すること」。

### 3.3 $n$ 次元へ — 最小二乗法は垂線を下ろす問題である

さて、最小二乗法に戻る。目的関数は

$$
E(\boldsymbol{\beta}) = \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2
$$

だった。ここで $\Phi \boldsymbol{\beta}$ の正体を見つめ直す。$\Phi$ の列ベクトル (第 $k$ 列) を $\boldsymbol{\phi}_k \in \mathbb{R}^n$ と書くと、行列×ベクトルの定義から

$$
\Phi \boldsymbol{\beta} = \beta_1 \boldsymbol{\phi}_1 + \beta_2 \boldsymbol{\phi}_2 + \cdots + \beta_m \boldsymbol{\phi}_m
$$

である。つまり $\boldsymbol{\beta}$ を自由に動かすと、$\Phi \boldsymbol{\beta}$ は「$m$ 本の列ベクトルのあらゆる組み合わせ」全体、すなわち $\mathbb{R}^n$ の中の**部分空間**

$$
S = \{ \Phi \boldsymbol{\beta} \mid \boldsymbol{\beta} \in \mathbb{R}^m \}
$$

を動き回る。3.2 節の「2 本のベクトル $\mathbf{a}_1, \mathbf{a}_2$ が張る平面」の、本数が $m$ 本・全体の次元が $n$ になった版である。次元は増えたが構造は同じで、$S$ は原点を通る「平ら」な集合である。

すると最小二乗法は次のように読める。

> $n$ 次元空間の中に、点 $\mathbf{y}$ と、モデルが表現できる点の全体 $S$ (部分空間) がある。
> **$S$ の中で $\mathbf{y}$ に最も近い点 $\hat{\mathbf{y}}$ を探せ。**

3 次元までの直感がそのまま答えを教えてくれる。最も近い点は $\mathbf{y}$ から $S$ に下ろした垂線の足、つまり直交射影である。ただし $n$ 次元では絵が描けないので、三平方の定理による証明を $n$ 次元の言葉で書き直して確認する。

**主張:** $\hat{\mathbf{y}} \in S$ を「$\mathbf{y} - \hat{\mathbf{y}}$ が $S$ 内のすべてのベクトルと直交する」ように取れば、$\hat{\mathbf{y}}$ は $S$ の中で $\mathbf{y}$ に最も近い唯一の点である。

**確認:** $S$ 内の任意の点 $\mathbf{v}$ を取る。$\mathbf{y} - \mathbf{v}$ を 2 つに分ける。

$$
\mathbf{y} - \mathbf{v} = \underbrace{(\mathbf{y} - \hat{\mathbf{y}})}_{S \text{ に直交}} + \underbrace{(\hat{\mathbf{y}} - \mathbf{v})}_{S \text{ の中}}
$$

$\hat{\mathbf{y}}$ も $\mathbf{v}$ も $S$ の中の点なので、その差 $\hat{\mathbf{y}} - \mathbf{v}$ も $S$ の中のベクトルである (部分空間は差を取っても中に留まる)。よって 2 つの成分は直交しており、ノルムの 2 乗を展開すると交差項 (内積) が消える。

$$
\begin{aligned}
\| \mathbf{y} - \mathbf{v} \|^2
&= \| (\mathbf{y} - \hat{\mathbf{y}}) + (\hat{\mathbf{y}} - \mathbf{v}) \|^2 \\
&= \| \mathbf{y} - \hat{\mathbf{y}} \|^2
+ 2 \, (\mathbf{y} - \hat{\mathbf{y}}) \cdot (\hat{\mathbf{y}} - \mathbf{v})
+ \| \hat{\mathbf{y}} - \mathbf{v} \|^2 \\
&= \| \mathbf{y} - \hat{\mathbf{y}} \|^2 + 0 + \| \hat{\mathbf{y}} - \mathbf{v} \|^2 \\
&\geq \| \mathbf{y} - \hat{\mathbf{y}} \|^2
\end{aligned}
$$

等号成立は $\| \hat{\mathbf{y}} - \mathbf{v} \| = 0$、すなわち $\mathbf{v} = \hat{\mathbf{y}}$ のときに限る。これは $n$ 次元版の三平方の定理そのものである。$\blacksquare$

### 3.4 直交条件を式にする — 正規方程式

あとは「$\mathbf{y} - \hat{\mathbf{y}}$ が $S$ に直交する」を計算できる式に直すだけである。3.2 節と同じ理屈で、$S$ 内のすべてのベクトルと直交することは、$S$ を張る $m$ 本の列ベクトル $\boldsymbol{\phi}_1, \dots, \boldsymbol{\phi}_m$ それぞれと直交することと同値である。残差 $\mathbf{r} = \mathbf{y} - \Phi\boldsymbol{\beta}$ を使って書けば

$$
\boldsymbol{\phi}_k \cdot \mathbf{r} = 0
\qquad (k = 1, 2, \dots, m)
$$

という $m$ 本の条件になる。ここで「$\Phi$ の各列と $\mathbf{r}$ の内積を並べたベクトル」は、転置の定義からちょうど $\Phi^\top \mathbf{r}$ である ($\Phi^\top$ の第 $k$ 行 = $\Phi$ の第 $k$ 列だから、$\Phi^\top \mathbf{r}$ の第 $k$ 成分 = $\boldsymbol{\phi}_k \cdot \mathbf{r}$)。よって $m$ 本の条件は 1 本の行列の式

$$
\Phi^\top ( \mathbf{y} - \Phi \boldsymbol{\beta} ) = \mathbf{0}
$$

にまとまり、展開して整理すると

$$
\Phi^\top \Phi \, \boldsymbol{\beta} = \Phi^\top \mathbf{y}
$$

が得られる。これを**正規方程式** (normal equations) と呼ぶ。$\Phi^\top \Phi$ は $m \times m$ の行列、$\Phi^\top \mathbf{y}$ は $m$ 次元ベクトルなので、これは**未知数 $m$ 個の連立 1 次方程式**である。もともと解けなかった $n$ 本の方程式が、「一番惜しい解」を求める問題に置き換えたことで、必ず解ける $m$ 本の方程式に化けたのである。

名前の由来にも触れておく。「正規 (normal)」は「正規化」や「正規分布」の正規ではなく、幾何用語の **normal = 垂直** である。残差がモデルの空間に垂直であることを表す方程式だから normal equations という。この直交条件こそが最小二乗法の心臓部である。

## 4. 導出 (微分) — 偏微分を書き下して同じ式にたどり着く

同じ正規方程式は、幾何を使わず「微分して 0 とおく」だけでも導ける。行列の微分公式を天下りに使うのではなく、高校流に**成分ごとの偏微分を Σ で書き下し**、最後にそれが行列の形にまとまることを確認する。この計算の流れは後続の [5. 最急降下法](./5_steepest_descent.md) 以降でも繰り返し登場するので、一度手を動かしておく価値がある。

### 4.1 偏微分 ∂E/∂β_k を計算する

目的関数を Σ の形で書く。

$$
E(\boldsymbol{\beta})
= \sum_{i=1}^{n} \Bigl( y_i - \sum_{j=1}^{m} \beta_j \phi_j(x_i) \Bigr)^2
$$

$E$ は $m$ 個の変数 $\beta_1, \dots, \beta_m$ の関数である。多変数関数の最小点では、どの変数の方向に見ても傾きが 0、すなわちすべての偏微分が 0 になる ([0. 数学の準備](./0_math_preliminaries.md) の「偏微分と勾配」を参照。偏微分 $\partial E / \partial \beta_k$ とは「$\beta_k$ 以外を定数とみなして $\beta_k$ で微分したもの」である)。

そこで $\partial E / \partial \beta_k$ を計算する。見やすくするため、括弧の中身を $r_i$ とおく。

$$
r_i = y_i - \sum_{j=1}^{m} \beta_j \phi_j(x_i)
$$

まず $r_i$ を $\beta_k$ で偏微分する。$y_i$ は定数。和 $\sum_j \beta_j \phi_j(x_i)$ のうち $\beta_k$ を含む項は $\beta_k \phi_k(x_i)$ の 1 つだけで、その係数は $\phi_k(x_i)$ である。よって

$$
\frac{\partial r_i}{\partial \beta_k} = - \phi_k(x_i)
$$

次に $E = \sum_i r_i^2$ を $\beta_k$ で偏微分する。合成関数の微分 ($u^2$ の微分は $2u \cdot u'$。連鎖律については [0. 数学の準備](./0_math_preliminaries.md) を参照) を各項に使うと

$$
\frac{\partial E}{\partial \beta_k}
= \sum_{i=1}^{n} 2 \, r_i \, \frac{\partial r_i}{\partial \beta_k}
= \sum_{i=1}^{n} 2 \, r_i \cdot \bigl( - \phi_k(x_i) \bigr)
= -2 \sum_{i=1}^{n} \phi_k(x_i) \, r_i
$$

となる。最小点ではこれが 0 なので、$-2$ で割って

$$
\sum_{i=1}^{n} \phi_k(x_i) \, r_i = 0
\qquad (k = 1, 2, \dots, m)
$$

を得る。$r_i$ を元に戻して書けば

$$
\sum_{i=1}^{n} \phi_k(x_i) \Bigl( y_i - \sum_{j=1}^{m} \beta_j \phi_j(x_i) \Bigr) = 0
\qquad (k = 1, 2, \dots, m)
$$

これが「微分して 0」の条件のすべてである。未知数 $\beta_1, \dots, \beta_m$ の連立 1 次方程式が $m$ 本並んでいる。

### 4.2 Σ の式を行列の形にまとめる

上の条件を行列の記法に翻訳する。まず左辺第 1 の形

$$
\sum_{i=1}^{n} \phi_k(x_i) \, r_i
$$

をよく見ると、これは「$\Phi$ の第 $k$ 列 $\boldsymbol{\phi}_k = (\phi_k(x_1), \dots, \phi_k(x_n))^\top$ と残差ベクトル $\mathbf{r} = (r_1, \dots, r_n)^\top$ の内積」である。内積の定義 $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ を思い出せばそのまま読める。したがって $m$ 本の条件は

$$
\boldsymbol{\phi}_k \cdot \mathbf{r} = 0 \qquad (k = 1, \dots, m)
\quad \Longleftrightarrow \quad
\Phi^\top \mathbf{r} = \mathbf{0}
\quad \Longleftrightarrow \quad
\Phi^\top ( \mathbf{y} - \Phi \boldsymbol{\beta} ) = \mathbf{0}
$$

とまとまる (2 番目の同値は 3.4 節と同じく「$\Phi^\top \mathbf{r}$ の第 $k$ 成分が $\boldsymbol{\phi}_k \cdot \mathbf{r}$」だから)。展開すれば

$$
\Phi^\top \Phi \, \boldsymbol{\beta} = \Phi^\top \mathbf{y}
$$

となり、第 3 節の正規方程式に完全に一致した。**幾何の「残差がモデル空間に直交する」と、微分の「すべての偏微分が 0」は、同じ 1 つの式の 2 通りの読み方**だったのである。偏微分の条件 $\sum_i \phi_k(x_i) r_i = 0$ が内積 $\boldsymbol{\phi}_k \cdot \mathbf{r} = 0$ そのものだった、という対応を見れば納得できるだろう。

なお、勾配 (偏微分を並べたベクトル) の記法で書けば

$$
\nabla E(\boldsymbol{\beta})
= \begin{bmatrix} \partial E / \partial \beta_1 \\ \vdots \\ \partial E / \partial \beta_m \end{bmatrix}
= -2 \, \Phi^\top ( \mathbf{y} - \Phi \boldsymbol{\beta} )
= 2 \, \Phi^\top \Phi \, \boldsymbol{\beta} - 2 \, \Phi^\top \mathbf{y}
$$

である。教科書によっては行列微分の公式としてこの結果だけが書かれているが、中身は今やったとおり、成分ごとの素朴な偏微分にすぎない。

### 4.3 それは本当に「最小」なのか — 凸性の確認

1 変数のときを思い出すと、微分が 0 になる点 (停留点) は最小とは限らない。最大かもしれないし、$y = x^3$ の原点のような平らな点かもしれない。多変数ではさらに、ある方向には谷でも別の方向には山という**鞍点**もありうる。だから「勾配 = 0 とおいて解いた点が本当に最小か」は確認が必要である。

幸い $E$ については心配がいらない。[0. 数学の準備](./0_math_preliminaries.md) で見たとおり、**凸関数なら停留点は自動的に大域的最小点**である。$E$ が凸であることは次のように直接確かめられる。任意の点 $\boldsymbol{\beta}$ と任意の方向 $\mathbf{u} \in \mathbb{R}^m$ を固定し、直線に沿った 1 変数関数 $g(t) = E(\boldsymbol{\beta} + t \mathbf{u})$ を考えると

$$
g(t) = \| \mathbf{y} - \Phi (\boldsymbol{\beta} + t \mathbf{u}) \|^2
= \| (\mathbf{y} - \Phi \boldsymbol{\beta}) - t \, \Phi \mathbf{u} \|^2
= \| \mathbf{r} \|^2 - 2 t \, (\mathbf{r} \cdot \Phi \mathbf{u}) + t^2 \, \| \Phi \mathbf{u} \|^2
$$

となる ($\mathbf{r} = \mathbf{y} - \Phi\boldsymbol{\beta}$ とおいた)。これは $t$ の 2 次関数で、$t^2$ の係数は $\| \Phi \mathbf{u} \|^2 \geq 0$。つまり **$E$ をどの方向に切っても、断面は下に凸 (または直線) の放物線**である。よって $E$ は凸関数であり、勾配が $\mathbf{0}$ となる点は必ず大域的最小点である。鞍点や極大点の心配はない。

## 5. 解の存在と一意性

正規方程式 $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$ は、いつでも解けるのか。解は 1 つに決まるのか。

**最小点は常に存在する。** 幾何の導出 (3.3 節) がそのまま答えである。部分空間 $S$ への直交射影 $\hat{\mathbf{y}}$ はどんな $\mathbf{y}$ に対しても必ず存在するので、$E$ の最小値を与える点は必ずある。

**解が 1 つに決まるのは、$\Phi$ の列ベクトルが線形独立なとき**、すなわち $\mathrm{rank} \, \Phi = m$ のときである ([0. 数学の準備](./0_math_preliminaries.md) の「線形独立とランク」を参照。線形独立とは、どの列も他の列の組み合わせでは作れないこと)。理由を確かめる。列が線形独立なら、$\mathbf{u} \neq \mathbf{0}$ に対して $\Phi \mathbf{u} \neq \mathbf{0}$ (列の非自明な組み合わせが $\mathbf{0}$ にならないのが線形独立の定義そのもの)。すると

$$
\mathbf{u}^\top ( \Phi^\top \Phi ) \, \mathbf{u}
= ( \Phi \mathbf{u} )^\top ( \Phi \mathbf{u} )
= \| \Phi \mathbf{u} \|^2 > 0
\qquad (\mathbf{u} \neq \mathbf{0})
$$

となる (最初の等号は公式 $(A\mathbf{u}) \cdot \mathbf{v} = \mathbf{u} \cdot (A^\top \mathbf{v})$ の応用である)。ここでもし $\Phi^\top \Phi \, \mathbf{u} = \mathbf{0}$ となる $\mathbf{u} \neq \mathbf{0}$ が存在したら、上の式の左辺は 0 になってしまい矛盾する。よって $\Phi^\top \Phi \, \mathbf{u} = \mathbf{0}$ の解は $\mathbf{u} = \mathbf{0}$ のみであり、$\Phi^\top \Phi$ は逆行列を持つ。このとき解は形式的に

$$
\boldsymbol{\beta}^* = ( \Phi^\top \Phi )^{-1} \Phi^\top \mathbf{y}
$$

と 1 つの式で書ける (ただしこの式をそのまま数値計算に使うのは推奨されない。第 6 節)。

**列が線形従属な場合はどうなるか。** 例えば基底関数に $\phi_1(x) = 1$、$\phi_2(x) = x$、$\phi_3(x) = 2x + 3$ を選んでしまうと、$\phi_3 = 2\phi_2 + 3\phi_1$ なので $\Phi$ の第 3 列は他の列の組み合わせで書けてしまう。このとき面白いのは、**射影 $\hat{\mathbf{y}}$ (= 当てはめた曲線そのもの) は変わらず 1 つに決まる**が、**それを実現する係数 $\boldsymbol{\beta}$ が無数に存在する**ことである。同じ点 $\hat{\mathbf{y}}$ を作る列の組み合わせ方が何通りもあるからである。この場合、慣例では「係数ベクトルのノルム $\|\boldsymbol{\beta}\|$ が最小のもの」を代表として選び、これは擬似逆行列 $\Phi^{+}$ を用いて $\boldsymbol{\beta}^* = \Phi^{+} \mathbf{y}$ と書ける。擬似逆行列は [3. 特異値分解](./3_singular_value_decomposition.md) で扱う。

## 6. 数値計算の実際 — 正規方程式を直接解いてはいけない

理論上、最小二乗解は正規方程式を解けば得られる。ところが実際のコンピュータで計算するとき、**$\Phi^\top \Phi$ を作ってから解く方法は数値的に危険**であることが知られている。

コンピュータの小数 (浮動小数点数) はおよそ 16 桁程度の精度しか持たない。連立 1 次方程式の解きやすさは行列の「性質の良さ」に依存し、性質の悪い行列では計算の途中で有効桁がどんどん失われる。この「性質の悪さ」を測る指標を**条件数**と呼ぶが、重要な事実は次の 1 つである。

> $\Phi^\top \Phi$ の条件数は、$\Phi$ 自身の条件数の **2 乗**になる。

つまり、$\Phi$ がそこそこ性質の悪い行列だった場合、$\Phi^\top \Phi$ を作った瞬間に問題の難しさが 2 乗に跳ね上がる。例えば $\Phi$ の悪さが「有効桁を 4 桁失う」程度なら、$\Phi^\top \Phi$ では 8 桁失う。せっかく元のデータには十分な情報があったのに、**$\Phi^\top \Phi$ という中間生成物を経由した時点で情報を自分から捨てている**のである。

これを回避する標準的な方法が **QR 分解**である。$\Phi$ を「列が互いに直交する行列 $Q$」と「上三角行列 $R$」の積 $\Phi = QR$ に分解すると、$\Phi^\top \Phi$ を一度も作ることなく、$R \boldsymbol{\beta} = Q^\top \mathbf{y}$ という解きやすい形に問題を変形できる。条件数は 2 乗にならずそのまま保たれる。仕組みの詳細は [2. QR 分解](./2_qr_decomposition.md) で扱うので、ここでは方法の比較だけ頭に入れておけばよい。

| 方法 | 手順 | 特徴 |
| --- | --- | --- |
| 正規方程式を直接解く | $\Phi^\top \Phi$ を作って解く | 計算は速いが、条件数が 2 乗に悪化し桁落ちしやすい |
| QR 分解 | $\Phi = QR$ として $R \boldsymbol{\beta} = Q^\top \mathbf{y}$ を解く | $\Phi^\top \Phi$ を作らないため数値的に安定。**通常はこれが標準** |
| 特異値分解 (SVD) | $\Phi = U \Sigma V^\top$ から擬似逆行列を構成 | 最も頑健。列が線形従属 (またはその寸前) でも動くが、計算コストは最大 ([3. 特異値分解](./3_singular_value_decomposition.md)) |

NumPy の `np.linalg.lstsq` や各種ライブラリの最小二乗ソルバは、内部で QR や SVD を使っている。実務では自前で $\Phi^\top \Phi$ を組み立てるのではなく、これらを呼ぶのが正しい。ただし理論を理解する上では正規方程式が出発点であることに変わりはない。

## 7. 具体例 — 直線フィッティングを最後まで手計算する

抽象論はここまでにして、小さなデータで最後まで手を動かす。

### 7.1 問題

次の 3 点に直線 $\hat{y} = \beta_1 + \beta_2 x$ を当てはめたい。

| $i$ | $x_i$ | $y_i$ |
| --- | --- | --- |
| 1 | 0 | 1 |
| 2 | 1 | 3 |
| 3 | 2 | 4 |

3 点は一直線上にない (点 1 と点 3 を結ぶ直線 $y = 1 + \frac{3}{2}x$ は $x = 1$ で $y = 2.5$ を通るが、点 2 は $y = 3$) ので、どんな直線を引いても必ずどこかにずれが残る。まさに過剰決定系である。

### 7.2 計画行列と正規方程式を組み立てる

基底関数は $\phi_1(x) = 1$、$\phi_2(x) = x$ である。計画行列 $\Phi$ の第 $i$ 行は $(\phi_1(x_i), \phi_2(x_i)) = (1, x_i)$ なので

$$
\Phi =
\begin{bmatrix}
1 & 0 \\
1 & 1 \\
1 & 2
\end{bmatrix},
\qquad
\mathbf{y} =
\begin{bmatrix}
1 \\ 3 \\ 4
\end{bmatrix},
\qquad
\boldsymbol{\beta} =
\begin{bmatrix}
\beta_1 \\ \beta_2
\end{bmatrix}
$$

$\Phi^\top \Phi$ を成分計算する。$\Phi^\top$ は $2 \times 3$、$\Phi$ は $3 \times 2$ なので積は $2 \times 2$ になる。

$$
\Phi^\top \Phi =
\begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 2
\end{bmatrix}
\begin{bmatrix}
1 & 0 \\
1 & 1 \\
1 & 2
\end{bmatrix}
=
\begin{bmatrix}
1 \cdot 1 + 1 \cdot 1 + 1 \cdot 1 & 1 \cdot 0 + 1 \cdot 1 + 1 \cdot 2 \\
0 \cdot 1 + 1 \cdot 1 + 2 \cdot 1 & 0 \cdot 0 + 1 \cdot 1 + 2 \cdot 2
\end{bmatrix}
=
\begin{bmatrix}
3 & 3 \\
3 & 5
\end{bmatrix}
$$

一般に直線フィッティングでは

$$
\Phi^\top \Phi =
\begin{bmatrix}
n & \sum_i x_i \\
\sum_i x_i & \sum_i x_i^2
\end{bmatrix}
$$

という形になる。今回は $n = 3$、$\sum x_i = 0 + 1 + 2 = 3$、$\sum x_i^2 = 0 + 1 + 4 = 5$ で確かに一致する。右辺も同様に

$$
\Phi^\top \mathbf{y} =
\begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 2
\end{bmatrix}
\begin{bmatrix}
1 \\ 3 \\ 4
\end{bmatrix}
=
\begin{bmatrix}
1 + 3 + 4 \\
0 \cdot 1 + 1 \cdot 3 + 2 \cdot 4
\end{bmatrix}
=
\begin{bmatrix}
8 \\ 11
\end{bmatrix}
$$

(一般形は $\Phi^\top \mathbf{y} = \bigl( \sum_i y_i, \ \sum_i x_i y_i \bigr)^\top$ である)。よって正規方程式は

$$
\begin{bmatrix}
3 & 3 \\
3 & 5
\end{bmatrix}
\begin{bmatrix}
\beta_1 \\ \beta_2
\end{bmatrix}
=
\begin{bmatrix}
8 \\ 11
\end{bmatrix}
\qquad \Longleftrightarrow \qquad
\begin{cases}
3 \beta_1 + 3 \beta_2 = 8 \\
3 \beta_1 + 5 \beta_2 = 11
\end{cases}
$$

### 7.3 解く

下の式から上の式を引くと

$$
(3 \beta_1 + 5 \beta_2) - (3 \beta_1 + 3 \beta_2) = 11 - 8
\quad \Longrightarrow \quad
2 \beta_2 = 3
\quad \Longrightarrow \quad
\beta_2 = \frac{3}{2}
$$

これを上の式に代入して

$$
3 \beta_1 + 3 \cdot \frac{3}{2} = 8
\quad \Longrightarrow \quad
3 \beta_1 = 8 - \frac{9}{2} = \frac{7}{2}
\quad \Longrightarrow \quad
\beta_1 = \frac{7}{6}
$$

求める直線は

$$
\hat{y} = \frac{7}{6} + \frac{3}{2} x
$$

である。

### 7.4 検算 — 残差は本当にモデル空間と直交しているか

理論どおりなら、残差ベクトル $\mathbf{r}$ は $\Phi$ の両方の列と直交しているはずである。まず各点での予測値と残差を計算する。

| $i$ | $x_i$ | $y_i$ | $\hat{y}(x_i) = \frac{7}{6} + \frac{3}{2} x_i$ | $r_i = y_i - \hat{y}(x_i)$ |
| --- | --- | --- | --- | --- |
| 1 | 0 | 1 | $\frac{7}{6}$ | $1 - \frac{7}{6} = -\frac{1}{6}$ |
| 2 | 1 | 3 | $\frac{7}{6} + \frac{3}{2} = \frac{16}{6} = \frac{8}{3}$ | $3 - \frac{8}{3} = \frac{1}{3}$ |
| 3 | 2 | 4 | $\frac{7}{6} + 3 = \frac{25}{6}$ | $4 - \frac{25}{6} = -\frac{1}{6}$ |

つまり $\mathbf{r} = \bigl( -\frac{1}{6}, \ \frac{1}{3}, \ -\frac{1}{6} \bigr)^\top$。列との内積を確かめる。

- 第 1 列 $(1, 1, 1)^\top$ との内積: $-\frac{1}{6} + \frac{1}{3} - \frac{1}{6} = -\frac{1}{6} + \frac{2}{6} - \frac{1}{6} = 0$ ✓
- 第 2 列 $(0, 1, 2)^\top$ との内積: $0 \cdot \bigl(-\frac{1}{6}\bigr) + 1 \cdot \frac{1}{3} + 2 \cdot \bigl(-\frac{1}{6}\bigr) = \frac{1}{3} - \frac{1}{3} = 0$ ✓

確かに直交している。なお第 1 列との直交は「残差の総和が 0」を意味する。切片項 $\phi_1 = 1$ を含む直線フィッティングでは、当てはめた直線の上側のずれと下側のずれが必ず打ち消し合うのである。残差二乗和 (この問題での $E$ の最小値) は

$$
E = \Bigl( -\frac{1}{6} \Bigr)^2 + \Bigl( \frac{1}{3} \Bigr)^2 + \Bigl( -\frac{1}{6} \Bigr)^2
= \frac{1}{36} + \frac{4}{36} + \frac{1}{36}
= \frac{6}{36} = \frac{1}{6}
$$

どんな直線を選んでもずれの二乗和は $\frac{1}{6}$ より小さくできない、というのが最小二乗法の結論である。

### 7.5 一般の公式 — 平均・分散・共分散との関係

同じ計算を一般の $n$ 点で実行すると、見通しの良い公式が得られる。正規方程式

$$
\begin{cases}
n \beta_1 + \bigl( \sum_i x_i \bigr) \beta_2 = \sum_i y_i \\
\bigl( \sum_i x_i \bigr) \beta_1 + \bigl( \sum_i x_i^2 \bigr) \beta_2 = \sum_i x_i y_i
\end{cases}
$$

を、平均 $\bar{x} = \frac{1}{n} \sum_i x_i$、$\bar{y} = \frac{1}{n} \sum_i y_i$ を使って整理する。第 1 式を $n$ で割ると

$$
\beta_1 + \beta_2 \bar{x} = \bar{y}
\quad \Longrightarrow \quad
\beta_1 = \bar{y} - \beta_2 \bar{x}
$$

これを第 2 式に代入し、$\sum_i x_i = n \bar{x}$ を使って整理すると

$$
n \bar{x} ( \bar{y} - \beta_2 \bar{x} ) + \beta_2 \sum_i x_i^2 = \sum_i x_i y_i
\quad \Longrightarrow \quad
\beta_2 \Bigl( \sum_i x_i^2 - n \bar{x}^2 \Bigr) = \sum_i x_i y_i - n \bar{x} \bar{y}
$$

ここで統計でおなじみの恒等式 $\sum_i (x_i - \bar{x})^2 = \sum_i x_i^2 - n \bar{x}^2$ と $\sum_i (x_i - \bar{x})(y_i - \bar{y}) = \sum_i x_i y_i - n \bar{x} \bar{y}$ を使うと

$$
\beta_2 = \frac{\sum_i ( x_i - \bar{x} )( y_i - \bar{y} )}{\sum_i ( x_i - \bar{x} )^2},
\qquad
\beta_1 = \bar{y} - \beta_2 \bar{x}
$$

が得られる。この形には明確な解釈がある。

- 傾き $\beta_2$ は **$x$ と $y$ の共分散を $x$ の分散で割ったもの** (分母分子を $n$ で割ればそのまま標本分散・標本共分散になる)。
- $\beta_1 = \bar{y} - \beta_2 \bar{x}$ を書き直すと $\bar{y} = \beta_1 + \beta_2 \bar{x}$。つまり**最小二乗直線は必ずデータの重心 $(\bar{x}, \bar{y})$ を通る**。

7.3 節の答えで確かめると、重心は $(\bar{x}, \bar{y}) = \bigl( 1, \frac{8}{3} \bigr)$ で、直線に $x = 1$ を代入すると $\hat{y} = \frac{7}{6} + \frac{3}{2} = \frac{16}{6} = \frac{8}{3}$。確かに重心を通っている。

また、傾きの公式の分母は $x$ の分散なので、**全部の $x_i$ が同じ値だと分母が 0 になり傾きが決まらない**。データ点が縦一列に並んでいたらどんな傾きの直線でも同じくらい「惜しい」のだから、当然である。行列の言葉で言えば、このとき $\Phi$ の第 2 列 $(x_1, \dots, x_n)^\top$ が第 1 列 $(1, \dots, 1)^\top$ の定数倍になり、列が線形従属になっている。第 5 節の「解が一意に決まらないケース」の最も単純な実例である。

## 8. よくある誤解と限界

**「線形」は直線のことではない。** 繰り返しになるが、線形なのは未知数 $\boldsymbol{\beta}$ についてであり、多項式や三角関数のフィッティングも線形最小二乗法である。逆に $\hat{y} = e^{\beta x}$ や $\hat{y} = \beta_1 \sin(\beta_2 x)$ のように未知数が非線形に入るモデルは対象外で、[4. 非線形最小二乗法](./4_nonlinear_least_squares.md) の手法 (ガウス・ニュートン法、レーベンバーグ・マーカート法など) が必要になる。

**誤差は $y$ 方向のみを仮定している。** 残差 $r_i = y_i - \hat{y}(x_i)$ は縦方向のずれである。つまり「$x_i$ は正確に分かっていて、誤差は $y_i$ にだけ乗る」という暗黙の仮定を置いている。$x$ にも測定誤差がある場合 (例えばどちらも測定値である場合) には、本来は点から直線への垂直距離を最小化する全最小二乗法 (total least squares) の領分になる。

**外れ値に弱い。** 第 2 節で述べたとおり、二乗はずれを強調する。1 点だけ大きく外れたデータがあると、その 1 点の残差を減らすために直線全体が引きずられる。対策としては、大きな残差への罰を二乗より緩くするロバスト回帰 (Huber 損失など) や、外れ値を含まない部分集合を探す RANSAC がある。

**基底を増やすほど良いわけではない。** 基底関数を増やせば表現力が上がり、残差二乗和は必ず減る (極端な話、$n$ 点のデータは $n-1$ 次多項式で残差 0 にできる)。しかしそれはデータに含まれる誤差まで忠実になぞっているだけで、新しいデータへの予測はむしろ悪化する。これを**過学習**という。対策の 1 つがリッジ回帰で、正規方程式の $\Phi^\top \Phi$ を $\Phi^\top \Phi + \lambda I$ に置き換えて係数が大きくなりすぎることに罰を与える。これは過学習の抑制と同時に、第 6 節で触れた数値的な不安定さ (と第 5 節の線形従属の問題) への処方箋にもなる。

## 9. まとめ

線形最小二乗法は「解けない連立方程式 $\Phi \boldsymbol{\beta} \approx \mathbf{y}$ を、残差の二乗和が最小という意味で最良に解く」手法である。この文書の内容は、次の 1 つの幾何的事実に集約される。

> 最小二乗解とは、観測ベクトル $\mathbf{y}$ をモデルの列が張る部分空間へ**直交射影**したものであり、正規方程式 $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$ は「残差がその部分空間に垂直」という直交条件の言い換えである。

道具立てを振り返る。

- 「点から平面に垂線を下ろす」という 2〜3 次元の直感が、そのまま $n$ 次元で通用した (三平方の定理)。
- 微分による導出 (偏微分を Σ で書き下して 0 とおく) も同じ正規方程式に到達し、しかも $E$ が凸なので停留点が大域的最小であることまで保証された。
- 解が一意に決まる条件は $\Phi$ の列の線形独立性 ($\mathrm{rank} \, \Phi = m$) であった。
- ただし数値計算では $\Phi^\top \Phi$ を作ると問題の難しさが 2 乗に悪化するため、実際には [2. QR 分解](./2_qr_decomposition.md) を経由して解くのが標準である。

次の文書では、この「$\Phi^\top \Phi$ を作らずに最小二乗問題を解く」ための道具である QR 分解を扱う。
