# ガウス・ニュートン法 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

## 1. 位置づけ — これまでに揃った道具と、残っている問題

[4. 非線形最小二乗法](./4_nonlinear_least_squares.md) で立てた問題をもう一度書く。データ $(x_1, y_1), \dots, (x_n, y_n)$ とモデル $f(x; \boldsymbol{\beta})$ に対して、残差 $r_i(\boldsymbol{\beta}) = y_i - f(x_i; \boldsymbol{\beta})$ の二乗和

$$
E(\boldsymbol{\beta}) = \sum_{i=1}^{n} r_i(\boldsymbol{\beta})^2 = \| \mathbf{r}(\boldsymbol{\beta}) \|^2
$$

を最小化したい。モデルがパラメータ $\boldsymbol{\beta} = (\beta_1, \dots, \beta_m)^\top$ について非線形なので閉じた式では解けず、「初期値から出発して $E$ が下がる方向に少しずつ動かす」反復法が必要だった。すでに 2 つの反復法を見た。

- [5. 最急降下法](./5_steepest_descent.md): 勾配 $\nabla E = 2 J^\top \mathbf{r}$ の逆方向に進む。1 階微分だけで済むが、収束が非常に遅い (指数減衰モデルの例では数千反復かかった)
- [6. ニュートン法](./6_newton_method.md): 関数を放物面で近似してその底へ飛ぶ。解の近くでは桁数が倍々で増える二次収束を達成するが、ヘッセ行列 $\nabla^2 E$ (2 階微分) が必要

ニュートン法の代償を最小二乗法の文脈で具体的に見ると、文書 4 で導いたように

$$
\nabla^2 E = 2 \Bigl( J^\top J + \sum_{i=1}^{n} r_i \nabla^2 r_i \Bigr)
$$

であり、第 2 項のために **残差 1 本ごとに** $m \times m$ の 2 階微分の行列 $\nabla^2 r_i$ を $n$ 個計算しなければならない。データが増えるほど、モデルが複雑になるほど、この負担は重くなる。

**ガウス・ニュートン法** (Gauss-Newton method) は、最小二乗問題に特有の構造を利用して、**2 階微分を一切計算せずに** ニュートン法に近い速さを実現する手法である。本文書では

1. 「残差を接線 (1 次式) で置き換える」という発想から更新式を導き (§2–4)、
2. 同じ式が「ニュートン法のヘッセ行列から第 2 項を捨てたもの」でもあることを確かめ (§5)、
3. サンプルコード `src/bin/7_gauss_newton_method.rs` の実験で、良い初期値からの速さと、2 通りの壊れ方を数値で追う (§6–8)。

## 2. 発想 — 1 変数の接線近似から出発する

### 2.1 接線近似の復習

出発点は [6. ニュートン法](./6_newton_method.md) でも使った、微分の最も基本的な使い方である。滑らかな関数 $g$ は、点 $\beta$ の近くでは接線で近似できる。

$$
g(\beta + \delta) \approx g(\beta) + g'(\beta) \, \delta
\qquad (\delta \text{ が小さいとき})
$$

右辺は $\delta$ の **1 次式** (直線) である。「曲がったものを、いま居る場所の傾きで、まっすぐなもので置き換える」— これがこの文書全体でやることのすべてである。ただしニュートン法が **目的関数 $E$ を放物面で** 置き換えたのに対し、ガウス・ニュートン法は **残差 $\mathbf{r}$ を接線で** 置き換える。何を近似するかの違いが手法の違いを生む。

### 2.2 残差 1 本をまっすぐにしてみる

まずデータ 1 点、パラメータ 1 個の最小の例で試す。データ $(x, y) = (1, 2)$ をモデル $f(x; \beta) = e^{\beta x}$ でフィットしたい。残差は

$$
r(\beta) = 2 - e^{\beta}
$$

であり、$E(\beta) = r(\beta)^2$ を最小化する。$r$ は $\beta$ について非線形 (指数関数) なので、$E$ は 2 次関数にならず、頂点の公式では解けない。

そこで、現在の推定値 $\beta$ の近くで $r$ を接線で置き換える。$r'(\beta) = -e^{\beta}$ だから

$$
r(\beta + \delta) \approx r(\beta) + r'(\beta) \, \delta
$$

右辺は $\delta$ の 1 次式なので、それを二乗した

$$
E(\beta + \delta) \approx \bigl( r(\beta) + r'(\beta) \, \delta \bigr)^2
$$

は $\delta$ の **2 次関数** である。2 次関数の最小化なら高校数学で解ける。今の場合は括弧の中を 0 にする $\delta$ を選べばよく、

$$
r + r' \delta = 0
\quad \Longleftrightarrow \quad
\delta = - \frac{r(\beta)}{r'(\beta)}
$$

となる。この $\delta$ だけ動かして $\beta \leftarrow \beta + \delta$ とし、新しい場所でまた接線を引き直す。実際に $\beta_0 = 0$ から回すと

| $k$ | $\beta_k$ | $r(\beta_k)$ |
| --- | --- | --- |
| 0 | $0$ | $1$ |
| 1 | $1$ | $-0.71828$ |
| 2 | $0.73576$ | $-0.08706$ |
| 3 | $0.69403$ | $-0.00177$ |
| 4 | $0.69315$ | $-0.0000008$ |

となり、真の解 $\beta^* = \ln 2 = 0.693147\ldots$ ($e^{\beta} = 2$ の解) に急速に収束する。正しい桁数が反復ごとにほぼ倍増しており、これは文書 6 で見た二次収束の挙動そのものである。実際、更新式 $\delta = -r / r'$ は **方程式 $r(\beta) = 0$ に対するニュートン法 (求根)** と完全に一致する。データが 1 点なら残差をちょうど 0 にできるから、「二乗和の最小化」が「方程式の求根」に退化するのである。

### 2.3 データが複数あるとき — 手計算で 1 反復ずつ追う

データが増えると事情が変わる。データ 2 点 $(1, 2), (2, 5)$ を同じモデル $f(x; \beta) = e^{\beta x}$ でフィットする。残差は 2 本になる。

$$
r_1(\beta) = 2 - e^{\beta},
\qquad
r_2(\beta) = 5 - e^{2\beta}
$$

$r_1 = 0$ なら $\beta = \ln 2 \approx 0.693$、$r_2 = 0$ なら $\beta = \frac{1}{2} \ln 5 \approx 0.805$ が必要で、両方を同時に 0 にする $\beta$ は存在しない。そこで二乗和 $E(\beta) = r_1^2 + r_2^2$ を最小化する。もはや求根では済まないが、接線近似の作戦はそのまま使える。2 本の残差をそれぞれ接線で置き換えると

$$
r_1(\beta + \delta) \approx r_1 + r_1' \, \delta,
\qquad
r_2(\beta + \delta) \approx r_2 + r_2' \, \delta
$$

であり ($r_1' = -e^{\beta}$、$r_2' = -2 e^{2\beta}$)、二乗和は

$$
E(\beta + \delta)
\approx (r_1 + r_1' \delta)^2 + (r_2 + r_2' \delta)^2
$$

となる。右辺を展開して $\delta$ について整理すると

$$
E(\beta + \delta)
\approx
\underbrace{\bigl( (r_1')^2 + (r_2')^2 \bigr)}_{a} \, \delta^2 + \underbrace{2 (r_1' r_1 + r_2' r_2)}_{b} \, \delta + \underbrace{r_1^2 + r_2^2}_{c}
$$

というただの **下に凸な 2 次関数** $a \delta^2 + b \delta + c$ である。頂点の公式 $\delta = -\frac{b}{2a}$ から

$$
\delta = - \frac{r_1' r_1 + r_2' r_2}{(r_1')^2 + (r_2')^2}
$$

が得られる。$\beta_0 = 1$ から実際に手計算で回してみる。

**1 反復目** ($\beta_0 = 1$)。$e^1 = 2.71828$、$e^2 = 7.38906$ より

$$
r_1 = -0.71828, \quad r_2 = -2.38906, \quad E = 6.2235
$$

$$
r_1' = -2.71828, \quad r_2' = -14.77811
$$

$$
\delta_0 = - \frac{(-2.71828)(-0.71828) + (-14.77811)(-2.38906)}{(-2.71828)^2 + (-14.77811)^2}
= - \frac{1.9525 + 35.3060}{7.3891 + 218.3926}
= - \frac{37.2585}{225.7817}
= -0.16502
$$

よって $\beta_1 = 1 - 0.16502 = 0.83498$。

**2 反復目** ($\beta_1 = 0.83498$)。$e^{0.83498} = 2.30477$、$e^{1.66996} = 5.31194$ より

$$
r_1 = -0.30477, \quad r_2 = -0.31194, \quad E = 0.1902
$$

$$
\delta_1 = - \frac{(-2.30477)(-0.30477) + (-10.62388)(-0.31194)}{(2.30477)^2 + (10.62388)^2}
= - \frac{4.0164}{118.1786}
= -0.03399
$$

よって $\beta_2 = 0.80099$。以降も同様に回すと、次の表のようになる。

| $k$ | $\beta_k$ | $E(\beta_k)$ |
| --- | --- | --- |
| 0 | $1.00000$ | $6.2235$ |
| 1 | $0.83498$ | $0.19019$ |
| 2 | $0.80099$ | $0.05325$ |
| 3 | $0.79965$ | $0.05306$ |

3 反復で最小点 $\beta^* \approx 0.79963$ (そこでの $E^* \approx 0.05306$) にほぼ到達した。ここで起きたことを言葉にすると:

> 非線形な残差を接線で置き換えた瞬間、「解けない問題 ($E$ の最小化)」が「解ける問題 (2 次関数の最小化)」に変わる。解ける問題を解いて少し進み、進んだ先でまた置き換え直す。

この「**線形化 → 解く → 更新 → 再線形化**」のループがガウス・ニュートン法である。あとはこれを多変数 (パラメータが $m$ 個、データが $n$ 個) に拡張するだけでよい。

## 3. 多変数への拡張 — 残差ベクトルの 1 次近似

### 3.1 残差 1 本の多変数テイラー展開

パラメータが $\boldsymbol{\beta} = (\beta_1, \dots, \beta_m)^\top$ とベクトルになったとき、§2 の接線近似はどうなるか。残差 1 本 $r_i(\boldsymbol{\beta})$ を考え、$\boldsymbol{\beta}$ から $\boldsymbol{\delta} = (\delta_1, \dots, \delta_m)^\top$ だけ動かす。

コツは「**一度に 1 方向ずつ動かす**」ことである ([0. 数学の準備](./0_math_preliminaries.md) で偏微分を導入したときと同じ考え方)。まず $\beta_1$ 方向にだけ $\delta_1$ 動かすと、他の変数は止まっているので 1 変数の接線近似がそのまま使えて、$r_i$ の変化量は約 $\frac{\partial r_i}{\partial \beta_1} \delta_1$ である。続いて $\beta_2$ 方向に $\delta_2$ 動かすと、さらに約 $\frac{\partial r_i}{\partial \beta_2} \delta_2$ 変わる (移動後の点では偏微分の値もわずかに変わっているが、その違いは「小さい量 × 小さい量」の 2 次の微小量なので、1 次近似では無視できる)。これを $m$ 方向すべてについて繰り返すと、変化量の合計は

$$
r_i(\boldsymbol{\beta} + \boldsymbol{\delta})
\approx
r_i(\boldsymbol{\beta}) + \frac{\partial r_i}{\partial \beta_1} \delta_1 + \frac{\partial r_i}{\partial \beta_2} \delta_2 + \cdots + \frac{\partial r_i}{\partial \beta_m} \delta_m
= r_i(\boldsymbol{\beta}) + \sum_{k=1}^{m} \frac{\partial r_i}{\partial \beta_k} \delta_k
$$

となる。これが 1 変数の接線近似 $g(\beta + \delta) \approx g + g' \delta$ の多変数版であり、右辺は $\delta_1, \dots, \delta_m$ の **1 次式** である。

### 3.2 $n$ 本まとめて行列で書く

残差は $n$ 本ある。上の近似を $i = 1, \dots, n$ について縦に並べると

$$
\begin{pmatrix}
r_1(\boldsymbol{\beta} + \boldsymbol{\delta}) \\
r_2(\boldsymbol{\beta} + \boldsymbol{\delta}) \\
\vdots \\
r_n(\boldsymbol{\beta} + \boldsymbol{\delta})
\end{pmatrix}
\approx
\begin{pmatrix}
r_1(\boldsymbol{\beta}) \\
r_2(\boldsymbol{\beta}) \\
\vdots \\
r_n(\boldsymbol{\beta})
\end{pmatrix}
+
\begin{pmatrix}
\dfrac{\partial r_1}{\partial \beta_1} & \cdots & \dfrac{\partial r_1}{\partial \beta_m} \\
\dfrac{\partial r_2}{\partial \beta_1} & \cdots & \dfrac{\partial r_2}{\partial \beta_m} \\
\vdots & & \vdots \\
\dfrac{\partial r_n}{\partial \beta_1} & \cdots & \dfrac{\partial r_n}{\partial \beta_m}
\end{pmatrix}
\begin{pmatrix}
\delta_1 \\
\vdots \\
\delta_m
\end{pmatrix}
$$

となる。右辺第 2 項の $n \times m$ 行列は、[4. 非線形最小二乗法](./4_nonlinear_least_squares.md) で導入した **ヤコビ行列** $J(\boldsymbol{\beta})$ ($J_{ik} = \frac{\partial r_i}{\partial \beta_k}$) そのものである。したがってベクトルの記法で簡潔に

$$
\boxed{\;
\mathbf{r}(\boldsymbol{\beta} + \boldsymbol{\delta})
\approx
\mathbf{r}(\boldsymbol{\beta}) + J(\boldsymbol{\beta}) \, \boldsymbol{\delta}
\;}
$$

と書ける。これを残差の **線形化** (1 次近似) と呼ぶ。$m = 1$ のときは $J$ が $n \times 1$ の縦ベクトル $(r_1', \dots, r_n')^\top$ になり、§2.3 の式に戻ることを確認してほしい。

### 3.3 指数減衰モデルでのヤコビ行列

以降の実験で使う指数減衰モデル $f(x; \boldsymbol{\beta}) = \beta_1 e^{\beta_2 x}$ (文書 4 と同じ) で $J$ を具体的に書いておく。残差は $r_i = y_i - \beta_1 e^{\beta_2 x_i}$ なので

$$
\frac{\partial r_i}{\partial \beta_1} = - e^{\beta_2 x_i},
\qquad
\frac{\partial r_i}{\partial \beta_2} = - \beta_1 x_i \, e^{\beta_2 x_i}
$$

であり、$J$ の第 $i$ 行は $\bigl( -e^{\beta_2 x_i}, \; -\beta_1 x_i e^{\beta_2 x_i} \bigr)$ となる。サンプルコードの `jacobian` 関数はこの式をそのまま実装している。第 2 列に $\beta_1$ が掛かっていることを覚えておいてほしい — §8 の故障モードの伏線である。

## 4. ガウス・ニュートン法 — 線形最小二乗の繰り返し

### 4.1 部分問題は文書 1 の問題そのものである

線形化した残差を目的関数に代入する。

$$
E(\boldsymbol{\beta} + \boldsymbol{\delta})
= \| \mathbf{r}(\boldsymbol{\beta} + \boldsymbol{\delta}) \|^2
\approx \| \mathbf{r} + J \boldsymbol{\delta} \|^2
$$

現在の点での $\mathbf{r}$ と $J$ は計算済みの **定数** であり、未知数は $\boldsymbol{\delta}$ だけである。つまり各反復で解くべき **部分問題** は

$$
\min_{\boldsymbol{\delta}} \; \| \mathbf{r} + J \boldsymbol{\delta} \|^2
= \min_{\boldsymbol{\delta}} \; \| J \boldsymbol{\delta} - (-\mathbf{r}) \|^2
$$

である。この形をよく見てほしい。[1. 線形最小二乗法](./1_least_squares_method.md) で解いた問題は $\min_{\boldsymbol{\beta}} \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2$ だった。記号を

$$
\Phi \;\to\; J,
\qquad
\mathbf{y} \;\to\; -\mathbf{r},
\qquad
\boldsymbol{\beta} \;\to\; \boldsymbol{\delta}
$$

と読み替えれば、部分問題は文書 1 の問題と **完全に同じ形** である。ここがガウス・ニュートン法の発想の転換である。

> 非線形最小二乗問題は一発では解けない。しかし残差を線形化すれば、各反復は「すでに完全に解ける」線形最小二乗問題になる。つまりガウス・ニュートン法とは、**非線形の問題を「線形最小二乗の繰り返し」に変える** 手法である。

文書 1 (正規方程式)、[2. QR 分解](./2_qr_decomposition.md) (安定な解法) で積み上げた道具が、そっくりそのまま毎反復の内側で再利用される。

### 4.2 部分問題の解 — 正規方程式

読み替えを使えば、文書 1 の正規方程式 $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$ から直ちに

$$
\boxed{\;
J^\top J \, \boldsymbol{\delta} = - J^\top \mathbf{r}
\;}
$$

が得られる。これがガウス・ニュートン法の更新量を定める方程式である。念のため、読み替えに頼らず直接も確かめておく。ノルムの 2 乗を内積で展開すると

$$
\begin{aligned}
\| \mathbf{r} + J \boldsymbol{\delta} \|^2
&= (\mathbf{r} + J \boldsymbol{\delta})^\top (\mathbf{r} + J \boldsymbol{\delta}) \\
&= \mathbf{r}^\top \mathbf{r} + \mathbf{r}^\top J \boldsymbol{\delta} + \boldsymbol{\delta}^\top J^\top \mathbf{r} + \boldsymbol{\delta}^\top J^\top J \boldsymbol{\delta} \\
&= \mathbf{r}^\top \mathbf{r} + 2 \, (J^\top \mathbf{r})^\top \boldsymbol{\delta} + \boldsymbol{\delta}^\top (J^\top J) \boldsymbol{\delta}
\end{aligned}
$$

(2 行目から 3 行目では、$\mathbf{r}^\top J \boldsymbol{\delta}$ がただの数なので転置しても値が変わらないこと $\mathbf{r}^\top J \boldsymbol{\delta} = \boldsymbol{\delta}^\top J^\top \mathbf{r}$ を使った)。これは $\boldsymbol{\delta}$ の 2 次関数であり、文書 1 の §4 とまったく同じ計算で、$\boldsymbol{\delta}$ についての勾配を 0 とおくと $2 J^\top \mathbf{r} + 2 J^\top J \boldsymbol{\delta} = \mathbf{0}$、すなわち上の正規方程式に一致する。

$m = 1$ の場合、$J^\top J = \sum_i (r_i')^2$、$J^\top \mathbf{r} = \sum_i r_i' r_i$ はどちらもただの数になり、$\delta = - \frac{\sum_i r_i' r_i}{\sum_i (r_i')^2}$ となって §2.3 で頂点の公式から出した式と一致する。

なお、右辺の $-J^\top \mathbf{r}$ は勾配 $\nabla E = 2 J^\top \mathbf{r}$ の $-\frac{1}{2}$ 倍である。つまりガウス・ニュートン法は「勾配の逆方向 (最急降下方向) を、行列 $J^\top J$ で変形した方向」に進んでいる、とも読める。

### 4.3 実際の計算は QR 分解で

正規方程式は理論の道具としては完璧だが、数値計算で $J^\top J$ を明示的に作って解くのは避けるべきである。理由は文書 1 の §6 で見たとおり: $J^\top J$ を作ると条件数が 2 乗されてしまい、桁落ちに弱くなる。部分問題 $\min_{\boldsymbol{\delta}} \| J \boldsymbol{\delta} - (-\mathbf{r}) \|^2$ は普通の線形最小二乗なのだから、[2. QR 分解](./2_qr_decomposition.md) の §5 の手順で $J = QR$ と分解し、$R \boldsymbol{\delta} = Q^\top (-\mathbf{r})$ を後退代入で解けばよい。サンプルコードでも各反復で `lstsq_qr(&j, &minus_r)` を呼んでおり、これは文書 1・2 で実装した関数の再利用である。

### 4.4 アルゴリズム全体

まとめると、ガウス・ニュートン法は次のループである。

1. 初期値 $\boldsymbol{\beta}_0$ を決める
2. 現在の $\boldsymbol{\beta}_k$ で残差 $\mathbf{r} = \mathbf{r}(\boldsymbol{\beta}_k)$ とヤコビ行列 $J = J(\boldsymbol{\beta}_k)$ を計算する
3. 線形最小二乗の部分問題 $\min_{\boldsymbol{\delta}} \| \mathbf{r} + J \boldsymbol{\delta} \|^2$ を QR 分解で解き、$\boldsymbol{\delta}_k$ を得る
4. $\boldsymbol{\beta}_{k+1} = \boldsymbol{\beta}_k + \boldsymbol{\delta}_k$ と更新する
5. $\| \boldsymbol{\delta}_k \|$ が十分小さければ終了。そうでなければ 2 に戻る

必要な微分は 1 階の偏微分 (ヤコビ行列) だけである。ニュートン法で問題だった「残差ごとの 2 階微分 $\nabla^2 r_i$」はどこにも現れない。それでいて、§6 で見るようにニュートン法並みに速い。なぜそんな都合のよいことが可能なのか — それを次節で明らかにする。

## 5. ニュートン法との関係 — 何を捨てたのか

§2〜4 の導出は「残差を線形化する」という見方だった。同じ更新式は「ニュートン法のヘッセ行列を近似する」という見方でも導ける。この 2 つ目の見方から、ガウス・ニュートン法が **何を捨てているのか**、そして **いつ捨ててよいのか** が見える。

### 5.1 $E$ のヘッセ行列を成分で計算する

$E(\boldsymbol{\beta}) = \sum_{i=1}^{n} r_i(\boldsymbol{\beta})^2$ を素直に 2 回偏微分する。まず 1 回目は合成関数の微分 (外側 $u^2$、内側 $u = r_i$) で

$$
\frac{\partial E}{\partial \beta_k}
= \sum_{i=1}^{n} 2 \, r_i \, \frac{\partial r_i}{\partial \beta_k}
$$

これは文書 4 で $\nabla E = 2 J^\top \mathbf{r}$ と行列表記した式の成分表示である。続いてこれを $\beta_l$ でもう一度微分する。和の中身 $r_i \cdot \frac{\partial r_i}{\partial \beta_k}$ は「関数 × 関数」なので **積の微分法則** を使う。

$$
\frac{\partial}{\partial \beta_l} \Bigl( r_i \, \frac{\partial r_i}{\partial \beta_k} \Bigr)
= \frac{\partial r_i}{\partial \beta_l} \, \frac{\partial r_i}{\partial \beta_k} + r_i \, \frac{\partial^2 r_i}{\partial \beta_l \, \partial \beta_k}
$$

よってヘッセ行列の $(k, l)$ 成分は

$$
\frac{\partial^2 E}{\partial \beta_l \, \partial \beta_k}
= 2 \sum_{i=1}^{n} \frac{\partial r_i}{\partial \beta_k} \frac{\partial r_i}{\partial \beta_l}
\;+\; 2 \sum_{i=1}^{n} r_i \, \frac{\partial^2 r_i}{\partial \beta_l \, \partial \beta_k}
$$

である。第 1 項をよく見ると、$(J^\top J)_{kl} = \sum_i J_{ik} J_{il} = \sum_i \frac{\partial r_i}{\partial \beta_k} \frac{\partial r_i}{\partial \beta_l}$ に一致している。したがって行列の形にまとめると

$$
\nabla^2 E
= 2 \Bigl( \underbrace{J^\top J}_{\text{1 階微分だけで計算できる}} + \underbrace{\sum_{i=1}^{n} r_i \, \nabla^2 r_i}_{\text{2 階微分が必要}} \Bigr)
$$

となる (文書 4 の式の再導出)。

### 5.2 ニュートン法の式と並べて比較する

ニュートン法の反復式は $\nabla^2 E \, \boldsymbol{\delta} = - \nabla E$ だった。上の分解と $\nabla E = 2 J^\top \mathbf{r}$ を代入すると

$$
2 \Bigl( J^\top J + \sum_i r_i \nabla^2 r_i \Bigr) \boldsymbol{\delta} = - 2 J^\top \mathbf{r}
$$

両辺の 2 を約分し、さらに **第 2 項 $\sum_i r_i \nabla^2 r_i$ を捨てる** と

$$
J^\top J \, \boldsymbol{\delta} = - J^\top \mathbf{r}
$$

— §4.2 のガウス・ニュートン法の式そのものである。つまり、

> **ガウス・ニュートン法 = ニュートン法のヘッセ行列 $\nabla^2 E$ を $2 J^\top J$ で置き換えたもの**。捨てたのは「残差 × 残差の 2 階微分」の項である。

残差の線形化 (§3) とヘッセ行列の近似 (本節) という 2 つの導出が同じ式に着地するのは偶然ではない。「残差を 1 次式で置き換える」ことは「残差の 2 階微分を 0 とみなす」ことと同じであり、そのとき消えるのがちょうど第 2 項だからである。

### 5.3 いつ捨ててよいか — 残差が小さいときに近似が良い理由

捨てた項 $\sum_i r_i \nabla^2 r_i$ には残差 $r_i$ が **掛け算で** 入っている。ここが決定的に重要で、次の 2 つの場合にこの項は小さくなる。

1. **残差が小さいとき** ($r_i \approx 0$)。モデルがデータによく合っているなら、解の近くで $r_i$ は小さく、第 2 項は $\nabla^2 r_i$ の大きさによらず自動的に小さくなる。フィッティングの目的はまさに残差を小さくすることなのだから、「うまくいく問題ほどこの近似は正確になる」という自己強化的な性質を持つ。
2. **モデルがほぼ線形なとき** ($\nabla^2 r_i \approx O$)。残差の曲がり自体が小さければ、線形化の誤差はもともと小さい。極端な例として完全に線形なモデルでは $\nabla^2 r_i$ が厳密に零行列になり、ガウス・ニュートン法はニュートン法と厳密に一致して 1 反復で収束する (文書 1 の正規方程式を 1 回解くことと同じ)。

指数減衰モデルで具体的に確かめると、$r_i = y_i - \beta_1 e^{\beta_2 x_i}$ の 2 階偏微分は

$$
\frac{\partial^2 r_i}{\partial \beta_1^2} = 0,
\qquad
\frac{\partial^2 r_i}{\partial \beta_1 \partial \beta_2} = - x_i e^{\beta_2 x_i},
\qquad
\frac{\partial^2 r_i}{\partial \beta_2^2} = - \beta_1 x_i^2 e^{\beta_2 x_i}
$$

であり、決して 0 ではない。それでも解の近くでは $r_i$ (ノイズ程度の大きさ) が掛かるため、捨てた項は $J^\top J$ に比べて十分小さい。

収束の速さもこの観点で説明できる。ニュートン法との差は捨てた項だけなので、**解での残差が 0 に近いほどニュートン法との差が消え、二次収束に近い速さになる**。逆に解でも残差が大きい問題 (モデルがデータに本質的に合っていない問題) では、近似の質が落ちて収束が遅くなり、収束しないこともある。

### 5.4 おまけの利点 — ガウス・ニュートン方向は必ず下り坂

第 2 項を捨てることには、計算量の節約以外の利点もある。任意のベクトル $\mathbf{v}$ に対して

$$
\mathbf{v}^\top (J^\top J) \mathbf{v} = (J \mathbf{v})^\top (J \mathbf{v}) = \| J \mathbf{v} \|^2 \geq 0
$$

なので、$J^\top J$ は常に **半正定値** (どの方向に切っても下に凸) である。真のヘッセ行列 $\nabla^2 E$ は解から遠い場所で正定値とは限らず、その場合ニュートン方向は下り方向ですらないことがあった (文書 6 §4)。一方ガウス・ニュートン方向 $\boldsymbol{\delta}$ は、$J^\top \mathbf{r} = -J^\top J \boldsymbol{\delta}$ を使うと

$$
\nabla E^\top \boldsymbol{\delta}
= 2 (J^\top \mathbf{r})^\top \boldsymbol{\delta}
= -2 (J^\top J \boldsymbol{\delta})^\top \boldsymbol{\delta}
= -2 \| J \boldsymbol{\delta} \|^2 \leq 0
$$

となり、$J$ の列が線形独立 (フルランク) で $\boldsymbol{\delta} \neq \mathbf{0}$ である限り厳密に負、すなわち **$\boldsymbol{\delta}$ の方向に少し進めば $E$ は必ず減る**。ただし注意: これは「方向が下り坂」という保証であって、「$\boldsymbol{\delta}$ の分だけ丸ごと進んでも減る」という保証ではない。この区別が §7 の故障につながる。

## 6. 実験 — 指数減衰モデルで動かす

### 6.1 設定

サンプルコード `src/bin/7_gauss_newton_method.rs` は、文書 4 と同じデータセットを使う。

- モデル: $f(x; \boldsymbol{\beta}) = \beta_1 e^{\beta_2 x}$、真値 $\boldsymbol{\beta}^* = (2.0, -1.5)^\top$
- データ: $x_i$ は $[0, 2]$ の等間隔 30 点、$y_i = 2.0 \, e^{-1.5 x_i} + (\pm 0.01 \text{ 程度の一様ノイズ})$

以下の数表はすべてこのコードの実行結果である (乱数シード固定なので再現できる)。

### 6.2 良い初期値 $[1.0, -1.0]$ — 数反復で収束する

初期値 $\boldsymbol{\beta}_0 = (1.0, -1.0)^\top$ から回した結果:

| 反復 $k$ | $\beta_1$ | $\beta_2$ | $E$ |
| --- | --- | --- | --- |
| 0 | $1.000000$ | $-1.000000$ | $3.87 \times 10^{0}$ |
| 1 | $1.954586$ | $-1.722002$ | $2.20 \times 10^{-1}$ |
| 2 | $1.994935$ | $-1.472780$ | $4.03 \times 10^{-3}$ |
| 3 | $2.001011$ | $-1.504544$ | $8.208 \times 10^{-4}$ |
| 4 | $2.001165$ | $-1.505036$ | $8.201 \times 10^{-4}$ |

わずか 4 反復で推定値 $(2.0012, -1.5050)$ に到達した。同じ問題を [5. 最急降下法](./5_steepest_descent.md) で解いたときは数千反復が必要だったから、圧倒的な差である。観察を 2 つ。

- **$E$ は 0 にならず $8.2 \times 10^{-4}$ で止まる。** データにノイズが乗っているため、真値ですら残差は 0 にならない。この値がこの問題の「残差の床」であり、それ以上は下がりようがない (実際、一様ノイズ $\pm 0.01$ の 30 点分の二乗和の期待値は約 $10^{-3}$ で、桁が合っている)。
- **収束は二次収束的である。** 更新量のノルム $\| \boldsymbol{\delta}_k \|$ は反復ごとに $1.20 \to 0.25 \to 0.032 \to 0.00052 \to 0.00000006$ と推移しており、およそ「前回の 2 乗」のペースで縮んでいる。残差の床が小さい (よくフィットする) 問題なので、§5.3 の議論どおりニュートン法並みの速さが出ている。

ここまでがガウス・ニュートン法の光の面である。ここからは影の面 — 同じコードが初期値を変えるだけで 2 通りに壊れる様子を、数値で追いかける。

## 7. 故障モード 1 — 飛びすぎて発散する (初期値 $[5.0, 5.0]$)

### 7.1 何が起きるか

初期値を $\boldsymbol{\beta}_0 = (5.0, 5.0)^\top$ にする。真のデータは減衰 ($\beta_2 < 0$) なのに、急増 ($\beta_2 = 5$) を仮定した悪い初期値である。各反復の $\boldsymbol{\beta}$ と $E$、および部分問題の解 $\boldsymbol{\delta}$ を出力すると次のようになる。

| 反復 $k$ | $\beta_1$ | $\beta_2$ | $E$ | $\boldsymbol{\delta}$ |
| --- | --- | --- | --- | --- |
| 0 | $5.0$ | $5.0$ | $2.43 \times 10^{10}$ | $(-4.9996, \; -0.00004)$ |
| 1 | $0.00039$ | $4.99996$ | $1.61 \times 10^{2}$ | $(0.00000003, \; -0.504)$ |
| 2 | $0.00039$ | $4.496$ | $3.92 \times 10^{1}$ | $(0.0007, \; -1.404)$ |
| 3 | $0.0011$ | $3.092$ | $2.12 \times 10^{1}$ | $(0.018, \; -9.081)$ |
| 4 | $0.019$ | $-5.989$ | $2.11 \times 10^{1}$ | $(1.673, \; +1072.7)$ |
| 5 | $1.692$ | $1066.7$ | $\infty$ | (打ち切り) |

$E$ は反復 4 まで単調に減っている ($2.4 \times 10^{10} \to 21.1$) のに、反復 4 → 5 で突然 $\beta_2$ が $+1066.7$ へ吹き飛び、$E = \infty$ となって計算が破綻する。

### 7.2 なぜ起きるか

順を追って読み解く。

**初期状態。** $\beta_2 = 5$ では、$x = 2$ で $e^{5 \times 2} = e^{10} \approx 2.2 \times 10^4$。モデルの値はデータ ($y \leq 2$) より 4 桁以上大きく、$E_0 = 2.4 \times 10^{10}$ という途方もない値になる。

**反復 0: $\beta_1$ が潰される。** モデル $\beta_1 e^{\beta_2 x}$ は $\beta_1$ については **線形** なので、この方向の線形化は厳密である。巨大な基底関数 $e^{5x}$ に対する最適なスケールはほぼ 0 であり、部分問題は正しく $\beta_1 \approx 0.0004$ と答える。$E$ は $161$ まで急落する。ここまでは合理的な挙動である。

**反復 1〜4: $\beta_2$ をじりじり下げる。** $\beta_1$ がほぼ 0 なので、モデルはほぼ恒等的に 0、残差はほぼデータそのもの ($E \approx \sum_i y_i^2 \approx 21$) で頭打ちになる。部分問題は $\beta_2$ を $5 \to 4.5 \to 3.1 \to -6.0$ と下げていくが、$E$ は 21 付近から動かない。

**反復 4: 破滅的なステップ。** ここで $\boldsymbol{\beta} = (0.019, -5.989)$ である。§3.3 で見たとおり、$J$ の第 2 列は $-\beta_1 x_i e^{\beta_2 x_i}$ で **$\beta_1$ に比例** する。$\beta_1 \approx 0.02$ と小さいため第 2 列の成分は最大でも $10^{-3}$ 程度しかなく、線形モデル上「$\beta_2$ を動かしても残差はほとんど変わらない」ことになる。一方で残差を減らすにはモデルをデータに近づける必要があり、その (ごくわずかな) 感度で効果を出すために、部分問題は $\delta_2 = +1072.7$ という **桁外れのステップ** を最適解として返す。実際、$\delta_1 = 1.67$ で $\beta_1 \approx 1.69$ と真値 2.0 に近づいており、線形モデルの「言い分」としては筋が通っている。

**なぜそれが致命傷になるか。** 線形化 $\mathbf{r}(\boldsymbol{\beta} + \boldsymbol{\delta}) \approx \mathbf{r} + J \boldsymbol{\delta}$ が有効なのは $\boldsymbol{\delta}$ が小さい範囲だけである (§3.1 で 2 次の微小量を捨てたことを思い出そう)。ところがガウス・ニュートン法には **$\boldsymbol{\delta}$ の大きさを抑える仕組みが何もない**。部分問題の厳密な最適解が $\delta_2 = 1073$ なら、その言い分を無条件に信じてそのまま飛ぶ。飛んだ先の $\beta_2 = 1066.7$ では $e^{\beta_2 x}$ が $x = 2$ で $e^{2133}$ となり、浮動小数点数で表現できる上限 ($\approx 10^{308}$) を遥かに超えてオーバーフローし、$E = \infty$ になる。§5.4 で「方向は下り坂」を証明したが、「その距離だけ進んでも下る」ことは何も保証されていない — その隙を突かれた形である。

> **故障モード 1 の本質:** 線形化には有効範囲があるのに、ステップ幅を制御する仕組みがないため、近似の有効範囲外へ飛び出して $E$ が発散し得る。

## 8. 故障モード 2 — $J^\top J$ が特異になって止まる (初期値 $[1.0, 3.0]$)

### 8.1 何が起きるか

今度は初期値を $\boldsymbol{\beta}_0 = (1.0, 3.0)^\top$ にする。主要な反復を抜き出すと:

| 反復 $k$ | $\beta_1$ | $\beta_2$ | $E$ | $\boldsymbol{\delta}$ |
| --- | --- | --- | --- | --- |
| 0 | $1.0$ | $3.0$ | $4.80 \times 10^{5}$ | $(-0.977, \; -0.012)$ |
| 1 | $0.0229$ | $2.988$ | $2.45 \times 10^{2}$ | $(0.0006, \; -0.530)$ |
| ⋮ | | | | |
| 4 | $0.710$ | $-4.914$ | $1.44 \times 10^{1}$ | $(1.007, \; +20.5)$ |
| 5 | $1.718$ | $15.59$ | $4.06 \times 10^{27}$ | $(-1.718, \; -0.0000000000001)$ |
| 6 | $3.7 \times 10^{-13}$ | $15.59$ | $2.08 \times 10^{2}$ | $(-0.000, \; -0.494)$ |
| ⋮ | | | | |
| 9 | $1.2 \times 10^{-11}$ | $7.53$ | $2.13 \times 10^{1}$ | $(0.0000026, \; -104518)$ |
| 10 | $2.6 \times 10^{-6}$ | $-104511$ | $2.13 \times 10^{1}$ | $(\mathrm{NaN}, \; -\infty)$ (打ち切り) |

今度は $E = \infty$ にはならない。代わりに、反復 10 で部分問題の解 $\boldsymbol{\delta}$ 自体が NaN (計算不能) になり、$\boldsymbol{\beta} = (2.6 \times 10^{-6}, \; -104511)$ という真値から程遠い場所でアルゴリズムが停止する。

### 8.2 なぜ起きるか

**反復 4 → 5: まず飛びすぎる。** 反復 4 で $\boldsymbol{\beta} = (0.710, -4.914)$。ここでも故障モード 1 と同様に大きめのステップ $\delta_2 = +20.5$ が出て、$\beta_2 = 15.59$ の高台 ($E = 4.1 \times 10^{27}$) に飛び上がってしまう。ここまでは故障モード 1 と同じ筋書きである。

**反復 5: $\beta_1$ が 0 に飛ばされる。** $\beta_2 = 15.59$ では $e^{\beta_2 x}$ が $x = 2$ で $e^{31.2} \approx 3.5 \times 10^{13}$ と巨大である。モデルは $\beta_1$ について線形なので、部分問題は「この巨大な基底に掛けるスケールはほぼ 0 にせよ」と正確に判断し、$\delta_1 = -1.718$ で $\beta_1$ を $3.7 \times 10^{-13}$、実質 0 に潰す。オーバーフローは免れたが、ここで罠に落ちた。

**$\beta_1 = 0$ はパラメータが 1 個死ぬ点である。** $\beta_1 = 0$ だとモデルは $0 \cdot e^{\beta_2 x} = 0$ となり、**$\beta_2$ の値が何であってもモデルの出力は変わらない**。式で見ると、$J$ の第 2 列 $-\beta_1 x_i e^{\beta_2 x_i}$ は $\beta_1 = 0$ で全成分が 0、つまり **零ベクトル** になる。列の 1 本が零ベクトルなら $J$ の列は線形独立でなく ([0. 数学の準備](./0_math_preliminaries.md) の言葉でランク落ち)、文書 1 の §5 で見たとおり $J^\top J$ は逆行列を持たない (**特異**) から、正規方程式 $J^\top J \boldsymbol{\delta} = -J^\top \mathbf{r}$ は一意に解けなくなる。

**反復 6〜10: 特異点への転落。** 反復 6 以降、$\beta_1 \approx 10^{-13}$ のまま $\beta_2$ だけが動く展開になる。$J$ の第 2 列は零ベクトル寸前で、部分問題は「ほぼ特異」— すると故障モード 1 と同じ理屈で $\delta_2$ が暴走し、反復 9 で $\delta_2 = -104518$、$\beta_2 = -104511$ へ飛ぶ。この点では $e^{\beta_2 x_i}$ が ($x_i > 0$ のすべての点で) 小さすぎて浮動小数点数では **厳密に 0** になり (アンダーフロー)、$J$ の第 2 列は厳密に零ベクトルとなる。QR 分解の $R$ の対角に 0 が現れ、後退代入の割り算が 0 除算となって $\boldsymbol{\delta} = (\mathrm{NaN}, -\infty)$ — 更新量が計算不能になり、打ち切られる。

> **故障モード 2 の本質:** $J$ がランク落ちする (またはそれに近い) 場所では、$J^\top J$ が特異 (または悪条件) になり、$\boldsymbol{\delta}$ が計算不能・無意味になる。この例では「$\beta_1$ が 0 に飛ばされ、モデルが $\beta_2$ に反応しなくなる」ことでランク落ちが発生した。

2 つの故障モードは無関係ではない。どちらも「大きすぎるステップで変な場所に飛ぶ」ことから始まり、飛んだ先が発散 (モード 1) か特異点 (モード 2) かの違いである。そして「$J^\top J$ が特異に近いとき、正規方程式の解 $\boldsymbol{\delta}$ は巨大になる」という事実を介して、特異性がさらなる飛びすぎを呼ぶ悪循環になっている。

## 9. Levenberg-Marquardt 法へ

2 つの故障モードを並べると、必要な手当てが見えてくる。

| 故障モード | 原因 | 必要な手当て |
| --- | --- | --- |
| 1. 発散 | $\boldsymbol{\delta}$ が線形化の有効範囲を超えて大きすぎる | ステップの大きさを抑える |
| 2. 特異 | $J^\top J$ が特異・悪条件で $\boldsymbol{\delta}$ が解けない/暴れる | $J^\top J$ を正則化する |

驚くべきことに、この 2 つは **1 つの仕掛けで同時に** 手当てできる。正規方程式の $J^\top J$ に $\lambda I$ ($\lambda > 0$、$I$ は単位行列) を足すのである。

$$
(J^\top J + \lambda I) \, \boldsymbol{\delta} = - J^\top \mathbf{r}
$$

$\lambda I$ を足すと $J^\top J + \lambda I$ は必ず正定値になって特異性が消え (モード 2 の手当て)、さらに $\lambda$ を大きくするほど $\boldsymbol{\delta}$ が小さく・最急降下方向に近くなってステップが抑えられる (モード 1 の手当て)。これは文書 6 §4 の修正ニュートン法 ($H + \tau I$) と同じ発想であり、この $\lambda$ を状況に応じて自動調節する仕組みを加えたものが、次の [8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) である。

## 10. まとめ

> ガウス・ニュートン法は、残差を 1 次近似 $\mathbf{r}(\boldsymbol{\beta} + \boldsymbol{\delta}) \approx \mathbf{r} + J \boldsymbol{\delta}$ で置き換えることにより、非線形最小二乗問題を「線形最小二乗問題の繰り返し」に変える手法である。2 階微分なしでニュートン法に近い速さが出るが、ステップ幅の制御がなく、$J^\top J$ が特異になると破綻する。

- 各反復の部分問題 $\min_{\boldsymbol{\delta}} \| \mathbf{r} + J \boldsymbol{\delta} \|^2$ は [1. 線形最小二乗法](./1_least_squares_method.md) の問題そのもの ($\Phi \to J$、$\mathbf{y} \to -\mathbf{r}$) であり、[2. QR 分解](./2_qr_decomposition.md) で安定に解ける。更新式は $J^\top J \boldsymbol{\delta} = -J^\top \mathbf{r}$。
- 同じ式は、ニュートン法のヘッセ行列 $\nabla^2 E = 2 (J^\top J + \sum_i r_i \nabla^2 r_i)$ から第 2 項を捨てた近似としても導ける。第 2 項には残差が掛かっているため、**残差が小さい (よくフィットする) 問題ほど近似が良く、二次収束に近づく**。
- 故障モードは 2 つ。(1) $\boldsymbol{\delta}$ の大きさに制限がなく、線形化の有効範囲外へ飛んで発散する。(2) $J$ がランク落ちすると $J^\top J$ が特異になり $\boldsymbol{\delta}$ が計算できない。サンプルコード `src/bin/7_gauss_newton_method.rs` では、初期値 $(5.0, 5.0)$ と $(1.0, 3.0)$ からそれぞれの故障が実際に再現される。
- この 2 つの故障を $\lambda I$ の追加 (ダンピング) で同時に手当てするのが [8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) である。
- 「線形化 → 解く → 更新 → 再線形化」というループは、非線形な問題に対する数値解法の普遍的なパターンである。ここで骨格を掴んでおくと、他の分野 (非線形方程式、最適制御など) でも同じ構造に再会する。
