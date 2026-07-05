# 非線形最小二乗法 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

サンプルコード: [`src/bin/4_nonlinear_least_squares.rs`](../../src/bin/4_nonlinear_least_squares.rs)

前提知識: [0. 数学の準備](./0_math_preliminaries.md) (Σ記法、ベクトルとノルム、行列と転置、偏微分・勾配・連鎖律、凸関数)、[1. 線形最小二乗法](./1_least_squares_method.md)

## 1. この文書の位置づけ

[線形最小二乗法](./1_least_squares_method.md) では、モデルが**パラメータについて線形**である限り、正規方程式を 1 回解くだけで最適なパラメータが求まることを見た。しかし現実のモデルには、パラメータが指数の中や分母に入っているものが多く、その場合は正規方程式のような「一発で解ける式」が存在しない。これが**非線形最小二乗法** (nonlinear least squares) の世界である。

この文書で扱うのは、次の 3 つである。

1. **問題設定** — 何を最小化したいのか。線形の場合と何が同じで、何が違うのか
2. **微分の構造** — 残差ベクトル $\mathbf{r}$ とヤコビ行列 $J$ を定義し、勾配が $\nabla E = 2 J^\top \mathbf{r}$ という形にまとまることを導く
3. **対数変換による線形化の注意点** — 手軽な近似解法だが「別の問題」を解いていること

ここでは**実際に解くところまでは進まない**。解くための反復法は、[5. 最急降下法](./5_steepest_descent.md)、[7. ガウス・ニュートン法](./7_gauss_newton_method.md)、[8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) がそれぞれ担当する。この文書はそれらすべての土台となる「問題の形」と「微分の道具立て」を準備する回である。

## 2. 問題設定 — 指数減衰モデルを例に

### 2.1 題材

サンプルコードと同じ題材を使う。**指数減衰モデル**

```math
f(x; \boldsymbol{\beta}) = \beta_1 e^{\beta_2 x}
```

をデータにフィットさせたい。$\boldsymbol{\beta} = (\beta_1, \beta_2)^\top$ が求めたい未知パラメータである。$\beta_2 < 0$ なら $x$ が増えるにつれて $y$ が指数的に減っていく曲線で、放射性物質の崩壊、コンデンサの放電、薬の血中濃度など、自然界のいたるところに現れる形である。

データは次のように作る (サンプルコードの `make_dataset`)。

- 真のパラメータを $\boldsymbol{\beta}_{\text{true}} = (2.0, \ -1.5)^\top$ とする
- $x_1, \dots, x_{30}$ を区間 $[0, 2]$ に等間隔に 30 点取る
- $y_i = 2.0 \, e^{-1.5 x_i} + (\text{小さなノイズ})$ とする (ノイズは $\pm 0.01$ の一様乱数)

つまり「答えを知った上でノイズ入りデータを作り、そこから答えを推定し直せるか」を試す設定である。フィットの良し悪しは、線形のときとまったく同じく**残差の二乗和**

```math
E(\boldsymbol{\beta}) = \sum_{i=1}^{n} \bigl( y_i - f(x_i; \boldsymbol{\beta}) \bigr)^2
```

で測り、これを最小にする $\boldsymbol{\beta}$ を探す。$n = 30$ である。目的関数 $E$ の定義は線形の場合と一字一句同じであり、**変わったのはモデル $f$ の中身だけ**である。

### 2.2 何が「非線形」なのか

[線形最小二乗法](./1_least_squares_method.md) で強調したとおり、「線形」とはモデルが**パラメータについて**線形、すなわち

```math
f(x; \boldsymbol{\beta}) = \beta_1 \phi_1(x) + \beta_2 \phi_2(x) + \cdots + \beta_m \phi_m(x)
```

という「係数 × 決まった関数」の和で書けることを指した。$\phi_k(x) = x^2$ や $\sin x$ のように $x$ について曲がっていても構わない。

指数減衰モデル $\beta_1 e^{\beta_2 x}$ はどうか。$\beta_1$ については線形である ($\beta_1$ 倍しているだけ)。しかし $\beta_2$ は**指数の肩に乗っている**。どんな関数 $\phi_1, \phi_2$ を選んでも

```math
\beta_1 e^{\beta_2 x} = \beta_1 \phi_1(x) + \beta_2 \phi_2(x)
```

の形には書けない。左辺で $\beta_2$ を動かすと曲線の「曲がり方」自体が変わるのに対し、右辺で $\beta_2$ を動かしても固定された形 $\phi_2(x)$ の混ぜる量が変わるだけだからである。このように、パラメータの少なくとも 1 つが線形結合の形に収まらないモデルを扱うのが非線形最小二乗法である。他の例としては

- ガウス関数: $f(x; \boldsymbol{\beta}) = \beta_1 \exp\Bigl( -\dfrac{(x - \beta_2)^2}{2 \beta_3^2} \Bigr)$ — 山の中心 $\beta_2$ と幅 $\beta_3$ が非線形に効く
- 点群への円のフィッティング — 中心座標と半径がパラメータ

など、計測・画像処理・統計モデリングで現れるモデルの多くがこちらに属する。

## 3. 線形の場合と何が変わるか

### 3.1 復習: 線形なら E は「1 つの谷しかない放物面」だった

線形の場合、$E(\boldsymbol{\beta}) = \|\mathbf{y} - \Phi \boldsymbol{\beta}\|^2$ を展開すると $\boldsymbol{\beta}$ の **2 次式**になった。1 変数でいえば $E(\beta) = a\beta^2 + b\beta + c$ ($a > 0$) という下に凸な放物線であり、多変数でも「お椀型の曲面」である。このとき

- 谷底 (最小点) はただ 1 つ
- 勾配 $\nabla E$ は $\boldsymbol{\beta}$ の **1 次式**になるので、$\nabla E = \mathbf{0}$ は連立 1 次方程式 (正規方程式) となり、一発で解ける
- どこから探し始めても答えは同じ。そもそも「探す」必要すらない

という三拍子が揃っていた。線形最小二乗法の解きやすさは、すべて「$E$ が凸 2 次関数である」という 1 つの事実から来ていたのである。

### 3.2 非線形にすると ∇E = 0 が解けなくなる

指数減衰モデルで $E$ を実際に展開してみる。

```math
E(\beta_1, \beta_2)
= \sum_{i=1}^{n} \bigl( y_i - \beta_1 e^{\beta_2 x_i} \bigr)^2
= \sum_{i=1}^{n} y_i^2
\ - \ 2 \beta_1 \sum_{i=1}^{n} y_i \, e^{\beta_2 x_i}
\ + \ \beta_1^2 \sum_{i=1}^{n} e^{2 \beta_2 x_i}
```

$\beta_1$ については確かに 2 次式である。しかし $\beta_2$ は $e^{\beta_2 x_i}$ の中に閉じ込められており、$E$ は $\beta_2$ の多項式ではない。「凸 2 次関数」という構造はここで壊れる。

それでも最小点では勾配が $\mathbf{0}$ になるはずなので ([0. 数学の準備](./0_math_preliminaries.md) の第 5 節)、$\nabla E = \mathbf{0}$ を書き下してみる。各成分を偏微分すると (計算の詳細は第 5 節で丁寧にやるので、ここでは結果だけ見る)

```math
\frac{\partial E}{\partial \beta_1}
= -2 \sum_{i=1}^{n} e^{\beta_2 x_i} \bigl( y_i - \beta_1 e^{\beta_2 x_i} \bigr) = 0,
\qquad
\frac{\partial E}{\partial \beta_2}
= -2 \sum_{i=1}^{n} \beta_1 x_i e^{\beta_2 x_i} \bigl( y_i - \beta_1 e^{\beta_2 x_i} \bigr) = 0
```

線形の場合はここが連立 **1 次**方程式になったから解けた。今回は未知数 $\beta_2$ が $e^{\beta_2 x_i}$ の中にいるため、この 2 本の式は**非線形連立方程式**である。移項して $\beta_2 = \cdots$ の形に変形する方法は存在せず、**閉じた形の解 (公式) は書けない**。

そこで発想を変える。「方程式を解いて一発で答えを出す」ことを諦め、

> 適当な出発点 $\boldsymbol{\beta}_0$ を決め、$E$ が下がる方向へ少しずつ $\boldsymbol{\beta}$ を動かしていく

という**反復法** (iterative method) で答えに近づくのである。その具体的な動かし方が文書 5, 7, 8 の主題になる。

### 3.3 E が凸でなくなる — 谷が複数あり得る

反復法には、線形のときには無かった落とし穴がある。それを 1 変数の目に見える例で確認しておく。

モデルを $f(x; \beta) = \sin(\beta x)$、データを 1 点だけ $(x_1, y_1) = (1, \ 0.9)$ とすると、目的関数は

```math
E(\beta) = (0.9 - \sin \beta)^2
```

である。$\sin \beta$ は $\beta$ を動かすと $-1$ と $1$ の間を永遠に往復するから、$E(\beta)$ のグラフは

- $\sin \beta = 0.9$ となる $\beta$ で谷底 ($E = 0$)
- $\sin \beta = -1$ となる $\beta$ で山頂 ($E = 3.61$)

が交互に無限に並ぶ、**波打った地形**になる。放物線のような「谷が 1 つだけのお椀」では全くない。データ点を増やせば各谷の深さは変わり、一番深い谷 (**大域的最小**, global minimum) と、それより浅い谷 (**局所的極小**, local minimum) ができる。

反復法を「地形の上でボールを転がして谷底を探す」ことに例えると、ボールは**出発点から見て最寄りの谷**に落ちる。出発点が悪ければ、浅い谷 (局所的極小) に落ちて止まり、本当の答え (大域的最小) には決してたどり着かない。これが

- **初期値が必要**であり、
- **結果が初期値に依存し得る**

という、非線形最小二乗法に固有の事情である。なお、今回の指数減衰モデルは幸い地形が素直で、常識的な初期値からなら正しい谷に落ちる。しかし一般のモデルではそうとは限らず、「凸 2 次」という保証を失った以上、常にこの落とし穴を意識する必要がある。

### 3.4 違いのまとめ

| | 線形最小二乗法 | 非線形最小二乗法 |
| --- | --- | --- |
| $E(\boldsymbol{\beta})$ の形 | 凸 2 次関数 (谷が 1 つのお椀) | 一般に非凸 (谷が複数あり得る) |
| $\nabla E = \mathbf{0}$ | 連立 1 次方程式 → 閉じた形で解ける | 非線形方程式 → 公式では解けない |
| 解き方 | 正規方程式を 1 回解く | 反復法で近づける |
| 初期値 | 不要 | 必要。結果は初期値に依存し得る |
| 見つかる解 | 大域的最小 (フルランク時、一意) | 局所的極小かもしれない |

## 4. 残差ベクトルとヤコビ行列

反復法の設計は次の文書に譲るとして、どの反復法にも共通して必要になる**材料**をここで揃える。それが残差ベクトル $\mathbf{r}$ とヤコビ行列 $J$ である。

### 4.1 残差ベクトル r

$i$ 番目のデータ点における「観測値とモデルのずれ」を**残差** (residual) と呼び、

```math
r_i(\boldsymbol{\beta}) = y_i - f(x_i; \boldsymbol{\beta})
\qquad (i = 1, \dots, n)
```

と定義する。$n$ 個の残差を縦に並べたものが**残差ベクトル**

```math
\mathbf{r}(\boldsymbol{\beta}) =
\begin{bmatrix}
r_1(\boldsymbol{\beta}) \\ r_2(\boldsymbol{\beta}) \\ \vdots \\ r_n(\boldsymbol{\beta})
\end{bmatrix}
\in \mathbb{R}^n
```

である。ここで大事なのは、$\mathbf{r}$ が「$\boldsymbol{\beta}$ を入れるとベクトルが返ってくる関数」だという見方である。目的関数はノルムを使って

```math
E(\boldsymbol{\beta}) = \sum_{i=1}^{n} r_i(\boldsymbol{\beta})^2 = \| \mathbf{r}(\boldsymbol{\beta}) \|^2
```

と書ける ([0. 数学の準備](./0_math_preliminaries.md) 第 2 節)。線形の場合は $\mathbf{r} = \mathbf{y} - \Phi \boldsymbol{\beta}$ という 1 次式だったが、今は $\boldsymbol{\beta}$ に対して非線形な関数である。

### 4.2 ヤコビ行列 J の定義 — 「偏微分を並べた表」

反復法では「今いる $\boldsymbol{\beta}$ を少し動かしたら、残差はどう変わるか」という情報が要る。残差は $n$ 個、パラメータは $m$ 個あるから、変化の仕方は $n \times m$ 通りある。それを全部並べた表が**ヤコビ行列** (Jacobian matrix) である。

```math
J(\boldsymbol{\beta}) =
\begin{bmatrix}
\dfrac{\partial r_1}{\partial \beta_1} & \dfrac{\partial r_1}{\partial \beta_2} & \cdots & \dfrac{\partial r_1}{\partial \beta_m} \\[2mm]
\dfrac{\partial r_2}{\partial \beta_1} & \dfrac{\partial r_2}{\partial \beta_2} & \cdots & \dfrac{\partial r_2}{\partial \beta_m} \\[2mm]
\vdots & \vdots & \ddots & \vdots \\[1mm]
\dfrac{\partial r_n}{\partial \beta_1} & \dfrac{\partial r_n}{\partial \beta_2} & \cdots & \dfrac{\partial r_n}{\partial \beta_m}
\end{bmatrix}
\in \mathbb{R}^{n \times m},
\qquad
J_{ik} = \frac{\partial r_i}{\partial \beta_k}
```

行と列の意味を言葉にしておく。

- **第 $i$ 行** は「$i$ 番目の残差 $r_i$ が、各パラメータの変化にどう反応するか」を並べた横ベクトル。実は $r_i$ の勾配 $\nabla r_i$ を横に寝かせたものである
- **第 $k$ 列** は「パラメータ $\beta_k$ を少し動かしたとき、$n$ 個の残差がそれぞれどれだけ動くか」を並べた縦ベクトル

つまり $J$ は「パラメータの微小な変化」と「残差ベクトルの微小な変化」をつなぐ変換表であり、1 変数関数における導関数 $f'$ の多変数・多出力版である。

### 4.3 指数減衰モデルでの計算 — 全成分を書き下す

定義だけでは掴みにくいので、指数減衰モデルで実際に全成分を計算する。残差は

```math
r_i(\beta_1, \beta_2) = y_i - \beta_1 e^{\beta_2 x_i}
```

である。$y_i$ と $x_i$ はデータであってただの定数であることに注意し、$\beta_1$ と $\beta_2$ で順に偏微分する。

**$\beta_1$ での偏微分。** $\beta_2$ を定数とみなすと、$e^{\beta_2 x_i}$ は定数であり、$r_i$ は「定数 − $\beta_1$ × 定数」という $\beta_1$ の 1 次式である。よって

```math
\frac{\partial r_i}{\partial \beta_1}
= \frac{\partial}{\partial \beta_1} \bigl( y_i - \beta_1 e^{\beta_2 x_i} \bigr)
= - e^{\beta_2 x_i}
```

**$\beta_2$ での偏微分。** 今度は $\beta_1$ を定数とみなす。$e^{\beta_2 x_i}$ を $\beta_2$ で微分するには、$u = \beta_2 x_i$ とおく連鎖律を使う ([0. 数学の準備](./0_math_preliminaries.md) 第 5 節)。$\dfrac{d}{du} e^u = e^u$、$\dfrac{\partial u}{\partial \beta_2} = x_i$ なので

```math
\frac{\partial}{\partial \beta_2} e^{\beta_2 x_i}
= e^{\beta_2 x_i} \cdot x_i
```

よって

```math
\frac{\partial r_i}{\partial \beta_2}
= \frac{\partial}{\partial \beta_2} \bigl( y_i - \beta_1 e^{\beta_2 x_i} \bigr)
= - \beta_1 x_i e^{\beta_2 x_i}
```

以上より、ヤコビ行列は $n \times 2$ の行列として全成分が求まった。

```math
J(\boldsymbol{\beta}) =
\begin{bmatrix}
- e^{\beta_2 x_1} & - \beta_1 x_1 e^{\beta_2 x_1} \\
- e^{\beta_2 x_2} & - \beta_1 x_2 e^{\beta_2 x_2} \\
\vdots & \vdots \\
- e^{\beta_2 x_n} & - \beta_1 x_n e^{\beta_2 x_n}
\end{bmatrix}
```

これはサンプルコードの `jacobian` 関数そのものである (行列の $(i, k)$ 成分を、$k = 0$ なら `-e`、$k = 1$ なら `-beta[0] * x * e` としている。$e$ は $e^{\beta_2 x_i}$ のこと)。

符号がすべてマイナスなのは、残差を $y_i - f$ (観測 − モデル) と定義したためである。モデルの値 $f$ が増えれば残差は減る、という当然の関係が符号に現れている。

### 4.4 手計算できる小さな例

数字で確かめたい人のために、データ 3 点だけのミニチュア版を手計算する。データを

```math
(x_1, y_1) = (0, \ 2.0), \qquad
(x_2, y_2) = (1, \ 0.5), \qquad
(x_3, y_3) = (2, \ 0.2)
```

とし、パラメータを $\boldsymbol{\beta} = (1, \ -1)^\top$ (つまり $\beta_1 = 1, \beta_2 = -1$) とする。まずモデルの値と残差を計算する。$e^{-1} \approx 0.368$、$e^{-2} \approx 0.135$ を使うと

| $i$ | $x_i$ | $y_i$ | $f(x_i) = 1 \cdot e^{-x_i}$ | $r_i = y_i - f(x_i)$ |
| --- | --- | --- | --- | --- |
| 1 | 0 | 2.0 | $1.000$ | $1.000$ |
| 2 | 1 | 0.5 | $0.368$ | $0.132$ |
| 3 | 2 | 0.2 | $0.135$ | $0.065$ |

次にヤコビ行列。第 1 列は $-e^{\beta_2 x_i} = -e^{-x_i}$、第 2 列は $-\beta_1 x_i e^{\beta_2 x_i} = -x_i e^{-x_i}$ なので

```math
J =
\begin{bmatrix}
-1.000 & 0 \\
-0.368 & -0.368 \\
-0.135 & -0.271
\end{bmatrix}
```

($J_{32} = -2 \times 0.135 = -0.271$)。第 1 行第 2 列が $0$ なのは、$x_1 = 0$ では $f = \beta_1 e^{0} = \beta_1$ となって $\beta_2$ が式から消えるためである。「$x = 0$ の点は $\beta_2$ を動かしても影響を受けない」という事実が、ヤコビ行列の成分にそのまま表れている。

### 4.5 数値微分による検証

手で導いた偏微分が正しいかどうかは、**数値微分**で機械的に確かめられる。偏微分の定義は「$\beta_k$ をほんの少し動かしたときの変化率」だから、小さな $h$ を使って

```math
\frac{\partial r_i}{\partial \beta_k}
\approx
\frac{r_i(\boldsymbol{\beta} + h \mathbf{e}_k) - r_i(\boldsymbol{\beta} - h \mathbf{e}_k)}{2h}
```

と近似できる ($\mathbf{e}_k$ は第 $k$ 成分だけ 1 のベクトル)。前後対称に取るこの形は**中心差分**と呼ばれ、片側だけの差分より精度が良い。

これが良い検証になる理由は、**同じ量を全く別のルートで計算している**からである。解析的な式 (紙の上の微分計算) と数値微分 (関数値の引き算) は独立した手続きなので、両者が一致すれば、微分計算のミス (符号の間違い、連鎖律の掛け忘れなど) はほぼ確実に検出できる。

サンプルコードの `jacobian_numeric` がこの計算で、実行すると

```
== ヤコビ行列の検証 (解析解 vs 数値微分) ==
最大差: 1.446e-10
```

全 $30 \times 2 = 60$ 成分の最大差が $10^{-10}$ のオーダーであり、4.3 節の手計算が正しいことが確認できる。

## 5. 勾配は ∇E = 2 Jᵀr にまとまる

材料が揃ったので、この文書の山場である**勾配の公式**を導く。目的関数

```math
E(\boldsymbol{\beta}) = \sum_{i=1}^{n} r_i(\boldsymbol{\beta})^2
```

の勾配 $\nabla E$ が、ヤコビ行列と残差ベクトルの積という極めて簡潔な形

```math
\nabla E = 2 J^\top \mathbf{r}
```

にまとまることを示す。

### 5.1 成分ごとの導出 — 連鎖律の Σ 計算

勾配の第 $k$ 成分、すなわち $\dfrac{\partial E}{\partial \beta_k}$ を計算する。和の微分は微分の和なので

```math
\frac{\partial E}{\partial \beta_k}
= \frac{\partial}{\partial \beta_k} \sum_{i=1}^{n} r_i^2
= \sum_{i=1}^{n} \frac{\partial}{\partial \beta_k} \bigl( r_i^2 \bigr)
```

各項 $r_i^2$ は「$\beta_k \mapsto r_i \mapsto r_i^2$」という合成関数である。外側は 2 乗する関数 $g(t) = t^2$ で $g'(t) = 2t$、内側は $r_i(\boldsymbol{\beta})$ だから、連鎖律 ([0. 数学の準備](./0_math_preliminaries.md) 第 5 節) により

```math
\frac{\partial}{\partial \beta_k} \bigl( r_i^2 \bigr)
= 2 r_i \cdot \frac{\partial r_i}{\partial \beta_k}
```

これを Σ に戻すと

```math
\frac{\partial E}{\partial \beta_k}
= \sum_{i=1}^{n} 2 r_i \frac{\partial r_i}{\partial \beta_k}
= 2 \sum_{i=1}^{n} \frac{\partial r_i}{\partial \beta_k} \, r_i
```

ここでヤコビ行列の定義 $J_{ik} = \dfrac{\partial r_i}{\partial \beta_k}$ を思い出すと

```math
\frac{\partial E}{\partial \beta_k}
= 2 \sum_{i=1}^{n} J_{ik} \, r_i
\tag{5.1}
```

これで勾配の各成分が「ヤコビ行列の第 $k$ 列と残差ベクトルの内積の 2 倍」であることが分かった。

### 5.2 行列の形にまとめる

式 (5.1) の Σ が、行列とベクトルの積の定義そのものになっていることを確認する。[0. 数学の準備](./0_math_preliminaries.md) 第 3 節のとおり、行列 $A$ とベクトル $\mathbf{v}$ の積の第 $k$ 成分は $(A \mathbf{v})_k = \sum_i A_{ki} v_i$、つまり「$A$ の第 $k$ **行**と $\mathbf{v}$ の内積」である。

式 (5.1) に現れるのは $\sum_i J_{ik} r_i$ で、添字の位置に注目すると $J$ の第 $k$ **列**と $\mathbf{r}$ の内積になっている。「列との内積」を「行との内積」に読み替えるには転置を使えばよい。転置の定義 $(J^\top)_{ki} = J_{ik}$ より

```math
\sum_{i=1}^{n} J_{ik} \, r_i
= \sum_{i=1}^{n} (J^\top)_{ki} \, r_i
= \bigl( J^\top \mathbf{r} \bigr)_k
```

よって式 (5.1) は

```math
\frac{\partial E}{\partial \beta_k} = 2 \bigl( J^\top \mathbf{r} \bigr)_k
\qquad (k = 1, \dots, m)
```

となり、$m$ 個の成分をまとめてベクトルとして書けば

```math
\boxed{\ \nabla E(\boldsymbol{\beta}) = 2 \, J(\boldsymbol{\beta})^\top \mathbf{r}(\boldsymbol{\beta}) \ }
```

が得られる。$J^\top$ は $m \times n$、$\mathbf{r}$ は $n$ 次元なので、積は $m$ 次元ベクトルとなり、勾配の次元と確かに一致している。

$m = 2$ の場合に中身を開いて見れば、まとめ方は一目瞭然である。

```math
2 J^\top \mathbf{r}
= 2
\begin{bmatrix}
\dfrac{\partial r_1}{\partial \beta_1} & \dfrac{\partial r_2}{\partial \beta_1} & \cdots & \dfrac{\partial r_n}{\partial \beta_1} \\[2mm]
\dfrac{\partial r_1}{\partial \beta_2} & \dfrac{\partial r_2}{\partial \beta_2} & \cdots & \dfrac{\partial r_n}{\partial \beta_2}
\end{bmatrix}
\begin{bmatrix}
r_1 \\ r_2 \\ \vdots \\ r_n
\end{bmatrix}
= 2
\begin{bmatrix}
\displaystyle \sum_{i=1}^{n} \frac{\partial r_i}{\partial \beta_1} r_i \\[3mm]
\displaystyle \sum_{i=1}^{n} \frac{\partial r_i}{\partial \beta_2} r_i
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{\partial E}{\partial \beta_1} \\[2mm]
\dfrac{\partial E}{\partial \beta_2}
\end{bmatrix}
```

この公式のありがたみは、**1 階微分の情報が $J$ に全部詰まっている**ことにある。モデルごとに $\nabla E$ を一から計算し直す必要はなく、ヤコビ行列さえ用意すれば、勾配は「転置して掛けて 2 倍」で機械的に得られる。文書 5 以降の反復法は、すべてこの形を通して勾配を計算する。

### 5.3 手計算の例のつづき

4.4 節のミニチュア例で $\nabla E = 2 J^\top \mathbf{r}$ を計算してみる。$\mathbf{r} = (1.000, \ 0.132, \ 0.065)^\top$ だった。

```math
J^\top \mathbf{r} =
\begin{bmatrix}
-1.000 & -0.368 & -0.135 \\
0 & -0.368 & -0.271
\end{bmatrix}
\begin{bmatrix}
1.000 \\ 0.132 \\ 0.065
\end{bmatrix}
```

第 1 成分: $(-1.000)(1.000) + (-0.368)(0.132) + (-0.135)(0.065) = -1.000 - 0.049 - 0.009 = -1.058$

第 2 成分: $(0)(1.000) + (-0.368)(0.132) + (-0.271)(0.065) = -0.049 - 0.018 = -0.066$

よって

```math
\nabla E = 2 J^\top \mathbf{r} \approx
\begin{bmatrix}
-2.115 \\ -0.132
\end{bmatrix}
```

意味を読み取っておく。どの残差も正 (データがモデルより上にある) なので、モデルを持ち上げるべき状況である。$\dfrac{\partial E}{\partial \beta_1} < 0$ は「$\beta_1$ を増やすと $E$ が減る」ことを意味し、$\beta_1$ を増やす = 曲線全体を持ち上げる、という直感と合致する。勾配は単なる記号ではなく「どちらへ動けば良くなるか」を指す矢印 (の逆向き) なのである。

### 5.4 整合性チェック: 線形の場合を当てはめる

新しい公式を得たら、既知の場合に帰着して検算するのが良い習慣である。線形モデルでは $\mathbf{r} = \mathbf{y} - \Phi \boldsymbol{\beta}$ であり、成分で書けば $r_i = y_i - \sum_k \Phi_{ik} \beta_k$ なので

```math
J_{ik} = \frac{\partial r_i}{\partial \beta_k} = - \Phi_{ik}
\quad \Longrightarrow \quad
J = -\Phi
```

ヤコビ行列が $\boldsymbol{\beta}$ によらない**定数行列**になるのが線形の特徴である。公式に代入すると

```math
\nabla E = 2 J^\top \mathbf{r} = 2 (-\Phi)^\top (\mathbf{y} - \Phi \boldsymbol{\beta}) = -2 \Phi^\top (\mathbf{y} - \Phi \boldsymbol{\beta})
= 2 \Phi^\top \Phi \boldsymbol{\beta} - 2 \Phi^\top \mathbf{y}
```

これは [線形最小二乗法](./1_least_squares_method.md) 第 4 節で導いた勾配と完全に一致し、$\nabla E = \mathbf{0}$ とおけば正規方程式が出てくる。つまり $\nabla E = 2 J^\top \mathbf{r}$ は線形の場合を特別な場合として含む、より一般的な公式である。

### 5.5 数値微分による検証

ヤコビ行列のときと同様、勾配の公式も数値微分で検証できる。今度は $E$ そのものを中心差分で微分し、$2 J^\top \mathbf{r}$ と比べる。サンプルコードは $\boldsymbol{\beta} = (1, -1)^\top$ (真値からわざと外した点) で両者を計算し、

```
== ∇E = 2 Jᵀ r の検証 (vs E の数値微分) ==
2Jᵀr     = [-9.808733, -2.269919]
数値微分 = [-9.808733, -2.269919]
```

小数第 6 位まで一致する。5.1〜5.2 節の導出 (連鎖律 → Σ → 行列の形) が数値的にも裏付けられた。

## 6. ヘッセ行列の構造 (予告)

勾配が求まったので、ついでに 2 階微分も見ておく。この節はガウス・ニュートン法への伏線であり、初読では「そういう構造がある」と眺めるだけでよい。

式 (5.1) をもう一度 $\beta_l$ で偏微分する。積の微分法則を使うと

```math
\frac{\partial^2 E}{\partial \beta_l \partial \beta_k}
= 2 \sum_{i=1}^{n} \frac{\partial}{\partial \beta_l} \Bigl( \frac{\partial r_i}{\partial \beta_k} \, r_i \Bigr)
= 2 \sum_{i=1}^{n} \Bigl(
\frac{\partial r_i}{\partial \beta_k} \frac{\partial r_i}{\partial \beta_l}
+ r_i \frac{\partial^2 r_i}{\partial \beta_l \partial \beta_k}
\Bigr)
```

第 1 項の Σ は $\sum_i J_{ik} J_{il} = (J^\top J)_{kl}$ とまとまるので、ヘッセ行列 (2 階偏微分を並べた行列) は

```math
\nabla^2 E = 2 \Bigl( J^\top J + \sum_{i=1}^{n} r_i \nabla^2 r_i \Bigr)
```

という 2 つの項に分かれる ($\nabla^2 r_i$ は $r_i$ のヘッセ行列)。ここに重要な観察が 2 つある。

1. **第 1 項 $J^\top J$ は 1 階微分だけで計算できる。** 2 階微分が要るのは第 2 項だけである
2. **第 2 項には残差 $r_i$ が掛かっている。** フィットがうまくいく問題 (残差が小さい) では、この項は小さい

つまり「良いフィットが見込める問題では、ヘッセ行列は $2 J^\top J$ でほぼ代用できる」。面倒な 2 階微分を計算せずに 2 次の情報 (もどき) が手に入るこの近似こそが、[7. ガウス・ニュートン法](./7_gauss_newton_method.md) の心臓部である。線形の場合は $J = -\Phi$ が定数なので $\nabla^2 r_i = O$、すなわち第 2 項が厳密に消えて $\nabla^2 E = 2 \Phi^\top \Phi$ となり、これも文書 1 の結果と一致する。

## 7. 対数変換による線形化 — 便利だが「別の問題」

### 7.1 対数を取ると直線フィットになる

指数減衰モデルには、昔からよく知られた近道がある。$y = \beta_1 e^{\beta_2 x}$ の両辺の対数を取ると

```math
\log y = \log \beta_1 + \beta_2 x
```

ここで $b_1 = \log \beta_1$, $b_2 = \beta_2$ とおけば、$(x, \log y)$ 平面では

```math
\log y = b_1 + b_2 x
```

という**直線**のフィッティングになる。これは [線形最小二乗法](./1_least_squares_method.md) 第 7 節でやった直線フィットそのものであり、公式一発で $b_1, b_2$ が求まる。最後に $\beta_1 = e^{b_1}$ と戻せば、指数モデルのパラメータが反復法なしで手に入る。サンプルコードの `log_linearized_fit` がこの計算である (なお $\log y_i$ を取るため、この方法は $y_i > 0$ のデータにしか使えない)。

一見これで問題が解決したように見える。しかし、そうは行かない。

### 7.2 最小化している量が違う

直線フィットが最小化しているのは

```math
E_{\log}(b_1, b_2) = \sum_{i=1}^{n} \bigl( \log y_i - (b_1 + b_2 x_i) \bigr)^2
```

すなわち **$\log y$ の残差**の二乗和である。一方、われわれが本当に最小化したいのは

```math
E(\beta_1, \beta_2) = \sum_{i=1}^{n} \bigl( y_i - \beta_1 e^{\beta_2 x_i} \bigr)^2
```

すなわち **$y$ そのものの残差**の二乗和である。$\log$ は値の大小を非線形に圧縮するので、$E_{\log}$ と $E$ は別の関数であり、**最小点も一般に異なる**。

違いの正体は「どのデータ点を重視するか」にある。対数の性質から、$\log y$ 方向のずれは $y$ の**相対誤差** (何 % ずれたか) にほぼ対応する。今回のデータでいうと、$x = 0$ 付近では $y \approx 2$ なのでノイズ $\pm 0.01$ は約 $0.5\%$ の揺れだが、$x = 2$ 付近では $y \approx 2 e^{-3} \approx 0.10$ なので同じ $\pm 0.01$ が約 $10\%$ の揺れになる。$E_{\log}$ の目にはこの末尾の揺れが 20 倍も大きく映る。つまり**対数変換は $y$ が小さいデータ点ほど重視する**フィットになっており、$y$ の絶対的なずれを均等に扱う元の $E$ とは判断基準が違うのである。

### 7.3 数値実験: log 線形化の解では ∇E ≠ 0

「別の問題の解である」ことは、勾配で定量的に確認できる。もし log 線形化の解 $\boldsymbol{\beta}_{\log}$ が元の $E$ の最小点なら、そこでは $\nabla E(\boldsymbol{\beta}_{\log}) = 2 J^\top \mathbf{r} = \mathbf{0}$ が成り立つはずである。サンプルコードで実際に計算すると

```
== 対数変換による線形化 (真値 β = [2.0, -1.5]) ==
log 線形化の解: [1.9980, -1.5030]
E(log線形化)  = 8.504817e-4
∇E(log線形化) = [-0.021149, -0.003475] (≠ 0 → E の停留点ではない)
```

$\boldsymbol{\beta}_{\log} = (1.9980, \ -1.5030)^\top$ は真値 $(2.0, \ -1.5)^\top$ にかなり近い。ノイズが小さい ($\pm 0.01$) ので当然ではある。しかし勾配は $\mathbf{0}$ ではない。勾配が $\mathbf{0}$ でないということは、**この点からまだ $E$ を減らせる方向が残っている**、つまり $\boldsymbol{\beta}_{\log}$ は $E$ の停留点 ($\nabla E = \mathbf{0}$ の点) ですらない、ということである。$E_{\log}$ の最小点ではあっても $E$ の最小点ではない — 「別の問題を解いている」とはこの意味である。

なお、ここで第 5 節の公式が早速役に立っていることに注意してほしい。「この点は最適か?」という問いに、$2 J^\top \mathbf{r}$ を計算するだけで答えられる。反復法を走らせる前から、勾配は解の**検算器**として機能するのである。

### 7.4 それでも初期値としては優秀

log 線形化を貶して終わるのは公平でない。3.3 節で見たとおり、反復法は初期値しだいで間違った谷に落ちる。log 線形化の解は (ノイズが極端でなければ) 正しい谷のすぐ近くに着地するので、**反復法の初期値**としては非常に優秀である。実務では

1. log 線形化で大まかな $\boldsymbol{\beta}_{\log}$ を求める (閉じた形、一瞬で計算できる)
2. それを初期値として反復法で $E$ を最小化し、仕上げる

という 2 段構えがよく使われる。「近似解法として割り切って使い、最終回答には使わない」が正しい付き合い方である。

## 8. どう解くか — 反復法の見取り図

材料は揃った。実際に $E$ を最小化する反復法は、どれも共通の骨格を持つ。

1. 初期値 $\boldsymbol{\beta}_0$ を決める (log 線形化、物理的な知識、複数の候補から試す、など)
2. 現在の $\boldsymbol{\beta}_k$ で更新量 $\boldsymbol{\delta}_k$ を計算し、$\boldsymbol{\beta}_{k+1} = \boldsymbol{\beta}_k + \boldsymbol{\delta}_k$ と更新する
3. 収束判定 (勾配 $\|\nabla E\|$ が十分小さい、更新量が十分小さい、など) を満たすまで 2 を繰り返す

手法の違いは「$\boldsymbol{\delta}_k$ をどう決めるか」に尽きる。以降の文書では、その代表的な選択肢を順に見ていく。

| 手法 | 更新量の決め方 | 使う微分 |
| --- | --- | --- |
| [5. 最急降下法](./5_steepest_descent.md) | $-\nabla E = -2 J^\top \mathbf{r}$ の方向に進む | 1 階 ($J$) |
| [6. ニュートン法](./6_newton_method.md) | 2 次近似の最小点へ飛ぶ | 1 階 + 2 階 |
| [7. ガウス・ニュートン法](./7_gauss_newton_method.md) | ヘッセ行列を $2 J^\top J$ で近似してニュートン法 | 1 階 ($J$) のみで 2 次相当 |
| [8. Levenberg-Marquardt 法](./8_levenberg_marquardt_method.md) | ガウス・ニュートン + ダンピングで安定化 | 1 階 ($J$) のみ |

どの行にも $J$ が現れることに注目してほしい。この文書で準備したヤコビ行列と $\nabla E = 2 J^\top \mathbf{r}$ は、以降のすべての手法で毎回使われる共通部品である。

## 9. まとめ

> 非線形最小二乗法は、モデルがパラメータについて非線形な場合の二乗誤差最小化である。$E$ は凸 2 次関数でなくなるため、$\nabla E = \mathbf{0}$ は閉じた形では解けず、谷 (極小) が複数あり得るので初期値から反復法で近づくしかない。その反復法の共通部品が、残差の偏微分を並べたヤコビ行列 $J$ と、そこから機械的に得られる勾配 $\nabla E = 2 J^\top \mathbf{r}$ である。

要点を再掲する。

- **問題の定義は線形と同じ** ($E = \|\mathbf{r}\|^2$ の最小化)。壊れたのは「$E$ が凸 2 次」という構造であり、それとともに閉形式解・大域的最小の保証・初期値不要という 3 つの恩恵を失った
- **ヤコビ行列** $J_{ik} = \partial r_i / \partial \beta_k$ は「各残差が各パラメータにどう反応するか」の表であり、指数減衰モデルなら全成分が手で計算できる
- **勾配の公式** $\nabla E = 2 J^\top \mathbf{r}$ は、成分ごとの連鎖律 $\partial E / \partial \beta_k = 2 \sum_i J_{ik} r_i$ を行列の形にまとめただけのものである。解析的な微分は数値微分と突き合わせて必ず検証する
- **ヘッセ行列は** $\nabla^2 E = 2 (J^\top J + \sum_i r_i \nabla^2 r_i)$ と分解でき、残差が小さければ $2 J^\top J$ で近似できる。これがガウス・ニュートン法の伏線となる
- **log 変換による線形化は「$\log y$ の残差」という別の量の最小化**であり、その解では $\nabla E \neq \mathbf{0}$ (数値実験で確認済み)。最終回答には使えないが、反復法の初期値としては優秀

実際に谷を降りていく方法は、[5. 最急降下法](./5_steepest_descent.md) から始まる。
