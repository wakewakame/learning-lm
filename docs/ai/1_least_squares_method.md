# 線形最小二乗法 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

著者による学習メモ: [`docs/1_least_squares_method.md`](../1_least_squares_method.md)

## 1. 問題設定

$n$ 個の観測データ $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$ が与えられたとき、これをよく説明する関数

$$
\hat{y}(x) = \sum_{k=1}^{m} \beta_k \, \phi_k(x)
$$

を求めたい。ここで $\phi_1, \dots, \phi_m$ は**基底関数**と呼ばれるあらかじめ選んだ関数であり、求めたい未知数は係数 $\beta_1, \dots, \beta_m$ である。例えば

- $\phi_1(x) = 1, \ \phi_2(x) = x$ とすれば直線 $\hat{y} = \beta_1 + \beta_2 x$
- $\phi_k(x) = x^{k-1}$ とすれば多項式
- $\phi_1(x) = \sin x, \ \phi_2(x) = \cos x$ のような三角関数

などが表現できる。重要なのは、モデルが**未知数 $\beta_k$ について線形**であることだけで、基底関数 $\phi_k$ 自体はいくらでも非線形でよい。これが「線形」最小二乗法の「線形」の意味であり、直線フィッティングに限る手法ではない。

観測をまとめてベクトルと行列で書く。

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

$\Phi$ は**計画行列** (design matrix) と呼ばれる。理想的には $\mathbf{y} = \Phi \boldsymbol{\beta}$ が成り立ってほしいが、通常はデータ数がパラメータ数より多く ($n > m$)、観測には誤差が乗るため、この連立方程式は厳密には解けない (**過剰決定系**)。そこで「厳密に解く」ことを諦め、「できるだけ近い」解を探す、というのが最小二乗法の出発点である。

## 2. 目的関数 — なぜ「二乗和」なのか

モデルとデータのずれを**残差** $\mathbf{r} = \mathbf{y} - \Phi \boldsymbol{\beta}$ で測り、その二乗和

$$
E(\boldsymbol{\beta})
= \| \mathbf{y} - \Phi \boldsymbol{\beta} \|^2
= \sum_{i=1}^{n} \Bigl( y_i - \sum_{k=1}^{m} \beta_k \phi_k(x_i) \Bigr)^2
$$

を最小化する。ずれの測り方は他にもある (絶対値の和、最大値など) が、二乗和には次の利点がある。

1. **解析的に解ける。** $E$ は $\boldsymbol{\beta}$ の 2 次関数なので、微分して 0 とおくだけで閉じた形の解が得られる。絶対値の和は微分できない点を持ち、反復計算が必要になる。
2. **幾何的に自然。** 二乗和の平方根はユークリッド距離そのものであり、最小化問題が「部分空間への直交射影」というきれいな幾何の問題になる (次節)。
3. **統計的な正当化がある。** 観測誤差が独立で等分散なら、最小二乗解は線形不偏推定量の中で分散最小になる (ガウス・マルコフの定理)。さらに誤差が正規分布に従うなら、最小二乗解は最尤推定と一致する。

一方で弱点もある。二乗はずれを強調するため、**外れ値に引きずられやすい**。外れ値が多いデータではロバスト回帰 (第 7 節) を検討すべきである。

## 3. 導出 (幾何) — 直交射影としての最小二乗法

最小二乗法の本質は幾何にある。$\boldsymbol{\beta}$ を動かすと $\Phi \boldsymbol{\beta}$ は $\mathbb{R}^n$ の中で $\Phi$ の列ベクトルが張る部分空間

$$
S = \{ \Phi \boldsymbol{\beta} \mid \boldsymbol{\beta} \in \mathbb{R}^m \}
$$

の全体を動く。つまり最小二乗法とは「$S$ の中で $\mathbf{y}$ に最も近い点 $\hat{\mathbf{y}}$ を探す」問題である。

最も近い点は $\mathbf{y}$ から $S$ に下ろした垂線の足、すなわち**直交射影**である。これは三平方の定理から直ちに分かる。$\hat{\mathbf{y}} \in S$ を $\mathbf{y} - \hat{\mathbf{y}}$ が $S$ に直交するように取ると、任意の $\mathbf{v} \in S$ に対して $\hat{\mathbf{y}} - \mathbf{v} \in S$ なので

$$
\| \mathbf{y} - \mathbf{v} \|^2
= \| (\mathbf{y} - \hat{\mathbf{y}}) + (\hat{\mathbf{y}} - \mathbf{v}) \|^2
= \| \mathbf{y} - \hat{\mathbf{y}} \|^2 + \| \hat{\mathbf{y}} - \mathbf{v} \|^2
\ \geq\ \| \mathbf{y} - \hat{\mathbf{y}} \|^2
$$

となり (交差項は直交性より消える)、等号は $\mathbf{v} = \hat{\mathbf{y}}$ のときに限る。

「残差が $S$ に直交する」という条件は、$S$ を張る各列 $\Phi$ との内積が 0、すなわち

$$
\Phi^\top ( \mathbf{y} - \Phi \boldsymbol{\beta} ) = \mathbf{0}
\quad \Longleftrightarrow \quad
\Phi^\top \Phi \, \boldsymbol{\beta} = \Phi^\top \mathbf{y}
$$

と書ける。これが**正規方程式** (normal equations) である。名前の由来もここにあり、残差がモデル空間に**垂直** (normal) であることを表す方程式だからである。

## 4. 導出 (微分) — 凸 2 次関数の最小化

同じ結論は微分からも得られる。$E$ を展開すると

$$
E(\boldsymbol{\beta})
= \mathbf{y}^\top \mathbf{y}
- 2 \boldsymbol{\beta}^\top \Phi^\top \mathbf{y}
+ \boldsymbol{\beta}^\top \Phi^\top \Phi \, \boldsymbol{\beta}
$$

であり (途中、スカラー $\mathbf{y}^\top \Phi \boldsymbol{\beta}$ とその転置 $\boldsymbol{\beta}^\top \Phi^\top \mathbf{y}$ が等しいことを使った)、勾配は

$$
\nabla E(\boldsymbol{\beta}) = 2 \Phi^\top \Phi \, \boldsymbol{\beta} - 2 \Phi^\top \mathbf{y}
$$

となる。これを $\mathbf{0}$ とおけば正規方程式が得られる。

ここで「勾配が $\mathbf{0}$ の点は本当に最小か」を確認しておく必要がある。$E$ のヘッセ行列は $2 \Phi^\top \Phi$ であり、任意の $\mathbf{u} \in \mathbb{R}^m$ に対して

$$
\mathbf{u}^\top ( \Phi^\top \Phi ) \mathbf{u} = \| \Phi \mathbf{u} \|^2 \geq 0
$$

なので半正定値である。したがって $E$ は凸関数であり、勾配が $\mathbf{0}$ となる点は必ず大域的最小点である。鞍点や極大点の心配はない。

## 5. 解の存在と一意性

- **最小点は常に存在する。** 幾何の導出から明らかで、部分空間への射影 $\hat{\mathbf{y}}$ は必ず存在する。
- **解が一意なのは $\Phi$ の列が線形独立なとき** ($\operatorname{rank} \Phi = m$)。このとき $\Phi^\top \Phi$ は正定値で可逆となり、解は形式的に

$$
\boldsymbol{\beta}^* = ( \Phi^\top \Phi )^{-1} \Phi^\top \mathbf{y}
$$

  と書ける (ただし数値計算でこの式を直接使うのは推奨されない。第 6 節)。

- **列が線形従属な場合** (基底関数が冗長、データが縮退しているなど)、射影 $\hat{\mathbf{y}}$ 自体は一意だが、それを実現する $\boldsymbol{\beta}$ が無数に存在する。この場合はノルム最小の解を選ぶのが慣例で、擬似逆行列 $\Phi^{+}$ を用いて $\boldsymbol{\beta}^* = \Phi^{+} \mathbf{y}$ と表される (SVD で計算できる)。

## 6. 数値計算の実際

正規方程式は理論上の解を与えるが、**どう計算するか**で数値的な品質が大きく変わる。

| 方法 | 手順 | 特徴 |
| --- | --- | --- |
| 正規方程式 + コレスキー分解 | $\Phi^\top \Phi$ を作って解く | 最速だが、条件数が $\kappa(\Phi^\top \Phi) = \kappa(\Phi)^2$ と二乗に悪化し、桁落ちしやすい |
| QR 分解 | $\Phi = QR$ として $R \boldsymbol{\beta} = Q^\top \mathbf{y}$ を後退代入で解く | $\Phi^\top \Phi$ を作らないため条件数が悪化せず、数値的に安定。**通常はこれが標準** |
| SVD | $\Phi = U \Sigma V^\top$ から擬似逆行列を構成 | 最も頑健。ランク落ち・ランク落ち寸前のデータにも対応できるが、計算コストは最大 |

要点は「$\Phi^\top \Phi$ を明示的に作った時点で情報が失われる」ことである。LAPACK や NumPy の `lstsq` などのライブラリ関数は内部で QR や SVD を使っており、実務では自前で正規方程式を組むのではなくこれらを呼ぶのが正しい。

## 7. 具体例 — 直線フィッティング

最も基本的な例として $\hat{y} = \beta_1 + \beta_2 x$ ($\phi_1 = 1, \ \phi_2 = x$) を正規方程式から解いてみる。

$$
\Phi^\top \Phi =
\begin{bmatrix}
n & \sum_i x_i \\
\sum_i x_i & \sum_i x_i^2
\end{bmatrix},
\qquad
\Phi^\top \mathbf{y} =
\begin{bmatrix}
\sum_i y_i \\
\sum_i x_i y_i
\end{bmatrix}
$$

なので、正規方程式は 2 元連立 1 次方程式になる。$\bar{x} = \frac{1}{n} \sum_i x_i, \ \bar{y} = \frac{1}{n} \sum_i y_i$ とおいて整理すると

$$
\beta_2 = \frac{\sum_i ( x_i - \bar{x} )( y_i - \bar{y} )}{\sum_i ( x_i - \bar{x} )^2},
\qquad
\beta_1 = \bar{y} - \beta_2 \bar{x}
$$

が得られる。この形には明確な解釈がある。

- 傾き $\beta_2$ は **$x$ と $y$ の標本共分散を $x$ の標本分散で割ったもの**。
- $\beta_1$ の式を書き直すと $\bar{y} = \beta_1 + \beta_2 \bar{x}$、つまり**最小二乗直線は必ずデータの重心 $(\bar{x}, \bar{y})$ を通る**。

また、分母が $x$ の分散であることから、全ての $x_i$ が等しい (分散 0) と傾きが定まらないことも直ちに分かる。これは第 5 節の「$\Phi$ の列が線形従属になるケース」($x_i$ の列が定数列の定数倍になる) の最も単純な例である。

## 8. よくある誤解と限界

- **「線形」は直線のことではない。** 線形なのはパラメータについてであり、多項式や三角関数のフィッティングも線形最小二乗法である。逆に $\hat{y} = e^{\beta x}$ のようにパラメータに関して非線形なモデルは対象外で、非線形最小二乗法 (ガウス・ニュートン法、レーベンバーグ・マーカート法など) が必要になる。
- **誤差は $y$ 方向のみを仮定している。** 残差は $y$ 方向のずれで測っており、$x$ にも誤差がある場合は本来、全最小二乗法 (total least squares) の領分である。
- **外れ値に弱い。** 二乗はずれを強調するため、少数の外れ値が解を大きく歪める。Huber 損失によるロバスト回帰や RANSAC が対策になる。
- **基底を増やすほど良いわけではない。** 基底関数を増やせば残差は必ず減るが、データの誤差まで拾う過学習に陥る。リッジ回帰 ($\Phi^\top \Phi + \lambda I$ を使う正則化) は過学習と条件数悪化の両方への処方箋になる。

## 9. まとめ

線形最小二乗法は「解けない連立方程式 $\Phi \boldsymbol{\beta} \approx \mathbf{y}$ を、残差の二乗和最小の意味で最良に解く」手法である。その核心は 1 つの幾何的事実に集約される。

> 最小二乗解とは、$\mathbf{y}$ をモデルの張る部分空間へ直交射影したものであり、正規方程式 $\Phi^\top \Phi \boldsymbol{\beta} = \Phi^\top \mathbf{y}$ はその直交条件の言い換えである。

理論はこの一行で閉じるが、数値計算では $\Phi^\top \Phi$ を作らず QR 分解や SVD で解くこと、そしてモデルの妥当性 (外れ値、過学習、誤差の方向) を常に疑うことが実務上の要点である。
