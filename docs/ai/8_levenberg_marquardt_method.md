# Levenberg-Marquardt 法 (お手本解説)

> [!WARNING]
> この文書は AI (Claude) が執筆したものであり、著者はまだ内容をレビューしていない。
> 誤りを含む可能性を前提に、検証しながら読むこと。
>
> - 執筆: Claude Fable 5 (2026-07-05)
> - 著者レビュー: 未

## 1. 位置づけ

[ガウス・ニュートン法](./7_gauss_newton_method.md) (以下 GN 法) には 2 つの故障モードがあった。

1. **飛びすぎ。** 線形化 $\mathbf{r}(\boldsymbol{\beta} + \boldsymbol{\delta}) \approx \mathbf{r} + J \boldsymbol{\delta}$ が有効なのは $\boldsymbol{\delta}$ が小さい範囲だけなのに、更新量の大きさに歯止めがなく、近似の有効範囲外へ飛び出して目的関数 $E$ がかえって増えることがある。実験では初期値 $[5.0, 5.0]$ から発散した。
2. **解けない・暴れる。** $J^\top J$ が特異 (逆行列を持たない) だと正規方程式が解けず、特異に近い (条件数が大きい) と $\boldsymbol{\delta}$ が巨大になって暴れる。実験では初期値 $[1.0, 3.0]$ で $J^\top J$ が特異化して破綻した。

Levenberg-Marquardt 法 (以下 LM 法) は、この両方を**ダンピング** (damping: 減衰) というたった 1 つの仕掛けで同時に解決する。仕掛けの中身は「正規方程式の係数行列に $\lambda I$ を足す」だけである。それだけの変更で、

- 方程式が**常に解ける**ようになり (故障モード 2 の解決)、
- 得られる $\boldsymbol{\delta}$ が**常に下り方向**になり、さらに $\lambda$ を大きくすれば**歩幅がいくらでも小さくなる** (故障モード 1 の解決)

ことが保証される。LM 法は非線形最小二乗法の実務における事実上の標準アルゴリズムであり、このリポジトリのロードマップの終着点である。ここまでの 7 つの文書で積んだ部品が全て組み合わさる様子を見ていく。

記号はこれまでの文書と同じである。データ $(x_i, y_i)$ ($i = 1, \dots, n$) にパラメータ $\boldsymbol{\beta} \in \mathbb{R}^m$ のモデル $f(x; \boldsymbol{\beta})$ を当てはめる。残差ベクトル $\mathbf{r}(\boldsymbol{\beta})$ の第 $i$ 成分は $r_i = y_i - f(x_i; \boldsymbol{\beta})$、目的関数は

$$
E(\boldsymbol{\beta}) = \| \mathbf{r}(\boldsymbol{\beta}) \|^2 = \sum_{i=1}^{n} r_i(\boldsymbol{\beta})^2
$$

であり、ヤコビ行列 $J$ ($n \times m$、第 $(i, k)$ 成分は $\partial r_i / \partial \beta_k$) を使うと勾配は $\nabla E = 2 J^\top \mathbf{r}$ と書けるのだった ([4. 非線形最小二乗法](./4_nonlinear_least_squares.md))。ベクトル・内積・勾配などの基礎は [0. 数学の準備](./0_math_preliminaries.md) を参照。

## 2. 更新式と 2 つの保証

GN 法の正規方程式 $J^\top J \boldsymbol{\delta} = - J^\top \mathbf{r}$ の係数行列に、**ダンピング項** $\lambda I$ ($\lambda > 0$、$I$ は $m \times m$ の単位行列) を足す。

$$
( J^\top J + \lambda I ) \, \boldsymbol{\delta} = - J^\top \mathbf{r}
$$

これが LM 法の更新式である。$\lambda$ を**ダンピングパラメータ**と呼ぶ。この一手で何が変わるのかを、2 つの保証として順に確かめる。

### 2.1 保証 1: 常に解ける

まず「$J^\top J$ が特異でも、$\lambda I$ を足せば必ず逆行列を持つ」ことを、手計算できる $2 \times 2$ の例で見る。

**具体例。** ヤコビ行列が

$$
J = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}
$$

だったとする。2 つの列 $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ と $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ は同じベクトルであり、線形独立でない (このとき $J$ は**ランク落ち**しているという)。正規方程式の係数行列を計算すると

$$
J^\top J
= \begin{bmatrix} 1 & 0 \\ 1 & 0 \end{bmatrix}
\begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}
= \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
$$

となる。$2 \times 2$ 行列が逆行列を持つ条件は行列式が 0 でないことだったが、

$$
\det (J^\top J) = 1 \cdot 1 - 1 \cdot 1 = 0
$$

なので逆行列を持たない。つまり GN 法の正規方程式はこの時点で解けない (これが [7 番の実験](./7_gauss_newton_method.md)で初期値 $[1.0, 3.0]$ のときに起きたことである)。

ここに $\lambda I$ を足してみる。

$$
J^\top J + \lambda I
= \begin{bmatrix} 1 + \lambda & 1 \\ 1 & 1 + \lambda \end{bmatrix},
\qquad
\det ( J^\top J + \lambda I )
= (1 + \lambda)^2 - 1
= \lambda^2 + 2 \lambda
= \lambda ( \lambda + 2 )
$$

$\lambda > 0$ なら行列式は必ず正になり、逆行列が持てる。たとえば $\lambda = 0.1$ なら $\det = 0.1 \times 2.1 = 0.21 \neq 0$ である。「対角に少し足すだけで特異性が消える」感覚を掴んでほしい。

**一般の場合。** この現象は $2 \times 2$ に限らない。鍵になるのは**正定値** (positive definite) という性質である。

> 対称行列 $A$ が**正定値**であるとは、すべての $\boldsymbol{\delta} \neq \mathbf{0}$ に対して $\boldsymbol{\delta}^\top A \boldsymbol{\delta} > 0$ が成り立つことをいう。等号を許して $\boldsymbol{\delta}^\top A \boldsymbol{\delta} \geq 0$ までしか言えないとき、**半正定値**という。

$J^\top J + \lambda I$ が正定値であることは直接計算で分かる。任意の $\boldsymbol{\delta} \neq \mathbf{0}$ に対して

$$
\boldsymbol{\delta}^\top ( J^\top J + \lambda I ) \, \boldsymbol{\delta}
= \boldsymbol{\delta}^\top J^\top J \boldsymbol{\delta} + \lambda \, \boldsymbol{\delta}^\top \boldsymbol{\delta}
= \| J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2
$$

である ($\boldsymbol{\delta}^\top J^\top J \boldsymbol{\delta} = (J \boldsymbol{\delta})^\top (J \boldsymbol{\delta}) = \| J \boldsymbol{\delta} \|^2$ に注意)。第 1 項 $\| J \boldsymbol{\delta} \|^2$ はノルムの 2 乗なので常に 0 以上 ($J$ がランク落ちしていると、ちょうど 0 になる $\boldsymbol{\delta}$ が存在する。これが $J^\top J$ が「半」正定値止まりである理由)。しかし第 2 項 $\lambda \| \boldsymbol{\delta} \|^2$ は $\boldsymbol{\delta} \neq \mathbf{0}$ である限り厳密に正なので、合計は必ず正になる。

$$
\boldsymbol{\delta}^\top ( J^\top J + \lambda I ) \, \boldsymbol{\delta}
\geq \lambda \| \boldsymbol{\delta} \|^2 > 0
$$

そして**正定値なら必ず逆行列を持つ**。理由は簡単で、もし逆行列を持たないなら $( J^\top J + \lambda I ) \boldsymbol{\delta} = \mathbf{0}$ となる $\boldsymbol{\delta} \neq \mathbf{0}$ が存在するが、そのとき $\boldsymbol{\delta}^\top ( J^\top J + \lambda I ) \boldsymbol{\delta} = \boldsymbol{\delta}^\top \mathbf{0} = 0$ となって正定値性に矛盾するからである。

**固有値の言葉で。** 同じことは固有値でも言える。ベクトル $\mathbf{v} \neq \mathbf{0}$ と数 $\mu$ が $A \mathbf{v} = \mu \mathbf{v}$ を満たすとき、$\mu$ を $A$ の**固有値**、$\mathbf{v}$ を**固有ベクトル**と呼ぶ ([3. 特異値分解](./3_singular_value_decomposition.md)では、$J^\top J$ の固有値が $J$ の特異値の 2 乗 $\sigma_j^2$ であることに触れた)。$J^\top J$ の固有値 $\mu$ は、固有ベクトル $\mathbf{v}$ に対して $\mu \| \mathbf{v} \|^2 = \mathbf{v}^\top J^\top J \mathbf{v} = \| J \mathbf{v} \|^2 \geq 0$ より、すべて 0 以上である。ここで $J^\top J \mathbf{v} = \mu \mathbf{v}$ の両辺に $\lambda \mathbf{v}$ を足すと

$$
( J^\top J + \lambda I ) \, \mathbf{v} = \mu \mathbf{v} + \lambda \mathbf{v} = ( \mu + \lambda ) \, \mathbf{v}
$$

つまり **$\lambda I$ を足すと、すべての固有値がちょうど $\lambda$ だけ持ち上がる**。$J^\top J$ の固有値は 0 以上だったから、$J^\top J + \lambda I$ の固有値はすべて $\lambda$ 以上、特にすべて正である (対称行列では「固有値がすべて正」と「正定値」は同値)。先の $2 \times 2$ の例で確かめると、$J^\top J = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$ の固有値は $\det \begin{bmatrix} 1 - \mu & 1 \\ 1 & 1 - \mu \end{bmatrix} = (1 - \mu)^2 - 1 = 0$ より $\mu = 0, 2$ であり、$\lambda I$ を足した後の固有値は $\lambda, \lambda + 2$。確かに全部 $\lambda$ 分だけ持ち上がり、0 だった固有値が $\lambda$ になって特異性が消えている。

### 2.2 保証 2: 常に下り方向

次に、解いて得た $\boldsymbol{\delta}$ が「その向きに (少しだけ) 進めば $E$ が減る方向」、すなわち**下り方向**であることを示す。

**下り方向とは何か。** 現在地 $\boldsymbol{\beta}$ から向き $\boldsymbol{\delta}$ に小さく $t$ だけ進んだときの $E$ の変化は、1 次近似で

$$
E(\boldsymbol{\beta} + t \boldsymbol{\delta}) \approx E(\boldsymbol{\beta}) + t \, \boldsymbol{\delta}^\top \nabla E
$$

と書ける (多変数のテイラー展開の 1 次まで。[0. 数学の準備](./0_math_preliminaries.md)の勾配の節を参照)。したがって $t > 0$ が十分小さいとき、

$$
\boldsymbol{\delta}^\top \nabla E < 0 \iff E \text{ が減る}
$$

である。この条件を満たす $\boldsymbol{\delta}$ を下り方向と呼ぶ。内積の幾何的意味を思い出すと、$\boldsymbol{\delta}$ と $\nabla E$ のなす角を $\theta$ として

$$
\boldsymbol{\delta}^\top \nabla E = \| \boldsymbol{\delta} \| \, \| \nabla E \| \cos \theta
$$

なので、$\boldsymbol{\delta}^\top \nabla E < 0$ は $\cos \theta < 0$、つまり **$\theta$ が鈍角** ($90^\circ < \theta \leq 180^\circ$) であることと同値である。勾配 $\nabla E$ は「最も急な上り」を指すベクトルだったから ([5. 最急降下法](./5_steepest_descent.md))、それと鈍角をなす向きは全部「下り側」に入っている、という素直な描像である。真下り ($\theta = 180^\circ$、最急降下方向) である必要はなく、鈍角でさえあれば下れる。

**LM 法の $\boldsymbol{\delta}$ は必ず鈍角側にある。** 更新式 $( J^\top J + \lambda I ) \boldsymbol{\delta} = - J^\top \mathbf{r}$ より $J^\top \mathbf{r} = - ( J^\top J + \lambda I ) \boldsymbol{\delta}$ である。これを $\nabla E = 2 J^\top \mathbf{r}$ に代入すると

$$
\boldsymbol{\delta}^\top \nabla E
= 2 \, \boldsymbol{\delta}^\top J^\top \mathbf{r}
= - 2 \, \boldsymbol{\delta}^\top ( J^\top J + \lambda I ) \, \boldsymbol{\delta}
$$

右辺のカッコの中身はまさに 2.1 で正定値と示した行列なので、$\boldsymbol{\delta} \neq \mathbf{0}$ なら

$$
\boldsymbol{\delta}^\top \nabla E = - 2 \bigl( \| J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2 \bigr) < 0
$$

となり、LM 法の一歩は**例外なく下り方向**である。GN 法でも $J$ がフルランクなら同じ性質が成り立ったが ([7 番](./7_gauss_newton_method.md)の導出 2)、LM 法では**ランク落ちしていても**成り立つ。ここが違いである。

ただし「下り方向」は「その向きに微小に進めば減る」ことしか保証しない。歩幅 $\| \boldsymbol{\delta} \|$ が大きすぎれば $E$ が増えることは依然あり得る。それを防ぐのが次節の $\lambda$ の役割である。

**Marquardt の変種。** なお Marquardt が提案した変種では、$\lambda I$ の代わりに $\lambda \mathrm{diag}(J^\top J)$ ($J^\top J$ の対角成分だけ取り出した対角行列) を足す。パラメータごとのスケールの違い (単位の違いなど) をダンピングが自動で吸収するため、スケーリングの悪い問題に強い。本文書では簡単のため $\lambda I$ で説明を続ける。

## 3. 2 つの極限 — ガウス・ニュートンと最急降下の補間

$\lambda$ を動かすと LM 法がどう振る舞うかを見る。結論から言うと、LM 法は既に学んだ 2 つの手法の**間を連続的に補間**している。

### 3.1 $\lambda \to 0$ の極限: ガウス・ニュートン法

更新式に $\lambda = 0$ を代入するだけである。

$$
( J^\top J + 0 \cdot I ) \, \boldsymbol{\delta} = - J^\top \mathbf{r}
\quad \Longrightarrow \quad
J^\top J \, \boldsymbol{\delta} = - J^\top \mathbf{r}
$$

これは GN 法の正規方程式そのものである。つまり $\lambda$ が小さいとき、LM 法は GN 法とほぼ同じ一歩を打つ。GN 法の長所 (残差が小さい問題での二次収束に近い速さ) がそのまま手に入る。

### 3.2 $\lambda$ が大きい極限: 最急降下法の小さな一歩

$\lambda$ が大きいときはどうなるか。更新式の両辺を $\lambda$ で割ってみる。

$$
( J^\top J + \lambda I ) \, \boldsymbol{\delta} = - J^\top \mathbf{r}
\quad \Longrightarrow \quad
\Bigl( \frac{1}{\lambda} J^\top J + I \Bigr) \boldsymbol{\delta} = - \frac{1}{\lambda} J^\top \mathbf{r}
$$

$\lambda$ を大きくしていくと、左辺の $\frac{1}{\lambda} J^\top J$ の各成分は 0 に近づく ($J^\top J$ は現在地で決まった定数行列であり、それを大きな数で割るため)。したがって左辺のカッコは単位行列 $I$ に近づき、

$$
\boldsymbol{\delta} \approx - \frac{1}{\lambda} J^\top \mathbf{r}
$$

を得る。ここで $\nabla E = 2 J^\top \mathbf{r}$、すなわち $J^\top \mathbf{r} = \frac{1}{2} \nabla E$ を代入すると

$$
\boldsymbol{\delta} \approx - \frac{1}{\lambda} \cdot \frac{1}{2} \nabla E = - \frac{1}{2 \lambda} \nabla E
$$

これは[最急降下法](./5_steepest_descent.md)の更新式 $\boldsymbol{\delta} = - \alpha \nabla E$ において、ステップ幅を $\alpha = \frac{1}{2 \lambda}$ と選んだものに他ならない。しかも $\lambda$ を大きくするほど $\alpha$ は小さくなる。つまり

> $\lambda$ が大きいとき、LM 法は「最急降下方向への、$\lambda$ に反比例した小さな一歩」になる。

最急降下法の一歩は、十分小さくすれば必ず $E$ を減らせるのだった (下り方向 + 十分小さい歩幅)。だから **$\lambda$ をどんどん大きくすれば、いつかは必ず $E$ が減る一歩が打てる**。これが故障モード 1 (飛びすぎ) への保険になる。

### 3.3 $\lambda$ は「GN 法をどれだけ信用するか」のつまみ

2 つの極限をまとめる。

| $\lambda$ | 挙動 | 得られるもの |
| --- | --- | --- |
| 小さい ($\to 0$) | GN 法に一致 | **速さ** (終盤の二次収束的な収束) |
| 大きい | 最急降下法の小さな一歩 | **頑健さ** (必ず下れる、暴れない) |

つまり $\lambda$ は「線形化 (GN 法のモデル) をどれだけ信用するか」を決める 1 つのつまみである。

> うまく進めているときは $\lambda$ を小さくして GN 法の**速さ**を使い、モデルが信用できないときは $\lambda$ を大きくして最急降下法の**頑健さ**に退避する。

「終盤に速いが遠方で危うい手法」と「遅いが必ず下る手法」の切り替えを、$\lambda$ という 1 つの実数の増減だけで実現しているのが LM 法の設計の妙である。[6. ニュートン法](./6_newton_method.md)で見た「$H$ が正定値でないとき $H + \tau I$ に置き換える」修正ニュートン法の発想と全く同じ形であることにも注意してほしい。LM 法のダンピングは、あの $\tau I$ を最小二乗の文脈で体系化したものと見ることができる。

## 4. 信頼領域としての解釈

ダンピングにはもう 1 つ、より現代的で幾何的な解釈がある。

GN 法の一歩は「線形化モデル $L(\boldsymbol{\delta}) = \| \mathbf{r} + J \boldsymbol{\delta} \|^2$ を最小化する $\boldsymbol{\delta}$」だった。故障の原因は、このモデルが**現在地の近くでしか正確でない**のに、最小化が $\boldsymbol{\delta}$ を遠くまで飛ばしてしまうことにある。ならば発想を変えて、最初から

> 「モデルを信用できる範囲」に半径 $\Delta$ の球を設定し、**その球の中でだけ**モデルを最小化する

ことにすればよい。式で書くと

$$
\min_{\boldsymbol{\delta}} \| \mathbf{r} + J \boldsymbol{\delta} \|^2
\qquad \text{s.t.} \quad \| \boldsymbol{\delta} \| \leq \Delta
$$

という制約付き最小化になる (s.t. は subject to、「〜という条件のもとで」の意)。この球を**信頼領域** (trust region)、$\Delta$ を**信頼半径**と呼ぶ。

実はこの制約付き問題の解は、ある $\lambda \geq 0$ に対する LM 法の更新式

$$
( J^\top J + \lambda I ) \, \boldsymbol{\delta} = - J^\top \mathbf{r}
$$

の解と一致することが知られている (厳密な証明はラグランジュ乗数法という道具を使うためここでは踏み込まないが、対応関係だけ述べる)。対応は次の通りである。

- 制約が**効いていない**とき (GN 法の一歩がもともと球の中に収まるとき) は $\lambda = 0$ でよく、LM 法は GN 法に一致する。
- 制約が**きつい** (球が小さく、モデルの最小点が球のずっと外にある) ほど、対応する $\lambda$ は大きくなる。

つまり **$\lambda$ は信頼半径 $\Delta$ の「逆向きのつまみ」**である。

> LM 法の $\boldsymbol{\delta}$ は「線形化モデルを信頼できる半径 $\Delta$ の球の中で、モデルを最小化する一歩」であり、$\lambda$ を増やすことは信頼半径を狭めることに、$\lambda$ を減らすことは信頼半径を広げることに対応する。

この見方をすると、次節の $\lambda$ の適応更新が自然に読める。「モデルの予測がよく当たった → モデルはもっと広い範囲で信用できそうだ → 半径を広げる ($\lambda$ を減らす)」「予測が外れた → 信用できる範囲を狭めるべきだ → 半径を狭める ($\lambda$ を増やす)」という運用である。

なお、直接計算でも直感の裏付けが取れる。第 6 節で確認するように、LM 法の一歩は

$$
\min_{\boldsymbol{\delta}} \Bigl( \| \mathbf{r} + J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2 \Bigr)
$$

の解でもある。第 2 項は「$\boldsymbol{\delta}$ が大きいこと」への罰金であり、$\lambda$ はその罰金の重さである。罰金が重いほど小さな一歩が選ばれる、というわけである。この形は線形最小二乗のリッジ正則化 (チホノフ正則化) と同型であり、[SVD](./3_singular_value_decomposition.md) の言葉では「小さい特異値 $\sigma_j$ による誤差増幅 $1 / \sigma_j$ を $\sigma_j / ( \sigma_j^2 + \lambda )$ に抑える」操作になっている。悪条件への耐性はここから来る。

## 5. ゲイン比 $\rho$ と $\lambda$ の適応更新

$\lambda$ は固定せず、反復のたびに「線形化モデルの予測がどれだけ当たったか」を測って調整する。この節がアルゴリズムの心臓部である。

### 5.1 ゲイン比の定義

一歩 $\boldsymbol{\delta}$ を試したとき、比べたい量が 2 つある。

- **実際の減少量**: $E(\boldsymbol{\beta}) - E(\boldsymbol{\beta} + \boldsymbol{\delta})$。本物の目的関数がどれだけ減ったか。
- **モデルが予測した減少量**: $L(\mathbf{0}) - L(\boldsymbol{\delta})$。線形化モデル $L(\boldsymbol{\delta}) = \| \mathbf{r} + J \boldsymbol{\delta} \|^2$ が「これだけ減るはず」と見積もった量 ($L(\mathbf{0}) = \| \mathbf{r} \|^2 = E(\boldsymbol{\beta})$ であることに注意。動かなければモデルと本物は一致する)。

この 2 つの比を**ゲイン比** (gain ratio) と呼ぶ。

$$
\rho = \frac{E(\boldsymbol{\beta}) - E(\boldsymbol{\beta} + \boldsymbol{\delta})}{L(\mathbf{0}) - L(\boldsymbol{\delta})}
= \frac{\text{実際の減少量}}{\text{モデルが予測した減少量}}
$$

$\rho$ の読み方は次の通り。

- $\rho \approx 1$: 予測どおり減った。線形化はこの歩幅でよく当たっている → もっと信用してよい ($\lambda$ を減らす)
- $0 < \rho \ll 1$: 減ったには減ったが、予測よりずっと少ない。線形化は当てにならなくなってきている
- $\rho \leq 0$: $E$ が**増えた**。モデルの予測は完全に外れた → この一歩は**棄却**し、$\lambda$ を増やして (信頼半径を狭めて) 小さく安全な一歩でやり直す

### 5.2 分母 $\boldsymbol{\delta}^\top ( \lambda \boldsymbol{\delta} - \mathbf{g} )$ の導出と、それが常に正であること

ゲイン比の分母は、以下のように閉じた式で計算できる。記号を軽くするため $\mathbf{g} = J^\top \mathbf{r}$ とおく ($\nabla E = 2 \mathbf{g}$)。

まず $L(\boldsymbol{\delta})$ を展開する。$\| \mathbf{a} \|^2 = \mathbf{a}^\top \mathbf{a}$ を使って

$$
L(\boldsymbol{\delta})
= ( \mathbf{r} + J \boldsymbol{\delta} )^\top ( \mathbf{r} + J \boldsymbol{\delta} )
= \mathbf{r}^\top \mathbf{r} + \mathbf{r}^\top J \boldsymbol{\delta} + \boldsymbol{\delta}^\top J^\top \mathbf{r} + \boldsymbol{\delta}^\top J^\top J \boldsymbol{\delta}
$$

第 2 項 $\mathbf{r}^\top J \boldsymbol{\delta}$ はスカラーなので転置しても変わらず、$\mathbf{r}^\top J \boldsymbol{\delta} = \boldsymbol{\delta}^\top J^\top \mathbf{r} = \boldsymbol{\delta}^\top \mathbf{g}$。よって

$$
L(\boldsymbol{\delta}) = \| \mathbf{r} \|^2 + 2 \, \boldsymbol{\delta}^\top \mathbf{g} + \| J \boldsymbol{\delta} \|^2
$$

これより予測した減少量は

$$
L(\mathbf{0}) - L(\boldsymbol{\delta})
= - 2 \, \boldsymbol{\delta}^\top \mathbf{g} - \| J \boldsymbol{\delta} \|^2
\tag{5.1}
$$

次に、$\boldsymbol{\delta}$ が更新式の解であることを使う。$( J^\top J + \lambda I ) \boldsymbol{\delta} = - \mathbf{g}$ の両辺に左から $\boldsymbol{\delta}^\top$ を掛けると

$$
\| J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2 = - \boldsymbol{\delta}^\top \mathbf{g}
\tag{5.2}
$$

(5.2) を (5.1) に代入して $\| J \boldsymbol{\delta} \|^2 = - \boldsymbol{\delta}^\top \mathbf{g} - \lambda \| \boldsymbol{\delta} \|^2$ を消すと

$$
L(\mathbf{0}) - L(\boldsymbol{\delta})
= - 2 \, \boldsymbol{\delta}^\top \mathbf{g} - \bigl( - \boldsymbol{\delta}^\top \mathbf{g} - \lambda \| \boldsymbol{\delta} \|^2 \bigr)
= - \boldsymbol{\delta}^\top \mathbf{g} + \lambda \| \boldsymbol{\delta} \|^2
= \boldsymbol{\delta}^\top ( \lambda \boldsymbol{\delta} - \mathbf{g} )
$$

これで分母の計算式が得られた。$L(\boldsymbol{\delta})$ を実際に評価し直さなくても、既に手元にある $\boldsymbol{\delta}, \lambda, \mathbf{g}$ だけで済むのが利点である。

さらに、この分母が**常に正**であることも分かる。(5.2) より $- \boldsymbol{\delta}^\top \mathbf{g} = \| J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2$ なので

$$
L(\mathbf{0}) - L(\boldsymbol{\delta})
= \underbrace{\| J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2}_{- \boldsymbol{\delta}^\top \mathbf{g}} + \lambda \| \boldsymbol{\delta} \|^2
= \| J \boldsymbol{\delta} \|^2 + 2 \lambda \| \boldsymbol{\delta} \|^2 > 0
\qquad ( \boldsymbol{\delta} \neq \mathbf{0} )
$$

つまり**線形化モデルは常に「減る」と予測する** (LM 法の一歩はモデル上では必ず改善になっている)。だからゲイン比の符号は分子だけで決まり、「$\rho > 0 \iff$ 実際に $E$ が減った」ときれいに読める。分母が 0 や負になって $\rho$ の解釈が壊れる心配はない。

### 5.3 Madsen–Nielsen 流のアルゴリズム

以上を組み立てた全体のアルゴリズムを示す。$\lambda$ の更新則は Madsen–Nielsen のテキストで知られる方式である (サンプルコード `src/bin/8_levenberg_marquardt_method.rs` もこの通りに実装している)。

```text
入力: 初期値 β, τ (例 1e-3), 停止閾値 ε₁ ε₂, 最大反復数
r, J を計算; A = JᵀJ; g = Jᵀr
λ = τ · max(diag(A)); ν = 2
repeat:
    (A + λI) δ = −g を解く
    if ‖δ‖ ≤ ε₂ (‖β‖ + ε₂): 収束 (更新量が微小)
    β_new = β + δ
    ρ = (E(β) − E(β_new)) / (δᵀ(λδ − g))
    if ρ > 0:                       # 目的関数が減った → 採択
        β = β_new
        r, J, A, g を再計算
        if ‖g‖∞ ≤ ε₁: 収束 (勾配が微小)
        λ = λ · max(1/3, 1 − (2ρ − 1)³);  ν = 2
    else:                           # 増えた → 棄却してやり直し
        λ = λ · ν;  ν = 2ν
```

各行の意味を順に解説する。

**`λ = τ · max(diag(A))` — 初期値の決め方。** $\lambda$ の初期値は問題のスケールに合わせる必要がある ($J^\top J$ の成分が $10^8$ の問題と $10^{-3}$ の問題で同じ $\lambda = 1$ を使っても意味が違う)。$\mathrm{diag}(A)$ の第 $k$ 成分は $J$ の第 $k$ 列のノルムの 2 乗であり、$J^\top J$ の対角成分の最大値はその問題の「スケール」の代表値になる。それに小さな係数 $\tau$ を掛けて出発する。初期値に自信があれば $\tau = 10^{-3}$ 程度 (ほぼ GN 法として出発)、自信がなければ $\tau = 1$ など大きめ (最急降下寄りで慎重に出発) を選ぶ。

**`ν = 2` — 棄却時の倍率の初期化。** $\nu$ は「棄却が続いたときに $\lambda$ をどんどん大胆に増やす」ための倍率である (後述)。

**`(A + λI) δ = −g を解く` — 一歩の計算。** 第 2 節の更新式そのもの。実装ではこの方程式を直接解くのではなく、第 6 節の拡大系を QR 分解で解く。

**`if ‖δ‖ ≤ ε₂ (‖β‖ + ε₂)` — 停止条件 1 (更新量が微小)。** 一歩の大きさが現在地のスケール $\| \boldsymbol{\beta} \|$ に対して相対的に十分小さければ、これ以上動いても変わらないので停止する ($+ \varepsilon_2$ は $\boldsymbol{\beta} \approx \mathbf{0}$ でも機能させるための保険)。

**`ρ = ...` — ゲイン比の計算。** 分子は残差を評価し直して得る実際の減少量、分母は 5.2 節で導いた閉じた式である。

**`if ρ > 0` — 採択判定。** 分母は常に正なので、$\rho > 0$ は「$E$ が実際に減った」ことと同値。減ったなら一歩を採択して $\boldsymbol{\beta}$ を進め、新しい地点で $\mathbf{r}, J, A, \mathbf{g}$ を計算し直す。減らなかったら $\boldsymbol{\beta}$ は動かさない (棄却)。**採択された反復では $E$ が必ず減る**ので、$E$ の値は単調に下がっていく。

**`if ‖g‖∞ ≤ ε₁` — 停止条件 2 (勾配が微小)。** $\| \mathbf{g} \|_\infty$ は $\mathbf{g} = J^\top \mathbf{r}$ の成分の絶対値の最大値。$\nabla E = 2 \mathbf{g} \approx \mathbf{0}$ は「谷底に着いた」ことを意味する ([4 番](./4_nonlinear_least_squares.md)の停留条件)。

**`λ = λ · max(1/3, 1 − (2ρ − 1)³)` — 採択時の $\lambda$ 更新。** 一見複雑だが、$\rho$ に応じて滑らかに $\lambda$ を調整する関数である。いくつか代入してみると

- $\rho = 1$ (予測が完璧): $1 - (2 \cdot 1 - 1)^3 = 1 - 1 = 0$ だが `max` で下限が効いて倍率 $1/3$。$\lambda$ を 3 分の 1 に減らし、次はもっと GN 法寄りに。
- $\rho = 1/2$ (予測の半分だけ減った): $1 - 0^3 = 1$。$\lambda$ は据え置き。
- $\rho \to 0^+$ (かろうじて減った): $1 - (-1)^3 = 2$。採択はするが $\lambda$ は 2 倍に**増やす**。「減ったけれど予測はほぼ外れている」ので信頼半径を狭める。

古典的な実装では「成功したら $\lambda$ を $1/10$、失敗したら $10$ 倍」のような不連続な更新がよく使われたが、この式は $\rho$ の値に応じた連続的な調整になっており、$\lambda$ の振動が起きにくい。

**`λ = λ · ν; ν = 2ν` — 棄却時の $\lambda$ 更新。** 棄却したら $\lambda$ を $\nu$ 倍し、さらに $\nu$ 自体を倍にする。つまり連続で棄却されると $\lambda$ は $2, 4, 8, \dots$ 倍と加速度的に増える。「連続で失敗しているなら、もっと大胆に信頼半径を狭めるべきだ」という発想である。3.2 節で見たとおり $\lambda$ が十分大きくなれば LM 法の一歩は最急降下法の微小ステップになり、いつかは必ず $E$ が減る。したがってこの棄却ループが無限に続くことはない。採択に成功したら $\nu$ は 2 にリセットされる。

## 6. 実装上の注意

### 6.1 $J^\top J$ を作らずに解く — 拡大系と QR 分解

[2. QR 分解](./2_qr_decomposition.md)で学んだとおり、正規方程式を作ると条件数が 2 乗になって精度が落ちる。$J^\top J + \lambda I$ を明示的に作るのも同じ問題を抱える。幸い、LM 法の各反復は次の**拡大系** (行列を縦に積んだ線形最小二乗問題) と等価である。

$$
\min_{\boldsymbol{\delta}}
\left\|
\begin{bmatrix} J \\ \sqrt{\lambda} \, I \end{bmatrix} \boldsymbol{\delta} -
\begin{bmatrix} - \mathbf{r} \\ \mathbf{0} \end{bmatrix}
\right\|^2
$$

ここで $\begin{bmatrix} J \\ \sqrt{\lambda} I \end{bmatrix}$ は $J$ ($n \times m$) の下に $\sqrt{\lambda} I$ ($m \times m$) を縦に積んだ $(n + m) \times m$ 行列、右辺のベクトルも $-\mathbf{r}$ の下に零ベクトル $\mathbf{0} \in \mathbb{R}^m$ を積んだものである。

**等価性の確認。** 実際に展開して、これが LM 法の正規方程式に一致することを確かめる。方法は 2 通りある。

*方法 1: ノルムを直接展開する。* 縦に積んだベクトルのノルムの 2 乗は、各ブロックのノルムの 2 乗の和である ($\| [\mathbf{a}; \mathbf{b}] \|^2 = \| \mathbf{a} \|^2 + \| \mathbf{b} \|^2$。成分の 2 乗和を上下で分けて足すだけ)。積んだ行列とベクトルの積を計算すると

$$
\begin{bmatrix} J \\ \sqrt{\lambda} \, I \end{bmatrix} \boldsymbol{\delta} -
\begin{bmatrix} - \mathbf{r} \\ \mathbf{0} \end{bmatrix} =
\begin{bmatrix} J \boldsymbol{\delta} + \mathbf{r} \\ \sqrt{\lambda} \, \boldsymbol{\delta} \end{bmatrix}
$$

なので、目的関数は

$$
\left\| \begin{bmatrix} \mathbf{r} + J \boldsymbol{\delta} \\ \sqrt{\lambda} \, \boldsymbol{\delta} \end{bmatrix} \right\|^2
= \| \mathbf{r} + J \boldsymbol{\delta} \|^2 + \| \sqrt{\lambda} \, \boldsymbol{\delta} \|^2
= \| \mathbf{r} + J \boldsymbol{\delta} \|^2 + \lambda \| \boldsymbol{\delta} \|^2
$$

となる。これは第 4 節の最後に出てきた「モデル + 歩幅への罰金」の形そのものである。最小点では $\boldsymbol{\delta}$ についての勾配が $\mathbf{0}$ になるので、5.2 節の展開 $\| \mathbf{r} + J \boldsymbol{\delta} \|^2 = \| \mathbf{r} \|^2 + 2 \boldsymbol{\delta}^\top J^\top \mathbf{r} + \boldsymbol{\delta}^\top J^\top J \boldsymbol{\delta}$ を使って微分すると ([1 番](./1_least_squares_method.md)の正規方程式の導出と同じ計算)

$$
2 J^\top \mathbf{r} + 2 J^\top J \boldsymbol{\delta} + 2 \lambda \boldsymbol{\delta} = \mathbf{0}
\quad \Longrightarrow \quad
( J^\top J + \lambda I ) \, \boldsymbol{\delta} = - J^\top \mathbf{r}
$$

確かに LM 法の更新式に戻った。

*方法 2: 拡大系の正規方程式を作る。* 線形最小二乗 $\min \| A \boldsymbol{\delta} - \mathbf{b} \|^2$ の正規方程式は $A^\top A \boldsymbol{\delta} = A^\top \mathbf{b}$ だった。$A = \begin{bmatrix} J \\ \sqrt{\lambda} I \end{bmatrix}$、$\mathbf{b} = \begin{bmatrix} - \mathbf{r} \\ \mathbf{0} \end{bmatrix}$ を当てはめると (ブロックごとの積の規則で)

$$
A^\top A
= \begin{bmatrix} J^\top & \sqrt{\lambda} \, I \end{bmatrix}
\begin{bmatrix} J \\ \sqrt{\lambda} \, I \end{bmatrix}
= J^\top J + \lambda I,
\qquad
A^\top \mathbf{b}
= J^\top ( - \mathbf{r} ) + \sqrt{\lambda} \, I \cdot \mathbf{0}
= - J^\top \mathbf{r}
$$

やはり $( J^\top J + \lambda I ) \boldsymbol{\delta} = - J^\top \mathbf{r}$ に一致する。

等価なら何が嬉しいのか。拡大系は「ただの線形最小二乗問題」なので、正規方程式を経由せずに **QR 分解で直接解ける**。$J^\top J$ を一度も作らないため条件数が 2 乗にならない。さらに拡大された行列の特異値は $\sqrt{\sigma_j^2 + \lambda}$ となり ($\sigma_j$ は $J$ の特異値)、$\lambda > 0$ である限り 0 にならないので、$J$ がランク落ちしていても拡大系は常にフルランクである。サンプルコード `src/bin/8_levenberg_marquardt_method.rs` もこの方法 (拡大行列を組んで `lstsq_qr` に渡す) で実装している。

### 6.2 ヤコビ行列の入手方法

解析的に導出するのが最良である (速く正確)。導出が困難なら数値微分 (実装が楽だが誤差と評価回数に注意) や自動微分を使う。ヤコビ行列の符号や転置の取り違えは LM 法の定番バグなので、解析的に書いた場合も数値微分との突き合わせで検証する価値がある。

### 6.3 停止条件は複数併用する

擬似コードにも入れたとおり、次の 3 つを併用するのが定石である。

1. **勾配の微小**: $\| J^\top \mathbf{r} \|_\infty \leq \varepsilon_1$ (停留点に到達)
2. **更新量の微小**: $\| \boldsymbol{\delta} \| \leq \varepsilon_2 ( \| \boldsymbol{\beta} \| + \varepsilon_2 )$ (これ以上動けない)
3. **最大反復数** (無限ループの保険)

どれか 1 つだけだと、問題によっては永遠に満たされないことがある。

### 6.4 保証されるのは局所解まで

LM 法は各反復で $E$ を減らすが、たどり着くのは**初期値が属する谷の底** (局所解) である。大域的最小の保証はなく、初期値の質が結果を左右するという事実は[非線形最小二乗法](./4_nonlinear_least_squares.md)の注意のまま変わらない。LM 法が改善するのは「良い初期値からの収束の速さ」と「そこそこの初期値からでも粘り強く谷を降りる頑健さ」であって、「どの谷に落ちるか」ではない。

## 7. 実験 — GN 法が破綻した初期値からの収束

サンプルコード `src/bin/8_levenberg_marquardt_method.rs` で、[4 番](./4_nonlinear_least_squares.md)以来おなじみの指数減衰モデル

$$
f(x; \boldsymbol{\beta}) = \beta_1 e^{\beta_2 x},
\qquad \text{真値 } \boldsymbol{\beta} = [2.0, \; -1.5]
$$

のフィッティング (データ 30 点、ノイズ付き) を、3 つの初期値から解く。うち 2 つは [7 番](./7_gauss_newton_method.md)で GN 法が破綻した初期値そのものである。実行は `cargo run --bin 8_levenberg_marquardt_method`。

| 初期値 | GN 法 (7 番) の結果 | LM 法の結果 |
| --- | --- | --- |
| $[1.0, -1.0]$ (良い初期値) | 数反復で収束 | **5 反復**で収束 (同等の速さ) |
| $[5.0, 5.0]$ | 飛びすぎて**発散** | **34 反復**で収束 |
| $[1.0, 3.0]$ | $J^\top J$ が特異化して**破綻** | **17 反復**で収束 |

3 つとも同じ推定値 $[2.001, -1.505]$ (真値 $[2.0, -1.5]$ にノイズ分だけずれた値) に到達する。それぞれの $\lambda$ の動きを見ると、これまでの理論がそのまま観察できる。

**良い初期値 $[1.0, -1.0]$。** 初期 $\lambda = \tau \cdot \max \mathrm{diag}(J^\top J)$ は $10^{-3}$ 程度と小さく、最初からほぼ GN 法として動く。毎反復 $\rho \approx 1$ で採択され、$\lambda$ は $2.5 \times 10^{-3} \to 3.1 \times 10^{-5}$ と 3 分の 1 ずつ下がり続け、5 反復で $E = 8.2 \times 10^{-4}$ (ノイズ由来の下限) に到達する。ダンピングがほぼ眠ったまま、GN 法の速さだけが出るケースである。

**GN 法が発散した初期値 $[5.0, 5.0]$。** この点ではモデル $\beta_1 e^{\beta_2 x}$ が爆発しており ($e^{5 \cdot 2} \approx 2 \times 10^4$)、$E \approx 3.3 \times 10^9$、ヤコビ行列の成分も巨大である。初期 $\lambda$ はスケール連動で $10^7$ 台という非常に大きな値になり、LM 法は**最急降下法側から極めて慎重に出発する**。トレースを見ると、$E$ は 1 反復も増えることなく (全ステップ採択) $3.3 \times 10^9 \to 8.2 \times 10^{-4}$ まで単調に 13 桁下がり、$\lambda$ も $5.5 \times 10^7 \to 1.4 \times 10^{-3}$ までほぼ単調に下がっていく。序盤は「小さく安全な一歩」の積み重ねで谷へ降り、モデルが当たり始めた終盤 (反復 27 以降) は $\lambda$ が $10^{-1}$ 以下まで落ちて GN 法の速さに切り替わり、最後の数反復で一気に収束する。GN 法が一歩目で吹き飛んだ初期値から、34 反復で完走する。

**GN 法が特異化で破綻した初期値 $[1.0, 3.0]$。** この点では $J^\top J$ が特異に近く、GN 法は $\boldsymbol{\delta}$ が解けずに破綻した。LM 法では $\lambda I$ のおかげで方程式が常に解け (保証 1)、17 反復で収束する。トレースには興味深い踊り場がある。反復 3〜9 で $E$ が $20.9$ 付近にほぼ張り付き、$\lambda$ も $0.2$ 前後で足踏みする。これは $\rho$ が中途半端 (減ってはいるが予測ほどではない) で、5.3 節の更新式 $\max(1/3, 1 - (2\rho - 1)^3)$ が 1 に近い倍率を返すためである。その間もアルゴリズムは特異な地形をじりじりと横切り、反復 10 あたりで線形化が当たる領域に入ると $\rho$ が 1 に近づき、$\lambda$ が 3 分の 1 ずつ落ちて一気に収束する。**うまく進めないときは信頼半径を保って粘り、進めるようになったら加速する**という適応更新の設計意図が、そのまま数字に現れている。

なお今回の 3 ケースではたまたま棄却 ($\rho \leq 0$ の分岐) が一度も発生しなかったが、より意地悪な初期値や強い非線形性では棄却と $\lambda$ の引き上げ (`(棄却)` の表示) が観察できる。テストコードでは「採択された反復で $E$ が単調減少すること」も検証している。

## 8. ロードマップ全体のまとめ

LM 法は、このリポジトリで積み上げてきた部品の集大成である。各トピックがどこで使われているかを振り返る。

| トピック | LM 法での役割 |
| --- | --- |
| [1. 線形最小二乗法](./1_least_squares_method.md) | 各反復で解く部分問題そのもの (正規方程式の導出は 6.1 節の展開で再利用) |
| [2. QR 分解](./2_qr_decomposition.md) | 拡大系 $[J; \sqrt{\lambda} I]$ を条件数を悪化させずに解く道具 |
| [3. SVD](./3_singular_value_decomposition.md) | 悪条件・ダンピングの効果を理解する視点 (特異値 $\sigma_j \to \sqrt{\sigma_j^2 + \lambda}$ の持ち上げ) |
| [4. 非線形最小二乗法](./4_nonlinear_least_squares.md) | 問題設定と $\nabla E = 2 J^\top \mathbf{r}$ という構造。局所解の注意もそのまま |
| [5. 最急降下法](./5_steepest_descent.md) | $\lambda$ 大の極限 (3.2 節)。頑健さの供給源 |
| [6. ニュートン法](./6_newton_method.md) | 速さの理想形。$H + \tau I$ の正定値化がダンピングの原型 |
| [7. ガウス・ニュートン法](./7_gauss_newton_method.md) | $\lambda \to 0$ の極限 (3.1 節)。速さの供給源 |

最後に一文でまとめる。

> Levenberg-Marquardt 法とは、残差の線形化 (ガウス・ニュートン法) で得た線形最小二乗の部分問題にダンピング $\lambda I$ を加えることで「常に解ける・常に下り方向」を保証し、モデルの当たり外れ (ゲイン比 $\rho$) を見ながら $\lambda$ を増減することで、ガウス・ニュートン法の速さと最急降下法の頑健さを 1 つのつまみで自動的に切り替える手法である。

ここまで理解できれば、あとは実装で確かめるだけである。良い初期値でゼロ残差に近い問題の二次収束的な速さを、悪い初期値で $\lambda$ が大きく取られて粘り強く降りる様子を、それぞれ `cargo run --bin 8_levenberg_marquardt_method` で観察できれば、このロードマップは完走と言ってよい。
