# 線形最小二乗法

## 概要

最小二乗法とは、何らかの測定値などの点群 $(x_i, y_i)$ が $y_i \approx \sum_{k=1}^{m} a_k f_k(x_i)$ のような数式モデルで表現できるとき、点群と数式モデルとの二乗誤差が最も小さくなるような $a_k$ を求める方法。  
このようにモデルがパラメータ $a_k$ について線形である場合を特に線形最小二乗法と呼び、このメモではこのケースを扱う。  
なお $f_k(x)$ 自体は $x^2$ や $\sin x$ のような非線形の関数でも構わない (例えば多項式フィッティングも線形最小二乗法である)。  
例えば、おおよそ直線上に並んでいる点群 $(x_1, y_1), (x_2, y_2), \dots$ に対して最もフィットする $y = a x + b$ の $a$ と $b$ を求めることができる。  

サンプル: [`1_least_squares_method.rs`](../src/bin/1_least_squares_method.rs)

## 解説

### 簡単な例

ここでは簡単のため、おおよそ直線上に並んでいる点群 $(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)$ にフィットするような $y = a x + b$ の $a$ と $b$ を求めるケースを考える。  
入力との誤差 (以降、残差と呼ぶ) の二乗和を $E$ とすると、$E$ は以下の式で表される。  

$$
E = \sum_{i=1}^{n} \bigl( y_i - ( a x_i + b ) \bigr)^2
$$

ここで $E$ が最小になるよう $a$, $b$ を調整したい。  
とりあえず式を展開して整理してみる。  

$$
\begin{aligned}
E &= \sum_{i=1}^{n} \bigl( a^2 x_i^2 + b^2 + 2 a b x_i - 2 a x_i y_i - 2 b y_i + y_i^2 \bigr) \\
&= a^2 \sum_{i=1}^{n} x_i^2 + b^2 n + 2 a b \sum_{i=1}^{n} x_i - 2 a \sum_{i=1}^{n} x_i y_i - 2 b \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} y_i^2
\end{aligned}
$$

ここで $E$ は $a$, $b$ の 1 次式 $y_i - ( a x_i + b )$ の二乗和なので、$(a, b)$ の 2 変数関数として下に凸になる。  
つまり $E$ を $a$ と $b$ で偏微分し、それぞれ 0 となる $a$, $b$ を求めると $E$ の最小値を得られる。  
(注意: 「$a$ を固定すれば $b$ について下に凸、$b$ を固定すれば $a$ について下に凸」というだけでは、偏微分が 0 の点が最小とは言えない。例えば $a^2 + b^2 - 4 a b$ は $a$ 単独でも $b$ 単独でも下に凸だが、原点は鞍点であり $a = b$ に沿って $-2 a^2$ となるためいくらでも小さくなる。2 変数関数としての凸性が必要。)  
では実際に $a$, $b$ を求める計算を行う。  

まずは $b$ から求める。  

$$
\begin{aligned}
\frac{\partial E}{\partial b} &= \frac{\partial}{\partial b} \biggl( a^2 \sum_{i=1}^{n} x_i^2 + b^2 n + 2 a b \sum_{i=1}^{n} x_i - 2 a \sum_{i=1}^{n} x_i y_i - 2 b \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} y_i^2 \biggr) \\
&= 2 b n + 2 a \sum_{i=1}^{n} x_i - 2 \sum_{i=1}^{n} y_i \\
&= 2 n \biggl( b + a \sum_{i=1}^{n} \frac{x_i}{n} - \sum_{i=1}^{n} \frac{y_i}{n} \biggr) \\
&= 2 n ( b + a \overline{x} - \overline{y} )
\end{aligned}
$$

これを 0 とおくと  

$$
b = - a \overline{x} + \overline{y}
$$

となる。  

次に $a$ を求める。  

$$
\begin{aligned}
\frac{\partial E}{\partial a} &= \frac{\partial}{\partial a} \biggl( a^2 \sum_{i=1}^{n} x_i^2 + b^2 n + 2 a b \sum_{i=1}^{n} x_i - 2 a \sum_{i=1}^{n} x_i y_i - 2 b \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} y_i^2 \biggr) \\
&= 2 a \sum_{i=1}^{n} x_i^2 + 2 b \sum_{i=1}^{n} x_i - 2 \sum_{i=1}^{n} y_i x_i \\
&= 2 n \biggl( a \sum_{i=1}^{n} \frac{x_i^2}{n} + b \sum_{i=1}^{n} \frac{x_i}{n} - \sum_{i=1}^{n} \frac{y_i x_i}{n} \biggr) \\
&= 2 n ( a \overline{x^2} + b \overline{x} - \overline{xy} ) \\
&= 2 n \bigl( a \overline{x^2} + ( - a \overline{x} + \overline{y} ) \overline{x} - \overline{xy} \bigr) \\
&= 2 n \bigl( a ( \overline{x^2} - \overline{x}^2 ) + \overline{x} \, \overline{y} - \overline{xy} \bigr)
\end{aligned}
$$

これを 0 とおくと  

$$
a = \frac{\overline{xy} - \overline{x} \, \overline{y}}{\overline{x^2} - \overline{x}^2}
$$

となる。  

まとめると $a$ と $b$ は以下の式で求められる。  

$$
\begin{aligned}
a &= \frac{\overline{xy} - \overline{x} \, \overline{y}}{\overline{x^2} - \overline{x}^2} \\
b &= \overline{y} - a \overline{x}
\end{aligned}
$$

ただし、全ての $x_i$ が等しい場合は $a$ の分母が $0$ になってしまうため直線を一意に求められない。  

### 一般化

まず  

$$
y_i \approx \sum_{k=1}^{m} a_k f_k(x_i)
$$

は  

$$
\underbrace{
    \begin{bmatrix}
        y_1 \\
        y_2 \\
        \vdots \\
        y_n
    \end{bmatrix}
}_{\mathbf{y}}
\approx
\underbrace{
    \begin{bmatrix}
        f_1(x_1) & f_2(x_1) & \cdots & f_m(x_1) \\
        f_1(x_2) & f_2(x_2) & \cdots & f_m(x_2) \\
        \vdots & \vdots & \ddots & \vdots \\
        f_1(x_n) & f_2(x_n) & \cdots & f_m(x_n)
    \end{bmatrix}
}_{\mathbf{F}}
\underbrace{
    \begin{bmatrix}
        a_1 \\
        a_2 \\
        \vdots \\
        a_m
    \end{bmatrix}
}_{\mathbf{a}}
$$

のように表現できる。  
したがって残差ベクトル $\mathbf{r}$ は $\mathbf{r} = \mathbf{y} - \mathbf{F} \mathbf{a}$ となり、その二乗和 $E$ は  

$$
\begin{aligned}
E &= \mathbf{r}^\top \mathbf{r} \\
&= (\mathbf{y} - \mathbf{F} \mathbf{a})^\top (\mathbf{y} - \mathbf{F} \mathbf{a}) \\
&= (\mathbf{y}^\top - (\mathbf{F} \mathbf{a})^\top) (\mathbf{y} - \mathbf{F} \mathbf{a}) \\
&= (\mathbf{y}^\top - \mathbf{a}^\top \mathbf{F}^\top) (\mathbf{y} - \mathbf{F} \mathbf{a}) \\
&= \mathbf{y}^\top \mathbf{y} - \mathbf{a}^\top \mathbf{F}^\top \mathbf{y} - \mathbf{y}^\top \mathbf{F} \mathbf{a} + \mathbf{a}^\top \mathbf{F}^\top \mathbf{F} \mathbf{a} \\
&= \mathbf{y}^\top \mathbf{y} - \mathbf{a}^\top \mathbf{F}^\top \mathbf{y} - \mathbf{a}^\top \mathbf{F}^\top \mathbf{y} + \mathbf{a}^\top \mathbf{F}^\top \mathbf{F} \mathbf{a} \\
&= \mathbf{y}^\top \mathbf{y} - 2 \mathbf{a}^\top \mathbf{F}^\top \mathbf{y} + \mathbf{a}^\top \mathbf{F}^\top \mathbf{F} \mathbf{a}
\end{aligned}
$$

と表現することができる。  
なお、途中で $\mathbf{y}^\top \mathbf{F} \mathbf{a}$ を $\mathbf{a}^\top \mathbf{F}^\top \mathbf{y}$ に置き換えたのは、$\mathbf{y}^\top \mathbf{F} \mathbf{a}$ がスカラーであり、スカラーは転置しても値が変わらない ($\mathbf{y}^\top \mathbf{F} \mathbf{a} = ( \mathbf{y}^\top \mathbf{F} \mathbf{a} )^\top = \mathbf{a}^\top \mathbf{F}^\top \mathbf{y}$) ためである。  

これをベクトル $\mathbf{a}$ で偏微分する。  
ここではベクトル微分の公式 $\frac{\partial}{\partial \mathbf{a}} \mathbf{a}^\top \mathbf{c} = \mathbf{c}$ と、対称行列 $\mathbf{M}$ に対する公式 $\frac{\partial}{\partial \mathbf{a}} \mathbf{a}^\top \mathbf{M} \mathbf{a} = 2 \mathbf{M} \mathbf{a}$ を用いる ($\mathbf{F}^\top \mathbf{F}$ は対称行列)。  

$$
\frac{\partial E}{\partial \mathbf{a}} = \mathbf{0} - 2 \mathbf{F}^\top \mathbf{y} + 2 \mathbf{F}^\top \mathbf{F} \mathbf{a}
$$

となる。  
ここでも $E$ は残差 ($\mathbf{a}$ の 1 次式) の二乗和なので $\mathbf{a}$ について下に凸であり、偏微分が $\mathbf{0}$ となる $\mathbf{a}$ が $E$ の最小値を与える。  
そして、これが $\mathbf{0}$ のときの $\mathbf{a}$ を求めたいので  

$$
\begin{aligned}
\mathbf{0} - 2 \mathbf{F}^\top \mathbf{y} + 2 \mathbf{F}^\top \mathbf{F} \mathbf{a} &= \mathbf{0} \\
\Leftrightarrow \quad \mathbf{F}^\top \mathbf{F} \mathbf{a} &= \mathbf{F}^\top \mathbf{y}
\end{aligned}
$$

と式変形でき、これを $\mathbf{a}$ について計算すれば良いこととなる。  
この式は正規方程式と呼ばれる。  

正規方程式の解 $\mathbf{a}$ が一意に定まるのは $\mathbf{F}^\top \mathbf{F}$ が可逆なとき、すなわち $\mathbf{F}$ の列が線形独立 ($\operatorname{rank} \mathbf{F} = m$) なときである。  
「簡単な例」で全ての $x_i$ が等しいと直線を一意に求められなかったのはこの特殊ケースで、$\mathbf{F}$ の第 1 列 ($x_i$ の列) と第 2 列 (全て 1 の列) が線形従属になるためである。  
なお、線形独立でない場合でも $E$ は下に凸なので最小値自体は存在し、それを実現する $\mathbf{a}$ が一意でなくなるだけである。  

正規方程式を解く方法はいくつかある。  

1. $\mathbf{a} = (\mathbf{F}^\top \mathbf{F})^{-1} \mathbf{F}^\top \mathbf{y}$ のように逆行列を明示的に求めて解く
    - ただし $\mathbf{F}^\top \mathbf{F}$ の条件数は $\kappa(\mathbf{F}^\top \mathbf{F}) = \kappa(\mathbf{F})^2$ と $\mathbf{F}$ の二乗に悪化するため、数値計算上は通常良い方法ではない
2. $\mathbf{F}$ 自体を直交分解して解く
    - 例えば QR分解 で $\mathbf{F} = \mathbf{Q} \mathbf{R}$ とすると、正規方程式は $\mathbf{R} \mathbf{a} = \mathbf{Q}^\top \mathbf{y}$ に帰着し、後退代入で解ける
    - $\mathbf{F}^\top \mathbf{F}$ を作ること自体を回避するため、条件数の悪化が起きない
    - 特異値分解(SVD) を用いる方法もあり、$\mathbf{F}$ の列が線形独立でない場合にも対応できる
