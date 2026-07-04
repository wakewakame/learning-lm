# 線形最小二乗法

## 概要

最小二乗法とは、何らかの測定値などの点群 $(x_i, y_i)$ が $y_i \approx \sum_{k=1}^{m} a_k f_k(x_i)$ のような数式モデルで表現できるとき、点群と数式モデルとの二乗誤差が最も小さくなるような $a_k$ を求める方法。  
例えば、おおよそ直線上に並んでいる点群 $(x_1, y_1), (x_2, y_2), \dots$ に対して最もフィットする $y = a x + b$ の $a$ と $b$ を求めることができる。  

サンプル: [`1_least_squares_method.rs`](../src/bin/1_least_squares_method.rs)

## 解説

### 簡単な例

ここでは簡単のため、おおよそ直線上に並んでいる点群 $(x_1, y_1), (x_2, y_2), \dots (x_n, y_n)$ にフィットするような $y = a x + b$ の $a$ と $b$ を求めるケースを考える。  
入力との誤差 (以降、残差と呼ぶ) の二乗和を $E$ とすると、$E$ は以下の式で表される。  

$$
E = \sum_{i=1}^{n} \biggl\lparen y_i - \lparen a x_i + b \rparen \biggr\rparen^2
$$

ここで $E$ が最小になるよう $a$, $b$ を調整したい。  
とりあえず式を展開して整理してみる。  

$$
\begin{aligned}
& E &=& \sum_{i=1}^{n} a^2 x_i^2 + b^2 + 2 a b x_i - 2 a x_i y_i - 2 b y_i + y_i^2 & \\
& &=& a^2 \sum_{i=1}^{n} x_i^2 + b^2 n + 2 a b \sum_{i=1}^{n} x_i - 2 a \sum_{i=1}^{n} x_i y_i - 2 b \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} y_i^2 &
\end{aligned}
$$

このとき $a$ と $E$ のグラフも $b$ と $E$ のグラフも下に凸な二次関数になる。  
つまり $E$ を $a$ と $b$ で偏微分し、それぞれ 0 となる $a$, $b$ を求めると $E$ の最小値を得られる。  
では実際に $a$, $b$ を求める計算を行う。  

まずは $b$ から求める。  

$$
\begin{aligned}
& \frac{\partial E}{\partial b} &=& \frac{\partial}{\partial b} \biggl\lparen a^2 \sum_{i=1}^{n} x_i^2 + b^2 n + 2 a b \sum_{i=1}^{n} x_i - 2 a \sum_{i=1}^{n} x_i y_i - 2 b \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} y_i^2 \biggr\rparen & \\
& &=& 2 b n + 2 a \sum_{i=1}^{n} x_i - 2 \sum_{i=1}^{n} y_i & \\
& &=& 2 n \biggl\lparen b + a \sum_{i=1}^{n} \frac{x_i}{n} - \sum_{i=1}^{n} \frac{y_i}{n} \biggr\rparen & \\
& &=& 2 n \lparen b + a \times \overline{x} - \overline{y} \rparen = 0 & \\
\Leftrightarrow & b &=& - a \times \overline{x} + \overline{y} &
\end{aligned}
$$

次に $a$ を求める。  

$$
\begin{aligned}
& \frac{\partial E}{\partial a} &=& \frac{\partial}{\partial a} \biggl\lparen a^2 \sum_{i=1}^{n} x_i^2 + b^2 n + 2 a b \sum_{i=1}^{n} x_i - 2 a \sum_{i=1}^{n} x_i y_i - 2 b \sum_{i=1}^{n} y_i + \sum_{i=1}^{n} y_i^2 \biggr\rparen & \\
& &=& 2 a \sum_{i=1}^{n} x_i^2 + 2 b \sum_{i=1}^{n} x_i - 2 \sum_{i=1}^{n} y_i x_i & \\
& &=& 2 n \biggl\lparen a \sum_{i=1}^{n} \frac{x_i^2}{n} + b \sum_{i=1}^{n} \frac{x_i}{n} - \sum_{i=1}^{n} \frac{y_i x_i}{n} \biggr\rparen & \\
& &=& 2 n \lparen a \times \overline{x^2} + b \times \overline{x} - \overline{xy} \rparen & \\
& &=& 2 n \lparen a \times \overline{x^2} + \lparen - a \times \overline{x} + \overline{y} \rparen \times \overline{x} - \overline{xy} \rparen & \\
& &=& 2 n \lparen a \lparen \overline{x^2} - \overline{x}^2 \rparen + \overline{x} \times \overline{y} - \overline{xy} \rparen = 0 & \\
\Leftrightarrow & a &=& \frac{\overline{xy} - \overline{x} \times \overline{y}}{\overline{x^2} - \overline{x}^2} &
\end{aligned}
$$

まとめると $a$ と $b$ は以下の式で求められる。  

$$
\begin{aligned}
& a &=& \frac{\overline{xy} - \overline{x} \times \overline{y}}{\overline{x^2} - \overline{x}^2} & \\
& b &=& \overline{y} - a \times \overline{x} &
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
        \vdots \\
        f_1(x_n) & f_2(x_n) & \cdots & f_m(x_n) \\
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
したがって残差ベクトル $\mathbf{r}$ は $\mathbf{r} = (\mathbf{y} - \mathbf{Fa})$ となり、その二乗和 $E$ は  

$$
\begin{aligned}
& E &=& \mathbf{r}^T \mathbf{r} & \\
& &=& (\mathbf{y} - \mathbf{Fa})^T (\mathbf{y} - \mathbf{Fa}) & \\
& &=& (\mathbf{y}^T - (\mathbf{Fa})^T) (\mathbf{y} - \mathbf{Fa}) & \\
& &=& (\mathbf{y}^T - \mathbf{a}^T \mathbf{F}^T) (\mathbf{y} - \mathbf{Fa}) & \\
& &=& \mathbf{y}^T \mathbf{y} - \mathbf{a}^T \mathbf{F}^T \mathbf{y} - \mathbf{y}^T \mathbf{F} \mathbf{a} + \mathbf{a}^T \mathbf{F}^T \mathbf{Fa} & \\
& &=& \mathbf{y}^T \mathbf{y} - \mathbf{a}^T \mathbf{F}^T \mathbf{y} - \mathbf{a}^T \mathbf{F}^T \mathbf{y} + \mathbf{a}^T \mathbf{F}^T \mathbf{Fa} & \\
& &=& \mathbf{y}^T \mathbf{y} - 2 \mathbf{a}^T \mathbf{F}^T \mathbf{y} + \mathbf{a}^T \mathbf{F}^T \mathbf{Fa} &
\end{aligned}
$$

と表現することができる。  
これをベクトル $\mathbf{a}$ で偏微分すると  

$$
\frac{\partial E}{\partial \mathbf{a}} = 0 - 2 \mathbf{F}^T \mathbf{y} + 2 \mathbf{F}^T \mathbf{Fa}
$$

となる。  
そして、これが 0 のときの $\mathbf{a}$ を求めたいので  

$$
\begin{aligned}
& 0 - 2 \mathbf{F}^T \mathbf{y} + 2 \mathbf{F}^T \mathbf{Fa} &=& 0 & \\
\Leftrightarrow & \mathbf{F}^T \mathbf{Fa} &=& \mathbf{F}^T \mathbf{y} &
\end{aligned}
$$

と式変形でき、これを $\mathbf{a}$ について計算すれば良いこととなる。  
この計算方法はいくつかある。  

1. $\mathbf{a} = (\mathbf{F}^T \mathbf{F})^{-1} \mathbf{F}^T \mathbf{y}$ のように解く
    - ただし Wikipedia によると $\mathbf{F}^T \mathbf{F}$ の逆行列を明示的に求めることは通常良い方法ではないらしい?
2. 直行分解で正規方程式を解く
    - 具体的には QR分解 や 特異値分解(SVD) を用いる方法がある