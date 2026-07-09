// 各サンプルコードで共有する線形代数ユーティリティ。
//
// 外部の線形代数クレートに頼らず、学習用にすべて自前で実装している。
// - `Mat`: 行優先の密行列
// - `solve_linear`: 部分ピボット付きガウス消去
// - `qr_thin` / `lstsq_qr`: ハウスホルダー変換による QR 分解と線形最小二乗 (docs/ai/2)
// - `jacobi_svd` / `svd_lstsq`: 片側ヤコビ法による SVD と最小ノルム最小二乗 (docs/ai/3)

use std::ops::{Index, IndexMut};

/// 行優先の密行列
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// 対角行列 diag(d) を作る
    pub fn diag(d: &[f64]) -> Self {
        let mut m = Self::zeros(d.len(), d.len());
        for (i, &di) in d.iter().enumerate() {
            m[(i, i)] = di;
        }
        m
    }

    pub fn from_fn(rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> f64) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                m[(i, j)] = f(i, j);
            }
        }
        m
    }

    pub fn transpose(&self) -> Mat {
        Mat::from_fn(self.cols, self.rows, |i, j| self[(j, i)])
    }

    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        Mat::from_fn(self.rows, other.cols, |i, j| {
            (0..self.cols).map(|k| self[(i, k)] * other[(k, j)]).sum()
        })
    }

    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self[(i, j)] * v[j]).sum())
            .collect()
    }

    pub fn sub(&self, other: &Mat) -> Mat {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Mat::from_fn(self.rows, self.cols, |i, j| self[(i, j)] - other[(i, j)])
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, j)]).collect()
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }
}

impl Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// a - b (成分ごとの引き算)
pub fn sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// x + α d を返す (反復法の更新 x ← x + αδ や x ← x - α∇f に使う)
pub fn add_scaled(x: &[f64], alpha: f64, d: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), d.len());
    x.iter().zip(d).map(|(xi, di)| xi + alpha * di).collect()
}

/// x にハウスホルダー鏡映を適用する: x ← H x = x - 2 (v·x)/(v·v) v
/// (docs/ai/2 の 5.1 節。v = 0 のときは H = I として何もしない)
fn reflect(v: &[f64], x: &mut [f64]) {
    let vtv = dot(v, v);
    if vtv == 0.0 {
        return;
    }
    let c = 2.0 * dot(v, x) / vtv;
    for (xi, vi) in x.iter_mut().zip(v) {
        *xi -= c * vi;
    }
}

/// 部分ピボット付きガウス消去で正方連立 1 次方程式 Ax = b を解く。
/// 特異 (に近い) 場合は None を返す。
pub fn solve_linear(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    assert_eq!(a.rows, a.cols);
    assert_eq!(a.rows, b.len());
    let n = a.rows;
    let mut w = a.clone();
    let mut x = b.to_vec();
    for k in 0..n {
        // ピボット選択 (第 k 列で絶対値最大の行を選ぶ)
        let pivot = (k..n).max_by(|&i, &j| w[(i, k)].abs().total_cmp(&w[(j, k)].abs()))?;
        if w[(pivot, k)].abs() < 1e-300 {
            return None;
        }
        if pivot != k {
            for j in 0..n {
                let (a, b) = (w[(k, j)], w[(pivot, j)]);
                w[(k, j)] = b;
                w[(pivot, j)] = a;
            }
            x.swap(k, pivot);
        }
        // 前進消去
        for i in (k + 1)..n {
            let factor = w[(i, k)] / w[(k, k)];
            for j in k..n {
                w[(i, j)] -= factor * w[(k, j)];
            }
            x[i] -= factor * x[k];
        }
    }
    // 前進消去で w は上三角になったので、あとは後退代入
    Some(back_substitution(&w, &x))
}

/// 上三角行列 R に対する後退代入 (Rx = b を解く)。
/// R が特異 (列が線形従属) な場合は 0 除算により inf/NaN が伝播するので、
/// 呼び出し側で結果の有限性を確認すること。
pub fn back_substitution(r: &Mat, b: &[f64]) -> Vec<f64> {
    assert_eq!(r.rows, r.cols);
    assert_eq!(r.rows, b.len());
    let m = r.rows;
    let mut x = b.to_vec();
    for k in (0..m).rev() {
        for j in (k + 1)..m {
            x[k] -= r[(k, j)] * x[j];
        }
        x[k] /= r[(k, k)];
    }
    x
}

/// ハウスホルダー変換を列ごとに適用し、鏡映ベクトル群と R を返す内部関数。
/// A (n x m, n >= m) に対し H_m ... H_1 A = [R; O] となる。
fn householder(a: &Mat) -> (Vec<Vec<f64>>, Mat) {
    let (n, m) = (a.rows, a.cols);
    assert!(n >= m, "行数 >= 列数を仮定");
    let mut w = a.clone();
    let mut vs: Vec<Vec<f64>> = Vec::with_capacity(m);
    for k in 0..m {
        // 第 k 列の対角以下を x とし、x を ±||x|| e1 に写す鏡映ベクトル v を作る。
        // 桁落ちを避けるため x_1 と同符号側を選ぶ: v = x + sign(x_1) ||x|| e1
        let mut v: Vec<f64> = (k..n).map(|i| w[(i, k)]).collect();
        let normx = norm(&v);
        v[0] += if v[0] >= 0.0 { normx } else { -normx };
        // W[k.., k..] の各列に鏡映 H = I - 2 v vᵀ / (vᵀv) を適用
        for j in k..m {
            let mut col: Vec<f64> = (k..n).map(|i| w[(i, j)]).collect();
            reflect(&v, &mut col);
            for (t, i) in (k..n).enumerate() {
                w[(i, j)] = col[t];
            }
        }
        vs.push(v);
    }
    let mut r = Mat::zeros(m, m);
    for i in 0..m {
        for j in i..m {
            r[(i, j)] = w[(i, j)];
        }
    }
    (vs, r)
}

/// 鏡映ベクトル群を b に順に適用する (b <- H_m ... H_1 b、つまり b <- Qᵀ b)
fn apply_householder(vs: &[Vec<f64>], b: &mut [f64]) {
    for (k, v) in vs.iter().enumerate() {
        // H_k は先頭 k 成分に触らないので、b[k..] だけに鏡映を適用すればよい
        reflect(v, &mut b[k..]);
    }
}

/// 薄い QR 分解。A (n x m, n >= m) に対し A = Q R となる
/// (Q: n x m 正規直交列, R: m x m 上三角) を返す。
pub fn qr_thin(a: &Mat) -> (Mat, Mat) {
    let (vs, r) = householder(a);
    let (n, m) = (a.rows, a.cols);
    // Q = H_1 ... H_m [I; O] を列ごとに構成する (鏡映を逆順に適用)
    let mut q = Mat::zeros(n, m);
    for j in 0..m {
        // Q の第 j 列 = H_1 ... H_m e_j
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        for (k, v) in vs.iter().enumerate().rev() {
            reflect(v, &mut e[k..]);
        }
        for i in 0..n {
            q[(i, j)] = e[i];
        }
    }
    (q, r)
}

/// QR 分解による線形最小二乗法。||Ax - b|| を最小化する x を返す。
/// AᵀA を作らないため、正規方程式のような条件数の悪化 (κ²) が起きない。
pub fn lstsq_qr(a: &Mat, b: &[f64]) -> Vec<f64> {
    assert_eq!(a.rows, b.len());
    let (vs, r) = householder(a);
    let mut qtb = b.to_vec();
    apply_householder(&vs, &mut qtb);
    back_substitution(&r, &qtb[..a.cols])
}

/// 列 p, q にギブンス回転を適用する:
/// (col_p, col_q) ← (c col_p - s col_q, s col_p + c col_q)
fn rotate_cols(a: &mut Mat, p: usize, q: usize, c: f64, s: f64) {
    for i in 0..a.rows {
        let (aip, aiq) = (a[(i, p)], a[(i, q)]);
        a[(i, p)] = c * aip - s * aiq;
        a[(i, q)] = s * aip + c * aiq;
    }
}

/// 片側ヤコビ法 (Hestenes 法) による薄い特異値分解。
/// A (n x m, n >= m) に対し A = U diag(σ) Vᵀ となる
/// (U: n x m, σ: 降順の特異値, V: m x m 直交) を返す。
/// σ_j = 0 に対応する U の列は零ベクトルとする。
pub fn jacobi_svd(a: &Mat) -> (Mat, Vec<f64>, Mat) {
    let (n, m) = (a.rows, a.cols);
    assert!(n >= m, "行数 >= 列数を仮定");
    let mut b = a.clone();
    let mut v = Mat::identity(m);
    let eps = 1e-15;
    // B の列同士が直交するまで、2 列ずつ回転で直交化するスイープを繰り返す
    for _sweep in 0..60 {
        let mut max_off = 0.0f64;
        for p in 0..m {
            for q in (p + 1)..m {
                // 列 p, q の内積 (2x2 のグラム行列 [app apq; apq aqq] の成分)
                let (bp, bq) = (b.col(p), b.col(q));
                let (app, aqq, apq) = (dot(&bp, &bp), dot(&bq, &bq), dot(&bp, &bq));
                let denom = (app * aqq).sqrt();
                if denom == 0.0 || apq.abs() <= eps * denom {
                    continue;
                }
                max_off = max_off.max(apq.abs() / denom);
                // 列 p, q を直交化するギブンス回転を求め、B と V の両方に適用する
                let zeta = (aqq - app) / (2.0 * apq);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                rotate_cols(&mut b, p, q, c, s);
                rotate_cols(&mut v, p, q, c, s);
            }
        }
        if max_off <= eps {
            break;
        }
    }
    // 特異値 = 直交化された列のノルム、U = 正規化した列
    let mut sigma: Vec<f64> = (0..m).map(|j| norm(&b.col(j))).collect();
    // 降順に並べ替える
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&i, &j| sigma[j].total_cmp(&sigma[i]));
    let mut u = Mat::zeros(n, m);
    let mut v_sorted = Mat::zeros(m, m);
    let mut sigma_sorted = vec![0.0; m];
    for (dst, &src) in order.iter().enumerate() {
        sigma_sorted[dst] = sigma[src];
        if sigma[src] > 0.0 {
            for i in 0..n {
                u[(i, dst)] = b[(i, src)] / sigma[src];
            }
        }
        for i in 0..m {
            v_sorted[(i, dst)] = v[(i, src)];
        }
    }
    sigma = sigma_sorted;
    (u, sigma, v_sorted)
}

/// SVD による最小二乗法。||Ax - b|| を最小化する x のうちノルム最小のものを返す。
/// rcond * σ_max 以下の特異値は 0 とみなす (ランク落ち・悪条件への対処)。
pub fn svd_lstsq(a: &Mat, b: &[f64], rcond: f64) -> Vec<f64> {
    let (u, sigma, v) = jacobi_svd(a);
    let tol = sigma.first().copied().unwrap_or(0.0) * rcond;
    // x = Σ_j (u_j·b / σ_j) v_j  (σ_j > tol の項だけ足す)
    let mut x = vec![0.0; a.cols];
    for (j, &s) in sigma.iter().enumerate() {
        if s > tol {
            x = add_scaled(&x, dot(&u.col(j), b) / s, &v.col(j));
        }
    }
    x
}