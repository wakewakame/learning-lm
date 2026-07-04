// 非線形最小二乗法 (docs/ai/4_nonlinear_least_squares.md)
//
// 指数減衰モデル y = β1 exp(β2 x) のフィッティングを題材に、
// - 残差ベクトル r とヤコビ行列 J の定義
// - 解析的なヤコビ行列が数値微分と一致すること
// - ∇E = 2 Jᵀ r という勾配の構造
// - 対数変換による線形化は「別の問題」を解いていること
// を確認する。実際に解くのは 5, 7, 8 番のサンプル。

use learning_lm::{dot, Mat};
use rand::prelude::*;

/// フィッティング対象の観測データ
pub struct Dataset {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
}

/// テスト用データを生成する。真のパラメータは β = (2.0, -1.5)。
pub fn make_dataset(noise: f64, seed: u64) -> Dataset {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 / (n - 1) as f64).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| 2.0 * (-1.5 * x).exp() + rng.random_range(-noise..noise))
        .collect();
    Dataset { xs, ys }
}

/// モデル f(x; β) = β1 exp(β2 x)
pub fn model(x: f64, beta: &[f64]) -> f64 {
    beta[0] * (beta[1] * x).exp()
}

/// 残差ベクトル r_i = y_i - f(x_i; β)
pub fn residual(data: &Dataset, beta: &[f64]) -> Vec<f64> {
    data.xs
        .iter()
        .zip(&data.ys)
        .map(|(&x, &y)| y - model(x, beta))
        .collect()
}

/// ヤコビ行列 J_ik = ∂r_i/∂β_k (解析的に導出したもの)
/// r_i = y_i - β1 exp(β2 x_i) なので
///   ∂r_i/∂β1 = -exp(β2 x_i)
///   ∂r_i/∂β2 = -β1 x_i exp(β2 x_i)
pub fn jacobian(data: &Dataset, beta: &[f64]) -> Mat {
    Mat::from_fn(data.xs.len(), 2, |i, k| {
        let x = data.xs[i];
        let e = (beta[1] * x).exp();
        match k {
            0 => -e,
            _ => -beta[0] * x * e,
        }
    })
}

/// 目的関数 E(β) = ||r(β)||²
pub fn objective(data: &Dataset, beta: &[f64]) -> f64 {
    let r = residual(data, beta);
    dot(&r, &r)
}

/// 中心差分による数値微分でヤコビ行列を計算する (解析解の検証用)
fn jacobian_numeric(data: &Dataset, beta: &[f64]) -> Mat {
    Mat::from_fn(data.xs.len(), beta.len(), |i, k| {
        let h = 1e-6 * beta[k].abs().max(1.0);
        let mut bp = beta.to_vec();
        let mut bm = beta.to_vec();
        bp[k] += h;
        bm[k] -= h;
        (residual(data, &bp)[i] - residual(data, &bm)[i]) / (2.0 * h)
    })
}

/// 対数変換による線形化: log y = log β1 + β2 x を直線フィットで解く。
/// 「log y の残差」の最小二乗なので、元の問題の最適解とは一致しない。
pub fn log_linearized_fit(data: &Dataset) -> [f64; 2] {
    let points: Vec<(f64, f64)> = data.xs.iter().zip(&data.ys).map(|(&x, &y)| (x, y.ln())).collect();
    let n = points.len() as f64;
    let xy_avg = points.iter().map(|(x, y)| x * y).sum::<f64>() / n;
    let x_avg = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_avg = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let x2_avg = points.iter().map(|(x, _)| x * x).sum::<f64>() / n;
    let slope = (xy_avg - x_avg * y_avg) / (x2_avg - x_avg * x_avg);
    let intercept = y_avg - slope * x_avg;
    [intercept.exp(), slope]
}

fn main() {
    let data = make_dataset(0.01, 42);
    let beta = [1.0, -1.0]; // 適当な点で微分の構造を確認する

    // 1. 解析的ヤコビ行列と数値微分の一致
    println!("== ヤコビ行列の検証 (解析解 vs 数値微分) ==");
    let j = jacobian(&data, &beta);
    let j_num = jacobian_numeric(&data, &beta);
    println!("最大差: {:.3e}", j.sub(&j_num).frobenius_norm());

    // 2. 勾配の構造 ∇E = 2 Jᵀ r の確認
    println!();
    println!("== ∇E = 2 Jᵀ r の検証 (vs E の数値微分) ==");
    let r = residual(&data, &beta);
    let grad_struct = j.transpose().matvec(&r).iter().map(|g| 2.0 * g).collect::<Vec<_>>();
    let grad_num: Vec<f64> = (0..2)
        .map(|k| {
            let h = 1e-6;
            let mut bp = beta.to_vec();
            let mut bm = beta.to_vec();
            bp[k] += h;
            bm[k] -= h;
            (objective(&data, &bp) - objective(&data, &bm)) / (2.0 * h)
        })
        .collect();
    println!("2Jᵀr     = {grad_struct:.6?}");
    println!("数値微分 = {grad_num:.6?}");

    // 3. 対数変換による線形化は「別の問題」
    println!();
    println!("== 対数変換による線形化 (真値 β = [2.0, -1.5]) ==");
    let beta_loglin = log_linearized_fit(&data);
    println!("log 線形化の解: {beta_loglin:.4?}");
    println!("E(log線形化)  = {:.6e}", objective(&data, &beta_loglin));
    // E の最小点なら ∇E = 2Jᵀr = 0 のはずだが、log 線形化の解ではゼロにならない。
    // つまりこれは「log y の残差の最小二乗」という別の問題の解である。
    let j_ll = jacobian(&data, &beta_loglin);
    let r_ll = residual(&data, &beta_loglin);
    let grad_ll: Vec<f64> = j_ll.transpose().matvec(&r_ll).iter().map(|g| 2.0 * g).collect();
    println!("∇E(log線形化) = {grad_ll:.6?} (≠ 0 → E の停留点ではない)");
    println!("(log 線形化は log y の残差を最小化する別の問題を解いている。");
    println!(" ただし 7, 8 番の反復法の初期値としては十分良い。");
    println!(" 実際の E の最小点は 7, 8 番の解法が求める)");
}

#[cfg(test)]
mod tests {
    use super::*;

    // 解析的ヤコビ行列が数値微分と一致すること
    #[test]
    fn test_jacobian_matches_numeric() {
        let data = make_dataset(0.01, 42);
        for beta in [[1.0, -1.0], [2.0, -1.5], [0.5, 0.5], [-1.0, 0.3]] {
            let diff = jacobian(&data, &beta).sub(&jacobian_numeric(&data, &beta));
            assert!(diff.frobenius_norm() < 1e-6, "β={beta:?}: {}", diff.frobenius_norm());
        }
    }

    // ∇E = 2 Jᵀ r が E の数値微分と一致すること
    #[test]
    fn test_gradient_structure() {
        let data = make_dataset(0.01, 42);
        let beta = [1.0, -1.0];
        let j = jacobian(&data, &beta);
        let r = residual(&data, &beta);
        let grad: Vec<f64> = j.transpose().matvec(&r).iter().map(|g| 2.0 * g).collect();
        for k in 0..2 {
            let h = 1e-6;
            let mut bp = beta.to_vec();
            let mut bm = beta.to_vec();
            bp[k] += h;
            bm[k] -= h;
            let num = (objective(&data, &bp) - objective(&data, &bm)) / (2.0 * h);
            assert!((grad[k] - num).abs() < 1e-4 * num.abs().max(1.0));
        }
    }

    // log 線形化はそこそこ近いが、E の停留点 (∇E = 0) ではないこと
    #[test]
    fn test_log_linearization_is_biased() {
        let data = make_dataset(0.01, 42);
        let beta_loglin = log_linearized_fit(&data);
        // 初期値としては使える程度に近い
        assert!((beta_loglin[0] - 2.0).abs() < 0.3);
        assert!((beta_loglin[1] + 1.5).abs() < 0.3);
        // しかし ∇E = 2Jᵀr はゼロにならない = E の最小点ではない
        let j = jacobian(&data, &beta_loglin);
        let r = residual(&data, &beta_loglin);
        let grad: Vec<f64> = j.transpose().matvec(&r).iter().map(|g| 2.0 * g).collect();
        assert!(learning_lm::norm(&grad) > 1e-4, "∇E = {grad:?}");
    }
}