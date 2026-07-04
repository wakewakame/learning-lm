// QR 分解 (docs/ai/2_qr_decomposition.md)
//
// - ハウスホルダー変換による QR 分解 (実装は src/lib.rs) の性質を確認する
// - QR 分解による最小二乗法が、正規方程式より数値的に有利なことを確かめる

use learning_lm::{lstsq_qr, qr_thin, solve_linear, Mat};
use rand::prelude::*;

/// 悪条件な最小二乗問題の例: ヴァンデルモンド行列による多項式フィッティング。
/// y = 1 + x + x^2 + ... + x^(m-1) (係数が全て 1) を厳密に満たすデータを作り、
/// 係数を復元できるかを「正規方程式」と「QR 分解」で比較する。
fn vandermonde_comparison(n: usize, m: usize) -> (f64, f64) {
    let phi = Mat::from_fn(n, m, |i, j| {
        let x = i as f64 / (n - 1) as f64;
        x.powi(j as i32)
    });
    let y: Vec<f64> = (0..n).map(|i| phi.matvec(&vec![1.0; m])[i]).collect();

    // 正規方程式 (ΦᵀΦ) c = Φᵀy を素朴に解く (条件数が κ² に悪化する)
    let phi_t = phi.transpose();
    let gram = phi_t.matmul(&phi);
    let rhs = phi_t.matvec(&y);
    let c_normal = solve_linear(&gram, &rhs).expect("正規方程式が特異");

    // QR 分解で解く (κ のまま)
    let c_qr = lstsq_qr(&phi, &y);

    let max_err = |c: &[f64]| c.iter().map(|ci| (ci - 1.0).abs()).fold(0.0f64, f64::max);
    (max_err(&c_normal), max_err(&c_qr))
}

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // 1. ランダム行列で QR 分解の性質を確認
    println!("== QR 分解の性質 (ランダムな 5x3 行列) ==");
    let a = Mat::from_fn(5, 3, |_, _| rng.random_range(-1.0..1.0));
    let (q, r) = qr_thin(&a);
    let qtq_err = q.transpose().matmul(&q).sub(&Mat::identity(3)).frobenius_norm();
    let qr_err = q.matmul(&r).sub(&a).frobenius_norm();
    println!("‖QᵀQ - I‖ = {qtq_err:.3e} (直交性)");
    println!("‖QR - A‖  = {qr_err:.3e} (再構成)");

    // 2. 悪条件な問題で正規方程式と比較
    println!();
    println!("== 多項式フィッティングでの比較 (真の係数は全て 1) ==");
    println!("次数が上がるほどヴァンデルモンド行列の条件数 κ が悪化し、");
    println!("κ² で効く正規方程式から先に破綻していく。");
    println!("{:>4}  {:>12}  {:>12}", "次数", "正規方程式", "QR 分解");
    for m in [4, 8, 12] {
        let (err_normal, err_qr) = vandermonde_comparison(50, m);
        println!("{:>4}  {:>12.3e}  {:>12.3e}", m - 1, err_normal, err_qr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use learning_lm::{dot, norm};

    // ランダム行列に対して QR 分解の定義を満たすことを確認
    #[test]
    fn test_qr_properties() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for (n, m) in [(3, 3), (5, 3), (10, 4), (20, 7)] {
            let a = Mat::from_fn(n, m, |_, _| rng.random_range(-10.0..10.0));
            let (q, r) = qr_thin(&a);
            // Q の列は正規直交
            let qtq_err = q.transpose().matmul(&q).sub(&Mat::identity(m)).frobenius_norm();
            assert!(qtq_err < 1e-12, "直交性: {qtq_err}");
            // R は上三角
            for i in 0..m {
                for j in 0..i {
                    assert_eq!(r[(i, j)], 0.0);
                }
            }
            // QR = A
            let qr_err = q.matmul(&r).sub(&a).frobenius_norm();
            assert!(qr_err < 1e-12 * a.frobenius_norm().max(1.0), "再構成: {qr_err}");
        }
    }

    // QR による最小二乗が直線フィットの閉形式解 (1_least_squares_method) と一致することを確認
    #[test]
    fn test_lstsq_matches_closed_form() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let points: Vec<(f64, f64)> = (0..100)
            .map(|_| {
                let x: f64 = rng.random_range(-10.0..10.0);
                let y = 2.5 * x - 3.0 + rng.random_range(-0.1..0.1);
                (x, y)
            })
            .collect();
        // 閉形式解
        let n = points.len() as f64;
        let xy_avg = points.iter().map(|(x, y)| x * y).sum::<f64>() / n;
        let x_avg = points.iter().map(|(x, _)| x).sum::<f64>() / n;
        let y_avg = points.iter().map(|(_, y)| y).sum::<f64>() / n;
        let x2_avg = points.iter().map(|(x, _)| x * x).sum::<f64>() / n;
        let a_closed = (xy_avg - x_avg * y_avg) / (x2_avg - x_avg * x_avg);
        let b_closed = y_avg - a_closed * x_avg;
        // QR 分解による解 (Φ の列は [x, 1])
        let phi = Mat::from_fn(points.len(), 2, |i, j| if j == 0 { points[i].0 } else { 1.0 });
        let y: Vec<f64> = points.iter().map(|(_, y)| *y).collect();
        let coef = lstsq_qr(&phi, &y);
        assert!((coef[0] - a_closed).abs() < 1e-10);
        assert!((coef[1] - b_closed).abs() < 1e-10);
    }

    // 最小二乗解の残差が Φ の列空間と直交すること (正規方程式) を確認
    #[test]
    fn test_residual_orthogonality() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let a = Mat::from_fn(20, 4, |_, _| rng.random_range(-1.0..1.0));
        let b: Vec<f64> = (0..20).map(|_| rng.random_range(-1.0..1.0)).collect();
        let x = lstsq_qr(&a, &b);
        let ax = a.matvec(&x);
        let r: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
        for j in 0..4 {
            assert!(dot(&a.col(j), &r).abs() < 1e-10 * norm(&r).max(1.0));
        }
    }

    // 悪条件な問題では QR が正規方程式より高精度なことを確認
    #[test]
    fn test_qr_beats_normal_equations() {
        let (err_normal, err_qr) = vandermonde_comparison(50, 12);
        assert!(err_qr < 1e-6, "QR の誤差が大きすぎる: {err_qr}");
        assert!(err_qr < err_normal, "QR ({err_qr}) が正規方程式 ({err_normal}) より悪い");
    }
}