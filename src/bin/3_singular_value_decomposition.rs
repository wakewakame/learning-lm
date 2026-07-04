// 特異値分解 (SVD) (docs/ai/3_singular_value_decomposition.md)
//
// - 片側ヤコビ法による SVD (実装は src/lib.rs) の性質を確認する
// - ランク検出、最小ノルム最小二乗解、低ランク近似という SVD ならではの応用を試す

use learning_lm::{jacobi_svd, norm, svd_lstsq, Mat};
use rand::prelude::*;

fn main() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    // 1. ランダム行列で SVD の性質を確認
    println!("== SVD の性質 (ランダムな 6x4 行列) ==");
    let a = Mat::from_fn(6, 4, |_, _| rng.random_range(-1.0..1.0));
    let (u, sigma, v) = jacobi_svd(&a);
    let s_mat = Mat::from_fn(4, 4, |i, j| if i == j { sigma[i] } else { 0.0 });
    let recon_err = u.matmul(&s_mat).matmul(&v.transpose()).sub(&a).frobenius_norm();
    let utu_err = u.transpose().matmul(&u).sub(&Mat::identity(4)).frobenius_norm();
    let vtv_err = v.transpose().matmul(&v).sub(&Mat::identity(4)).frobenius_norm();
    println!("特異値: {sigma:.4?}");
    println!("‖UΣVᵀ - A‖ = {recon_err:.3e} (再構成)");
    println!("‖UᵀU - I‖  = {utu_err:.3e}, ‖VᵀV - I‖ = {vtv_err:.3e} (直交性)");
    println!("条件数 κ = σ_max/σ_min = {:.3}", sigma[0] / sigma[3]);

    // 2. ランク落ちの検出と最小ノルム解
    println!();
    println!("== ランク落ち行列の最小二乗 ==");
    // 第 3 列 = 第 1 列 + 第 2 列 という線形従属な 5x3 行列を作る
    let base = Mat::from_fn(5, 2, |_, _| rng.random_range(-1.0..1.0));
    let a_def = Mat::from_fn(5, 3, |i, j| match j {
        0 | 1 => base[(i, j)],
        _ => base[(i, 0)] + base[(i, 1)],
    });
    let (_, sigma_def, _) = jacobi_svd(&a_def);
    println!("特異値: {sigma_def:.4?} (最後がほぼ 0 → 数値ランク 2)");
    // 解が一意でない問題でも、SVD ならノルム最小の解が得られる
    let x_true = [1.0, 2.0, 0.0];
    let b = a_def.matvec(&x_true);
    let x_mn = svd_lstsq(&a_def, &b, 1e-10);
    let ax = a_def.matvec(&x_mn);
    let res: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
    println!("最小ノルム解: {x_mn:.4?}");
    println!("残差 = {:.3e}, ‖x‖ = {:.4} (元の解 {x_true:?} の ‖x‖ = {:.4})",
        norm(&res), norm(&x_mn), norm(&x_true));

    // 3. 低ランク近似 (エッカート・ヤングの定理)
    println!();
    println!("== 低ランク近似 ==");
    let a2 = Mat::from_fn(8, 5, |_, _| rng.random_range(-1.0..1.0));
    let (u2, s2, v2) = jacobi_svd(&a2);
    println!("特異値: {s2:.4?}");
    for k in 1..5 {
        // 上位 k 個の特異値だけで再構成した A_k
        let a_k = Mat::from_fn(8, 5, |i, j| {
            (0..k).map(|l| s2[l] * u2[(i, l)] * v2[(j, l)]).sum()
        });
        let err = a_k.sub(&a2).frobenius_norm();
        // 理論値: ‖A - A_k‖_F = sqrt(σ_{k+1}² + ... + σ_m²)
        let theory = s2[k..].iter().map(|s| s * s).sum::<f64>().sqrt();
        println!("k={k}: ‖A - A_k‖_F = {err:.6} (理論値 {theory:.6})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use learning_lm::dot;

    // 特異値が既知の行列で確認 (対角成分 3, -4 → 特異値は 4, 3)
    #[test]
    fn test_known_singular_values() {
        let mut a = Mat::zeros(3, 2);
        a[(0, 0)] = 3.0;
        a[(1, 1)] = -4.0;
        let (_, sigma, _) = jacobi_svd(&a);
        assert!((sigma[0] - 4.0).abs() < 1e-12);
        assert!((sigma[1] - 3.0).abs() < 1e-12);
    }

    // ランダム行列に対して SVD の定義を満たすことを確認
    #[test]
    fn test_svd_properties() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        for (n, m) in [(3, 3), (5, 3), (10, 4), (20, 7)] {
            let a = Mat::from_fn(n, m, |_, _| rng.random_range(-10.0..10.0));
            let (u, sigma, v) = jacobi_svd(&a);
            // 特異値は非負かつ降順
            for j in 0..m {
                assert!(sigma[j] >= 0.0);
                if j > 0 {
                    assert!(sigma[j - 1] >= sigma[j]);
                }
            }
            // U, V の直交性
            let utu_err = u.transpose().matmul(&u).sub(&Mat::identity(m)).frobenius_norm();
            let vtv_err = v.transpose().matmul(&v).sub(&Mat::identity(m)).frobenius_norm();
            assert!(utu_err < 1e-10, "U の直交性: {utu_err}");
            assert!(vtv_err < 1e-10, "V の直交性: {vtv_err}");
            // A = U Σ Vᵀ
            let s_mat = Mat::from_fn(m, m, |i, j| if i == j { sigma[i] } else { 0.0 });
            let recon_err = u.matmul(&s_mat).matmul(&v.transpose()).sub(&a).frobenius_norm();
            assert!(recon_err < 1e-10 * a.frobenius_norm(), "再構成: {recon_err}");
        }
    }

    // ランク落ち行列: 特異値からランクを検出できること
    #[test]
    fn test_rank_detection() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let base = Mat::from_fn(5, 2, |_, _| rng.random_range(-1.0..1.0));
        let a = Mat::from_fn(5, 3, |i, j| match j {
            0 | 1 => base[(i, j)],
            _ => base[(i, 0)] + base[(i, 1)],
        });
        let (_, sigma, _) = jacobi_svd(&a);
        let rank = sigma.iter().filter(|&&s| s > sigma[0] * 1e-10).count();
        assert_eq!(rank, 2);
    }

    // ランク落ちでも SVD による最小二乗は最小ノルム解を返すこと
    #[test]
    fn test_min_norm_solution() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let base = Mat::from_fn(5, 2, |_, _| rng.random_range(-1.0..1.0));
        let a = Mat::from_fn(5, 3, |i, j| match j {
            0 | 1 => base[(i, j)],
            _ => base[(i, 0)] + base[(i, 1)],
        });
        let b = a.matvec(&[1.0, 2.0, 0.0]);
        let x = svd_lstsq(&a, &b, 1e-10);
        // 残差はゼロ (b は列空間内)
        let ax = a.matvec(&x);
        let res: Vec<f64> = b.iter().zip(&ax).map(|(bi, axi)| bi - axi).collect();
        assert!(norm(&res) < 1e-10);
        // 零空間方向 (1, 1, -1) と直交している = ノルム最小
        assert!(dot(&x, &[1.0, 1.0, -1.0]).abs() < 1e-10);
        // 実際に元の解よりノルムが小さい
        assert!(norm(&x) <= norm(&[1.0, 2.0, 0.0]) + 1e-12);
    }
}