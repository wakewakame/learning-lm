// Levenberg-Marquardt 法 (docs/ai/8_levenberg_marquardt_method.md)
//
// ガウス・ニュートン法にダンピング λ を加え、ゲイン比で λ を適応更新する
// (Madsen-Nielsen 流)。各反復の方程式 (JᵀJ + λI)δ = -Jᵀr は、
// 行列を縦に積んだ線形最小二乗 min ||[J; √λ I]δ - [-r; 0]|| として
// QR 分解で解く (JᵀJ を作らないため条件数が悪化しない)。
//
// - 良い初期値ではガウス・ニュートン法と同等の速さ
// - ガウス・ニュートン法が発散した悪い初期値 (7 番参照) でも収束する

use learning_lm::{add_scaled, dot, lstsq_qr, norm, Mat};
use rand::prelude::*;

/// テスト用データ: y = 2.0 exp(-1.5 x) + ノイズ (4, 7 番と同じ設定)
fn make_dataset(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 / (n - 1) as f64).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| 2.0 * (-1.5 * x).exp() + rng.random_range(-0.01..0.01))
        .collect();
    (xs, ys)
}

fn residual(xs: &[f64], ys: &[f64], b: &[f64]) -> Vec<f64> {
    xs.iter().zip(ys).map(|(&x, &y)| y - b[0] * (b[1] * x).exp()).collect()
}

fn jacobian(xs: &[f64], b: &[f64]) -> Mat {
    Mat::from_fn(xs.len(), 2, |i, k| {
        let e = (b[1] * xs[i]).exp();
        match k {
            0 => -e,
            _ => -b[0] * xs[i] * e,
        }
    })
}

/// 1 反復分の記録 (観察用)
pub struct LmTrace {
    pub lambda: f64,
    pub e: f64,
    pub accepted: bool,
}

/// Levenberg-Marquardt 法 (Madsen-Nielsen 流の λ 更新)。
/// (解, 履歴) を返す。
pub fn levenberg_marquardt(
    residual: &dyn Fn(&[f64]) -> Vec<f64>,
    jacobian: &dyn Fn(&[f64]) -> Mat,
    beta0: &[f64],
    tau: f64,
    eps1: f64, // 勾配の停止閾値
    eps2: f64, // 更新量の停止閾値
    max_iter: usize,
) -> (Vec<f64>, Vec<LmTrace>) {
    let mut beta = beta0.to_vec();
    let m = beta.len();
    let mut r = residual(&beta);
    let mut j = jacobian(&beta);
    let mut e = dot(&r, &r);
    // g = Jᵀr, λ の初期値は τ × max(diag(JᵀJ))
    let mut g = j.transpose().matvec(&r);
    let max_diag = (0..m)
        .map(|k| dot(&j.col(k), &j.col(k)))
        .fold(0.0f64, f64::max);
    let mut lambda = tau * max_diag;
    let mut nu = 2.0;
    let mut trace = vec![];
    for _ in 0..max_iter {
        if g.iter().map(|gi| gi.abs()).fold(0.0f64, f64::max) <= eps1 {
            break; // 勾配が十分小さい
        }
        // (JᵀJ + λI)δ = -g を、拡大系 min ||[J; √λ I]δ - [-r; 0]|| として QR で解く
        let n = j.rows;
        let aug = Mat::from_fn(n + m, m, |i, k| {
            if i < n {
                j[(i, k)]
            } else if i - n == k {
                lambda.sqrt()
            } else {
                0.0
            }
        });
        let mut rhs = vec![0.0; n + m];
        for i in 0..n {
            rhs[i] = -r[i];
        }
        let delta = lstsq_qr(&aug, &rhs);
        if norm(&delta) <= eps2 * (norm(&beta) + eps2) {
            break; // 更新量が十分小さい
        }
        let beta_new = add_scaled(&beta, 1.0, &delta); // β + δ
        let r_new = residual(&beta_new);
        let e_new = dot(&r_new, &r_new);
        // ゲイン比 ρ = 実際の減少量 / 線形化モデルが予測した減少量
        // 分母は δᵀ(λδ - g) で、常に正になる
        let predicted: f64 = delta.iter().zip(&g).map(|(di, gi)| di * (lambda * di - gi)).sum();
        let rho = (e - e_new) / predicted;
        if e_new.is_finite() && rho > 0.0 {
            // 採択: モデルの当たり具合に応じて λ を減らす
            beta = beta_new;
            r = r_new;
            e = e_new;
            j = jacobian(&beta);
            g = j.transpose().matvec(&r);
            lambda *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
            nu = 2.0;
            trace.push(LmTrace { lambda, e, accepted: true });
        } else {
            // 棄却: λ を増やして小さく安全な一歩でやり直す
            lambda *= nu;
            nu *= 2.0;
            trace.push(LmTrace { lambda, e, accepted: false });
        }
    }
    (beta, trace)
}

fn main() {
    let (xs, ys) = make_dataset(42);
    let res = |b: &[f64]| residual(&xs, &ys, b);
    let jac = |b: &[f64]| jacobian(&xs, b);

    let print_trace = |trace: &[LmTrace]| {
        for (k, t) in trace.iter().enumerate() {
            println!(
                "反復 {k}: E = {:.6e}, λ = {:.3e} {}",
                t.e,
                t.lambda,
                if t.accepted { "" } else { "(棄却)" }
            );
        }
    };

    // 1. 良い初期値 (真値は β = [2.0, -1.5])
    println!("== 初期値 [1.0, -1.0] (良い初期値) ==");
    let (beta, trace) = levenberg_marquardt(&res, &jac, &[1.0, -1.0], 1e-3, 1e-8, 1e-10, 100);
    print_trace(&trace);
    println!("推定値: {beta:.6?} (ガウス・ニュートン法と同等の速さ)");

    // 2. ガウス・ニュートン法 (7 番) が発散した初期値
    println!();
    println!("== 初期値 [5.0, 5.0] (7 番で GN 法が発散した初期値) ==");
    let (beta_bad, trace_bad) = levenberg_marquardt(&res, &jac, &[5.0, 5.0], 1e-3, 1e-8, 1e-10, 200);
    print_trace(&trace_bad);
    println!("推定値: {beta_bad:.6?}");

    // 3. ガウス・ニュートン法が JᵀJ の特異化で破綻した初期値
    println!();
    println!("== 初期値 [1.0, 3.0] (7 番で GN 法が JᵀJ の特異化で破綻した初期値) ==");
    let (beta_bad2, trace_bad2) = levenberg_marquardt(&res, &jac, &[1.0, 3.0], 1e-3, 1e-8, 1e-10, 200);
    print_trace(&trace_bad2);
    println!("推定値: {beta_bad2:.6?}");
    println!();
    println!("初期 λ (= τ × max diag(JᵀJ)) が大きく取られて最急降下法側から慎重に出発し、");
    println!("モデルが当たり始めると λ が下がってガウス・ニュートン法の速さに移行していく。");
    println!("もし途中で予測が外れれば、その一歩は棄却され λ が引き上げられる (ρ ≤ 0 の分岐)。");
}

#[cfg(test)]
mod tests {
    use super::*;

    // 良い初期値から真値近くに収束すること
    #[test]
    fn test_converges_from_good_init() {
        let (xs, ys) = make_dataset(42);
        let res = |b: &[f64]| residual(&xs, &ys, b);
        let jac = |b: &[f64]| jacobian(&xs, b);
        let (beta, trace) = levenberg_marquardt(&res, &jac, &[1.0, -1.0], 1e-3, 1e-8, 1e-10, 100);
        assert!((beta[0] - 2.0).abs() < 0.05, "β1 = {}", beta[0]);
        assert!((beta[1] + 1.5).abs() < 0.05, "β2 = {}", beta[1]);
        assert!(trace.len() <= 30, "反復回数が多すぎる: {}", trace.len());
    }

    // ガウス・ニュートン法が破綻する悪い初期値でも収束すること (7 番のテストと対になる)
    #[test]
    fn test_converges_from_bad_init() {
        let (xs, ys) = make_dataset(42);
        let res = |b: &[f64]| residual(&xs, &ys, b);
        let jac = |b: &[f64]| jacobian(&xs, b);
        // GN が発散した初期値と、JᵀJ の特異化で破綻した初期値の両方から収束する
        for init in [[5.0, 5.0], [1.0, 3.0]] {
            let (beta, _) = levenberg_marquardt(&res, &jac, &init, 1e-3, 1e-8, 1e-10, 500);
            assert!((beta[0] - 2.0).abs() < 0.05, "init {init:?}: β1 = {}", beta[0]);
            assert!((beta[1] + 1.5).abs() < 0.05, "init {init:?}: β2 = {}", beta[1]);
        }
    }

    // 採択された反復では E が単調減少すること
    #[test]
    fn test_accepted_steps_decrease_e() {
        let (xs, ys) = make_dataset(42);
        let res = |b: &[f64]| residual(&xs, &ys, b);
        let jac = |b: &[f64]| jacobian(&xs, b);
        let (_, trace) = levenberg_marquardt(&res, &jac, &[5.0, 5.0], 1e-3, 1e-8, 1e-10, 500);
        let es: Vec<f64> = trace.iter().filter(|t| t.accepted).map(|t| t.e).collect();
        for k in 1..es.len() {
            assert!(es[k] <= es[k - 1] * (1.0 + 1e-12));
        }
    }
}