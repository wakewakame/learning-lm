// ガウス・ニュートン法 (docs/ai/7_gauss_newton_method.md)
//
// 残差を線形化し「線形最小二乗 (QR 分解で解く) の繰り返し」で
// 非線形最小二乗問題を解く。
// - 良い初期値からは少ない反復で収束する (5 番の最急降下法と比べてみる)
// - ステップ制御がないため、悪い初期値からは発散し得る (8 番の LM 法の動機)

use learning_lm::{lstsq_qr, norm, Mat};
use rand::prelude::*;

/// テスト用データ: y = 2.0 exp(-1.5 x) + ノイズ (4 番と同じ設定)
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

/// ガウス・ニュートン法。
/// 各反復で min ||r + Jδ|| という線形最小二乗を QR 分解で解き、β += δ とする。
/// (解, 各反復の E = ||r||² の履歴) を返す。
/// E が非有限 (発散) になるか、δ が計算できなくなった (JᵀJ が特異) 場合は打ち切る。
pub fn gauss_newton(
    residual: &dyn Fn(&[f64]) -> Vec<f64>,
    jacobian: &dyn Fn(&[f64]) -> Mat,
    beta0: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut beta = beta0.to_vec();
    let mut history = vec![];
    for _ in 0..max_iter {
        let r = residual(&beta);
        let e = learning_lm::dot(&r, &r);
        history.push(e);
        if !e.is_finite() {
            break; // 発散 (故障モード 1: 線形化の範囲外へ飛びすぎた)
        }
        // 線形化した部分問題 min ||Jδ - (-r)|| を解く
        let j = jacobian(&beta);
        let minus_r: Vec<f64> = r.iter().map(|ri| -ri).collect();
        let delta = lstsq_qr(&j, &minus_r);
        if !delta.iter().all(|d| d.is_finite()) {
            break; // JᵀJ が特異 (故障モード 2: δ が計算できない)
        }
        for (bi, di) in beta.iter_mut().zip(&delta) {
            *bi += di;
        }
        if norm(&delta) <= tol * (norm(&beta) + tol) {
            history.push(learning_lm::dot(&residual(&beta), &residual(&beta)));
            break;
        }
    }
    (beta, history)
}

fn main() {
    let (xs, ys) = make_dataset(42);
    let res = |b: &[f64]| residual(&xs, &ys, b);
    let jac = |b: &[f64]| jacobian(&xs, b);

    // 1. 良い初期値からの収束 (真値は β = [2.0, -1.5])
    println!("== 初期値 [1.0, -1.0] (良い初期値) ==");
    let (beta, history) = gauss_newton(&res, &jac, &[1.0, -1.0], 1e-12, 50);
    for (k, e) in history.iter().enumerate() {
        println!("反復 {k}: E = {e:.6e}");
    }
    println!("推定値: {beta:.6?}");
    println!("(5 番の最急降下法では同じ問題に数千回の反復が必要だった)");

    // 2. 故障モード 1: 飛びすぎて発散
    println!();
    println!("== 初期値 [5.0, 5.0] (減衰ではなく急増を仮定した悪い初期値) ==");
    let (beta_bad, history_bad) = gauss_newton(&res, &jac, &[5.0, 5.0], 1e-12, 20);
    for (k, e) in history_bad.iter().enumerate() {
        println!("反復 {k}: E = {e:.6e}");
    }
    println!("最終値: {beta_bad:.3?}");
    println!("途中まで E が減るが、線形化の有効範囲を超えて飛び、発散する。");

    // 3. 故障モード 2: JᵀJ が特異になって破綻
    println!();
    println!("== 初期値 [1.0, 3.0] ==");
    let (beta_bad2, history_bad2) = gauss_newton(&res, &jac, &[1.0, 3.0], 1e-12, 20);
    for (k, e) in history_bad2.iter().enumerate() {
        println!("反復 {k}: E = {e:.6e}");
    }
    println!("最終値: {beta_bad2:.3?}");
    println!("β1 が 0 に飛ばされると J の第 2 列 (-β1 x e^(β2 x)) が消えて");
    println!("JᵀJ が特異になり、δ が計算できなくなる。");
    println!("この 2 つの故障モードへの対処が LM 法 (8 番) の動機。");
}

#[cfg(test)]
mod tests {
    use super::*;

    // 良い初期値から少ない反復で真値近くに収束すること
    #[test]
    fn test_converges_from_good_init() {
        let (xs, ys) = make_dataset(42);
        let res = |b: &[f64]| residual(&xs, &ys, b);
        let jac = |b: &[f64]| jacobian(&xs, b);
        let (beta, history) = gauss_newton(&res, &jac, &[1.0, -1.0], 1e-12, 50);
        assert!((beta[0] - 2.0).abs() < 0.05, "β1 = {}", beta[0]);
        assert!((beta[1] + 1.5).abs() < 0.05, "β2 = {}", beta[1]);
        assert!(history.len() <= 15, "反復回数が多すぎる: {}", history.len());
    }

    // 悪い初期値では実際に破綻すること (8 番の LM 法はここから収束する)
    #[test]
    fn test_fails_from_bad_init() {
        let (xs, ys) = make_dataset(42);
        let res = |b: &[f64]| residual(&xs, &ys, b);
        let jac = |b: &[f64]| jacobian(&xs, b);
        // 故障モード 1: E が非有限になる (発散)
        let (_, history) = gauss_newton(&res, &jac, &[5.0, 5.0], 1e-12, 20);
        assert!(!history.last().unwrap().is_finite());
        // 故障モード 2: JᵀJ が特異になり真値に到達できない
        let (beta, _) = gauss_newton(&res, &jac, &[1.0, 3.0], 1e-12, 20);
        assert!((beta[0] - 2.0).abs() > 0.5 || (beta[1] + 1.5).abs() > 0.5);
    }

    // 各反復で E が単調減少すること (良い初期値の場合)
    #[test]
    fn test_monotone_decrease_from_good_init() {
        let (xs, ys) = make_dataset(42);
        let res = |b: &[f64]| residual(&xs, &ys, b);
        let jac = |b: &[f64]| jacobian(&xs, b);
        let (_, history) = gauss_newton(&res, &jac, &[1.0, -1.0], 1e-12, 50);
        for k in 1..history.len() {
            assert!(history[k] <= history[k - 1] * (1.0 + 1e-12));
        }
    }
}