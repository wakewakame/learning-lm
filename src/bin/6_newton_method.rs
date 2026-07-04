// この実装は AI (Claude) が作成したもので、著者はまだレビューしていない。
//
// ニュートン法 (docs/ai/6_newton_method.md)
//
// - 求根 (g(x) = 0) としてのニュートン法で二次収束 (正しい桁数が倍々) を観察する
// - 最適化 (∇f = 0) としてのニュートン法を実装する
//   - 2 次関数ならどんな条件数でも 1 反復で厳密解に到達する
//   - 滑らかな凸関数でも解の近くでは二次収束する

use learning_lm::{norm, solve_linear, Mat};

/// 求根のニュートン法: x_{k+1} = x_k - g(x_k) / g'(x_k)
pub fn newton_root(
    g: &dyn Fn(f64) -> f64,
    dg: &dyn Fn(f64) -> f64,
    x0: f64,
    iters: usize,
) -> Vec<f64> {
    let mut xs = vec![x0];
    for _ in 0..iters {
        let x = *xs.last().unwrap();
        xs.push(x - g(x) / dg(x));
    }
    xs
}

/// 最適化のニュートン法: H(x) δ = -∇f(x) を解いて x += δ を繰り返す。
/// ||∇f|| <= tol になるまで反復し、(解, 各反復の ||∇f|| の履歴) を返す。
pub fn newton_optimize(
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    hess: &dyn Fn(&[f64]) -> Mat,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut x = x0.to_vec();
    let mut history = vec![];
    for _ in 0..max_iter {
        let g = grad(&x);
        history.push(norm(&g));
        if norm(&g) <= tol {
            break;
        }
        let minus_g: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let delta = solve_linear(&hess(&x), &minus_g).expect("ヘッセ行列が特異");
        for (xi, di) in x.iter_mut().zip(&delta) {
            *xi += di;
        }
    }
    (x, history)
}

fn main() {
    // 1. 求根: √2 を g(x) = x² - 2 の根として求める
    println!("== 求根: x² - 2 = 0 (√2 = 1.41421356237...) ==");
    let xs = newton_root(&|x| x * x - 2.0, &|x| 2.0 * x, 2.0, 5);
    for (k, x) in xs.iter().enumerate() {
        println!("x_{k} = {x:.15} (誤差 {:+.3e})", x - 2f64.sqrt());
    }
    println!("誤差の指数がほぼ倍々で減る = 二次収束");

    // 2. 2 次関数は 1 反復で厳密解
    // 最急降下法 (5 番) が κ = 1000 で何千回も反復した問題が、ニュートン法なら 1 回で終わる。
    println!();
    println!("== 2 次関数 f = (x² + 1000 y²)/2 を (1, 1) から ==");
    let kappa = 1000.0;
    let (x, history) = newton_optimize(
        &|x| vec![x[0], kappa * x[1]],
        &|_| Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (1, 1) => kappa,
            _ => 0.0,
        }),
        &[1.0, 1.0],
        1e-10,
        10,
    );
    println!("解: {x:?}, 反復回数: {}", history.len() - 1);

    // 3. 滑らかな凸関数での二次収束
    // f(x, y) = exp(x+y-1) + exp(x-y-1) + exp(-x-1)
    println!();
    println!("== 凸関数 f = e^(x+y-1) + e^(x-y-1) + e^(-x-1) を (-1, 1) から ==");
    let grad = |x: &[f64]| {
        let (e1, e2, e3) = ((x[0] + x[1] - 1.0).exp(), (x[0] - x[1] - 1.0).exp(), (-x[0] - 1.0).exp());
        vec![e1 + e2 - e3, e1 - e2]
    };
    let hess = |x: &[f64]| {
        let (e1, e2, e3) = ((x[0] + x[1] - 1.0).exp(), (x[0] - x[1] - 1.0).exp(), (-x[0] - 1.0).exp());
        Mat::from_fn(2, 2, |i, j| match (i, j) {
            (0, 0) => e1 + e2 + e3,
            (1, 1) => e1 + e2,
            _ => e1 - e2,
        })
    };
    let (x, history) = newton_optimize(&grad, &hess, &[-1.0, 1.0], 1e-14, 50);
    for (k, g_norm) in history.iter().enumerate() {
        println!("反復 {k}: ‖∇f‖ = {g_norm:.3e}");
    }
    println!("解: {x:.10?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    // √2 が少ない反復で機械精度まで求まること
    #[test]
    fn test_sqrt2() {
        let xs = newton_root(&|x| x * x - 2.0, &|x| 2.0 * x, 2.0, 6);
        assert!((xs.last().unwrap() - 2f64.sqrt()).abs() < 1e-14);
    }

    // 二次収束: 誤差 e_{k+1} <= C e_k² を数値的に確認
    #[test]
    fn test_quadratic_convergence_rate() {
        let xs = newton_root(&|x| x * x - 2.0, &|x| 2.0 * x, 2.0, 4);
        let errs: Vec<f64> = xs.iter().map(|x| (x - 2f64.sqrt()).abs()).collect();
        for k in 0..3 {
            // e_{k+1} / e_k² が有界 (この問題では約 0.35)
            assert!(errs[k + 1] / (errs[k] * errs[k]) < 1.0);
        }
    }

    // 2 次関数は条件数によらず 1 反復で収束すること
    #[test]
    fn test_one_step_on_quadratic() {
        for kappa in [1.0, 100.0, 1e6] {
            let (x, history) = newton_optimize(
                &move |x: &[f64]| vec![x[0], kappa * x[1]],
                &move |_: &[f64]| {
                    Mat::from_fn(2, 2, |i, j| match (i, j) {
                        (0, 0) => 1.0,
                        (1, 1) => kappa,
                        _ => 0.0,
                    })
                },
                &[1.0, 1.0],
                1e-10,
                10,
            );
            assert!(norm(&x) < 1e-10);
            assert_eq!(history.len(), 2); // 初期評価 + 1 反復後の評価
        }
    }

    // 凸関数で停留点 (∇f = 0) に到達すること
    #[test]
    fn test_convex_function() {
        let grad = |x: &[f64]| {
            let (e1, e2, e3) =
                ((x[0] + x[1] - 1.0).exp(), (x[0] - x[1] - 1.0).exp(), (-x[0] - 1.0).exp());
            vec![e1 + e2 - e3, e1 - e2]
        };
        let hess = |x: &[f64]| {
            let (e1, e2, e3) =
                ((x[0] + x[1] - 1.0).exp(), (x[0] - x[1] - 1.0).exp(), (-x[0] - 1.0).exp());
            Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => e1 + e2 + e3,
                (1, 1) => e1 + e2,
                _ => e1 - e2,
            })
        };
        let (x, history) = newton_optimize(&grad, &hess, &[-1.0, 1.0], 1e-12, 50);
        assert!(norm(&grad(&x)) < 1e-12);
        assert!(history.len() < 20, "収束が遅すぎる: {} 反復", history.len());
    }
}