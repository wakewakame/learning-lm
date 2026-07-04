// この実装は AI (Claude) が作成したもので、著者はまだレビューしていない。
//
// 最急降下法 (docs/ai/5_steepest_descent.md)
//
// - バックトラッキング直線探索付きの最急降下法を実装する
// - 収束の速さが条件数 κ に強く依存する (ジグザグする) ことを観察する
// - 非線形最小二乗 (指数減衰モデル) にも適用してみる

use learning_lm::norm;
use rand::prelude::*;

/// バックトラッキング直線探索付きの最急降下法。
/// ||∇f|| <= tol になるまで反復し、(解, 反復回数) を返す。
pub fn steepest_descent(
    f: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x0: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<f64>, usize) {
    let mut x = x0.to_vec();
    for iter in 0..max_iter {
        let g = grad(&x);
        let g_norm = norm(&g);
        if g_norm <= tol {
            return (x, iter);
        }
        // アルミホ条件を満たすまでステップ幅を半分にする
        let fx = f(&x);
        let c = 1e-4;
        let mut alpha = 1.0;
        loop {
            let x_new: Vec<f64> = x.iter().zip(&g).map(|(xi, gi)| xi - alpha * gi).collect();
            if f(&x_new) <= fx - c * alpha * g_norm * g_norm {
                x = x_new;
                break;
            }
            alpha *= 0.5;
            if alpha < 1e-20 {
                return (x, iter); // これ以上進めない
            }
        }
    }
    (x, max_iter)
}

fn main() {
    // 1. 条件数と収束速度の関係
    // f(x, y) = (x² + κ y²) / 2 は条件数 κ の 2 次関数。
    // κ が大きいほど等高線が細長くなり、ジグザグして収束が遅くなる。
    println!("== 条件数 κ と反復回数 (f = (x² + κy²)/2 を (1, 1) から) ==");
    println!("{:>8}  {:>8}", "κ", "反復回数");
    for kappa in [1.0, 10.0, 100.0, 1000.0] {
        let f = move |x: &[f64]| (x[0] * x[0] + kappa * x[1] * x[1]) / 2.0;
        let grad = move |x: &[f64]| vec![x[0], kappa * x[1]];
        let (_, iters) = steepest_descent(&f, &grad, &[1.0, 1.0], 1e-6, 1_000_000);
        println!("{kappa:>8}  {iters:>8}");
    }

    // 2. 非線形最小二乗への適用: y = β1 exp(β2 x) のフィッティング
    // (データ生成は 4_nonlinear_least_squares.rs と同じ設定。真値 β = (2.0, -1.5))
    println!();
    println!("== 指数減衰モデルのフィッティング (初期値 [1.0, -1.0]) ==");
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let n = 30;
    let xs: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 / (n - 1) as f64).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&x| 2.0 * (-1.5 * x).exp() + rng.random_range(-0.01..0.01))
        .collect();
    let e = {
        let (xs, ys) = (xs.clone(), ys.clone());
        move |b: &[f64]| -> f64 {
            xs.iter()
                .zip(&ys)
                .map(|(&x, &y)| {
                    let r = y - b[0] * (b[1] * x).exp();
                    r * r
                })
                .sum()
        }
    };
    // ∇E = 2 Jᵀ r を成分で直接書いたもの
    let grad_e = {
        let (xs, ys) = (xs.clone(), ys.clone());
        move |b: &[f64]| -> Vec<f64> {
            let mut g = vec![0.0, 0.0];
            for (&x, &y) in xs.iter().zip(&ys) {
                let ex = (b[1] * x).exp();
                let r = y - b[0] * ex;
                g[0] += 2.0 * r * (-ex);
                g[1] += 2.0 * r * (-b[0] * x * ex);
            }
            g
        }
    };
    let (beta, iters) = steepest_descent(&e, &grad_e, &[1.0, -1.0], 1e-8, 1_000_000);
    println!("推定値: {beta:.4?} (反復 {iters} 回, E = {:.6e})", e(&beta));
    println!("勾配だけでも解けるが、7 番のガウス・ニュートン法と反復回数を比べてみてほしい。");
}

#[cfg(test)]
mod tests {
    use super::*;

    // 2 次関数の最小点 (原点) に収束すること
    #[test]
    fn test_quadratic() {
        for kappa in [1.0, 10.0, 100.0] {
            let f = move |x: &[f64]| (x[0] * x[0] + kappa * x[1] * x[1]) / 2.0;
            let grad = move |x: &[f64]| vec![x[0], kappa * x[1]];
            let (x, _) = steepest_descent(&f, &grad, &[1.0, 1.0], 1e-8, 1_000_000);
            assert!(norm(&x) < 1e-6, "κ={kappa}: x={x:?}");
        }
    }

    // 条件数が大きいほど反復回数が増えること
    #[test]
    fn test_iterations_grow_with_condition_number() {
        let count = |kappa: f64| {
            let f = move |x: &[f64]| (x[0] * x[0] + kappa * x[1] * x[1]) / 2.0;
            let grad = move |x: &[f64]| vec![x[0], kappa * x[1]];
            steepest_descent(&f, &grad, &[1.0, 1.0], 1e-6, 1_000_000).1
        };
        assert!(count(1.0) < count(100.0));
        assert!(count(100.0) < count(10000.0));
    }
}