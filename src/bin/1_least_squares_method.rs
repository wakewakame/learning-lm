// 最小二乗法
//
// `[(x1, y1), (x2, y2), ...]` のような入力に対して最もフィットする $y=ax+b$ の `(a, b)` を返す。
fn least_squares_method(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let xy_avg = points.iter().map(|(x, y)| x * y).sum::<f64>() / n;
    let x_avg = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_avg = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let x2_avg = points.iter().map(|(x, _)| x * x).sum::<f64>() / n;
    let a = (xy_avg - x_avg * y_avg) / (x2_avg - x_avg * x_avg);
    let b = y_avg - a * x_avg;
    (a, b)
}

fn main() {
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    // 10 パターンでテスト
    for _ in 0..10 {
        // 適当な a, b を生成
        let a: f64 = rng.random_range(-10.0..10.0);
        let b: f64 = rng.random_range(-100.0..100.0);

        // ノイズを加えたデータ点を生成
        let points: Vec<(f64, f64)> = (0..1000)
            .map(|_| {
                let x = rng.random_range(-100.0..100.0);
                let y = a * x + b + rng.random_range(-1f64..1f64);
                (x, y)
            })
            .collect();

        // 最小二乗法で a, b を推定
        let (a_est, b_est) = least_squares_method(&points);

        // 推定値が真の値に近いことを確認
        println!(
            "a: {:+.5}, b: {:+.5}, a-a_est: {:+.5}, b-b_est: {:+.5}",
            a,
            b,
            a - a_est,
            b - b_est,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    #[test]
    fn test_least_squares_method() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        // 10 パターンでテスト
        for _ in 0..10 {
            // 適当な a, b を生成
            let a: f64 = rng.random_range(-10.0..10.0);
            let b: f64 = rng.random_range(-100.0..100.0);

            // ノイズを加えたデータ点を生成
            let points: Vec<(f64, f64)> = (0..1000)
                .map(|_| {
                    let x = rng.random_range(-100.0..100.0);
                    let y = a * x + b + rng.random_range(-1f64..1f64);
                    (x, y)
                })
                .collect();

            // 最小二乗法で a, b を推定
            let (a_est, b_est) = least_squares_method(&points);

            // 推定値が真の値に近いことを確認
            assert!((a - a_est).abs() < 0.1);
            assert!((b - b_est).abs() < 0.1);
        }
    }
}
