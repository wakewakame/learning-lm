// 三角測量 (docs/ai/10_triangulation.md)
//
// 姿勢が既知の複数のカメラに写った対応点から、その 3D 座標を復元する。
// 9 番 (点が既知でカメラが未知) のちょうど逆問題 (カメラが既知で点が未知)。
//
// - 線形三角測量 (DLT): 視線の条件を線形方程式に直し QR 分解で解く
//   (実装は共通部品 src/geometry.rs の triangulate_linear)
// - 中点法: 各視線への距離の 2 乗和を最小化する幾何的な解法
// - 非線形リファイン: 再投影誤差を LM 法で最小化 (9 番と同じ簡素版 LM)
// - 視差 (パララックス) が小さいと奥行きが急激に不確かになる退化を実験で観察する

use learning_lm::geometry::{
    camera_center, projection_jacobian, rodrigues, transform, triangulate_linear, Camera, View,
};
use learning_lm::{dot, norm, solve_linear, sub, Mat};
use rand::prelude::*;

/// 中点法。各視線 (カメラ中心 Cᵢ + 単位方向 d̂ᵢ) への距離の 2 乗和を最小化する点を返す。
/// 点 p から視線 i への距離² は ‖(I − d̂ᵢd̂ᵢᵀ)(p − Cᵢ)‖² なので、停留条件
///   Σᵢ (I − d̂ᵢd̂ᵢᵀ) p = Σᵢ (I − d̂ᵢd̂ᵢᵀ) Cᵢ
/// の 3×3 連立 1 次方程式を解けばよい (docs/ai/10 の 4 節)。
pub fn triangulate_midpoint(views: &[View], obs: &[[f64; 2]]) -> Vec<f64> {
    assert!(views.len() >= 2, "三角測量には 2 視点以上が必要");
    assert_eq!(views.len(), obs.len());
    let mut a = Mat::zeros(3, 3);
    let mut b = vec![0.0; 3];
    for (v, uv) in views.iter().zip(obs) {
        let [xh, yh] = v.camera.normalize(uv);
        // 視線方向 (ワールド座標): Rᵀ [x̂, ŷ, 1] を正規化
        let d = v.r.transpose().matvec(&[xh, yh, 1.0]);
        let dn = norm(&d);
        let d: Vec<f64> = d.iter().map(|x| x / dn).collect();
        let c = camera_center(&v.r, &v.t);
        for i in 0..3 {
            for j in 0..3 {
                let pij = (if i == j { 1.0 } else { 0.0 }) - d[i] * d[j];
                a[(i, j)] += pij;
                b[i] += pij * c[j];
            }
        }
    }
    solve_linear(&a, &b).expect("すべての視線が平行で交点が定まらない")
}

/// 点 p の再投影誤差の 2 乗和。カメラ背後に回ったら inf を返す
fn reprojection_cost(views: &[View], obs: &[[f64; 2]], p: &[f64]) -> f64 {
    let mut cost = 0.0;
    for (v, uv) in views.iter().zip(obs) {
        let pc = transform(&v.r, &v.t, p);
        if pc[2] <= 0.0 {
            return f64::INFINITY;
        }
        let proj = v.camera.project(&pc);
        cost += (proj[0] - uv[0]).powi(2) + (proj[1] - uv[1]).powi(2);
    }
    cost
}

/// 初期値 p0 から再投影誤差を LM 法で最小化して 3D 点を磨く (9 番と同じ簡素版 LM)。
/// 未知数は点の座標 3 つだけで、ヤコビ行列は ∂r/∂p = dproj · R (2×3) を積んだもの。
pub fn triangulate_refine(
    views: &[View],
    obs: &[[f64; 2]],
    p0: &[f64],
    max_iterations: usize,
) -> Vec<f64> {
    let m = views.len();
    let mut p = p0.to_vec();
    let mut lambda = 1e-3;
    for _ in 0..max_iterations {
        let mut residuals = vec![0.0; 2 * m];
        let mut j = Mat::zeros(2 * m, 3);
        let mut has_behind = false;
        for (i, (v, uv)) in views.iter().zip(obs).enumerate() {
            let pc = transform(&v.r, &v.t, &p);
            if pc[2] <= 0.0 {
                has_behind = true;
                break;
            }
            let proj = v.camera.project(&pc);
            residuals[2 * i] = proj[0] - uv[0];
            residuals[2 * i + 1] = proj[1] - uv[1];
            // ∂r/∂p = ∂π/∂Pc · ∂Pc/∂p = dproj · R (docs/ai/10 の 5 節)
            let jp = projection_jacobian(v.camera.focal, &pc).matmul(&v.r);
            for c in 0..3 {
                j[(2 * i, c)] = jp[(0, c)];
                j[(2 * i + 1, c)] = jp[(1, c)];
            }
        }
        if has_behind {
            break;
        }
        // (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr (9 番と同じ Marquardt 変種)
        let jt = j.transpose();
        let jtj = jt.matmul(&j);
        let jtr = jt.matvec(&residuals);
        let a = Mat::from_fn(3, 3, |i, k| {
            jtj[(i, k)] + if i == k { lambda * jtj[(i, i)] } else { 0.0 }
        });
        let rhs: Vec<f64> = jtr.iter().map(|g| -g).collect();
        let Some(delta) = solve_linear(&a, &rhs) else {
            break;
        };
        let p_new: Vec<f64> = p.iter().zip(&delta).map(|(a, b)| a + b).collect();
        let cost_old = dot(&residuals, &residuals);
        let cost_new = reprojection_cost(views, obs, &p_new);
        if cost_new < cost_old {
            p = p_new;
            lambda = (lambda * 0.1).clamp(1e-10, 1e10);
            if norm(&delta) < 1e-12 {
                break;
            }
        } else {
            lambda = (lambda * 10.0).clamp(1e-10, 1e10);
        }
    }
    p
}

/// 2 台のカメラ。cam0 は原点で +z 方向を向き、cam1 は x 方向に baseline だけ
/// 離れた位置からシーン中心 (z = 5 付近) の方を向く
fn make_two_views(baseline: f64) -> Vec<View> {
    let camera = Camera { focal: 800.0, cx: 320.0, cy: 240.0 };
    let view0 = View {
        camera: camera.clone(),
        r: Mat::identity(3),
        t: vec![0.0; 3],
    };
    let r1 = rodrigues(&[0.0, (baseline / 5.0).atan(), 0.0]);
    let t1: Vec<f64> = r1.matvec(&[baseline, 0.0, 0.0]).iter().map(|x| -x).collect();
    let view1 = View { camera, r: r1, t: t1 };
    vec![view0, view1]
}

/// 点 p を各カメラへ投影し、±noise のノイズを加えた観測を返す
fn observe(views: &[View], p: &[f64], noise: f64, rng: &mut impl Rng) -> Vec<[f64; 2]> {
    views
        .iter()
        .map(|v| {
            let proj = v.camera.project(&transform(&v.r, &v.t, p));
            [
                proj[0] + rng.random_range(-noise..noise.max(1e-300)),
                proj[1] + rng.random_range(-noise..noise.max(1e-300)),
            ]
        })
        .collect()
}

/// テスト用の点群 (シーン中心 z ≈ 5 の直方体内に一様分布)
fn make_points(seed: u64, n: usize) -> Vec<[f64; 3]> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            [
                rng.random_range(-1.0..1.0),
                rng.random_range(-1.0..1.0),
                rng.random_range(4.0..6.0),
            ]
        })
        .collect()
}

fn main() {
    let points = make_points(7, 100);

    // 1. ノイズなしなら線形三角測量・中点法とも厳密に復元できる
    println!("== 実験 1: ノイズなし (baseline = 1.0, 100 点) ==");
    let views = make_two_views(1.0);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0);
    let (mut max_lin, mut max_mid) = (0.0f64, 0.0f64);
    for p in &points {
        let obs = observe(&views, p, 0.0, &mut rng);
        max_lin = max_lin.max(norm(&sub(&triangulate_linear(&views, &obs), p)));
        max_mid = max_mid.max(norm(&sub(&triangulate_midpoint(&views, &obs), p)));
    }
    println!("3D 誤差の最大値: 線形 = {max_lin:.3e}, 中点法 = {max_mid:.3e}");

    // 2. ノイズあり: 3 手法の比較
    println!();
    println!("== 実験 2: 観測ノイズ ±0.5 px (baseline = 1.0, 100 点) ==");
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let (mut e3d, mut erep) = ([0.0f64; 3], [0.0f64; 3]);
    for p in &points {
        let obs = observe(&views, p, 0.5, &mut rng);
        let lin = triangulate_linear(&views, &obs);
        let mid = triangulate_midpoint(&views, &obs);
        let refined = triangulate_refine(&views, &obs, &lin, 50);
        for (k, est) in [&lin, &mid, &refined].iter().enumerate() {
            e3d[k] += norm(&sub(est, p));
            erep[k] += reprojection_cost(&views, &obs, est);
        }
    }
    let n = points.len() as f64;
    for (k, name) in ["線形 (DLT)", "中点法", "線形 + LM リファイン"].iter().enumerate() {
        println!(
            "{name}: 平均 3D 誤差 = {:.6}, 平均再投影誤差 E = {:.6}",
            e3d[k] / n,
            erep[k] / n
        );
    }

    // 3. 視差と精度: baseline を狭めると奥行き誤差が爆発する
    println!();
    println!("== 実験 3: baseline と 3D 誤差 (ノイズ ±0.5 px, 100 点, 線形 + LM リファイン) ==");
    println!("baseline | 平均 3D 誤差");
    for baseline in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05] {
        let views = make_two_views(baseline);
        let mut rng = rand::rngs::StdRng::seed_from_u64(2);
        let mut e = 0.0;
        for p in &points {
            let obs = observe(&views, p, 0.5, &mut rng);
            let est = triangulate_refine(&views, &obs, &triangulate_linear(&views, &obs), 50);
            e += norm(&sub(&est, p));
        }
        println!("{baseline:8.2} | {:.4}", e / n);
    }
    println!();
    println!("baseline を半分にすると誤差はほぼ 2 倍になる (視差に反比例)。");
    println!("2 本の視線が平行に近づくほど、交点の奥行きがノイズに敏感になるためである。");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ノイズなしなら線形三角測量・中点法とも厳密に復元すること
    #[test]
    fn test_exact_without_noise() {
        let views = make_two_views(1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for p in make_points(7, 20) {
            let obs = observe(&views, &p, 0.0, &mut rng);
            assert!(norm(&sub(&triangulate_linear(&views, &obs), &p)) < 1e-9);
            assert!(norm(&sub(&triangulate_midpoint(&views, &obs), &p)) < 1e-9);
        }
    }

    // 3 視点以上でも動くこと (2 視点より精度が上がる)
    #[test]
    fn test_multi_view() {
        let mut views = make_two_views(1.0);
        // 3 台目: 反対側 (x = −1) から
        let r2 = rodrigues(&[0.0, -(1.0f64 / 5.0).atan(), 0.0]);
        let t2: Vec<f64> = r2.matvec(&[-1.0, 0.0, 0.0]).iter().map(|x| -x).collect();
        views.push(View { camera: views[0].camera.clone(), r: r2, t: t2 });
        let points = make_points(7, 50);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let (mut e2, mut e3) = (0.0, 0.0);
        for p in &points {
            let obs = observe(&views, p, 0.5, &mut rng);
            e2 += norm(&sub(&triangulate_linear(&views[..2], &obs[..2]), p));
            e3 += norm(&sub(&triangulate_linear(&views, &obs), p));
        }
        assert!(e3 < e2, "3 視点 ({e3}) が 2 視点 ({e2}) より悪化した");
    }

    // LM リファインは再投影誤差を悪化させない (線形解より良いか同等)
    #[test]
    fn test_refine_improves_reprojection() {
        let views = make_two_views(1.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        for p in make_points(7, 20) {
            let obs = observe(&views, &p, 0.5, &mut rng);
            let lin = triangulate_linear(&views, &obs);
            let refined = triangulate_refine(&views, &obs, &lin, 50);
            let e_lin = reprojection_cost(&views, &obs, &lin);
            let e_ref = reprojection_cost(&views, &obs, &refined);
            assert!(e_ref <= e_lin * (1.0 + 1e-12), "E: {e_lin} -> {e_ref}");
        }
    }

    // baseline が狭いほど 3D 誤差が大きいこと (視差の退化)
    #[test]
    fn test_small_baseline_degrades() {
        let points = make_points(7, 50);
        let err_at = |baseline: f64| {
            let views = make_two_views(baseline);
            let mut rng = rand::rngs::StdRng::seed_from_u64(2);
            points
                .iter()
                .map(|p| {
                    let obs = observe(&views, p, 0.5, &mut rng);
                    norm(&sub(&triangulate_linear(&views, &obs), p))
                })
                .sum::<f64>()
        };
        assert!(err_at(0.05) > 5.0 * err_at(1.0));
    }
}
