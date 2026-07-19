// バンドル調整 (docs/ai/11_bundle_adjustment.md)
//
// 複数のカメラの姿勢と 3D 点の座標を「全部未知」として、全観測の再投影誤差を
// 一斉に LM 法で最小化する。9 番 (点が既知) と 10 番 (カメラが既知) の集大成で、
// ヤコビ行列は両者の部品 (カメラブロック 2×6 と点ブロック 2×3) を並べたもの。
//
// - シーン全体を回転・並進・スケールしても E が変わらない「ゲージ自由度」(7 自由度)
//   があるため、最初のカメラを固定して 6 自由度を殺す。残るスケール 1 自由度は
//   JᵀJ をランク落ちさせるが、LM 法のダンピングのおかげで方程式は常に解ける
//   (docs/ai/8 の保証 1 の実演)
// - 初期値は「摂動した姿勢 + それによる線形三角測量 (10 番)」で作る

use learning_lm::geometry::{
    camera_center, camera_frame_jacobian, projection_jacobian, rodrigues, rotation_error_deg,
    transform, triangulate_linear, Camera, View,
};
use learning_lm::{dot, norm, solve_linear, sub, Mat};
use rand::prelude::*;

/// 観測: カメラ cam に 3D 点 point がスクリーン座標 uv で写っている
pub struct Observation {
    pub cam: usize,
    pub point: usize,
    pub uv: [f64; 2],
}

/// 1 反復分の記録 (観察用)
pub struct LmTrace {
    pub lambda: f64,
    pub cost: f64,
    pub accepted: bool,
}

/// 全観測の再投影誤差の 2 乗和。点がカメラ背後に回ったら inf を返す
fn total_cost(views: &[View], points: &[[f64; 3]], observations: &[Observation]) -> f64 {
    let mut cost = 0.0;
    for o in observations {
        let v = &views[o.cam];
        let pc = transform(&v.r, &v.t, &points[o.point]);
        if pc[2] <= 0.0 {
            return f64::INFINITY;
        }
        let proj = v.camera.project(&pc);
        cost += (proj[0] - o.uv[0]).powi(2) + (proj[1] - o.uv[1]).powi(2);
    }
    cost
}

/// バンドル調整。初期値 (views_init, points_init) から全観測の再投影誤差を
/// LM 法で最小化し、((姿勢, 点群), 履歴) を返す。
/// fix_first_camera = true のとき最初のカメラを動かさない (ゲージの固定、
/// docs/ai/11 の 5 節)。false にするとゲージ 7 自由度がまるごと残り、
/// JᵀJ がランク落ちするが、ダンピングのおかげでそれでも収束する。
pub fn bundle_adjust(
    views_init: &[View],
    points_init: &[[f64; 3]],
    observations: &[Observation],
    max_iterations: usize,
    fix_first_camera: bool,
) -> ((Vec<View>, Vec<[f64; 3]>), Vec<LmTrace>) {
    let m = views_init.len();
    let np = points_init.len();
    let fixed = if fix_first_camera { 1 } else { 0 };
    let cam_cols = 6 * (m - fixed);
    let ncols = cam_cols + 3 * np;
    let nrows = 2 * observations.len();
    assert!(nrows >= ncols, "未知数 {ncols} に対して観測 {} 本では足りない", nrows);

    let mut views = views_init.to_vec();
    let mut points = points_init.to_vec();
    let mut lambda = 1e-3;
    let mut trace = vec![];

    for _ in 0..max_iterations {
        // 残差ベクトル (2×観測数) とヤコビ行列を組み立てる。
        // 列の並びは [カメラ fixed.. の 6 列ずつ | 点 0.. の 3 列ずつ]。
        // 観測 (cam, point) の 2 行は、そのカメラと点の列にしか値を持たない
        // (ブロック疎構造。docs/ai/11 の 3 節)
        let mut residuals = vec![0.0; nrows];
        let mut j = Mat::zeros(nrows, ncols);
        let mut has_behind = false;
        for (k, o) in observations.iter().enumerate() {
            let v = &views[o.cam];
            let pc = transform(&v.r, &v.t, &points[o.point]);
            if pc[2] <= 0.0 {
                has_behind = true;
                break;
            }
            let proj = v.camera.project(&pc);
            residuals[2 * k] = proj[0] - o.uv[0];
            residuals[2 * k + 1] = proj[1] - o.uv[1];
            // カメラブロック (2×6): 9 番の camera_frame_jacobian そのまま
            if o.cam >= fixed {
                let jc = camera_frame_jacobian(v.camera.focal, &pc);
                let base = 6 * (o.cam - fixed);
                for c in 0..6 {
                    j[(2 * k, base + c)] = jc[(0, c)];
                    j[(2 * k + 1, base + c)] = jc[(1, c)];
                }
            }
            // 点ブロック (2×3): 10 番のリファインと同じ dproj · R
            let jp = projection_jacobian(v.camera.focal, &pc).matmul(&v.r);
            let base = cam_cols + 3 * o.point;
            for c in 0..3 {
                j[(2 * k, base + c)] = jp[(0, c)];
                j[(2 * k + 1, base + c)] = jp[(1, c)];
            }
        }
        if has_behind {
            break;
        }

        // (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr (9 番と同じ Marquardt 変種)
        let jt = j.transpose();
        let jtj = jt.matmul(&j);
        let jtr = jt.matvec(&residuals);
        let a = Mat::from_fn(ncols, ncols, |i, k| {
            jtj[(i, k)] + if i == k { lambda * jtj[(i, i)] } else { 0.0 }
        });
        let rhs: Vec<f64> = jtr.iter().map(|g| -g).collect();
        let Some(delta) = solve_linear(&a, &rhs) else {
            break;
        };

        // 候補: カメラは 9 番と同じ増分更新、点は素直な加算
        let mut views_new = views.clone();
        for i in fixed..m {
            let base = 6 * (i - fixed);
            let dr = rodrigues(&delta[base..base + 3]);
            views_new[i].r = dr.matmul(&views[i].r);
            views_new[i].t = dr
                .matvec(&views[i].t)
                .iter()
                .zip(&delta[base + 3..base + 6])
                .map(|(a, b)| a + b)
                .collect();
        }
        let points_new: Vec<[f64; 3]> = points
            .iter()
            .enumerate()
            .map(|(pi, p)| {
                let base = cam_cols + 3 * pi;
                [p[0] + delta[base], p[1] + delta[base + 1], p[2] + delta[base + 2]]
            })
            .collect();

        let cost_old = dot(&residuals, &residuals);
        let cost_new = total_cost(&views_new, &points_new, observations);

        if cost_new < cost_old {
            views = views_new;
            points = points_new;
            lambda = (lambda * 0.1).clamp(1e-10, 1e10);
            trace.push(LmTrace { lambda, cost: cost_new, accepted: true });
            if norm(&delta) < 1e-10 {
                break;
            }
        } else {
            lambda = (lambda * 10.0).clamp(1e-10, 1e10);
            trace.push(LmTrace { lambda, cost: cost_old, accepted: false });
        }
    }

    ((views, points), trace)
}

/// 復元結果と真値の間のスケール係数 s = argmin Σ‖s·p_est − p_true‖² を推定する。
/// 単眼の再構成はスケールが原理的に決まらない (docs/ai/11 の 5 節) ため、
/// 評価はスケールを合わせてから行う。cam0 (原点) を固定した場合、残る
/// ゲージはこの 1 自由度だけである
fn estimate_scale(points_est: &[[f64; 3]], points_true: &[[f64; 3]]) -> f64 {
    let num: f64 = points_est.iter().zip(points_true).map(|(e, t)| dot(e, t)).sum();
    let den: f64 = points_est.iter().map(|e| dot(e, e)).sum();
    num / den
}

/// テスト用シーン: 3 台のカメラ (cam0 は原点で固定の基準) と 25 個の 3D 点。
/// (真の姿勢, 真の点群, ノイズ付き観測) を返す
fn make_scene(seed: u64, noise: f64) -> (Vec<View>, Vec<[f64; 3]>, Vec<Observation>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let camera = Camera { focal: 800.0, cx: 320.0, cy: 240.0 };
    // cam0 は原点で +z を向く基準カメラ。cam1, cam2 は左右からシーン中心を向く
    let mut views = vec![View { camera: camera.clone(), r: Mat::identity(3), t: vec![0.0; 3] }];
    for (w, c) in [
        ([0.05, 0.29, 0.02], [1.5, 0.2, 0.0]),
        ([-0.03, -0.25, 0.05], [-1.2, -0.3, 0.5]),
    ] {
        let r = rodrigues(&w);
        let t: Vec<f64> = r.matvec(&c).iter().map(|x| -x).collect();
        views.push(View { camera: camera.clone(), r, t });
    }
    let points: Vec<[f64; 3]> = (0..25)
        .map(|_| {
            [
                rng.random_range(-1.5..1.5),
                rng.random_range(-1.5..1.5),
                rng.random_range(4.0..7.0),
            ]
        })
        .collect();
    let mut observations = vec![];
    for (ci, v) in views.iter().enumerate() {
        for (pi, p) in points.iter().enumerate() {
            let pc = transform(&v.r, &v.t, p);
            assert!(pc[2] > 0.0, "生成したシーンで点がカメラ背後にある");
            let proj = v.camera.project(&pc);
            observations.push(Observation {
                cam: ci,
                point: pi,
                uv: [
                    proj[0] + rng.random_range(-noise..noise.max(1e-300)),
                    proj[1] + rng.random_range(-noise..noise.max(1e-300)),
                ],
            });
        }
    }
    (views, points, observations)
}

/// 初期値: cam0 は真値のまま、cam1, cam2 の姿勢を摂動し (回転 約 2°、並進 約 0.1)、
/// その誤った姿勢とノイズ付き観測から線形三角測量 (10 番) で点群を作る
fn make_initial(
    views_true: &[View],
    observations: &[Observation],
    n_points: usize,
) -> (Vec<View>, Vec<[f64; 3]>) {
    let mut views = views_true.to_vec();
    for (i, (dw, dt)) in [
        ([0.02, -0.03, 0.015], [0.08, -0.05, 0.06]),
        ([-0.025, 0.02, -0.02], [-0.06, 0.09, 0.05]),
    ]
    .iter()
    .enumerate()
    {
        let v = &mut views[i + 1];
        let dr = rodrigues(dw);
        v.r = dr.matmul(&v.r);
        v.t = dr.matvec(&v.t).iter().zip(dt).map(|(a, b)| a + b).collect();
    }
    let points: Vec<[f64; 3]> = (0..n_points)
        .map(|pi| {
            let (vs, obs): (Vec<View>, Vec<[f64; 2]>) = observations
                .iter()
                .filter(|o| o.point == pi)
                .map(|o| (views[o.cam].clone(), o.uv))
                .unzip();
            let p = triangulate_linear(&vs, &obs);
            [p[0], p[1], p[2]]
        })
        .collect();
    (views, points)
}

/// スケールを合わせた上での点群の RMS 誤差とカメラの誤差を表示する
fn print_errors(views: &[View], points: &[[f64; 3]], views_true: &[View], points_true: &[[f64; 3]]) {
    let s = estimate_scale(points, points_true);
    let rms = (points
        .iter()
        .zip(points_true)
        .map(|(e, t)| {
            let d = [s * e[0] - t[0], s * e[1] - t[1], s * e[2] - t[2]];
            dot(&d, &d)
        })
        .sum::<f64>()
        / points.len() as f64)
        .sqrt();
    println!("スケール s = {s:.6}, 点群 RMS 誤差 (スケール合わせ後) = {rms:.6}");
    for (i, (v, vt)) in views.iter().zip(views_true).enumerate() {
        let c: Vec<f64> = camera_center(&v.r, &v.t).iter().map(|x| s * x).collect();
        let ct = camera_center(&vt.r, &vt.t);
        println!(
            "cam{i}: 回転誤差 = {:.4}°, 中心誤差 (スケール合わせ後) = {:.6}",
            rotation_error_deg(&v.r, &vt.r),
            norm(&sub(&c, &ct))
        );
    }
}

fn main() {
    let (views_true, points_true, observations) = make_scene(42, 0.2);
    let (views_init, points_init) = make_initial(&views_true, &observations, points_true.len());

    println!("== 実験 1: バンドル調整 (cam0 固定, カメラ 3 台, 点 25 個, ノイズ ±0.2 px) ==");
    println!("初期値 (摂動した姿勢 + 線形三角測量):");
    print_errors(&views_init, &points_init, &views_true, &points_true);
    println!("初期 E = {:.6e}", total_cost(&views_init, &points_init, &observations));
    let ((views, points), trace) =
        bundle_adjust(&views_init, &points_init, &observations, 100, true);
    for (k, t) in trace.iter().enumerate() {
        println!(
            "反復 {k}: E = {:.6e}, λ = {:.3e} {}",
            t.cost,
            t.lambda,
            if t.accepted { "" } else { "(棄却)" }
        );
    }
    println!("最適化後:");
    print_errors(&views, &points, &views_true, &points_true);

    println!();
    println!("== 実験 2: ゲージを固定しない (cam0 も動かす) ==");
    let ((views2, points2), trace2) =
        bundle_adjust(&views_init, &points_init, &observations, 100, false);
    let accepted = trace2.iter().filter(|t| t.accepted).count();
    println!(
        "最終 E = {:.6e} ({} 反復, 採択 {accepted})",
        trace2.last().map(|t| t.cost).unwrap_or(f64::NAN),
        trace2.len()
    );
    print_errors(&views2, &points2, &views_true, &points_true);
    println!();
    println!("E は実験 1 と同じ下限まで下がるのに、cam0 が真値からずれている。");
    println!("シーン全体が一緒に動いた (ゲージ自由度) だけで、再投影誤差は変わらない。");
    println!("JᵀJ は 7 自由度ぶんランク落ちしているが、ダンピングのおかげで解けている。");
}

#[cfg(test)]
mod tests {
    use super::*;

    // ノイズなしなら真のシーンをスケールを除いて厳密に復元すること
    #[test]
    fn test_exact_without_noise() {
        let (views_true, points_true, observations) = make_scene(42, 0.0);
        let (views_init, points_init) = make_initial(&views_true, &observations, points_true.len());
        let ((views, points), _) =
            bundle_adjust(&views_init, &points_init, &observations, 100, true);
        assert!(total_cost(&views, &points, &observations) < 1e-12);
        // 回転はスケールの影響を受けないので直接比較できる
        for (v, vt) in views.iter().zip(&views_true) {
            assert!(rotation_error_deg(&v.r, &vt.r) < 1e-5);
        }
        // 点群はスケールを合わせれば一致する
        let s = estimate_scale(&points, &points_true);
        for (e, t) in points.iter().zip(&points_true) {
            for k in 0..3 {
                assert!((s * e[k] - t[k]).abs() < 1e-6);
            }
        }
    }

    // ノイズありでも E が大きく下がり、真のシーンの近くに収束すること
    #[test]
    fn test_converges_with_noise() {
        let (views_true, points_true, observations) = make_scene(42, 0.2);
        let (views_init, points_init) = make_initial(&views_true, &observations, points_true.len());
        let e_init = total_cost(&views_init, &points_init, &observations);
        let ((views, points), _) =
            bundle_adjust(&views_init, &points_init, &observations, 100, true);
        let e_final = total_cost(&views, &points, &observations);
        assert!(e_final < e_init / 100.0, "E: {e_init} -> {e_final}");
        let s = estimate_scale(&points, &points_true);
        for (e, t) in points.iter().zip(&points_true) {
            for k in 0..3 {
                assert!((s * e[k] - t[k]).abs() < 0.02);
            }
        }
        for (v, vt) in views.iter().zip(&views_true) {
            assert!(rotation_error_deg(&v.r, &vt.r) < 0.1);
        }
    }

    // cam0 を固定すると cam0 の姿勢が変化しないこと
    #[test]
    fn test_first_camera_fixed() {
        let (views_true, points_true, observations) = make_scene(42, 0.2);
        let (views_init, points_init) = make_initial(&views_true, &observations, points_true.len());
        let ((views, _), _) = bundle_adjust(&views_init, &points_init, &observations, 30, true);
        assert!(views[0].r.sub(&views_init[0].r).frobenius_norm() == 0.0);
        assert_eq!(views[0].t, views_init[0].t);
    }

    // ゲージを固定しなくても (JᵀJ がランク落ちしていても) ダンピングのおかげで収束すること
    #[test]
    fn test_converges_without_gauge_fix() {
        let (views_true, points_true, observations) = make_scene(42, 0.2);
        let (views_init, points_init) = make_initial(&views_true, &observations, points_true.len());
        let e_init = total_cost(&views_init, &points_init, &observations);
        let ((views, points), _) =
            bundle_adjust(&views_init, &points_init, &observations, 100, false);
        let e_final = total_cost(&views, &points, &observations);
        assert!(e_final < e_init / 100.0, "E: {e_init} -> {e_final}");
    }

    // 採択された反復では E が単調減少すること
    #[test]
    fn test_accepted_steps_decrease_cost() {
        let (views_true, points_true, observations) = make_scene(42, 0.2);
        let (views_init, points_init) = make_initial(&views_true, &observations, points_true.len());
        let (_, trace) = bundle_adjust(&views_init, &points_init, &observations, 100, true);
        let costs: Vec<f64> = trace.iter().filter(|t| t.accepted).map(|t| t.cost).collect();
        for k in 1..costs.len() {
            assert!(costs[k] <= costs[k - 1] * (1.0 + 1e-12));
        }
    }
}
