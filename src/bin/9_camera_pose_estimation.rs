// カメラ姿勢の推定 (docs/ai/9_camera_pose_estimation.md)
//
// 形状が既知の 3D モデルを写した写真から、カメラの位置と向き (回転 R と並進 t)
// を推定する。モデルの頂点とその写真上の位置 (スクリーン座標) の対応が最低 3 点
// あれば、再投影誤差の最小化として Levenberg-Marquardt 法で解ける (PnP 問題)。
//
// 8 番 (曲線フィッティング) との違い:
// - パラメータが回転行列を含むため、Rodrigues ベクトルによる
//   局所パラメータ化 (R ← dR·R) で更新する
// - ダンピングは Marquardt 変種 (JᵀJ + λ·diag(JᵀJ))
// - λ の更新はゲイン比を使わない古典的な「成功で 1/10、失敗で 10 倍」
// - dof_mask で最適化する自由度を選べる (例: 回転のみ)

use learning_lm::geometry::{
    camera_center, camera_frame_jacobian, rodrigues, rotation_error_deg, transform, Camera,
};
use learning_lm::{dot, norm, solve_linear, Mat};
use rand::prelude::*;

/// 現在の姿勢での再投影誤差の 2 乗和。点がカメラ背後に回ったら inf を返す
fn reprojection_cost(
    camera: &Camera,
    points_3d: &[[f64; 3]],
    points_2d: &[[f64; 2]],
    r: &Mat,
    t: &[f64],
) -> f64 {
    let mut cost = 0.0;
    for (p3, p2) in points_3d.iter().zip(points_2d) {
        let pc = transform(r, t, p3);
        if pc[2] <= 0.0 {
            return f64::INFINITY;
        }
        let proj = camera.project(&pc);
        cost += (proj[0] - p2[0]).powi(2) + (proj[1] - p2[1]).powi(2);
    }
    cost
}

/// 1 反復分の記録 (観察用)
pub struct LmTrace {
    pub lambda: f64,
    pub cost: f64,
    pub accepted: bool,
}

/// 最適化する自由度の選択 (パラメータ順: [wx, wy, wz, tx, ty, tz])
pub const DOF_ALL: [bool; 6] = [true; 6];
pub const DOF_ROTATION: [bool; 6] = [true, true, true, false, false, false];
pub const DOF_TRANSLATION: [bool; 6] = [false, false, false, true, true, true];

/// 初期姿勢 (r_init, t_init) から再投影誤差を LM 法で最小化し、
/// 推定した姿勢 ((R, t), 履歴) を返す。
/// dof_mask で false にした自由度は最適化しない (docs/ai/9 の 5 節)。
pub fn estimate_pose(
    camera: &Camera,
    points_3d: &[[f64; 3]],
    points_2d: &[[f64; 2]],
    r_init: &Mat,
    t_init: &[f64],
    max_iterations: usize,
    dof_mask: [bool; 6],
) -> ((Mat, Vec<f64>), Vec<LmTrace>) {
    assert_eq!(points_3d.len(), points_2d.len());
    let n = points_3d.len();
    assert!(n >= 1, "対応点が 1 点もない");

    // 有効な自由度のインデックス (= ヤコビ行列から抜き出す列番号) を収集
    let active: Vec<usize> = (0..6).filter(|&i| dof_mask[i]).collect();
    let ndof = active.len();
    if ndof == 0 {
        return ((r_init.clone(), t_init.to_vec()), vec![]);
    }
    assert!(2 * n >= ndof, "自由度 {ndof} に対して対応点 {n} 点では足りない");

    let mut r = r_init.clone();
    let mut t = t_init.to_vec();
    let mut lambda = 1e-3;
    let mut trace = vec![];

    for _ in 0..max_iterations {
        // 残差ベクトル (2n) とヤコビ行列 (2n × ndof) を組み立てる
        let mut residuals = vec![0.0; 2 * n];
        let mut j = Mat::zeros(2 * n, ndof);
        let mut has_behind = false;
        for (i, (p3, p2)) in points_3d.iter().zip(points_2d).enumerate() {
            let pc = transform(&r, &t, p3);
            if pc[2] <= 0.0 {
                has_behind = true;
                break;
            }
            let proj = camera.project(&pc);
            residuals[2 * i] = proj[0] - p2[0];
            residuals[2 * i + 1] = proj[1] - p2[1];
            let ji_full = camera_frame_jacobian(camera.focal, &pc);
            for (col, &param) in active.iter().enumerate() {
                j[(2 * i, col)] = ji_full[(0, param)];
                j[(2 * i + 1, col)] = ji_full[(1, param)];
            }
        }
        if has_behind {
            break; // 点がカメラ背後に回った (異常な姿勢)。現状の解を返す
        }

        // LM 更新: (JᵀJ + λ·diag(JᵀJ)) δ = −Jᵀr (Marquardt 変種、docs/ai/8 の 2.2 節)
        let jt = j.transpose();
        let jtj = jt.matmul(&j);
        let jtr = jt.matvec(&residuals);
        let a = Mat::from_fn(ndof, ndof, |i, k| {
            jtj[(i, k)] + if i == k { lambda * jtj[(i, i)] } else { 0.0 }
        });
        let rhs: Vec<f64> = jtr.iter().map(|g| -g).collect();
        let Some(delta_active) = solve_linear(&a, &rhs) else {
            break; // 数値的に解けない (通常は起きない)
        };

        // 6 自由度に展開 (マスクで固定した自由度は 0 のまま)
        let mut delta = [0.0; 6];
        for (col, &param) in active.iter().enumerate() {
            delta[param] = delta_active[col];
        }

        // カメラフレームの更新を適用: R_new = dR·R, t_new = dR·t + δt
        let dr = rodrigues(&delta[..3]);
        let r_new = dr.matmul(&r);
        let t_new: Vec<f64> = dr
            .matvec(&t)
            .iter()
            .zip(&delta[3..])
            .map(|(a, b)| a + b)
            .collect();

        let cost_old = dot(&residuals, &residuals);
        let cost_new = reprojection_cost(camera, points_3d, points_2d, &r_new, &t_new);

        if cost_new < cost_old {
            // 採択: λ を下げて次はガウス・ニュートン法寄りに
            r = r_new;
            t = t_new;
            lambda = (lambda * 0.1).clamp(1e-10, 1e10);
            trace.push(LmTrace { lambda, cost: cost_new, accepted: true });
            if norm(&delta_active) < 1e-10 {
                break; // 更新量が十分小さい
            }
        } else {
            // 棄却: λ を上げて小さく安全な一歩でやり直す
            lambda = (lambda * 10.0).clamp(1e-10, 1e10);
            trace.push(LmTrace { lambda, cost: cost_old, accepted: false });
        }
    }

    ((r, t), trace)
}

/// テスト用シーン: 真の姿勢 (R, t) で 3D モデルの頂点を投影し、観測ノイズを加える
fn make_scene(seed: u64, noise: f64) -> (Camera, Vec<[f64; 3]>, Vec<[f64; 2]>, Mat, Vec<f64>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let camera = Camera { focal: 800.0, cx: 320.0, cy: 240.0 };
    let r_true = rodrigues(&[0.1, -0.2, 0.15]);
    let t_true = vec![0.1, -0.05, 0.2];
    let n = 20;
    let mut points_3d = vec![];
    let mut points_2d = vec![];
    for _ in 0..n {
        let p3 = [
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(3.0..6.0),
        ];
        let pc = transform(&r_true, &t_true, &p3);
        let proj = camera.project(&pc);
        points_3d.push(p3);
        points_2d.push([
            proj[0] + rng.random_range(-noise..noise.max(1e-300)),
            proj[1] + rng.random_range(-noise..noise.max(1e-300)),
        ]);
    }
    (camera, points_3d, points_2d, r_true, t_true)
}

fn main() {
    let (camera, points_3d, points_2d, r_true, t_true) = make_scene(42, 0.2);

    let print_trace = |trace: &[LmTrace]| {
        for (k, t) in trace.iter().enumerate() {
            println!(
                "反復 {k}: E = {:.6e}, λ = {:.3e} {}",
                t.cost,
                t.lambda,
                if t.accepted { "" } else { "(棄却)" }
            );
        }
    };
    let print_errors = |r: &Mat, t: &[f64]| {
        let dt: f64 = norm(&t.iter().zip(&t_true).map(|(a, b)| a - b).collect::<Vec<_>>());
        println!(
            "回転誤差 = {:.4}°, 並進誤差 = {:.6}",
            rotation_error_deg(r, &r_true),
            dt
        );
    };

    // 1. 全 6 自由度: 少しずれた初期値から推定する
    println!("== 実験 1: 全 6 自由度 (初期値は回転 約 5°、並進 0.2 ほどのずれ) ==");
    let r_init = rodrigues(&[0.05, -0.04, 0.06]).matmul(&r_true);
    let t_init = vec![t_true[0] + 0.1, t_true[1] - 0.08, t_true[2] + 0.15];
    print!("初期値: ");
    print_errors(&r_init, &t_init);
    let ((r, t), trace) =
        estimate_pose(&camera, &points_3d, &points_2d, &r_init, &t_init, 100, DOF_ALL);
    print_trace(&trace);
    print!("最適化後: ");
    print_errors(&r, &t);

    // 2. dof_mask で回転のみ最適化 (並進は正しい値で固定)
    println!();
    println!("== 実験 2: 回転のみ最適化 (dof_mask = DOF_ROTATION、並進は真値で固定) ==");
    let ((r2, t2), trace2) =
        estimate_pose(&camera, &points_3d, &points_2d, &r_init, &t_true, 100, DOF_ROTATION);
    print_trace(&trace2);
    println!("最適化後: 回転誤差 = {:.4}°", rotation_error_deg(&r2, &r_true));
    // t 自体は dR·t で動くが、カメラ中心 C = −Rᵀt は不変 (docs/ai/9 の 5 節)
    println!("カメラ中心 (初期値):   {:.6?}", camera_center(&r_init, &t_true));
    println!("カメラ中心 (最適化後): {:.6?}", camera_center(&r2, &t2));

    // 3. 大きくずれた初期値 (回転 約 30°、並進 1.0 ほど) からでも収束する
    println!();
    println!("== 実験 3: 大きくずれた初期値 (回転 約 30°、並進 約 1.0 のずれ) ==");
    let r_bad = rodrigues(&[0.3, -0.25, 0.3]).matmul(&r_true);
    let t_bad = vec![t_true[0] + 0.5, t_true[1] + 0.6, t_true[2] - 0.7];
    print!("初期値: ");
    print_errors(&r_bad, &t_bad);
    let ((r3, t3), trace3) =
        estimate_pose(&camera, &points_3d, &points_2d, &r_bad, &t_bad, 100, DOF_ALL);
    print_trace(&trace3);
    print!("最適化後: ");
    print_errors(&r3, &t3);
    println!();
    println!("採択のたびに λ が 1/10 になってガウス・ニュートン法の速さに移行し、");
    println!("予測が外れた反復では λ が 10 倍に引き上げられて (棄却) 慎重にやり直す。");
}

#[cfg(test)]
mod tests {
    use super::*;

    // rodrigues が回転行列 (RᵀR = I, det R = 1) を返すこと
    #[test]
    fn test_rodrigues_is_rotation() {
        let r = rodrigues(&[0.3, -0.5, 0.7]);
        let rtr = r.transpose().matmul(&r);
        assert!(rtr.sub(&Mat::identity(3)).frobenius_norm() < 1e-12);
        let det = r[(0, 0)] * (r[(1, 1)] * r[(2, 2)] - r[(1, 2)] * r[(2, 1)])
            - r[(0, 1)] * (r[(1, 0)] * r[(2, 2)] - r[(1, 2)] * r[(2, 0)])
            + r[(0, 2)] * (r[(1, 0)] * r[(2, 1)] - r[(1, 1)] * r[(2, 0)]);
        assert!((det - 1.0).abs() < 1e-12);
    }

    // ノイズなしデータなら真の姿勢にほぼ厳密に収束すること
    #[test]
    fn test_converges_exactly_without_noise() {
        let (camera, p3, p2, r_true, t_true) = make_scene(42, 0.0);
        let r_init = rodrigues(&[0.05, -0.04, 0.06]).matmul(&r_true);
        let t_init = vec![t_true[0] + 0.1, t_true[1] - 0.08, t_true[2] + 0.15];
        let ((r, t), _) = estimate_pose(&camera, &p3, &p2, &r_init, &t_init, 100, DOF_ALL);
        assert!(rotation_error_deg(&r, &r_true) < 1e-6);
        for k in 0..3 {
            assert!((t[k] - t_true[k]).abs() < 1e-7, "t[{k}] = {}", t[k]);
        }
    }

    // ノイズありでも真の姿勢の近くに収束すること
    #[test]
    fn test_converges_with_noise() {
        let (camera, p3, p2, r_true, t_true) = make_scene(42, 0.2);
        let r_init = rodrigues(&[0.3, -0.25, 0.3]).matmul(&r_true);
        let t_init = vec![t_true[0] + 0.5, t_true[1] + 0.6, t_true[2] - 0.7];
        let ((r, t), _) = estimate_pose(&camera, &p3, &p2, &r_init, &t_init, 100, DOF_ALL);
        assert!(rotation_error_deg(&r, &r_true) < 0.1);
        for k in 0..3 {
            assert!((t[k] - t_true[k]).abs() < 0.01, "t[{k}] = {}", t[k]);
        }
    }

    // 対応点が最低 3 点あれば全 6 自由度を推定できること
    #[test]
    fn test_minimum_3_points() {
        let (camera, p3, p2, r_true, t_true) = make_scene(42, 0.0);
        let r_init = rodrigues(&[0.05, -0.04, 0.06]).matmul(&r_true);
        let t_init = vec![t_true[0] + 0.1, t_true[1] - 0.08, t_true[2] + 0.15];
        let ((r, t), _) = estimate_pose(&camera, &p3[..3], &p2[..3], &r_init, &t_init, 100, DOF_ALL);
        assert!(rotation_error_deg(&r, &r_true) < 1e-6);
        for k in 0..3 {
            assert!((t[k] - t_true[k]).abs() < 1e-7, "t[{k}] = {}", t[k]);
        }
    }

    // dof_mask で並進を固定すると、カメラ中心 C = −Rᵀt が動かないこと
    // (t 自体は t_new = dR·t で更新されるが、C は不変になる。docs/ai/9 の 5 節)
    #[test]
    fn test_dof_mask_keeps_camera_center_fixed() {
        let (camera, p3, p2, r_true, t_true) = make_scene(42, 0.2);
        let r_init = rodrigues(&[0.05, -0.04, 0.06]).matmul(&r_true);
        let ((r, t), _) = estimate_pose(&camera, &p3, &p2, &r_init, &t_true, 100, DOF_ROTATION);
        let c_init = camera_center(&r_init, &t_true);
        let c_result = camera_center(&r, &t);
        for k in 0..3 {
            assert!((c_result[k] - c_init[k]).abs() < 1e-9, "C[{k}] = {}", c_result[k]);
        }
        assert!(rotation_error_deg(&r, &r_true) < rotation_error_deg(&r_init, &r_true));
    }

    // 採択された反復ではコストが単調減少すること
    #[test]
    fn test_accepted_steps_decrease_cost() {
        let (camera, p3, p2, r_true, t_true) = make_scene(42, 0.2);
        let r_init = rodrigues(&[0.3, -0.25, 0.3]).matmul(&r_true);
        let t_init = vec![t_true[0] + 0.5, t_true[1] + 0.6, t_true[2] - 0.7];
        let (_, trace) = estimate_pose(&camera, &p3, &p2, &r_init, &t_init, 100, DOF_ALL);
        let costs: Vec<f64> = trace.iter().filter(|t| t.accepted).map(|t| t.cost).collect();
        for k in 1..costs.len() {
            assert!(costs[k] <= costs[k - 1] * (1.0 + 1e-12));
        }
    }

    // 自由度に対して対応点が足りないとき panic すること (2 点 × 2 = 4 < 6 自由度)
    #[test]
    #[should_panic(expected = "足りない")]
    fn test_insufficient_points() {
        let (camera, p3, p2, r_true, t_true) = make_scene(42, 0.0);
        estimate_pose(&camera, &p3[..2], &p2[..2], &r_true, &t_true, 10, DOF_ALL);
    }
}
