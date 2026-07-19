// カメラ幾何の共有部品 (docs/ai/9 以降のサンプルコードで共用)。
//
// - `Camera` / `View`: ピンホールカメラの内部パラメータと、姿勢付きカメラ
// - `rodrigues`: Rodrigues ベクトル → 回転行列 (docs/ai/9 の 3.2 節)
// - `projection_jacobian` / `camera_frame_jacobian`: 投影の微分 (2×3) と
//   姿勢 6 自由度の微分 (2×6) (docs/ai/9 の 4 節)
// - `triangulate_linear`: 線形三角測量 (docs/ai/10 の 3 節)

use crate::{lstsq_qr, norm, Mat};

/// ピンホールカメラの内部パラメータ (歪みなし)
#[derive(Clone)]
pub struct Camera {
    pub focal: f64,
    pub cx: f64,
    pub cy: f64,
}

impl Camera {
    /// カメラ座標 Pc をスクリーン座標へ投影: u = f X/Z + cx, v = f Y/Z + cy
    pub fn project(&self, pc: &[f64]) -> [f64; 2] {
        [
            self.focal * pc[0] / pc[2] + self.cx,
            self.focal * pc[1] / pc[2] + self.cy,
        ]
    }

    /// スクリーン座標を正規化画像座標 x̂ = (u − cx)/f, ŷ = (v − cy)/f へ戻す
    pub fn normalize(&self, uv: &[f64; 2]) -> [f64; 2] {
        [(uv[0] - self.cx) / self.focal, (uv[1] - self.cy) / self.focal]
    }
}

/// 姿勢付きカメラ。ワールド座標の点 p はカメラ座標 Pc = R p + t に写る
#[derive(Clone)]
pub struct View {
    pub camera: Camera,
    pub r: Mat,
    pub t: Vec<f64>,
}

/// Rodrigues ベクトル (回転軸 k × 回転角 θ) から回転行列へ変換。
/// R = I + sinθ K + (1 − cosθ) K²,  K = [k]× (docs/ai/9 の 3.2 節)
pub fn rodrigues(w: &[f64]) -> Mat {
    let theta = norm(w);
    if theta < 1e-12 {
        return Mat::identity(3);
    }
    let k: Vec<f64> = w.iter().map(|x| x / theta).collect();
    #[rustfmt::skip]
    let kx = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 1) => -k[2], (0, 2) =>  k[1],
        (1, 0) =>  k[2], (1, 2) => -k[0],
        (2, 0) => -k[1], (2, 1) =>  k[0],
        _ => 0.0,
    });
    let kx2 = kx.matmul(&kx);
    Mat::from_fn(3, 3, |i, j| {
        (if i == j { 1.0 } else { 0.0 }) + theta.sin() * kx[(i, j)] + (1.0 - theta.cos()) * kx2[(i, j)]
    })
}

/// Pc = R p + t (ワールド座標からカメラ座標への変換)
pub fn transform(r: &Mat, t: &[f64], p: &[f64]) -> Vec<f64> {
    let rp = r.matvec(p);
    rp.iter().zip(t).map(|(a, b)| a + b).collect()
}

/// 投影の微分 d(u,v)/d(Pc) (2×3)。
/// u = f X/Z + cx, v = f Y/Z + cy の偏微分 (docs/ai/9 の 4 節の部品 1)
pub fn projection_jacobian(focal: f64, pc: &[f64]) -> Mat {
    let (x, y, z) = (pc[0], pc[1], pc[2]);
    let zinv = 1.0 / z;
    #[rustfmt::skip]
    let dproj = Mat::from_fn(2, 3, |i, j| match (i, j) {
        (0, 0) => focal * zinv,
        (0, 2) => -focal * x * zinv * zinv,
        (1, 1) => focal * zinv,
        (1, 2) => -focal * y * zinv * zinv,
        _ => 0.0,
    });
    dproj
}

/// カメラフレームでの姿勢 6 自由度のヤコビアン (2×6) を計算する。
/// パラメータ順: [δwx, δwy, δwz, δtx, δty, δtz] (すべてカメラ座標系)
/// 摂動モデル: Pc_new = rodrigues(δw)·Pc + δt ≈ Pc + δw×Pc + δt (docs/ai/9 の 4 節)
pub fn camera_frame_jacobian(focal: f64, pc: &[f64]) -> Mat {
    let (x, y, z) = (pc[0], pc[1], pc[2]);
    let dproj = projection_jacobian(focal, pc);
    // d(Pc)/d(δw) = −[Pc]× (δw × Pc を δw で微分したもの)
    #[rustfmt::skip]
    let skew = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 1) =>  z, (0, 2) => -y,
        (1, 0) => -z, (1, 2) =>  x,
        (2, 0) =>  y, (2, 1) => -x,
        _ => 0.0,
    });
    let jw = dproj.matmul(&skew);
    // 並進は d(Pc)/d(δt) = I なので dproj がそのまま並ぶ
    Mat::from_fn(2, 6, |i, j| if j < 3 { jw[(i, j)] } else { dproj[(i, j - 3)] })
}

/// カメラ中心 C = −Rᵀt (ワールド座標でのカメラの位置)
pub fn camera_center(r: &Mat, t: &[f64]) -> Vec<f64> {
    r.transpose().matvec(t).iter().map(|x| -x).collect()
}

/// 2 つの回転行列のずれを角度 (度) で返す: R_a R_bᵀ の回転角
pub fn rotation_error_deg(ra: &Mat, rb: &Mat) -> f64 {
    let rel = ra.matmul(&rb.transpose());
    let tr = rel[(0, 0)] + rel[(1, 1)] + rel[(2, 2)];
    ((tr - 1.0) / 2.0).clamp(-1.0, 1.0).acos().to_degrees()
}

/// 線形三角測量 (DLT)。姿勢既知の 2 台以上のカメラの観測から 3D 点を復元する。
/// 各視点の「観測方向と点の方向が一致する」条件
///   x̂ (r₃ᵀp + t₃) = r₁ᵀp + t₁,  ŷ (r₃ᵀp + t₃) = r₂ᵀp + t₂
/// を p について整理して縦に積み、線形最小二乗として QR 分解で解く
/// (docs/ai/10 の 3 節。rᵢᵀ は R の第 i 行)。
pub fn triangulate_linear(views: &[View], obs: &[[f64; 2]]) -> Vec<f64> {
    assert!(views.len() >= 2, "三角測量には 2 視点以上が必要");
    assert_eq!(views.len(), obs.len());
    let m = views.len();
    let mut a = Mat::zeros(2 * m, 3);
    let mut b = vec![0.0; 2 * m];
    for (i, (v, uv)) in views.iter().zip(obs).enumerate() {
        let [xh, yh] = v.camera.normalize(uv);
        for k in 0..3 {
            a[(2 * i, k)] = xh * v.r[(2, k)] - v.r[(0, k)];
            a[(2 * i + 1, k)] = yh * v.r[(2, k)] - v.r[(1, k)];
        }
        b[2 * i] = v.t[0] - xh * v.t[2];
        b[2 * i + 1] = v.t[1] - yh * v.t[2];
    }
    lstsq_qr(&a, &b)
}
