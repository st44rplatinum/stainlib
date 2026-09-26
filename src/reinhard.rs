//! Reinhard colour normalisation, matching torchstain 1.4.1 (numpy backend).

use crate::{validate_rgb, StainError};

const RGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.412453, 0.357580, 0.180423],
    [0.212671, 0.715160, 0.072169],
    [0.019334, 0.119193, 0.950227],
];

const XYZ_TO_RGB: [[f64; 3]; 3] = [
    [3.2404813432005266, -1.5371515162713185, -0.4985363261688878],
    [-0.9692549499965682, 1.8759900014898907, 0.04155592655829284],
    [0.05564663913517716, -0.20404133836651123, 1.0573110696453443],
];

const WHITE: [f64; 3] = [0.95047, 1.0, 1.08883];

/// Per-channel mean and standard deviation in LAB (L in 0..100).
#[derive(Debug, Clone, Copy)]
pub struct ReinhardFit {
    /// Channel means.
    pub mean: [f64; 3],
    /// Channel standard deviations.
    pub std: [f64; 3],
}

/// Reinhard normalisation.
#[derive(Debug, Clone, Copy, Default)]
pub struct Reinhard;

impl Reinhard {
    /// Fit channel statistics to an RGB image.
    pub fn fit(&self, rgb: &[u8], width: usize, height: usize) -> Result<ReinhardFit, StainError> {
        validate_rgb(rgb, width, height)?;
        if rgb.is_empty() {
            return Err(StainError::InvalidDimensions { width, height, buffer_len: 0 });
        }

        let lab: Vec<[f64; 3]> = rgb.chunks_exact(3).map(rgb_to_lab).collect();
        let n = lab.len() as f64;

        let mut mean = [0.0; 3];
        let mut std = [0.0; 3];
        for c in 0..3 {
            mean[c] = lab.iter().map(|p| p[c]).sum::<f64>() / n;
            // population sd, as np.std
            std[c] = (lab.iter().map(|p| (p[c] - mean[c]).powi(2)).sum::<f64>() / n).sqrt();
        }

        Ok(ReinhardFit { mean, std })
    }

    /// Map `rgb` from the source statistics onto the target statistics.
    pub fn normalize(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        source: &ReinhardFit,
        target: &ReinhardFit,
    ) -> Result<Vec<u8>, StainError> {
        validate_rgb(rgb, width, height)?;

        let mut out = vec![0u8; rgb.len()];
        for (src, dst) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
            let lab = rgb_to_lab(src);
            let mut mapped = [0.0; 3];
            for c in 0..3 {
                // a flat channel has nothing to scale
                let z = if source.std[c] > 0.0 { (lab[c] - source.mean[c]) / source.std[c] } else { 0.0 };
                mapped[c] = z * target.std[c] + target.mean[c];
            }
            let rgb = lab_to_rgb(mapped);
            for c in 0..3 {
                // torchstain truncates with astype(uint8)
                dst[c] = (rgb[c] * 255.0) as u8;
            }
        }

        Ok(out)
    }
}

/// sRGB bytes to LAB with L in 0..100 and a, b centred on 0.
fn rgb_to_lab(px: &[u8]) -> [f64; 3] {
    let lin = |x: u8| {
        let v = f64::from(x) / 255.0;
        if v > 0.04045 { ((v + 0.055) / 1.055).powf(2.4) } else { v / 12.92 }
    };
    let rgb = [lin(px[0]), lin(px[1]), lin(px[2])];

    let mut f = [0.0; 3];
    for i in 0..3 {
        let xyz = (0..3).map(|j| RGB_TO_XYZ[i][j] * rgb[j]).sum::<f64>() / WHITE[i];
        f[i] = if xyz > 0.008856 { xyz.cbrt() } else { 7.787 * xyz + 16.0 / 116.0 };
    }

    [116.0 * f[1] - 16.0, 500.0 * (f[0] - f[1]), 200.0 * (f[1] - f[2])]
}

/// LAB (as from `rgb_to_lab`) back to RGB in [0, 1].
fn lab_to_rgb(lab: [f64; 3]) -> [f64; 3] {
    let y = (lab[0] + 16.0) / 116.0;
    let f = [lab[1] / 500.0 + y, y, y - lab[2] / 200.0];

    let mut xyz = [0.0; 3];
    for i in 0..3 {
        let v = if f[i] > 0.2068966 { f[i].powi(3) } else { (f[i] - 16.0 / 116.0) / 7.787 };
        xyz[i] = v * WHITE[i];
    }

    let mut rgb = [0.0; 3];
    for i in 0..3 {
        let v: f64 = (0..3).map(|j| XYZ_TO_RGB[i][j] * xyz[j]).sum();
        let v = if v > 0.0031308 { 1.055 * v.powf(1.0 / 2.4) - 0.055 } else { v * 12.92 };
        rgb[i] = v.clamp(0.0, 1.0);
    }
    rgb
}
