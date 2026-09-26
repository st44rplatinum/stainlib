//! Macenko stain normalisation, matching torchstain 1.4.1 (numpy backend).

use crate::{
    StainError,
    linalg::{covariance3, percentile, symmetric_eigen3},
    od::macenko_od,
    validate_rgb,
};

const IO: f64 = 240.0;
const ALPHA: f64 = 1.0;
const BETA: f64 = 0.15;

const HE_REF: [[f64; 2]; 3] = [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]];

const MAX_C_REF: [f64; 2] = [1.9705, 1.0308];

/// Fitted Macenko stain model.
#[derive(Debug, Clone, Copy)]
pub struct MacenkoFit {
    /// Estimated source H/E stain vectors.
    pub he: [[f64; 2]; 3],
    /// 99th-percentile stain concentrations.
    pub max_concentration: [f64; 2],
}

impl MacenkoFit {
    /// torchstain's built-in reference (`HERef`, `maxCRef`).
    pub fn reference() -> Self {
        Self {
            he: HE_REF,
            max_concentration: MAX_C_REF,
        }
    }
}

/// Macenko stain normalization.
#[derive(Debug, Clone, Copy)]
pub struct Macenko {
    /// Transmitted light intensity.
    pub io: f64,
    /// Percentile for the extreme stain angles.
    pub alpha: f64,
    /// Optical density below which a pixel counts as background.
    pub beta: f64,
}

impl Default for Macenko {
    fn default() -> Self {
        Self {
            io: IO,
            alpha: ALPHA,
            beta: BETA,
        }
    }
}

impl Macenko {
    /// Fit a stain model to an RGB image.
    pub fn fit(&self, rgb: &[u8], width: usize, height: usize) -> Result<MacenkoFit, StainError> {
        validate_rgb(rgb, width, height)?;

        let mut od = Vec::with_capacity(width * height);
        let mut tissue = Vec::with_capacity(width * height);

        for px in rgb.chunks_exact(3) {
            let x = [
                macenko_od(px[0], self.io),
                macenko_od(px[1], self.io),
                macenko_od(px[2], self.io),
            ];

            od.push(x);

            if x[0] >= self.beta && x[1] >= self.beta && x[2] >= self.beta {
                tissue.push(x);
            }
        }

        if tissue.len() < 3 {
            return Err(StainError::InsufficientTissue {
                available: tissue.len(),
                required: 3,
            });
        }

        let cov = covariance3(&tissue);
        let (mut eigenvalues, mut eigenvectors) =
            symmetric_eigen3(cov).ok_or(StainError::SingularMatrix)?;

        // Sort eigenpairs descending.
        for i in 0..3 {
            for j in (i + 1)..3 {
                if eigenvalues[j] > eigenvalues[i] {
                    eigenvalues.swap(i, j);

                    for k in 0..3 {
                        eigenvectors[k].swap(i, j);
                    }
                }
            }
        }

        // torchstain order: middle eigenvector, then largest, with the
        // largest on the y axis. Its projection has one sign for all tissue,
        // so the angles never straddle the ±ð wrap whatever sign the solver
        // gave it.
        let largest = [eigenvectors[0][0], eigenvectors[1][0], eigenvectors[2][0]];
        let middle = [eigenvectors[0][1], eigenvectors[1][1], eigenvectors[2][1]];

        let mut phi = Vec::with_capacity(tissue.len());
        for x in &tissue {
            let u = x[0] * middle[0] + x[1] * middle[1] + x[2] * middle[2];
            let w = x[0] * largest[0] + x[1] * largest[1] + x[2] * largest[2];
            phi.push(w.atan2(u));
        }

        let min_phi = percentile(&phi, self.alpha).ok_or(StainError::InvalidPercentile)?;
        let max_phi = percentile(&phi, 100.0 - self.alpha).ok_or(StainError::InvalidPercentile)?;

        let direction = |angle: f64| {
            [
                middle[0] * angle.cos() + largest[0] * angle.sin(),
                middle[1] * angle.cos() + largest[1] * angle.sin(),
                middle[2] * angle.cos() + largest[2] * angle.sin(),
            ]
        };
        let v_min = direction(min_phi);
        let v_max = direction(max_phi);

        let he = if v_min[0] > v_max[0] {
            [
                [v_min[0], v_max[0]],
                [v_min[1], v_max[1]],
                [v_min[2], v_max[2]],
            ]
        } else {
            [
                [v_max[0], v_min[0]],
                [v_max[1], v_min[1]],
                [v_max[2], v_min[2]],
            ]
        };

        let c = concentrations(&od, he)?;

        let mut max_c = [
            percentile(&c.0, 99.0).unwrap_or(0.0),
            percentile(&c.1, 99.0).unwrap_or(0.0),
        ];

        max_c[0] = max_c[0].max(1e-12);
        max_c[1] = max_c[1].max(1e-12);

        Ok(MacenkoFit {
            he,
            max_concentration: max_c,
        })
    }

    /// Normalize `rgb`, fitted as `source`, onto `target`.
    ///
    /// Use `MacenkoFit::reference()` as the target for torchstain's default,
    /// or `fit` a reference image. On a whole slide, fit `source` once per
    /// slide and reuse it for every tile.
    pub fn normalize(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
        source: &MacenkoFit,
        target: &MacenkoFit,
    ) -> Result<Vec<u8>, StainError> {
        validate_rgb(rgb, width, height)?;

        let od: Vec<[f64; 3]> = rgb
            .chunks_exact(3)
            .map(|px| {
                [
                    macenko_od(px[0], self.io),
                    macenko_od(px[1], self.io),
                    macenko_od(px[2], self.io),
                ]
            })
            .collect();

        let c = concentrations(&od, source.he)?;

        let scale = [
            source.max_concentration[0] / target.max_concentration[0],
            source.max_concentration[1] / target.max_concentration[1],
        ];

        let mut out = vec![0u8; rgb.len()];

        for (i, dst) in out.chunks_exact_mut(3).enumerate() {
            let c0 = c.0[i] / scale[0];
            let c1 = c.1[i] / scale[1];

            for ch in 0..3 {
                let od_new = target.he[ch][0] * c0 + target.he[ch][1] * c1;
                // torchstain caps at 255, then truncates with astype(uint8)
                dst[ch] = (self.io * (-od_new).exp()).min(255.0) as u8;
            }
        }

        Ok(out)
    }
}

fn concentrations(od: &[[f64; 3]], he: [[f64; 2]; 3]) -> Result<(Vec<f64>, Vec<f64>), StainError> {
    let a = he[0][0];
    let b = he[0][1];
    let c = he[1][0];
    let d = he[1][1];
    let e = he[2][0];
    let f = he[2][1];

    let ata00 = a * a + c * c + e * e;
    let ata01 = a * b + c * d + e * f;
    let ata11 = b * b + d * d + f * f;

    let det = ata00 * ata11 - ata01 * ata01;

    if det.abs() < 1e-15 {
        return Err(StainError::SingularMatrix);
    }

    let inv00 = ata11 / det;
    let inv01 = -ata01 / det;
    let inv11 = ata00 / det;

    let mut h = Vec::with_capacity(od.len());
    let mut e_conc = Vec::with_capacity(od.len());

    for x in od {
        let aty0 = a * x[0] + c * x[1] + e * x[2];
        let aty1 = b * x[0] + d * x[1] + f * x[2];

        h.push(inv00 * aty0 + inv01 * aty1);
        e_conc.push(inv01 * aty0 + inv11 * aty1);
    }

    Ok((h, e_conc))
}
