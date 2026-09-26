//! HED colour deconvolution, matching scikit-image's `rgb2hed` / `hed2rgb`.

use crate::{linalg::inverse3, validate_rgb, StainError};

/// scikit-image's `rgb_from_hed`. Rows are not normalised.
const RGB_FROM_HED: [[f64; 3]; 3] = [
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
];

/// Floor applied before the log, as in scikit-image.
const MIN: f64 = 1e-6;

/// Interleaved RGB to interleaved H, E, D stain values.
///
/// Same maths as `skimage.color.rgb2hed`: scale to [0, 1], floor at 1e-6,
/// natural log divided by ln(1e-6), times `hed_from_rgb`, clip below at 0.
pub fn rgb2hed(rgb: &[u8], width: usize, height: usize) -> Result<Vec<f64>, StainError> {
    validate_rgb(rgb, width, height)?;

    let hed_from_rgb = inverse3(RGB_FROM_HED).ok_or(StainError::SingularMatrix)?;
    let log_min = MIN.ln();
    let mut out = vec![0.0; rgb.len()];

    for (src, dst) in rgb.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        let x = [
            (f64::from(src[0]) / 255.0).max(MIN).ln() / log_min,
            (f64::from(src[1]) / 255.0).max(MIN).ln() / log_min,
            (f64::from(src[2]) / 255.0).max(MIN).ln() / log_min,
        ];

        // row vector times matrix
        for j in 0..3 {
            dst[j] = (x[0] * hed_from_rgb[0][j]
                + x[1] * hed_from_rgb[1][j]
                + x[2] * hed_from_rgb[2][j])
                .max(0.0);
        }
    }

    Ok(out)
}

/// Interleaved H, E, D stain values back to RGB in [0, 1].
///
/// Same maths as `skimage.color.hed2rgb`, which returns floats rather than
/// bytes: `exp((stains · rgb_from_hed) · ln(1e-6))`, clipped to [0, 1].
pub fn hed2rgb(hed: &[f64], width: usize, height: usize) -> Result<Vec<f64>, StainError> {
    let expected = width.checked_mul(height).and_then(|n| n.checked_mul(3));
    if expected != Some(hed.len()) {
        return Err(StainError::InvalidDimensions {
            width,
            height,
            buffer_len: hed.len(),
        });
    }
    if hed.iter().any(|v| !v.is_finite()) {
        return Err(StainError::NonFiniteInput);
    }

    let log_min = MIN.ln();
    let mut out = vec![0.0; hed.len()];

    for (src, dst) in hed.chunks_exact(3).zip(out.chunks_exact_mut(3)) {
        for c in 0..3 {
            let s = src[0] * RGB_FROM_HED[0][c]
                + src[1] * RGB_FROM_HED[1][c]
                + src[2] * RGB_FROM_HED[2][c];
            dst[c] = (s * log_min).exp().clamp(0.0, 1.0);
        }
    }

    Ok(out)
}
