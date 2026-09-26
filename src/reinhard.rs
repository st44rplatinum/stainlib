//! Reinhard colour normalisation.

use crate::StainError;

/// Per-channel mean and standard deviation in LAB.
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
    pub fn fit(
        &self,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> Result<ReinhardFit, StainError> {
        let _ = (rgb, width, height);
        todo!("Reinhard::fit")
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
        let _ = (rgb, width, height, source, target);
        todo!("Reinhard::normalize")
    }
}
