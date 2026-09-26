//! Stain deconvolution and normalisation for RGB histology images.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod linalg;
mod od;

pub mod hed;
pub mod macenko;
pub mod reinhard;

pub use hed::{hed2rgb, rgb2hed};
pub use macenko::{Macenko, MacenkoFit};
pub use reinhard::{Reinhard, ReinhardFit};

/// Errors returned by stain operations.
#[derive(Debug, Clone, PartialEq)]
pub enum StainError {
    /// Width multiplied by height does not match the RGB buffer length.
    InvalidDimensions {
        /// Width in pixels.
        width: usize,
        /// Height in pixels.
        height: usize,
        /// Length of the buffer that was passed.
        buffer_len: usize,
    },

    /// An operation requires enough tissue pixels but the image does not
    /// contain enough pixels after background rejection.
    InsufficientTissue {
        /// Tissue pixels found.
        available: usize,
        /// Tissue pixels needed.
        required: usize,
    },

    /// A matrix could not be inverted or solved.
    SingularMatrix,

    /// The input contained a non-finite floating-point value.
    NonFiniteInput,

    /// The requested percentile could not be computed.
    InvalidPercentile,
}

impl core::fmt::Display for StainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions {
                width,
                height,
                buffer_len,
            } => write!(
                f,
                "RGB buffer length {buffer_len} does not match {width}x{height}"
            ),
            Self::InsufficientTissue {
                available,
                required,
            } => write!(
                f,
                "insufficient tissue pixels: {available}, need at least {required}"
            ),
            Self::SingularMatrix => write!(f, "singular matrix"),
            Self::NonFiniteInput => write!(f, "input contains non-finite values"),
            Self::InvalidPercentile => write!(f, "invalid percentile"),
        }
    }
}

impl std::error::Error for StainError {}

fn validate_rgb(
    rgb: &[u8],
    width: usize,
    height: usize,
) -> Result<(), StainError> {
    let expected = width
        .checked_mul(height)
        .and_then(|n| n.checked_mul(3))
        .unwrap_or(usize::MAX);

    if rgb.len() != expected {
        return Err(StainError::InvalidDimensions {
            width,
            height,
            buffer_len: rgb.len(),
        });
    }

    Ok(())
}

#[cfg(feature = "image")]
impl From<image::RgbImage> for RgbBuffer {
    fn from(image: image::RgbImage) -> Self {
        // read size before into_raw moves the image
        let width = image.width() as usize;
        let height = image.height() as usize;
        Self {
            data: image.into_raw(),
            width,
            height,
        }
    }
}

/// A borrowed RGB image.
#[derive(Debug, Clone, Copy)]
pub struct RgbView<'a> {
    /// Interleaved RGB bytes.
    pub data: &'a [u8],
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl<'a> RgbView<'a> {
    /// Construct an RGB view after validating dimensions.
    pub fn new(
        data: &'a [u8],
        width: usize,
        height: usize,
    ) -> Result<Self, StainError> {
        validate_rgb(data, width, height)?;
        Ok(Self {
            data,
            width,
            height,
        })
    }
}

/// An owned RGB image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbBuffer {
    /// Interleaved RGB bytes.
    pub data: Vec<u8>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl RgbBuffer {
    /// Create an RGB buffer after validating dimensions.
    pub fn new(
        data: Vec<u8>,
        width: usize,
        height: usize,
    ) -> Result<Self, StainError> {
        validate_rgb(&data, width, height)?;
        Ok(Self {
            data,
            width,
            height,
        })
    }
}

