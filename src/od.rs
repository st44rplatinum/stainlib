/// Optical density as torchstain computes it: `-ln((x + 1) / io)`.
#[inline]
pub(crate) fn macenko_od(x: u8, io: f64) -> f64 {
    -((f64::from(x) + 1.0) / io).ln()
}
