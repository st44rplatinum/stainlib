//! Golden tests against scikit-image and torchstain. Fixtures come from
//! golden/make_golden.py; versions are in golden/manifest.json.

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::read_npy;
use stainlib::{hed2rgb, rgb2hed, Macenko, MacenkoFit, Reinhard, StainError};

fn path(name: &str) -> String {
    format!("{}/golden/{name}.npy", env!("CARGO_MANIFEST_DIR"))
}

fn image(name: &str) -> (Vec<u8>, usize, usize) {
    let a: Array3<u8> = read_npy(path(name)).expect(name);
    let (h, w, _) = a.dim();
    (a.iter().copied().collect(), w, h)
}

fn floats3(name: &str) -> Vec<f64> {
    let a: Array3<f64> = read_npy(path(name)).expect(name);
    a.iter().copied().collect()
}

fn matrix(name: &str) -> [[f64; 2]; 3] {
    let a: Array2<f64> = read_npy(path(name)).expect(name);
    [[a[[0, 0]], a[[0, 1]]], [a[[1, 0]], a[[1, 1]]], [a[[2, 0]], a[[2, 1]]]]
}

fn vector2(name: &str) -> [f64; 2] {
    let a: Array1<f64> = read_npy(path(name)).expect(name);
    [a[0], a[1]]
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn column(m: &[[f64; 2]; 3], j: usize) -> [f64; 3] {
    [m[0][j], m[1][j], m[2][j]]
}

fn cosine(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot: f64 = (0..3).map(|i| a[i] * b[i]).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

#[test]
fn hed_matches_scikit_image() {
    let (rgb, w, h) = image("hed_input");
    let ours = rgb2hed(&rgb, w, h).unwrap();
    let diff = max_abs_diff(&ours, &floats3("hed_expected"));
    assert!(diff <= 1e-12, "max abs diff {diff}");
}

#[test]
fn hed_round_trip_matches_scikit_image() {
    let (_, w, h) = image("hed_input");
    let ours = hed2rgb(&floats3("hed_expected"), w, h).unwrap();
    let diff = max_abs_diff(&ours, &floats3("hed_roundtrip"));
    assert!(diff <= 1e-12, "max abs diff {diff}");
}

#[test]
fn macenko_recovers_synthetic_stains() {
    // torchstain itself gets 1.0000 and 0.9995 on this image
    let (rgb, w, h) = image("macenko_source");
    let fit = Macenko::default().fit(&rgb, w, h).unwrap();
    let truth = matrix("macenko_true_he");

    for (j, stain) in ["H", "E"].iter().enumerate() {
        let c = cosine(column(&fit.he, j), column(&truth, j));
        assert!(c >= 0.999, "{stain} cosine {c}");
    }
}

#[test]
fn macenko_fit_matches_torchstain() {
    for name in ["source", "target"] {
        let (rgb, w, h) = image(&format!("macenko_{name}"));
        let fit = Macenko::default().fit(&rgb, w, h).unwrap();

        let he = matrix(&format!("macenko_{name}_he"));
        let he_diff = max_abs_diff(
            &fit.he.concat(),
            &he.concat(),
        );
        assert!(he_diff <= 1e-6, "{name} HE diff {he_diff}");

        let max_c = vector2(&format!("macenko_{name}_max_c"));
        for k in 0..2 {
            let rel = (fit.max_concentration[k] - max_c[k]).abs() / max_c[k];
            assert!(rel <= 1e-6, "{name} maxC[{k}] relative diff {rel}");
        }
    }
}

#[test]
fn macenko_normalize_matches_torchstain() {
    let macenko = Macenko::default();
    let (source, w, h) = image("macenko_source");
    let (target, tw, th) = image("macenko_target");

    // our own fits end to end, not torchstain's numbers
    let source_fit = macenko.fit(&source, w, h).unwrap();
    let target_fit = macenko.fit(&target, tw, th).unwrap();
    let ours = macenko.normalize(&source, w, h, &source_fit, &target_fit).unwrap();

    let (expected, _, _) = image("macenko_normalized");
    let worst = ours
        .iter()
        .zip(&expected)
        .map(|(a, b)| (i16::from(*a) - i16::from(*b)).abs())
        .max()
        .unwrap();
    // truncation to u8 can land either side of a boundary
    assert!(worst <= 1, "worst channel difference {worst}");
}

#[test]
fn macenko_reference_target_is_torchstain_default() {
    let r = MacenkoFit::reference();
    assert_eq!(r.he[0], [0.5626, 0.2159]);
    assert_eq!(r.max_concentration, [1.9705, 1.0308]);
}

#[test]
fn macenko_blank_tile_errors() {
    let white = vec![255u8; 16 * 16 * 3];
    let err = Macenko::default().fit(&white, 16, 16).unwrap_err();
    assert!(matches!(err, StainError::InsufficientTissue { .. }), "{err:?}");
}

#[test]
fn wrong_buffer_size_errors() {
    let rgb = vec![128u8; 10];
    assert!(matches!(
        Macenko::default().fit(&rgb, 4, 4),
        Err(StainError::InvalidDimensions { .. })
    ));
    assert!(matches!(rgb2hed(&rgb, 4, 4), Err(StainError::InvalidDimensions { .. })));
}

fn stats(name: &str) -> ([f64; 3], [f64; 3]) {
    let a: Array2<f64> = read_npy(path(name)).expect(name);
    ([a[[0, 0]], a[[0, 1]], a[[0, 2]]], [a[[1, 0]], a[[1, 1]], a[[1, 2]]])
}

#[test]
fn reinhard_fit_matches_torchstain() {
    // torchstain works in float32, so agree to about 1e-4 in LAB units
    for name in ["source", "target"] {
        let (rgb, w, h) = image(&format!("macenko_{name}"));
        let fit = Reinhard.fit(&rgb, w, h).unwrap();
        let (mean, std) = stats(&format!("reinhard_{name}_stats"));
        for c in 0..3 {
            assert!((fit.mean[c] - mean[c]).abs() <= 1e-3, "{name} mean[{c}] {} vs {}", fit.mean[c], mean[c]);
            assert!((fit.std[c] - std[c]).abs() <= 1e-3, "{name} std[{c}] {} vs {}", fit.std[c], std[c]);
        }
    }
}

#[test]
fn reinhard_normalize_matches_torchstain() {
    let (source, w, h) = image("macenko_source");
    let (target, tw, th) = image("macenko_target");
    let source_fit = Reinhard.fit(&source, w, h).unwrap();
    let target_fit = Reinhard.fit(&target, tw, th).unwrap();
    let ours = Reinhard.normalize(&source, w, h, &source_fit, &target_fit).unwrap();

    let (expected, _, _) = image("reinhard_normalized");
    let worst = ours
        .iter()
        .zip(&expected)
        .map(|(a, b)| (i16::from(*a) - i16::from(*b)).abs())
        .max()
        .unwrap();
    assert!(worst <= 1, "worst channel difference {worst}");
}

#[test]
fn reinhard_identity_when_source_is_target() {
    let (rgb, w, h) = image("macenko_target");
    let fit = Reinhard.fit(&rgb, w, h).unwrap();
    let out = Reinhard.normalize(&rgb, w, h, &fit, &fit).unwrap();
    let worst = out.iter().zip(&rgb).map(|(a, b)| (i16::from(*a) - i16::from(*b)).abs()).max().unwrap();
    assert!(worst <= 1, "worst channel difference {worst}");
}
