"""Generate golden fixtures for the Rust tests.

References are pinned and recorded in manifest.json. Never overwrites
existing fixtures unless --update is given.

    python golden/make_golden.py --update
"""

import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import numpy as np
import skimage
from skimage.color import hed2rgb, rgb2hed
import torchstain

ROOT = Path(__file__).parent
IO = 240.0
SIZE = 64


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def synthetic_he(h_vec, e_vec, seed):
    """Beer-Lambert H&E tile from known stain vectors.

    Nuclei (mostly H), stroma (mostly E), mixed, and background. Inverts
    torchstain's OD = -ln((I + 1) / Io), so the true vectors are exact.
    """
    rng = np.random.default_rng(seed)
    he = np.stack([unit(h_vec), unit(e_vec)], axis=1)  # 3x2

    n = SIZE * SIZE
    kind = rng.choice(4, size=n, p=[0.25, 0.35, 0.3, 0.1])
    c = np.zeros((2, n))
    c[0, kind == 0] = rng.uniform(0.8, 1.6, (kind == 0).sum())   # nuclei
    c[1, kind == 0] = rng.uniform(0.0, 0.1, (kind == 0).sum())
    c[0, kind == 1] = rng.uniform(0.0, 0.1, (kind == 1).sum())   # stroma
    c[1, kind == 1] = rng.uniform(0.5, 1.2, (kind == 1).sum())
    c[:, kind == 2] = rng.uniform(0.2, 0.9, (2, (kind == 2).sum()))  # mixed
    # kind 3 stays zero: background

    od = he @ c
    rgb = IO * np.exp(-od) - 1.0
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return rgb.T.reshape(SIZE, SIZE, 3).copy(), he


def save(name, array):
    np.save(ROOT / f"{name}.npy", np.ascontiguousarray(array))


def generate_hed():
    rgb, _ = synthetic_he([0.65, 0.70, 0.29], [0.07, 0.99, 0.11], seed=1)
    # extremes: the 1e-6 floor and pure white
    rgb[0, 0] = [0, 0, 0]
    rgb[0, 1] = [255, 255, 255]
    rgb[0, 2] = [0, 255, 128]

    hed = rgb2hed(rgb)
    save("hed_input", rgb)
    save("hed_expected", hed.astype(np.float64))
    save("hed_roundtrip", hed2rgb(hed).astype(np.float64))


def compute_matrices(normalizer, rgb):
    # torchstain keeps this private; the name-mangled call is the only way
    # to get the fitted source HE and maxC it uses inside normalize()
    he, _, max_c = normalizer._NumpyMacenkoNormalizer__compute_matrices(rgb, IO, 1, 0.15)
    return he, max_c


def generate_macenko():
    source, true_he = synthetic_he([0.60, 0.75, 0.28], [0.10, 0.95, 0.30], seed=2)
    target, _ = synthetic_he([0.65, 0.70, 0.29], [0.07, 0.99, 0.11], seed=3)

    # an attribute, not an importable submodule, in 1.4.1
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend="numpy")
    source_he, source_max = compute_matrices(normalizer, source)
    target_he, target_max = compute_matrices(normalizer, target)

    normalizer.fit(target)
    normalized, _, _ = normalizer.normalize(source, stains=False)

    save("macenko_source", source)
    save("macenko_target", target)
    save("macenko_true_he", true_he)
    save("macenko_source_he", source_he)
    save("macenko_source_max_c", source_max)
    save("macenko_target_he", target_he)
    save("macenko_target_max_c", target_max)
    save("macenko_normalized", normalized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    manifest_path = ROOT / "manifest.json"
    if manifest_path.exists() and not args.update:
        raise SystemExit("fixtures exist; use --update to regenerate")

    generate_hed()
    generate_macenko()

    # written last, so a crash above leaves nothing claiming to be complete
    manifest = {
        "generator": "golden/make_golden.py",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_image": skimage.__version__,
        # torchstain's own __version__ says 1.3.0 in the 1.4.1 release
        "torchstain": metadata.version("torchstain"),
        "references": {
            "hed": "skimage.color.rgb2hed / hed2rgb",
            "macenko": "torchstain.normalizers.MacenkoNormalizer(backend='numpy')",
        },
        "macenko": {"io": IO, "alpha": 1, "beta": 0.15},
        "images": f"{SIZE}x{SIZE} synthetic Beer-Lambert H&E, fixed seeds",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote fixtures and manifest.json")


if __name__ == "__main__":
    main()
