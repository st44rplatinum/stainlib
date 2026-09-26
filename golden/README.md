# Golden fixtures

These fixtures are generated, not hand-authored.

## References

- scikit-image 0.26.0 for HED
- torchstain 1.4.1 for Macenko and Reinhard

Run:

    python golden/make_golden.py

Existing fixtures are never overwritten unless:

    python golden/make_golden.py --update

Needs numpy, scikit-image and torchstain 1.4.1. Note torchstain 1.4.1
reports `__version__ == "1.3.0"`; the manifest records the real version from
package metadata.

## Images

All fixtures are 64x64 synthetic H&E made with Beer-Lambert from known stain
vectors, fixed seeds, no third-party image material. `macenko_true_he.npy`
holds the vectors each synthetic source was built from, so Macenko is checked
against ground truth as well as against torchstain.

No real image yet. One with a redistribution-compatible licence would add a
check the synthetic tiles cannot; record its attribution and source URL here.

## Tolerances

| test | tolerance |
|---|---|
| HED, both directions | 1e-12 absolute |
| Macenko fit vs torchstain | 1e-6 on stain vectors and max concentrations |
| Macenko vs ground truth | cosine >= 0.999 (torchstain gets 1.0000 / 0.9995) |
| Macenko normalised image | 1 grey level |
