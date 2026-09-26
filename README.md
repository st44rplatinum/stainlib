# stainlib

Stain deconvolution and normalisation for H&E histology images, in pure Rust.

- **HED** — split an image into haematoxylin, eosin and DAB
- **Macenko** — normalise stain colour
- **Reinhard** — normalise colour statistics

Every method is tested against the Python reference: scikit-image for HED,
torchstain for Macenko and Reinhard.

## Install

```toml
[dependencies]
stainlib = "0.1"
```

## Images

All functions take interleaved RGB bytes plus width and height:

```rust
let rgb: Vec<u8> = /* width * height * 3 bytes, RGBRGB... */;
```

## Separate stains (HED)

```rust
use stainlib::{rgb2hed, hed2rgb};

let hed = rgb2hed(&rgb, width, height)?;   // H, E, D per pixel
let back = hed2rgb(&hed, width, height)?;  // RGB in 0.0..=1.0
```

## Normalise with Macenko

```rust
use stainlib::{Macenko, MacenkoFit};

let macenko = Macenko::default();

let source = macenko.fit(&rgb, width, height)?;
let target = MacenkoFit::reference();       // or macenko.fit(&reference_image, ...)

let out = macenko.normalize(&rgb, width, height, &source, &target)?;
```

## Normalise with Reinhard

```rust
use stainlib::Reinhard;

let source = Reinhard.fit(&rgb, width, height)?;
let target = Reinhard.fit(&reference_image, ref_width, ref_height)?;

let out = Reinhard.normalize(&rgb, width, height, &source, &target)?;
```

## Whole slides

Fit once per slide — on a thumbnail or a few tissue tiles — and reuse that fit
for every tile. Fitting each tile separately gives each tile slightly
different colours, and the seams show.

## Errors

| error | meaning |
|---|---|
| `InvalidDimensions` | buffer length is not `width * height * 3` |
| `InsufficientTissue` | Macenko found almost no tissue (e.g. a blank tile) |
| `SingularMatrix` | the stain matrix could not be solved |

## Accuracy

| method | reference | agreement |
|---|---|---|
| HED | scikit-image 0.26 | 1e-12 |
| Macenko | torchstain 1.4.1 | stain vectors 1e-6, output within 1 grey level |
| Reinhard | torchstain 1.4.1 | output within 1 grey level |

Fixtures and how to regenerate them: `golden/README.md`.

## License

MIT or Apache-2.0, at your option.
