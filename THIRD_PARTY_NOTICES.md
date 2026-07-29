# Third-Party Notices

Colortina combines original application code with selected third-party model
architectures, inference helpers, weights, and implementation ideas. This file
summarizes the known upstream projects and license notes. It is not legal
advice; downstream users should review the upstream repositories and package
metadata before redistribution or commercial use.

## Vendored, Adapted, or Closely Referenced Code

| Component | Upstream | How it is used | License / notice |
| --- | --- | --- | --- |
| Manga colorization core | [qweasdd/manga-colorization-v2](https://github.com/qweasdd/manga-colorization-v2) | Core generator/colorizer architecture, inference flow, and official generator/denoiser weight compatibility. | Upstream repository does not expose a license file in the GitHub listing checked for this notice; treat usage and redistribution with care and preserve attribution. |
| Manual hint workflow | [qweasdd/manga-colorization](https://github.com/qweasdd/manga-colorization) | Manual color-hint interaction and early web-demo workflow ideas. | Upstream repository does not expose a license file in the GitHub listing checked for this notice; treat usage and redistribution with care and preserve attribution. |
| Guided hint and panel workflow ideas | [vikast908/ColorComic](https://github.com/vikast908/ColorComic) | Hint-point API design, guided automatic hints, panel/block inference ideas. | MIT License. See `LICENSE_ColorComic_MIT.txt`. |
| Weight compatibility and release ideas | [xiaogdgenuine/Manga-Colorization-FJ](https://github.com/xiaogdgenuine/Manga-Colorization-FJ) | Compatibility handling for different checkpoint containers and release/runtime adaptation ideas. | Preserve upstream attribution; review upstream terms before redistributing derived code. |
| Structural line extraction | [ljsabc/MangaLineExtraction_PyTorch](https://github.com/ljsabc/MangaLineExtraction_PyTorch) | `vendor/manga_line_extraction/model.py` keeps the inference network layout compatible with the official `erika.pth` checkpoint. | MIT License. |
| Real-ESRGAN anime upscaler | [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Anime6B super-resolution checkpoint and inference behavior. | BSD-3-Clause License. |
| RRDBNet / restoration architecture | [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) | `vendor/realesrgan_min/rrdbnet.py` is a minimal torch-only RRDBNet compatible with Real-ESRGAN anime6B weights. | Apache-2.0 License. |
| FFDNet denoiser | [cszn/FFDNet](https://github.com/cszn/FFDNet), IPOL FFDNet-related releases | `vendor/manga_colorization_v2/denoising/` contains FFDNet-style denoiser code inherited through the manga-colorization lineage. | The included source header states GPLv3-or-later. Preserve GPL notices and comply with GPL obligations when redistributing. |
| Anime face cascade | [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) | Optional OpenCV-based anime/manga face fallback detector. | MIT License. See `LICENSE_lbpcascade_animeface_MIT.txt`. |

## Models and Weights

- `generator.zip` and denoiser weights are expected to follow the
  manga-colorization-v2 / manga-colorization release formats.
- `erika.pth` is used with the MangaLineExtraction_PyTorch-compatible network.
- `RealESRGAN_x4plus_anime_6B`-compatible weights are used for optional anime
  upscaling.
- ONNX or PyTorch weights bundled or downloaded by downstream packages may have
  separate model licenses. Do not assume the application code license covers
  model weights.

## Python Runtime Dependencies

Colortina depends on third-party packages listed in `requirements.txt` and,
for macOS lockfile builds, `requirements-macos.lock.txt`. Key dependencies
include:

- PyTorch / torchvision
- PySide6 / Qt for Python
- OpenCV
- NumPy
- Pillow
- Hugging Face Transformers and related packages
- ONNX Runtime
- scikit-image
- gdown
- PyMuPDF
- Pydantic

Each package remains under its own upstream license. In particular, PyMuPDF is
commonly distributed under AGPL/commercial licensing; review its current terms
before redistributing binaries or using the project commercially.

## Project License Scope

Unless otherwise stated in individual files or third-party notices, Colortina's
own glue code, UI code, and application-specific orchestration are separate
from the third-party components above. Third-party copyright notices and
license obligations continue to apply to the relevant code, model weights, and
binary dependencies.
