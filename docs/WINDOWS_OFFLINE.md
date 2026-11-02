# Colortina v5.20.0 — Windows portable (editable source layout)

This is a fully portable Windows x64 CPU build. The ZIP contains a small official
CPython executable in runtime/python.exe, dependencies in runtime/Lib/site-packages,
and editable application source files under app/. It does NOT contain one giant
self-extracting EXE, and no network connection is needed to install Python or pip
dependencies after extracting the ZIP.

## First launch: offline mode

1. Extract the whole archive to a writable folder (not Program Files).
2. Double-click Open_Models_Folder.bat to open models/weights/.
3. Copy the original generator.zip into models/weights/generator.zip.
   **Do not unzip or rename generator.zip**.
4. Double-click Run_Colortina.bat. This launcher defaults to offline mode.
5. Import a black-and-white image and run normal mc-v2 Auto Colorize.

The generator is the ONLY required model for standard coloring. Optional
model-only features still require their own weights; missing optional models
must not make the main generator unusable.

## Optional model file layout

| Relative path | Purpose | Without it |
| --- | --- | --- |
| models/weights/generator.zip | REQUIRED mc-v2 generator | AI coloring unavailable |
| models/weights/extractor.pth | Optional standalone encoder | Use generator's built-in encoder |
| models/weights/denoiser/net_rgb.pth | Optional FFDNet denoiser | Skip learned denoising, still colorize |
| models/weights/region_control_adapter.pth | Optional region adapter | Use stock generator |
| models/weights/generator_semantic_head.pth | Optional semantic head | Skip semantic-head model |
| models/weights/tunnel2_semantic_color_adapter.pth | Optional adapter | Skip adapter |
| models/weights/erika.pth | Optional line detection | Dedicated line detection unavailable |
| Other optional hair/segmentation/ESRGAN models | Special features | Those features degrade or show missing weights |

Network failure never deletes a valid user-provided generator.zip. You may
install a missing optional model at any time, then restart or retry its feature.

## Allow a confirmed model download

Run Run_Colortina_Allow_Download.bat instead. Only explicitly confirmed
downloads may access the network. If the download fails, you can still
manually place generator.zip and run the offline launcher.

## Contents and customization

- app/: editable Python source, including main.py, config.py, core/, ui/, vendor/
- runtime/: offline CPython and package dependencies; keep the entire folder
- models/weights/: manually managed checkpoint files
- Open_Models_Folder.bat: opens the correct weight directory
- Run_Colortina.bat: offline-first launcher
- Run_Colortina_Allow_Download.bat: interactive model download mode
- COLORTINA_WEIGHTS_DIR: optionally point to another drive with more free space
- COLORTINA_DATA_DIR: optionally move the writable data directory

This is a CPU-only build. NVIDIA GPU users can use source installation with
matching CUDA PyTorch, or create their own GPU portable build.

The generator.zip is a PyTorch checkpoint stored in the ZIP file format;
it is not an archive that users should unpack. Use only trusted weight sources.
A synthetic test checkpoint can verify file handling, but only real, validated
upstream weights can prove visual output correctness.
