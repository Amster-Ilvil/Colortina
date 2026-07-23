# Security Policy

Colortina is designed as a local desktop application for manga colorization.

## Local Data

- Imported images, PDFs, project files, generated results, and user-created
  color data are processed locally on your computer.
- The project does not intentionally upload user images, pages, project files,
  palettes, or generated outputs to any server.
- Local runtime folders, model weights, cache files, project assets, and user
  outputs are excluded from Git by `.gitignore`.

## Network Access

Colortina may access the network only for setup or optional model downloads:

- Windows one-click setup may download embedded Python, pip, PyTorch, and Python
  package dependencies.
- First-time colorization may download required model weights.
- Optional detectors or enhancement models may download their public weights
  when needed.

These downloads are for local installation and local inference.

## Sensitive Files

Do not commit:

- `.env` files
- API keys, access tokens, passwords, or private keys
- `venv/`, `.venv/`, or `runtime/`
- `models/weights/`
- `.ccproject`, `.ccpalette`, `.ccscene`, `.ccstyle`, output images, or project
  asset folders containing user work

## Reporting Issues

If you find a security or privacy issue, please open a GitHub issue with a clear
description and avoid including private images, credentials, or personal data.
