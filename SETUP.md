# Colortina — Phase 1 Setup (Mac / Apple Silicon)

This is Phase 1 of the project: a working command-line pipeline that proves
mc-v2 runs on your M-series GPU (MPS) with the auto + manual hint merge
logic in place. No UI yet — that's Phase 2, built on top of this once we
confirm this runs cleanly on your machine.

## What's in here

```
Colortina/
├── config.py                  # weight paths, device selection
├── pipeline.py                 # the Auto pipeline (single entry point)
├── test_pipeline.py             # CLI smoke test — run this first
├── requirements.txt
├── core/
│   ├── ml_colorizer.py         # mc-v2 wrapper (patched for MPS)
│   ├── hint_manager.py          # NEW — auto/manual hint priority merge
│   ├── guided_colorist.py       # auto color hints (CLIP + palette)
│   ├── region_classifier.py     # CLIP zero-shot region labeling (patched for MPS)
│   ├── region_segmenter.py      # lineart-bounded region segmentation
│   ├── panel_detector.py
│   ├── paint_bucket.py          # flood-fill recolor tool (for the future editor)
│   ├── color_director.py        # static color palette
│   ├── pdf_handler.py           # PDF -> page images
│   └── model_downloader.py      # auto-downloads mc-v2 weights on first run
├── vendor/manga_colorization_v2/  # vendored mc-v2 model code (MIT-adjacent, from qweasdd)
└── models/schemas.py
```

Everything under `core/` and `vendor/` (except `hint_manager.py`, which is
new) is adapted from `vikast908/ColorComic` (MIT licensed) — it already
had the hint-point API, guided auto-hints, and panel/tiling logic your
architecture doc called for. I patched every device-selection spot to
support MPS (Apple GPU) in addition to CUDA.

## 1. Set up the environment

```bash
cd Colortina
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

On Mac, `pip install torch` pulls the MPS-enabled build automatically —
no special index URL needed (that's only for CUDA on Linux/Windows).

## 2. Run the smoke test

```bash
python test_pipeline.py path/to/a_bw_manga_page.png
```

What happens on first run:
1. Downloads mc-v2 weights (~400MB) from Google Drive into `models/weights/`.
   This needs your Mac's normal internet — no VPN/proxy weirdness.
2. Downloads the CLIP checkpoint (`openai/clip-vit-base-patch32`, ~600MB)
   from Hugging Face on first use of guided hints.
3. Runs: auto color-hint generation → mc-v2 colorize → saves
   `<name>_colorized.png` next to your input.

Useful flags:
```bash
python test_pipeline.py page.png --device mps      # force Apple GPU
python test_pipeline.py page.png --device cpu       # force CPU (slower, for comparison)
python test_pipeline.py page.png --no-guided         # skip CLIP hints, plain mc-v2 auto-colorize
```

## 3. What to check

- Does it pick `mps` automatically? The script prints
  `[pipeline] mc-v2 loaded on device: mps` if so.
- Does colorization finish in a reasonable time (should be a few seconds
  per page on M-series GPU at the default 768px size)?
- Does the CLIP-guided version (default) produce noticeably better,
  more targeted colors than `--no-guided`?

## 4. If something breaks

- **`RuntimeError` mentioning MPS + an unsupported op**: some PyTorch ops
  aren't implemented on MPS yet. Tell me the exact error and I'll add a
  fallback (e.g. run just that op on CPU) — this is a known rough edge
  in PyTorch's MPS backend, not a bug in this code.
- **gdown fails to download from Google Drive**: Google sometimes
  rate-limits anonymous downloads. Tell me and I'll give you direct
  manual-download links + where to place the files.
- **Import errors**: send me the traceback — I already verified every
  file in this package imports cleanly in a Linux sandbox, so a Mac-side
  import error is almost certainly a missing dependency I can add to
  `requirements.txt`.

## Next steps (Phase 2, once Phase 1 is confirmed working)

Build the PySide6 (Qt6) desktop UI: import → auto-colorize → preview →
editor (canvas + brush + eyedropper, feeding `HintManager.add_manual_hint`)
→ re-run `colorize_page()` → export. The `paint_bucket.py` flood-fill tool
already ported over is meant for that editor's fill tool.

---

# Phase 2 — Desktop Editor

```bash
source venv/bin/activate
pip install -r requirements.txt   # picks up PySide6
python main.py
```

## What's new

```
Colortina/
├── main.py                  # app entry point — run this
├── ui/
│   ├── main_window.py         # 3-panel layout: pages | canvas | controls
│   ├── canvas.py               # brush/eyedropper canvas (QGraphicsView)
│   └── worker.py               # background thread so colorizing doesn't freeze the UI
```

## Workflow

1. **＋ 导入图片 / ＋ 导入 PDF** (left panel) — add pages. PDF is split into
   one PNG per page automatically (via `pdf_handler.extract_pages`).
2. Select a page, click **自动上色** — runs the same pipeline as
   `test_pipeline.py`, in a background thread (Ctrl+scroll to zoom while
   it's still the B&W original if you want a closer look first).
3. Not happy with a spot? Switch to **画笔**, pick a color (swatch or
   **吸管** to sample an existing color from the image), and paint dabs
   directly on the canvas. Each dab is a manual hint point.
4. Click **重新生成** — re-runs mc-v2 with your manual hints layered on
   top of the existing auto hints (manual always wins locally — that's
   the priority-merge logic in `hint_manager.py`). Auto hints are *not*
   regenerated on this path, only on **自动上色**.
5. **撤销上一笔** / **清除手动笔画** if you want to back out edits before
   regenerating.
6. **导出当前页 / 导出全部页面** when done.

## Known rough edges (tell me what you hit and I'll fix it)

- No autosave / project file yet — closing the app loses hint history
  for pages you haven't exported. Fine for testing, but say the word and
  I'll add project save/load (matches the `project.ccproj` idea from the
  original design doc).
- No batch "colorize all pages" button yet — one page at a time for now.
- No per-page undo history beyond manual hints (i.e. can't undo an
  auto-colorize back to the original).
- Large PDFs (100+ pages) will import fine but you'll want the batch
  button before doing a whole volume — ping me when Phase 1+2 feel solid
  and I'll build that next along with the GPU/queue monitor from the
  design doc.
