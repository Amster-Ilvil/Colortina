"""Generates a 32x32 pixel-art "girl with paint-splash hair" icon for
Colortina, upscaled crisply (nearest-neighbor) to 256x256, saved as both
.png (used by the Qt app) and a multi-size .ico (for Windows shortcuts)."""

import numpy as np
from PIL import Image

N = 32
# RGBA grid, default transparent
grid = np.zeros((N, N, 4), dtype=np.uint8)

# Palette — a colorizer app, so the hair is a rainbow "paint" gradient.
SKIN = (255, 219, 186, 255)
SKIN_SHADE = (240, 195, 160, 255)
BLUSH = (255, 150, 150, 200)
HAIR = [
    (230, 60, 90, 255),   # magenta-red (top)
    (240, 120, 60, 255),  # orange
    (250, 200, 60, 255),  # yellow
    (110, 200, 110, 255), # green
    (70, 150, 220, 255),  # blue
    (130, 90, 200, 255),  # violet (ends)
]
HAIR_DARK = (90, 60, 140, 255)
EYE = (50, 40, 70, 255)
EYE_HI = (255, 255, 255, 255)
OUTLINE = (40, 30, 50, 255)
COLLAR = (255, 255, 255, 255)
COLLAR_SHADE = (225, 225, 235, 255)


def set_px(x, y, color):
    if 0 <= x < N and 0 <= y < N:
        grid[y, x] = color


def set_row_range(y, x0, x1, color):
    for x in range(x0, x1 + 1):
        set_px(x, y, color)


# ── Hair (back layer, wide bob + pigtail-ish flips), rainbow banded top->bottom
hair_rows = {
    3: (8, 23),
    4: (6, 25),
    5: (5, 26),
    6: (4, 27),
    7: (3, 28),
    8: (3, 28),
    9: (3, 9, 22, 28),   # split row placeholder handled below
}
for y in range(3, 9):
    x0, x1 = hair_rows[y]
    band = HAIR[min((y - 3), len(HAIR) - 1)]
    set_row_range(y, x0, x1, band)

# Side locks framing the face down to shoulders, rainbow gradient continues
for i, y in enumerate(range(9, 20)):
    band = HAIR[min(2 + i // 2, len(HAIR) - 1)]
    set_row_range(y, 3, 6, band)
    set_row_range(y, 25, 28, band)

# Outline around hair silhouette (cheap approximation: darken edge pixels)
for y in range(3, 20):
    for x in range(N):
        if tuple(grid[y, x]) != (0, 0, 0, 0):
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            if any(not (0 <= nx < N and 0 <= ny < N) or tuple(grid[ny, nx]) == (0, 0, 0, 0)
                   for nx, ny in neighbors):
                pass  # keep banded color; outline handled by face silhouette below

# ── Face (skin) oval
face_rows = {
    8: (10, 21),
    9: (9, 22),
    10: (8, 23),
    11: (8, 23),
    12: (8, 23),
    13: (8, 23),
    14: (9, 22),
    15: (10, 21),
    16: (11, 20),
}
for y, (x0, x1) in face_rows.items():
    set_row_range(y, x0, x1, SKIN)

# Jaw shading
set_row_range(15, 10, 12, SKIN_SHADE)
set_row_range(15, 19, 21, SKIN_SHADE)

# Bangs across forehead (front hair layer over top of face rows 8-9)
set_row_range(8, 9, 22, HAIR[0])
for x in (11, 14, 17, 20):
    set_px(9, 8, HAIR[0])

# ── Eyes
for ex in (11, 19):
    set_px(ex, 12, OUTLINE)
    set_px(ex + 1, 12, EYE)
    set_px(ex, 13, EYE)
    set_px(ex + 1, 13, EYE)
    set_px(ex, 11, OUTLINE)
    set_px(ex + 1, 11, OUTLINE)
    set_px(ex, 12, EYE_HI)

# Blush
set_px(9, 14, BLUSH)
set_px(22, 14, BLUSH)

# Mouth (small smile)
set_px(14, 16, OUTLINE)
set_px(15, 16, OUTLINE)
set_px(16, 16, OUTLINE)
set_px(17, 16, OUTLINE)

# ── Collar / shoulders
for y in range(20, 27):
    width = (y - 20)
    set_row_range(y, max(0, 9 - width), min(N - 1, 22 + width), COLLAR if y < 24 else COLLAR_SHADE)

# A little paint-splash / color-drop accent on the collar (nod to "Colortina")
for (x, y, c) in [(15, 22, HAIR[2]), (16, 22, HAIR[2]), (15, 23, HAIR[2]),
                  (17, 23, HAIR[3]), (16, 23, HAIR[3])]:
    set_px(x, y, c)

img = Image.fromarray(grid, mode="RGBA")
big = img.resize((N * 8, N * 8), resample=Image.NEAREST)

out_dir = "assets"
import os
os.makedirs(out_dir, exist_ok=True)
big.save(os.path.join(out_dir, "icon.png"))

# Multi-size .ico for Windows builds/shortcuts
sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
big.save(os.path.join(out_dir, "icon.ico"), sizes=sizes)

print("done")
