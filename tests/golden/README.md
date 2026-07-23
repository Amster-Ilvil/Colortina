# Real-page golden regression set

This folder intentionally contains no copyrighted manga pages. Add authorized
black-and-white pages and generated results locally using this layout:

```text
tests/golden/
  bw_pages/
  results/
  references/
  manifest.json
```

Copy `manifest.example.json` to `manifest.json`, then record stable sample
points for hair, eyes and skin. Normalized coordinates (`0..1`) are recommended.

Evaluate an existing result set:

```bash
python tools/evaluate_golden.py tests/golden/manifest.json \
  --output tests/golden/report.json
```

Key outputs:

- `same_character_delta_e_mean`: lower is more consistent across pages.
- `different_character_delta_e_min`: higher means distinct characters remain separated.
- `line_bleed_ratio_mean`: lower means less colour on black line art.
- `ambiguous_match_ratio`: how often candidate identities were too close.
- `lock_region_coverage`: percentage of candidate regions that safely passed locking gates.

Do not commit copyrighted source pages unless you have redistribution rights.
