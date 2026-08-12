# manga-colorization-v2 integration

Colortina supports `qweasdd/manga-colorization-v2`, but the upstream repository currently does not publish an explicit software license. A public GitHub repository without a license is not automatically licensed for copying, modification, or redistribution.

For that reason, this public repository **does not vendor or redistribute the upstream manga-colorization-v2 source code**.

If you have obtained an upstream checkout directly from its author/source and have the right to use it, point Colortina to that checkout with:

```bash
export COLORTINA_MANGA_COLORIZATION_V2_PATH="/path/to/manga-colorization-v2"
```

The path must contain at least `colorizator.py` and the upstream `networks/` directory. Colortina also checks conventional per-user application-data locations through `vendor.manga_colorization_v2.__init__`.

Model weights are not licensed by Colortina. Obtain and use them only under terms supplied by their copyright holder or other applicable permission.

If the upstream project later publishes a clear license or grants redistribution permission, this integration can be revisited.
