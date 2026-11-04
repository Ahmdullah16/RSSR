# Source and model provenance

This separate minor application was requested by Ahmdullah. It reuses the model definitions, model factory, architecture registry, numerical metrics, bounded asset and COG loaders, upload validation and pinned model catalogs from [Mini-Project-ML-Lab-RSSR](https://github.com/Ahmdullah16/Mini-Project-ML-Lab-RSSR), source commit `9006d1b3ee14241b461e6f0ec915e4e1549754fc`. The original project's entrypoint and deployment are not modified.

The catalogs retain immutable artifact URLs, byte budgets, SHA-256 checksums, source acquisition date, license and each checkpoint's training limitations. No weights or datasets are committed here. Downloads are explicitly bounded and verified before restricted `weights_only=True` loading. No random-weight fallback and no training are performed by this app.

M1 is the original 100-step RGB CNN pilot. M2–M6 are distinct 100-step compatibility pilots; M7 is the frozen 1000-step hybrid. Their training budgets are not matched, so comparing them is a teaching demonstration, not a controlled architecture study. Every architecture retains convolutional input/output projections and a bicubic residual baseline. Family names describe the C/A/T feature branches, not the literal absence of every convolution.

The optional NAIP sample is a cached training example acquired on 2021-09-17, not a current satellite acquisition or held-out validation image. Retain USDA NAIP / OpenSR attribution; the catalog records NAIP public-domain provenance. The procedural default image is a synthetic pattern created by the app, not satellite data. PNG and JPEG exports are RGB display products; they do not preserve geographic CRS, physical reflectance or multispectral bands.

The reused COG range loader additionally allows the current Element 84 c1-L2A S3 hostname advertised by the Earth Search visual asset; unrelated hosts remain rejected.
