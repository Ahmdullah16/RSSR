# RSSR Minor Project — Image Reconstruction Lab

A separate Streamlit application for **CNN-, attention- and Transformer-based remote-sensing image super-resolution**, prepared for Ahmdullah, M.Tech in Signal Processing and Machine Learning, IIT Guwahati.

The main research repository is preserved. This compact app reuses its genuine checksum-pinned trained RGB pilots and adds an independent interface.

## Deploy on Streamlit Community Cloud

| Setting | Value |
|---|---|
| Repository | `Ahmdullah16/RSSR` |
| Branch | `main` |
| Main file path | `streamlit_app.py` |
| Python | `3.12` |
| Secrets | None for the public image/checkpoint workflows |

Create a new app in your Streamlit account with these settings. `requirements.txt` declares CPU-only PyTorch and the tested runtime. No training starts at import. The URL is assigned by Streamlit; this repository does not claim a live deployment until it is verified.

## Five main sections

| Section | What it does |
|---|---|
| Results | Recalculates the current input, displays LR, reference/bicubic, SR, difference map, metrics and parameter meanings. |
| Sensitivity | Runs three bounded scale, tile-size or input-noise cases; shows numerical tables, charts and images. |
| Compare | Runs up to three selected models on the same LR input and settings. |
| Concepts & mathematics | Explains degradation, CNNs, attention, Transformers, hybrid fusion, training loss, scale, PSNR, SSIM and export conventions. |
| Downloads | Exports actual PNG/JPEG image bytes, measured CSV results, JSON provenance and ZIP bundles. |

Results update when a control is committed (Enter or leaving a numeric field). Expensive three-case sensitivity and model-comparison batches run only on request. Any changed input/settings invalidates displayed batch results; old images are never relabeled as new runs.

## Image sources and satellite identity

- **Live satellite:** bounded public Element 84 Earth Search query for Sentinel-2 L2A/c1-L2A true-color display imagery. The exact spacecraft (`platform`), acquisition timestamp, item ID, collection, source CRS, pixel sampling, query, validity and license links are reported from the selected observation. MSI RGB uses B04/B03/B02. “Live” means a current archive request, not a live camera or a guarantee of a recent acquisition. Query/filter failures are shown explicitly.
- **Upload LR:** opaque RGB/grayscale PNG/JPEG, with optional exact-size aligned HR reference and user-supplied sensor/capture information. Unknown sensor metadata is not guessed.
- **Upload HR:** reproducible synthetic LR generation from the uploaded HR image for reference-based experiments.
- **Cached NAIP sample:** aerial training imagery acquired 2021-09-17; explicitly not satellite data or held-out evaluation imagery. Preserve USDA NAIP/OpenSR attribution.
- **Synthetic demonstration:** a procedural test pattern, explicitly not remote-sensing observations. This offline default uses bicubic and needs no checkpoint download.

Live acquisition takes a fixed native-pixel square crop centered on the AOI/scene intersection, not a resampled representation of the full bounding box. Scene cloud-cover filtering is not a pixel cloud mask. Invalid source pixels are black in display exports; PNG/JPEG does not retain the raster validity mask or CRS. The source transform is retained in JSON.

## Controls and limits

| Control | Limit / interpretation |
|---|---|
| Families | Bicubic plus M1 CNN, M2 Attention, M3 Transformer, M4 CNN+Attention, M5 CNN+Transformer, M6 Attention+Transformer, M7 full hybrid. |
| Scale | ×2, ×4, ×8, ×16, ×32, ×64. Native learned ×2; higher scales are explicit repeated ×2 cascades. |
| Input | ≤8 MB, 1–1024 pixels on each side, opaque RGB/grayscale PNG/JPEG. |
| Output | ≤1024 per side and ≤1,048,576 pixels in this minor app; exceeding requests are rejected without resizing. |
| Synthetic blur | Gaussian radius 0–3 HR pixels; synthetic-pair sources only. |
| Noise | Gaussian standard deviation 0–20 uint8 levels; seeded, clipped and rounded. |
| Tile / halo | 64/128/256 LR pixels and 16/32/64 LR context pixels; learned-model memory/context controls. |
| Metric crop | 0–16 HR border pixels, must leave at least one pixel; output itself is not cropped. |
| JPEG quality | 75–100, RGB 4:4:4; affects compression after metrics, not learned weights. |
| Live request | AOI edges ≤0.25 degrees, ≤366-day interval, one newest matching item, 64/128/256 native crop; ≤2 MB STAC metadata and ≤32 MB COG range-transfer budget. |

Model widths, attention heads, training epochs and weights are fixed checkpoint properties, not inference sliders. Larger output dimensions alone do not establish true recovered detail. The seven checkpoints have distinct hashes but unequal training budgets, so comparisons are educational rather than controlled scientific rankings.

## Evaluation conventions

Metrics use RGB float samples in [0,1], the selected border crop and exact reference alignment. HR references are never silently resized. SSIM uses valid uniform 7×7 population moments with K1=.01 and K2=.03; too-small images show unavailable SSIM. Perfect equality produces infinite PSNR. No reference means no PSNR/SSIM/RMSE. Edge activity and bicubic disagreement are descriptive, not accuracy scores. Metrics precede JPEG compression.

See [source/model provenance](PROVENANCE.md) for model and source limitations.

## Local execution and checks

```bash
python -m pip install -r requirements-dev.txt
python -m streamlit run streamlit_app.py
python -m pytest -q
# Optional bounded download/inference checks using real pinned checkpoints:
python scripts/check_models.py
```

The app stores image results and checkpoint caches temporarily. It does not persist uploads, train models or create paid cloud resources. PNG/JPEG exports are display RGB products, not calibrated/georeferenced scientific rasters.
