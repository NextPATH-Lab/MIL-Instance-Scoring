# Experiments — Script Index

This is a reference index for every top-level script in `experiments/` (everything
*except* the shared library in `experiments/src/`, which is documented via
docstrings in place). For setup and the end-to-end run order, see the main
[README.md](../README.md).

## Conventions

All batch/CLI scripts here follow the same environment-driven layout, configured
via the repo-root `.env` file (loaded with `python-dotenv`, `override=True`):

| Variable | Used for |
| --- | --- |
| `GLOBAL_LOG_DIR` | Root directory for log files. Each script writes a dated log to `GLOBAL_LOG_DIR/<pipeline>/<timestamp> <script-name>.log` (DDP scripts append `_rank<N>` per process). |
| `OUTPUTS_DIR` | Root directory for generated artifacts (figures, checkpoints, result CSVs), namespaced the same way as logs: `OUTPUTS_DIR/<pipeline>/...`. |
| `CONFIG_DIR` | Where generated JSON configs (feature lists, train/test ID splits) are written/read for the LDO pipeline. |
| `TRAINING_DATA_DIR` | Root of the LDO patient-level `.csv` data. |
| `CAMELYON16_EMBEDS_DIR` | Directory of per-slide UNI2-H embedding `.pt` files, consumed by the CAMELYON16 training/analysis scripts. |
| `ACCELERATION_DEVICE` | Default torch device (`"cuda"`/`"cpu"`) for training scripts. |
| `SEED` | Global seed for reproducibility helpers. |
| `HF_TOKEN` / `HF_READ_TOKEN` | HuggingFace token used once, to download UNI2-H weights. |

The three pipelines below each get their own subfolder under `GLOBAL_LOG_DIR` and
`OUTPUTS_DIR`: `ProMNIST`, `camelyon16`, and `LDO-Prostate`.

GUI/interactive tools (napari viewers) still write a log file for diagnostics, but
don't produce `OUTPUTS_DIR` artifacts themselves — they're for visualization, not
data generation.

---

## 1. Proportion MNIST Bags (synthetic, `ProMNIST`)

Bags of MNIST digits where the bag label is a threshold (`tau`) on the proportion
of a target digit present, used to study how bag composition affects whether
attention or instance logit better predicts instance-level identity.

| Script | Type | What it does |
| --- | --- | --- |
| [experiment-promnist-bags.py](experiment-promnist-bags.py) | experiment | Generates `ProportionMnistBags` datasets across a sweep of `tau` thresholds and seeds, trains an `ABMIL` classifier per (dataset, seed) combination, computes bag-level metrics plus instance logit/attention ROC-AUC and correlation, and writes `promnist.csv`. Run this first. |
| [experiment-promnist-za-dists.py](experiment-promnist-za-dists.py) | experiment | Re-trains `ABMIL` on a fixed subset of `tau` values from the datasets above and dumps the raw per-instance (logit, attention, label, tau) tuples to `promnist_za_dist.csv`, for plotting the joint distribution. Depends on `experiment-promnist-bags.py` having already created the `.pt` dataset files. |
| [figures-promnist.py](figures-promnist.py) | figures | Reads `promnist.csv` and `promnist_za_dist.csv` and renders the manuscript figures (ROC/correlation/bag-performance vs. `tau`, and the logit-vs-attention joint plot) to `OUTPUTS_DIR/ProMNIST/figures/`. |

## 2. CAMELYON16 + UNI2-H Embeddings (`camelyon16`)

WSI preprocessing → UNI2-H patch embedding → ABMILite training → instance-level
attention/logit/clustering analysis, on real whole-slide tumor data.

| Script | Type | What it does |
| --- | --- | --- |
| [utils-autodetect-tissue-wsi.py](utils-autodetect-tissue-wsi.py) | CLI utility | Batch tissue auto-detection: thresholds a downsampled WSI to find tissue regions, extracts polygon ROIs, and writes them as Getafics-style `.sec` annotation files next to each slide. Takes `-d/--directory`, `-t/--type`, `-l/--level` args. |
| [utils-download-uni2-h.py](utils-download-uni2-h.py) | one-shot utility | Logs into HuggingFace with `HF_TOKEN`/`HF_READ_TOKEN` and downloads/caches the `MahmoodLab/UNI2-h` foundation model weights locally, so later steps can run offline (`HF_HUB_OFFLINE=1`). |
| [utils-embed-wsi.py](utils-embed-wsi.py) | CLI utility (DDP-aware) | **Recommended path.** Slices each WSI's tissue ROI into overlapping tiles, splits them into tumor/benign using the ASAP XML tumor annotation (if present, same logic as `utils-zarrify-wsi.py`), and embeds each patch with UNI2-H immediately as it's read — no Zarr store is ever written, since only `{features, labels, coords}` is needed downstream and persisting raw patches would cost real disk (a large slide can be 100k+ patches). Supports multi-GPU/multi-node execution via `torch.distributed` when launched with `RANK`/`LOCAL_RANK`/`WORLD_SIZE` set (e.g. via `torchrun`); each rank processes a shard of the slide list and writes to its own log file. Takes `-d/--directory`, `-s/--save-dir`, `-i/--indices`, `-t/--tile-size`, `-o/--overlap-size`, `-b/--batch-size` args. |
| [utils-zarrify-wsi.py](utils-zarrify-wsi.py) | CLI utility | Slices each WSI's tissue ROI into overlapping tiles, splits them into tumor/benign using the ASAP XML tumor annotation (if present), and writes them plus their level-0 coordinates into a Zarr v3 store per slide. Useful standalone if you actually need the persisted raw patches (e.g. visual QA, a different backbone later); otherwise superseded by `utils-embed-wsi.py`. Takes `-d/--directory`, `-i/--indices`, `-s/--size`, `-o/--overlap-size` args. |
| [utils-uni2-embedding.py](utils-uni2-embedding.py) | CLI utility (DDP-aware) | Loads UNI2-H offline from the local HF cache and embeds every tumor/benign patch in each `.zarr` store, saving `{features, labels, coords}` per slide as a `.pt` file. Only needed if you already have Zarr stores from `utils-zarrify-wsi.py` and want to embed them separately; otherwise superseded by `utils-embed-wsi.py`. Supports multi-GPU/multi-node execution via `torch.distributed` when launched with `RANK`/`LOCAL_RANK`/`WORLD_SIZE` set (e.g. via `torchrun`); each rank processes a shard of the slide list and writes to its own log file. Takes `-d/--dataset-dir`, `-s/--save-dir`, `-b/--batch-size` args (output directory is explicit, not `.env`-driven, since it's typically pointed at scratch/shared storage per job). |
| [experiment-uni2-camelyon16-abmil.py](experiment-uni2-camelyon16-abmil.py) | experiment | Trains `ABMILite` on the UNI2-H slide embeddings (`CAMELYON16_EMBEDS_DIR`) with one stratified train/val split, evaluates on the held-out `test_*` slides, then runs instance-level inference to build the logit/attention "ZA space" and compares attention, logit, and a Bayesian Gaussian Mixture cluster score as tumor-patch predictors. Saves the trained checkpoint (`abmilite_camelyon16_uni2h.pt`) and figures to `OUTPUTS_DIR/camelyon16/`. Run this before the two scripts below. |
| [experiment-camelyon-uncertainty.py](experiment-camelyon-uncertainty.py) | experiment | Loads the checkpoint from the script above and bootstraps (slide-level resampling) the patch-level ROC-AUC of attention, instance logit, and GMM cluster score to get confidence intervals, plus a repeated-measures ANOVA + Tukey HSD post-hoc test across the three methods. Writes `bootstrap-CI-camelyon16.csv` and a violin/box comparison figure. |
| [figures-camelyon16-attention-overlay.py](figures-camelyon16-attention-overlay.py) | GUI tool | Interactive napari viewer: overlays a trained `ABMILite` checkpoint's per-patch attention and instance-logit scores directly on a WSI as colored rectangles, for qualitative inspection. Takes `-w/--wsi-path`, `-f/--features-path`, `-c/--checkpoint` (plus colormap/opacity options) as CLI args; produces no saved files. |
| [utils-wsi-viewer.py](utils-wsi-viewer.py) | GUI tool | General-purpose napari WSI viewer (any pyramidal `.svs`/`.tif*`), with a `magicgui` panel to load a slide and overlay `.sec`/ASAP-XML section and ROI annotations (as produced by `utils-autodetect-tissue-wsi.py`), colored either by section or by a custom JSON color config. |

## 3. Prostate Cancer LDO (`LDO-Prostate`)

Large-scale DNA Organization (LDO) nuclear features from prostate core needle
biopsies, used as MIL bags of nuclei per patient to predict clinical outcome
(`DW2Y` — death within 2 years).

| Script | Type | What it does |
| --- | --- | --- |
| [utils-ldo-feature-selection.py](utils-ldo-feature-selection.py) | data prep | Drops known-unusable/auxiliary columns, computes the train/test patient split from `clinical_deidentified.csv`, then filters features by removing zero-variance columns, invalid rows, and correlated pairs (favoring the one with higher mutual-information score against `DW2Y`). Writes the filtered feature config, scaler, and a per-patient `filtered_copy/` of cleaned, scaled CSVs used by training. Run this before `experiment-ldo-abmilite-training.py`. |
| [experiment-ldo-abmilite-training.py](experiment-ldo-abmilite-training.py) | experiment | Runs stratified 5-fold CV of `ABMILite` over the filtered LDO features, then a full train/test fit. Saves the trained checkpoint, loss and feature-importance figures, per-patient/per-biopsy LDO scores (`score_mapping.csv`), and a logit-vs-attention cluster plot to `OUTPUTS_DIR/LDO-Prostate/`. |

---

## Data-layout notes

- `experiments/config/` holds the generated JSON artifacts from
  `utils-ldo-feature-selection.py` (`feature-config.json`,
  `filtered-feature-config.json`, `train-test-ids-dw2y.json`) — not hand-edited,
  regenerate by re-running that script.
- `data/ProportionMNISTBags/` holds the synthetic bag `.pt` files and per-run
  metrics `.json` files produced by `experiment-promnist-bags.py`.
