# AlbaniaSAT — Path to 90% Accuracy
**Diplomarbeit Improvement Plan**
*Updated: April 2026*

---

## Current State

| Configuration | Test Accuracy |
|---|---|
| EuroSAT zero-shot (no training) | 19.08% |
| Best 8-class model — SoftCon, 4-band, summer | 64.58% |
| Best 8-class model — SoftCon, 6-band, summer | 65.33% |
| Best 7-class model — SoftCon, 6-band (Shrubland merged) | 73.33% |

**Goal**: 90% accuracy across all 8 classes.

**Realistic target**: 82–87% with the full pipeline. 90% is achievable but depends heavily on label quality over Albanian terrain — no model reaches it if the underlying labels are noisy. Treat 90% as a stretch goal, not a guarantee.

**Gap**: ~20–25 percentage points. Cannot be closed by training longer or collecting more of the same data.

---

## Root Cause Analysis

The three failing classes each fail for a distinct, fixable reason:

| Class | Current Accuracy | Root Cause | Fix |
|---|---|---|---|
| Shrubland | 22.7% | CORINE 100m labels on 10m patches — label noise at class boundaries | Switch to ESA WorldCover 2021 (10m labels) + purity filter |
| Agricultural | 54.8% | Summer dry crops are spectrally identical to summer dry grassland | Add spring composite (crops actively growing) |
| Grassland | 51.9% | Overlaps with Agricultural in summer, with Shrubland year-round | Spring + winter composites |

These three classes average ~43% accuracy and pull the overall score down by ~15 percentage points. Fixing them is the entire task.

---

## Known Limitations (State in Thesis)

- **WorldCover label quality over Albania is unknown.** WorldCover is produced by an automated pipeline and may have its own Shrubland/Grassland errors in Mediterranean terrain. Validate label quality in Phase 2 before building the full pipeline on top of it.
- **Accuracy projections are upper bounds.** They assume clean labels and no structural spectral overlap between classes. Real numbers may be lower.
- **22-channel input was considered and rejected.** Adding computed indices on top of 18 raw bands risks overfitting with 8,000 patches. Indices are reserved for Phase 4 only if 18-band results plateau.

---

## Phase 1 — Fix Existing Issues (1–2 weeks)
*Use data already collected. No new GEE exports required.*

### 1a — Train the 9-band model

9-band data (B2, B3, B4, B5, B6, B7, B8, B11, B12) was already collected and saved to `processed_v4` but never trained on. Red edge bands (B5, B6, B7) are designed for vegetation type discrimination — they should help separate forest classes and partially help Shrubland.

Adapt SoftCon `conv1` from 13 channels to 9:

```python
import torch
import torch.nn as nn

def adapt_conv1(model, n_channels=9):
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        n_channels, 64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # SoftCon order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
    # Your 9-band order: B2,B3,B4,B5,B6,B7,B8,B11,B12
    # Matching indices in SoftCon: 1,2,3,4,5,6,7,11,12
    softcon_indices = [1, 2, 3, 4, 5, 6, 7, 11, 12]
    with torch.no_grad():
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        for new_pos, sc_pos in enumerate(softcon_indices):
            new_conv.weight[:, new_pos, :, :] = old_conv.weight[:, sc_pos, :, :]
    model.conv1 = new_conv
    return model
```

### 1b — Fix 7-class imbalance bug

When Shrubland was merged into Grassland, Grassland became 2,000 patches while all other classes stayed at 1,000. CrossEntropyLoss with no class weights silently biases the model toward predicting Grassland.

```python
import numpy as np
import torch
import torch.nn as nn

def compute_class_weights(y):
    counts = np.bincount(y)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.FloatTensor(weights)

weights = compute_class_weights(y_train).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

### 1c — Fix band order inconsistency

The 4-band data was exported as `[B4, B3, B2, B8]` (Red, Green, Blue, NIR) but the 6-band data as `[B2, B3, B4, B8, B11, B12]` (Blue, Green, Red, NIR, SWIR). Channel order flipped between versions. For all future exports, standardize to Sentinel-2 natural order:

```python
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # do not change between versions
```

**Expected gain from Phase 1**: ~3–5pp overall.

---

## Phase 2 — Fix the Labels (2–3 weeks)
*This is the most important phase. Validate WorldCover quality before proceeding to Phase 3.*

### Switch from CORINE to ESA WorldCover 2021

CORINE Land Cover is produced at 100m resolution. Your imagery is 10m. When a 100m CORINE pixel labeled "Shrubland" is used to place a 640m×640m patch, the actual content at 10m resolution is often a mix of shrub, young forest, and grassland. This is the primary reason Shrubland achieves only 22.7%.

**ESA WorldCover 2021** is produced at **10m resolution** using Sentinel-1 + Sentinel-2. Available in GEE:

```python
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
```

**Class mapping:**

| WorldCover Class | Code | AlbaniaSAT Class |
|---|---|---|
| Cropland | 40 | Agricultural |
| Shrubland | 20 | Shrubland |
| Grassland | 30 | Grassland |
| Built-up | 50 | Urban |
| Permanent Water Bodies | 80 | Water |

**Keep CORINE** only for classes WorldCover cannot distinguish:
- Broad-leaved Forest (311)
- Coniferous Forest (312)
- Olive Groves (223)

### Add a Purity Filter

Erode each class mask before sampling so that only pixels where a 200m surrounding area is predominantly the same class are eligible:

```python
# In GEE
pure_mask = class_mask.focal_min(radius=200, units="meters")

points = pure_mask.selfMask().stratifiedSample(
    numPoints=1000,
    classBand="landcover",
    region=albania.geometry(),
    scale=10,
    seed=42,
    geometries=True
)
```

### WorldCover Validation Step (Do This First)

Before re-exporting 8,000 patches, visually inspect WorldCover labels over Albania in GEE for the three hard classes (Shrubland, Grassland, Agricultural). Check whether WorldCover's Shrubland class in Albania looks coherent or still shows obvious misclassifications. If it appears noisier than expected, the label-quality ceiling may not improve enough to justify the full re-export.

**Expected gain from Phase 2**: ~5–10pp, concentrated on Shrubland. Uncertain until WorldCover quality is validated.

---

## Phase 3 — Multi-Temporal Data (3–4 weeks)
*Only start after Phase 2 results are evaluated.*

### Collect Three Seasonal Composites

| Season | Date Range | Key Signal |
|---|---|---|
| Spring | March 15 – May 31 | Crops actively growing → high NDVI; bare agricultural fields starting to green |
| Summer | June 1 – Sept 15 | Existing data. Peak biomass, dry grassland |
| Winter | December 1 – February 28 | Deciduous leaf-off; bare agricultural soil; dormant grassland |

**Why each season helps:**

- **Agricultural vs Grassland**: Spring crops grow rapidly (steep NDVI rise). Grassland greens slowly. In winter, agricultural fields are often bare plowed soil while grassland retains partial green cover.
- **Broad-leaved vs Coniferous Forest**: Winter leaf-off causes NIR to drop sharply in broad-leaved forest. Coniferous stays elevated. Already good classes (76–82%) will improve further.
- **Shrubland vs Forest**: Partially deciduous shrubs show intermediate winter NIR — between bare grassland and evergreen forest.
- **Olive Groves**: Evergreen, stable NIR year-round. Distinguishable from deciduous agricultural land in winter.

### Winter GEE Code

Albanian winters have higher cloud cover. Relax the cloud threshold and add a snow mask using the Scene Classification Layer (SCL):

```python
def mask_snow_and_cloud(image):
    scl = image.select("SCL")
    # SCL 3=cloud shadow, 8=cloud medium, 9=cloud high, 11=snow/ice
    valid = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(11))
    return image.updateMask(valid)

winter_composite = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(albania.geometry())
    .filterDate("2021-12-01", "2022-02-28")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .map(mask_snow_and_cloud)
    .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    .median()
)
```

### Final Band Stack

Collect 6 bands per season × 3 seasons = **18 channels per patch**. This is the input ceiling — do not add computed indices yet.

```
Spring:  [B2_spr, B3_spr, B4_spr, B8_spr, B11_spr, B12_spr]
Summer:  [B2_sum, B3_sum, B4_sum, B8_sum, B11_sum, B12_sum]
Winter:  [B2_win, B3_win, B4_win, B8_win, B11_win, B12_win]
```

Use the same 8,000 patch center coordinates from your existing GEE exports — only the date filter and band selection change. This avoids re-running stratified sampling.

### Adapt SoftCon conv1 for 18 Channels

```python
def adapt_conv1_for_multitemporal(model, n_channels=18):
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        n_channels, 64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    # SoftCon order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 (indices 0–12)
    # Your stack: [spring 6 bands, summer 6 bands, winter 6 bands]
    # Copy pretrained weights for the 6 summer bands (positions 6–11) from SoftCon
    # SoftCon indices for B2,B3,B4,B8,B11,B12: 1,2,3,7,11,12
    softcon_indices = [1, 2, 3, 7, 11, 12]
    summer_positions = [6, 7, 8, 9, 10, 11]

    with torch.no_grad():
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        for new_pos, sc_pos in zip(summer_positions, softcon_indices):
            new_conv.weight[:, new_pos, :, :] = old_conv.weight[:, sc_pos, :, :]

    model.conv1 = new_conv
    return model
```

**Expected gain from Phase 3**: ~10–15pp. This is the largest single expected improvement.

---

## Phase 4 — Computed Indices or Foundation Model (2–3 weeks)
*Only if Phase 3 plateaus below target.*

### Option A — Add Computed Indices (18 → 22 channels)

Only attempt this if the 18-channel model has plateaued and you have confirmed no overfitting. With 8,000 patches, adding channels increases the risk of fitting noise rather than signal. Evaluate on validation set before training fully.

| Index | Formula | What It Separates |
|---|---|---|
| NDVI summer | (B8_sum − B4_sum) / (B8_sum + B4_sum) | Active vegetation density |
| NDVI spring | (B8_spr − B4_spr) / (B8_spr + B4_spr) | Crop growth signal |
| NDVI amplitude | NDVI_summer − NDVI_winter | Evergreen vs deciduous — most powerful single feature |
| BSI (Bare Soil Index) | (B11+B4 − B8−B2) / (B11+B4 + B8+B2) | Bare soil vs vegetated |

NDVI amplitude expected values per class:

| Class | Amplitude | Reason |
|---|---|---|
| Coniferous Forest | ~0.05–0.15 | Evergreen needles, minimal seasonal change |
| Olive Groves | ~0.05–0.15 | Evergreen broadleaf |
| Broad-leaved Forest | ~0.30–0.50 | Dramatic leaf-off in winter |
| Shrubland | ~0.20–0.35 | Partially deciduous |
| Grassland | ~0.20–0.30 | Dries in summer, partial green in winter |
| Agricultural | ~0.35–0.55 | Bare soil in winter, crops in spring |
| Urban | ~0.05–0.10 | Stable impervious surface |
| Water | ~0.00–0.05 | Stable, low NDVI |

### Option B — Prithvi Foundation Model

IBM and NASA released **Prithvi-100M**, a vision transformer pretrained on multi-temporal Harmonized Landsat-Sentinel (HLS) data. It natively handles multi-temporal 6-channel Sentinel-2 input — exactly your setup — without needing to modify `conv1`.

Available at: `ibm-nasa-geospatial/Prithvi-100M` on Hugging Face.

**Do not write the fine-tuning code from scratch.** Prithvi's forward pass involves temporal position encodings and patch embedding across time steps that are more complex than a standard ViT. Follow the official fine-tuning examples provided in the model repository:
- Fine-tuning guide: https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
- Example notebooks: https://github.com/NASA-IMPACT/hls-foundation-os

Using Prithvi as a second experiment alongside modified SoftCon is a publishable scientific contribution — comparing a domain-specific SSL pretrained model against a geospatial foundation model on the same Albanian dataset.

---

## Geographic Block Splitting

The current 70/15/15 random split allows nearby patches (spatially correlated) to appear in both train and test sets, inflating test accuracy. Geographic block splitting creates spatially disjoint regions:

```python
import numpy as np

def geographic_split(coords, n_blocks=20, val_frac=0.15, test_frac=0.15, seed=42):
    lons, lats = coords[:, 0], coords[:, 1]

    n = int(np.sqrt(n_blocks))
    lon_bins = np.linspace(lons.min(), lons.max(), n + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), n + 1)

    lon_idx = np.clip(np.digitize(lons, lon_bins) - 1, 0, n - 1)
    lat_idx = np.clip(np.digitize(lats, lat_bins) - 1, 0, n - 1)
    block_idx = lon_idx * n + lat_idx

    unique_blocks = np.unique(block_idx)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_blocks)

    n_test = int(len(unique_blocks) * test_frac)
    n_val  = int(len(unique_blocks) * val_frac)

    test_blocks  = set(unique_blocks[:n_test])
    val_blocks   = set(unique_blocks[n_test:n_test + n_val])
    train_blocks = set(unique_blocks[n_test + n_val:])

    train_idx = np.where([b in train_blocks for b in block_idx])[0]
    val_idx   = np.where([b in val_blocks   for b in block_idx])[0]
    test_idx  = np.where([b in test_blocks  for b in block_idx])[0]

    return train_idx, val_idx, test_idx
```

Note: this will likely lower your reported test numbers vs the current random split. That is expected and desirable — it is a more honest estimate of real-world performance, which is important for thesis validity.

---

## Accuracy Projections

These are upper-bound estimates assuming labels improve as expected. If WorldCover labels are noisy over Albania, all projections shift down.

| Configuration | Expected 8-class Accuracy |
|---|---|
| Current best (summer, 6-band, SoftCon) | 65% |
| + 9-band red edge — Phase 1 (existing data) | ~68–70% |
| + WorldCover labels + purity filter — Phase 2 | ~73–78% |
| + Multi-temporal 18-band — Phase 3 | ~82–87% |
| + Computed indices (22-band) or Prithvi — Phase 4 | ~85–90% (stretch goal) |

90% requires all phases to work as expected. If any phase underperforms (especially label quality), the ceiling drops accordingly.

---

## Implementation Order Summary

| Phase | What | Effort | Expected Gain |
|---|---|---|---|
| 1 | 9-band training + imbalance fix + band order fix | 1–2 weeks | +3–5pp |
| 2 | WorldCover labels + purity filter + **validate quality** | 2–3 weeks | +5–10pp |
| 3 | Spring + winter composites, 18-band stacking, SoftCon conv1 | 3–4 weeks | +10–15pp |
| 4 | Computed indices OR Prithvi (if Phase 3 plateaus) | 2–3 weeks | +3–5pp |

**Do not skip the WorldCover validation step in Phase 2.** If WorldCover is not cleaner than CORINE for Albanian terrain, the label quality ceiling cannot be raised regardless of model complexity. Knowing this early saves 4–6 weeks of wasted effort.

---

## Key References

- **ESA WorldCover 2021**: Zanaga et al. (2022). ESA WorldCover 10m 2021 v200. Zenodo. DOI: 10.5281/zenodo.7254221
- **Prithvi-100M**: Jakubik et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv:2310.18660
- **SoftCon / SSL4EO-S12**: Wang et al. (2023). Self-supervised pretraining for multi-modal remote sensing.
- **Multi-temporal LULC**: Pelletier et al. (2019). Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series. Remote Sensing, 11(5), 523
- **EuroSAT benchmark**: Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS
- **Discriminative fine-tuning**: Howard & Ruder (2018). Universal Language Model Fine-tuning for Text Classification. ACL 2018
