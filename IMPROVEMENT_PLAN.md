# AlbaniaSAT — Path to 90% Accuracy
**Diplomarbeit Improvement Plan**
*Generated: April 2026*

---

## Current State

| Configuration | Test Accuracy |
|---|---|
| EuroSAT zero-shot (no training) | 19.08% |
| Best 8-class model — SoftCon, 4-band, summer | 64.58% |
| Best 8-class model — SoftCon, 6-band, summer | 65.33% |
| Best 7-class model — SoftCon, 6-band (Shrubland merged) | 73.33% |

**Goal**: 90% accuracy across all 8 classes.

**Gap**: ~25 percentage points. Cannot be closed by training longer or collecting more of the same data.

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

## Step 1 — Fix the Labels (Highest Impact)

### Switch from CORINE to ESA WorldCover 2021

CORINE Land Cover is produced at 100m resolution. Your imagery is 10m. When a 100m CORINE pixel labeled "Shrubland" is used to place a 640m×640m patch, the actual content at 10m resolution is often a mix of shrub, young forest, and grassland. This is the primary reason Shrubland achieves only 22.7%.

**ESA WorldCover 2021** is produced at **10m resolution** using Sentinel-1 + Sentinel-2, making it perfectly matched to your imagery. It is freely available in Google Earth Engine.

```python
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first()
```

**Class mapping from WorldCover to AlbaniaSAT:**

| WorldCover Class | Code | AlbaniaSAT Class |
|---|---|---|
| Cropland | 40 | Agricultural |
| Shrubland | 20 | Shrubland |
| Grassland | 30 | Grassland |
| Built-up | 50 | Urban |
| Permanent Water Bodies | 80 | Water |

**Keep CORINE** (100m) only for:
- Broad-leaved Forest (311)
- Coniferous Forest (312)
- Olive Groves (223)

WorldCover does not distinguish forest types or olive groves, so CORINE remains necessary for those three classes.

### Add a Purity Filter

Even with 10m labels, sample points near class boundaries introduce noise. Before sampling, erode each class mask so that only pixels where a 200m surrounding area is predominantly the same class are eligible for sampling.

```python
# In GEE — erode the class mask before stratified sampling
pure_mask = class_mask.focal_min(radius=200, units="meters")

points = pure_mask.selfMask().stratifiedSample(
    numPoints=1000,
    classBand="landcover",
    region=albania.geometry(),
    scale=10,           # sample at 10m, not 100m
    seed=42,
    geometries=True
)
```

This removes transition-zone samples where the label is ambiguous. Expected to reduce Shrubland noise substantially.

---

## Step 2 — Multi-Temporal Feature Stack

### Collect Three Seasonal Composites

Each season captures different land surface states that are spectrally distinct:

| Season | Date Range | Key Signal |
|---|---|---|
| Spring | March 15 – May 31 | Crops actively growing → high NDVI; bare agricultural fields starting to green |
| Summer | June 1 – Sept 15 | Your existing data. Peak biomass, dry grassland |
| Winter | December 1 – February 28 | Deciduous trees lose leaves; bare agricultural soil; dormant grassland |

**Why each season helps your specific classes:**

- **Agricultural vs Grassland**: In spring, crops grow rapidly (NDVI rises steeply). Grassland greens slowly. In winter, agricultural fields are often bare plowed soil while grassland retains some green cover. Strong discriminator in both seasons.
- **Broad-leaved vs Coniferous Forest**: In winter, broad-leaved trees lose their leaves (NIR drops sharply). Coniferous keeps needles (NIR stays elevated). These two classes are already at 76–82% accuracy but winter will push them higher.
- **Shrubland vs Forest**: Shrubland (CORINE 324 = transitional woodland-shrub) is partially deciduous. Winter shows intermediate NIR between bare grassland and evergreen forest.
- **Olive Groves**: Evergreen — stable NIR year-round. Distinguishable from deciduous agricultural land in winter.

### Winter Collection Considerations

Albanian winters have higher cloud cover than summer. Use these settings:

```python
winter_composite = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(albania.geometry())
    .filterDate("2021-12-01", "2022-02-28")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))  # relax threshold slightly
    .filter(ee.Filter.lt("SNOW_ICE_PERCENTAGE", 15))      # exclude heavy snow scenes
    .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    .median()
)
```

Add a snow mask using the Scene Classification Layer (SCL) to exclude snow-covered pixels from the median composite:

```python
def mask_snow(image):
    scl = image.select("SCL")
    snow_free = scl.neq(11)  # SCL class 11 = snow/ice
    return image.updateMask(snow_free)

winter_composite = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(albania.geometry())
    .filterDate("2021-12-01", "2022-02-28")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(mask_snow)
    .select(["B2", "B3", "B4", "B8", "B11", "B12"])
    .median()
)
```

### Final Band Stack Per Patch

Collect 6 bands per season × 3 seasons = **18 raw channels per patch**.

```
Spring:  [B2_spr, B3_spr, B4_spr, B8_spr, B11_spr, B12_spr]
Summer:  [B2_sum, B3_sum, B4_sum, B8_sum, B11_sum, B12_sum]
Winter:  [B2_win, B3_win, B4_win, B8_win, B11_win, B12_win]
```

---

## Step 3 — Computed Spectral Indices

Add 4 computed indices as extra input channels. These capture temporal dynamics more explicitly than raw bands.

| Index | Formula | Separates |
|---|---|---|
| NDVI summer | (B8_sum − B4_sum) / (B8_sum + B4_sum) | Active vegetation density |
| NDVI spring | (B8_spr − B4_spr) / (B8_spr + B4_spr) | Crop growth signal |
| **NDVI amplitude** | NDVI_summer − NDVI_winter | **Evergreen vs deciduous** — single most powerful feature |
| BSI (Bare Soil Index) | (B11+B4 − B8−B2) / (B11+B4 + B8+B2) | Bare soil vs vegetated |

**NDVI amplitude is the most important single feature** for this problem. Expected values per class:

| Class | NDVI Amplitude | Reasoning |
|---|---|---|
| Coniferous Forest | ~0.05–0.15 | Evergreen needles, minimal seasonal change |
| Olive Groves | ~0.05–0.15 | Evergreen broadleaf, stable year-round |
| Broad-leaved Forest | ~0.30–0.50 | Dramatic leaf-off in winter |
| Shrubland | ~0.20–0.35 | Partially deciduous |
| Grassland | ~0.20–0.30 | Dries in summer, partially green in winter |
| Agricultural | ~0.35–0.55 | Bare soil in winter, crops in spring/summer |
| Urban | ~0.05–0.10 | Stable impervious surface |
| Water | ~0.00–0.05 | Stable, low NDVI year-round |

Compute indices in GEE before export, or compute from the stacked arrays in Python after loading.

**Total input channels: 18 raw + 4 indices = 22 channels per patch.**

---

## Step 4 — Model Architecture

### Option A — Modified SoftCon (Practical, Lower Effort)

SoftCon's `conv1` was pretrained for 13 channels (full Sentinel-2). For 22 channels, initialize a new `conv1` and copy the weights for the 6 bands that overlap with SoftCon's original pretraining (B2, B3, B4, B8, B11, B12). Initialize the remaining 16 channels randomly.

```python
import torch
import torch.nn as nn

def adapt_conv1_for_multitemporal(model, n_channels=22):
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        n_channels, 64,
        kernel_size=7, stride=2, padding=3, bias=False
    )
    
    # Copy pretrained weights for B2, B3, B4, B8, B11, B12
    # SoftCon channel order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
    # Map indices:   B2=1, B3=2, B4=3, B8=7, B11=11, B12=12
    softcon_indices = [1, 2, 3, 7, 11, 12]
    
    # Summer bands at positions 6–11 in your 22-channel stack
    summer_positions = [6, 7, 8, 9, 10, 11]  # adjust to your stacking order
    
    with torch.no_grad():
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        for new_pos, softcon_pos in zip(summer_positions, softcon_indices):
            new_conv.weight[:, new_pos, :, :] = old_conv.weight[:, softcon_pos, :, :]
    
    model.conv1 = new_conv
    return model
```

Fine-tune with the same 3-stage discriminative strategy (freeze early layers, gradually unfreeze).

### Option B — Prithvi Foundation Model (Better Science, Recommended for Thesis)

IBM and NASA released **Prithvi-100M**, a vision transformer pretrained on multi-temporal Harmonized Landsat-Sentinel (HLS) data. It takes **6-channel patches across multiple time steps** — exactly the multi-temporal Sentinel-2 setup you're building.

Available on Hugging Face: `ibm-nasa-geospatial/Prithvi-100M`

```python
pip install transformers huggingface_hub
```

```python
from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn

# Load pretrained Prithvi backbone
config = AutoConfig.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")
backbone = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")

# Prithvi expects: (batch, time, channels, H, W) where channels=6, time=3
# Your input: stack 3 seasons × 6 bands → reshape to (B, 3, 6, 64, 64)

class PrithviClassifier(nn.Module):
    def __init__(self, backbone, n_classes=8):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        # x: (batch, 18, 64, 64) → reshape to (batch, 3, 6, 64, 64)
        B = x.shape[0]
        x = x.reshape(B, 3, 6, 64, 64)
        features = self.backbone(x).last_hidden_state
        # features: (B, seq_len, 768) → reshape to spatial
        # ... (follow Prithvi fine-tuning examples)
        return self.classifier(features)
```

**Why Prithvi is better for a thesis**: It is the state-of-the-art geospatial foundation model, designed for exactly this task. Comparing SoftCon fine-tuning vs Prithvi fine-tuning is a publishable scientific contribution in itself.

---

## Step 5 — Fix the 7-Class Class Imbalance Bug

When Shrubland was merged into Grassland, Grassland became 2,000 patches while all other classes stayed at 1,000. CrossEntropyLoss with no class weights silently biases the model toward predicting Grassland.

**Fix**: Add inverse-frequency class weights:

```python
import torch
import numpy as np

def compute_class_weights(y):
    counts = np.bincount(y)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)  # normalize
    return torch.FloatTensor(weights)

# In training setup:
weights = compute_class_weights(y_train).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

For the 7-class case, Grassland+Shrubland (2000 samples) gets weight ~0.5 relative to others. This ensures the model is not incentivized to predict Grassland by default.

---

## Step 6 — Fix Band Order Inconsistency

Your 4-band data is ordered `[B4, B3, B2, B8]` (Red, Green, Blue, NIR) but your 6-band data is ordered `[B2, B3, B4, B8, B11, B12]` (Blue, Green, Red, NIR, SWIR). The channel order was flipped between versions.

For the new multi-temporal pipeline, standardize to **Sentinel-2 natural order** for all composites:

```python
# Standard order for all exports — do not change between versions
BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]  # Blue, Green, Red, NIR, SWIR1, SWIR2
```

This ensures SoftCon/Prithvi pretrained weights (which expect standard Sentinel-2 order) are correctly aligned.

---

## Step 7 — Geographic Block Splitting

The current 70/15/15 random split allows nearby patches (which are spatially correlated) to appear in both train and test sets. This inflates test accuracy because the model has seen spectrally similar patches from the same area.

**Geographic block splitting** creates spatially disjoint train, validation, and test regions:

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Assign each patch to a geographic grid cell
# Then split grid cells into train/val/test (not individual patches)

def geographic_split(coords, labels, n_blocks=20, val_frac=0.15, test_frac=0.15, seed=42):
    lons = coords[:, 0]
    lats = coords[:, 1]
    
    # Create grid blocks
    lon_bins = np.linspace(lons.min(), lons.max(), int(np.sqrt(n_blocks)) + 1)
    lat_bins = np.linspace(lats.min(), lats.max(), int(np.sqrt(n_blocks)) + 1)
    
    lon_idx = np.digitize(lons, lon_bins) - 1
    lat_idx = np.digitize(lats, lat_bins) - 1
    block_idx = lon_idx * int(np.sqrt(n_blocks)) + lat_idx
    
    unique_blocks = np.unique(block_idx)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_blocks)
    
    n_test = int(len(unique_blocks) * test_frac)
    n_val = int(len(unique_blocks) * val_frac)
    
    test_blocks = unique_blocks[:n_test]
    val_blocks = unique_blocks[n_test:n_test + n_val]
    train_blocks = unique_blocks[n_test + n_val:]
    
    train_idx = np.where(np.isin(block_idx, train_blocks))[0]
    val_idx = np.where(np.isin(block_idx, val_blocks))[0]
    test_idx = np.where(np.isin(block_idx, test_blocks))[0]
    
    return train_idx, val_idx, test_idx
```

Note: geographic splitting may slightly lower reported test accuracy vs random splitting, but gives a more honest estimate of real-world performance. This is important for thesis validity.

---

## Realistic Accuracy Projections

| Configuration | Expected 8-class Accuracy |
|---|---|
| Current best (summer, 6-band, SoftCon) | 65% |
| + 9-band red edge (data already collected) | ~68–70% |
| + WorldCover 10m labels + purity filter | ~72–76% |
| + Multi-temporal (spring + summer + winter, 18-band) | ~80–85% |
| + NDVI amplitude + BSI indices (22-band) | ~83–87% |
| + Prithvi foundation model | **~87–92%** |

90% with all 8 classes requires the full stack: better labels, multi-temporal data, and a multi-temporal-aware model. With modified SoftCon + band stacking, expect to plateau around 83–87%.

---

## Implementation Order

Execute in this order — each step builds on the previous and has a measurable accuracy checkoint.

### Phase 1 — Low Effort, Existing Data (1–2 weeks)
1. Train the 9-band model (data already collected in `processed_v4`)
2. Fix the class imbalance bug in the 7-class model (add `weight=` to CrossEntropyLoss)
3. Fix band order inconsistency — standardize to `[B2, B3, B4, B8, B11, B12]`

**Expected gain**: ~3–5pp overall.

### Phase 2 — Better Labels (2–3 weeks)
4. Re-implement GEE sampling pipeline with ESA WorldCover 2021
5. Add purity filter (200m erosion before sampling)
6. Re-export and process dataset with new labels
7. Retrain SoftCon 6-band with new labels

**Expected gain**: ~5–10pp, especially Shrubland.

### Phase 3 — Multi-Temporal Data (3–4 weeks)
8. Collect spring composite (March–May 2021) — same 8,000 patch centers
9. Collect winter composite (Dec 2021 – Feb 2022) with snow masking
10. Compute NDVI amplitude and BSI per patch
11. Stack all 3 seasons + indices → 22-channel input
12. Adapt SoftCon conv1 for 22 channels
13. Retrain with 3-stage discriminative fine-tuning

**Expected gain**: ~15–20pp total from Phase 2 + 3.

### Phase 4 — Foundation Model (2–3 weeks, highest upside)
14. Install and test Prithvi-100M from Hugging Face
15. Fine-tune on 22-channel multi-temporal AlbaniaSAT
16. Compare Prithvi vs modified SoftCon — include both in thesis

**Expected gain**: Additional 3–5pp, potential to break 90%.

---

## Key References

- **ESA WorldCover 2021**: Zanaga et al. (2022). ESA WorldCover 10m 2021 v200. Zenodo. DOI: 10.5281/zenodo.7254221
- **Prithvi-100M**: Jakubik et al. (2023). Foundation Models for Generalist Geospatial Artificial Intelligence. arXiv:2310.18660
- **SoftCon**: Wang et al. (2023). Self-supervised pretraining for multi-modal remote sensing. SSL4EO-S12
- **Multi-temporal LULC**: Pelletier et al. (2019). Temporal Convolutional Neural Network for the Classification of Satellite Image Time Series. Remote Sensing, 11(5), 523
- **EuroSAT benchmark**: Helber et al. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. IEEE JSTARS
- **Discriminative fine-tuning**: Howard & Ruder (2018). Universal Language Model Fine-tuning for Text Classification. ACL 2018
