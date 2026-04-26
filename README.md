# AlbaniaSAT

**Land Use and Land Cover Classification for Albania using Sentinel-2 Satellite Imagery and Deep Learning**

HTL Diplomarbeit

---

## Overview

EuroSAT (Helber et al., 2019) established a benchmark for land use and land cover (LULC) classification across Europe using Sentinel-2 satellite imagery, achieving 98.57% accuracy with a ResNet-50 across 10 classes. **AlbaniaSAT** extends this work to Albania — a country with active land use changes, unique Mediterranean and Alpine geography, and no existing ML-ready satellite dataset.

We demonstrate a significant domain gap: a ResNet-50 model trained on EuroSAT achieves only **19.08% accuracy** on Albanian imagery — marginally above the 12.5% random baseline — motivating the need for a country-specific dataset. Through progressive improvements — fine-tuning, better labels (ESA WorldCover 2021), and multi-temporal composites — accuracy reaches **77.35%** (8-class, 18-band spring+summer+winter) and **73.33%** (7-class, summer-only baseline).

---

## Team

| Name | Role |
|---|---|
| Virvi Huta | Dataset, model, training, evaluation, paper |
| Suisa Shehi | Frontend, map UI, design |
| Regis Balla | Infrastructure, deployment |
| Reuf Kozlica | Supervisor |

---

## Dataset

### AlbaniaSAT v1 (current)

- **8 land cover classes**
- **1,000 patches per class** — 8,000 total
- **64×64 pixels** at 10m resolution (640m × 640m per patch)
- **Source**: Sentinel-2 L2A, <10% cloud cover, median composite
- **Seasons**: Summer 2021 (Jun–Sep), Spring 2021 (Mar–May), Winter 2021–22 (Dec–Feb)
- **Labels**: CORINE Land Cover 2018 (forest + olive groves) + ESA WorldCover 2021 (remaining classes)
- **Spatial disjointness**: v2 sample points are ≥640m from v1 points (exclusion zone enforced in GEE)

### Class Definitions

| Class | CORINE Codes | Patches |
|---|---|---|
| Broad-leaved Forest | 311 | 1,000 |
| Coniferous Forest | 312 | 1,000 |
| Shrubland | 323, 324 | 1,000 |
| Agricultural | 211, 242, 243 | 1,000 |
| Grassland | 231, 321 | 1,000 |
| Olive Groves | 223 | 1,000 |
| Urban | 112 | 1,000 |
| Water | 512 | 1,000 |

Olive Groves is the key Albania-specific class absent from EuroSAT, reflecting the country's Mediterranean agricultural landscape. Broad-leaved and Coniferous Forest are kept separate — they are spectrally distinct in the NIR band and merging them would discard real signal.

### Dataset Versions

| Version | Classes | Patches | Bands | Labels | Notes |
|---|---|---|---|---|---|
| `processed/` | 8 | 4,000 | 4 (RGB+NIR) | CORINE | v1 only (500 per class) |
| `processed_v2/` | 8 | 8,000 | 4 (RGB+NIR) | CORINE | v1 + v2, spatially disjoint |
| `processed_v3/` | 8 | 8,000 | 6 (RGB+NIR+SWIR) | CORINE | B2,B3,B4,B8,B11,B12 |
| `processed_v4/` | 7 | 8,000 | 6 (RGB+NIR+SWIR) | CORINE | Shrubland merged into Grassland |
| `processed_phase2/` | 8 | 8,000 | 6 (RGB+NIR+SWIR) | WorldCover+CORINE | Hybrid labels, summer only |
| `processed_phase3/` | 8 | ~8,000 | 18 (6 bands × 3 seasons) | WorldCover+CORINE | Multi-temporal: spring+summer+winter |
| `processed_phase3_ndvi/` | 8 | ~8,000 | 22 (18 raw + 4 indices) | WorldCover+CORINE | + NDVI amplitude, BSI, seasonal NDVI |

---

## Results

### Full Progression

| Phase | Model | Labels | Bands | Seasons | Test Accuracy |
|---|---|---|---|---|---|
| Baseline | Random | — | — | — | 12.50% |
| Baseline | ResNet-50 zero-shot | EuroSAT | RGB | — | 19.08% |
| Baseline | ResNet-50 fine-tuned | CORINE | RGB | Summer | 63.50% |
| Baseline | SoftCon fine-tuned | CORINE | 4-band | Summer | 64.33% |
| Baseline | SoftCon fine-tuned | CORINE | 4-band | Summer | 64.58% |
| Baseline | SoftCon fine-tuned | CORINE | 6-band | Summer | 65.33% |
| Phase 1 | SoftCon fine-tuned | CORINE | 6-band | Summer | 73.33% *(7-class)* |
| Phase 2 | SoftCon fine-tuned | WorldCover+CORINE | 6-band | Summer | 70.50% |
| Phase 3 | SoftCon fine-tuned | WorldCover+CORINE | 18-band | Spring+Summer+Winter | **77.35%** |
| Phase 3+ | SoftCon fine-tuned | WorldCover+CORINE | 22-band | Spring+Summer+Winter | 77.61% |

### Per-Class Accuracy — Best 8-Class Model (Phase 3, 18-band multi-temporal)

| Class | Phase 2 (6-band, summer) | Phase 3 (18-band, multi-temporal) | Change |
|---|---|---|---|
| Water | 96.1% | ~96% | ≈0 |
| Agricultural | 84.5% | ~85% | ≈0 |
| Urban | 77.2% | ~83% | +6pp |
| Coniferous Forest | 80.3% | ~83% | +3pp |
| Olive Groves | 78.3% | ~79% | +1pp |
| Broad-leaved Forest | 71.9% | ~78% | +6pp |
| Grassland | 44.8% | ~58% | +13pp |
| Shrubland | 31.8% | ~34% | +2pp |

### Key Findings

1. **Domain gap is real and large**: EuroSAT drops from 98.57% on Europe to 19.08% on Albania — 80 percentage points — motivating a country-specific dataset.
2. **Fine-tuning recovers most performance**: +45pp over zero-shot transfer from AlbaniaSAT fine-tuning alone.
3. **Label quality matters more than band count**: Switching from CORINE (100m) to WorldCover (10m) improved Agricultural accuracy by +29.7pp. Adding SWIR bands alone gave only +0.75pp.
4. **Multi-temporal is the largest single improvement**: Going from single-season (summer) to three seasons (spring+summer+winter) delivered +6.85pp — the biggest jump in the project.
5. **Computed indices are redundant given raw temporal bands**: Adding NDVI amplitude and BSI to the 18-band stack gave only +0.26pp. The raw seasonal bands already encode phenological information.
6. **Shrubland remains the structural ceiling**: 22–34% across all configurations. CORINE 323/324 are transition classes — the label itself is ambiguous regardless of band count or season.

---

## Training Strategy

**Discriminative fine-tuning** (gradual unfreezing) across 3 stages:

| Stage | Trainable Layers | Learning Rate | Epochs |
|---|---|---|---|
| 1 | fc only | 1e-3 | 5 |
| 2 | fc + layer4 | 1e-4 | 10 |
| 3 | fc + layer4 + layer3 | 1e-5 | 10 |

Early layers (layer1, layer2) remain frozen throughout to preserve universal satellite features. Data augmentation (random horizontal/vertical flips, 90° rotations) is applied to training data only. Optimizer: Adam. Scheduler: CosineAnnealingLR.

---

## Repository Structure

```
AlbaniaSAT/
│
├── data/                             # NOT tracked by git
│   ├── AlbaniaSAT/
│   │   ├── raw/                      # v1 GeoJSON exports from GEE (500 per class)
│   │   ├── raw_v2/                   # v2 GeoJSON exports (500 new per class)
│   │   ├── raw_v3/                   # v3 GeoJSON exports (6-band, split v1/v2)
│   │   ├── processed/                # 4,000 patches — 4-band, CORINE
│   │   ├── processed_v2/             # 8,000 patches — 4-band, CORINE
│   │   ├── processed_v3/             # 8,000 patches — 6-band, CORINE
│   │   ├── processed_v4/             # 8,000 patches — 6-band, 7-class, CORINE
│   │   ├── processed_phase2/         # 8,000 patches — 6-band, WorldCover+CORINE
│   │   ├── processed_phase3/         # ~8,000 patches — 18-band, multi-temporal
│   │   ├── processed_phase3_ndvi/    # ~8,000 patches — 22-band, multi-temporal + indices
│   │   └── images_v2/                # JPG exports for visual inspection
│
├── notebooks/
│   ├── data_pipeline.ipynb           # GEE collection — composites, sampling, export
│   ├── albaniasat.ipynb              # Dataset building, QC, visualization
│   ├── training.ipynb                # All training runs — ResNet-50 and SoftCon
│   ├── evaluation_v2.ipynb           # Final evaluation, confusion matrices, comparison
│   └── eurostat_reproduction.ipynb   # EuroSAT zero-shot on AlbaniaSAT (domain gap)
│
├── results/
│   ├── models/                       # Saved model weights (not tracked by git)
│   └── figures/                      # Confusion matrices, band distributions, samples
│
├── IMPROVEMENT_PLAN.md               # Roadmap to 90% accuracy
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/virvihuta/AlbaniaSAT
cd AlbaniaSAT

conda create -n albaniasat python=3.11
conda activate albaniasat

pip install -r requirements.txt

# Authenticate Google Earth Engine
earthengine authenticate --project albaniasat
```

---

## Data Pipeline

Data collection runs entirely in Google Earth Engine. Open `notebooks/data_pipeline.ipynb` and run cells in order:

1. **GEE initialization** — connect with project `albaniasat`
2. **Albania boundary** — loaded from `USDOS/LSIB_SIMPLE/2017`
3. **Sentinel-2 composites** — L2A, <10% cloud cover, median; three seasons: spring (Mar–May), summer (Jun–Sep), winter (Dec–Feb) with SCL-based snow/cloud masking
4. **Label sources** — CORINE 2018 for forest + olive groves; ESA WorldCover 2021 for remaining classes; 200m purity filter applied before sampling
5. **Stratified sampling** — 1,000 points per class with 640m exclusion zones between v1 and v2
6. **Patch export** — 64×64 pixel neighborhoods exported to Google Drive as GeoJSON

Download exports from Google Drive and place under `data/AlbaniaSAT/`. Then run `notebooks/albaniasat.ipynb` to build the NumPy arrays and stack seasons into 18-channel patches.

---

## Known Issues

- **Shrubland ceiling**: 22–34% accuracy across all configurations. CORINE 323/324 (transitional woodland-shrub / sclerophyllous vegetation) are inherently ambiguous at 10m resolution. Merging into Grassland (7-class) is one practical resolution.
- **WorldCover Grassland noise**: Switching to WorldCover improved Agricultural (+29.7pp) but degraded Grassland (−13.6pp), suggesting WorldCover's Grassland boundary is noisier than CORINE for Albanian terrain specifically.
- **3 duplicate patches**: Detected via mean fingerprinting in `albaniasat.ipynb`. Negligible impact (0.04% of dataset).

---

## Roadmap

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the full strategy.

| Phase | What | Actual Result |
|---|---|---|
| 1 | 7-class imbalance fix + band order fix | 73.33% (7-class) ✓ |
| 2 | ESA WorldCover 2021 labels + purity filter | 70.50% (8-class) ✓ |
| 3 | Multi-temporal 18-band (spring+summer+winter) | **77.35%** (8-class) ✓ |
| 3+ | Computed indices (22-band) | 77.61% — negligible gain ✓ |
| 4 | Prithvi foundation model | pending |

Current best: **77.35%** on 8 classes. Next step: retrain 7-class model with 18-band multi-temporal input — expected to reach 85–88%.

---

## Requirements

```
torch>=2.0
torchvision
numpy
earthengine-api
geemap
Pillow
matplotlib
scikit-learn
huggingface_hub
timm
```

---

## References

```bibtex
@article{helber2019eurosat,
  title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019}
}

@article{wang2022ssl4eo,
  title={SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation},
  author={Wang, Yi and Braham, Nassim Ait Ali and Xiong, Zhitong and Liu, Chenying and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2211.07044},
  year={2022}
}

@misc{wang2024softcon,
  title={Multi-Label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining},
  author={Wang, Yi and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2405.20462},
  year={2024}
}

@misc{zanaga2022worldcover,
  title={ESA WorldCover 10m 2021 v200},
  author={Zanaga, D and Van De Kerchove, R and Daems, D and De Keersmaecker, W and Brockmann, C and Kirches, G and Wevers, J and Cartus, O and Santoro, M and Fritz, S and Lesiv, M and Herold, M and Tsendbazar, N E and Xu, P and Ramoino, F and Arino, O},
  year={2022},
  doi={10.5281/zenodo.7254221}
}
```

---

## License

Code: MIT License
Dataset: MIT License
