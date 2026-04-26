# AlbaniaSAT

**Land Use and Land Cover Classification for Albania using Sentinel-2 Satellite Imagery and Deep Learning**

HTL Diplomarbeit

---

## Overview

EuroSAT (Helber et al., 2019) established a benchmark for land use and land cover (LULC) classification across Europe using Sentinel-2 satellite imagery, achieving 98.57% accuracy with a ResNet-50 across 10 classes. **AlbaniaSAT** extends this work to Albania — a country with active land use changes, unique Mediterranean and Alpine geography, and no existing ML-ready satellite dataset.

We demonstrate a significant domain gap: a ResNet-50 trained on EuroSAT achieves only **19.08% accuracy** on Albanian imagery — marginally above the 12.5% random baseline. Through iterative improvements across labels, spectral bands, and temporal coverage, we reach **80.94%** on a 7-class model using 18-band multi-temporal Sentinel-2 composites (spring + summer + winter).

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

### AlbaniaSAT v1

- **7 land cover classes** (8-class variant also available)
- **~7,800 patches** across 3 seasons (some classes slightly under 1,000 due to winter availability)
- **64×64 pixels** at 10m resolution (640m × 640m per patch)
- **Source**: Sentinel-2 L2A — spring (Mar–May), summer (Jun–Sep), winter (Dec–Feb) 2021
- **Labels**: CORINE Land Cover 2018 for forests + olive groves; ESA WorldCover 2021 for remaining classes
- **Spatial disjointness**: v2 sample points are ≥640m from v1 points (exclusion zone enforced in GEE)
- **Snow masking**: SCL-based per-pixel masking applied to winter composite (classes 3, 8, 9, 11)

### Class Definitions

| Class | Label Source | CORINE Codes | Patches |
|---|---|---|---|
| Broad-leaved Forest | CORINE | 311 | ~1,000 |
| Coniferous Forest | CORINE | 312 | ~1,000 |
| Natural Vegetation (Grassland + Shrubland) | WorldCover | 20, 30 | ~1,884 |
| Agricultural | WorldCover | 40 | ~1,000 |
| Olive Groves | CORINE | 223 | ~1,000 |
| Urban | WorldCover | 50 | ~1,000 |
| Water | WorldCover | 80 | ~1,000 |

Grassland and Shrubland are merged into a single **Natural Vegetation** class — the two were consistently confused across every configuration (summer-only, multi-temporal, 4-band, 18-band) due to genuine spectral and ecological overlap. This reflects the classification reality at 10m resolution and matches ESA WorldCover's own class scheme. An 8-class variant (Shrubland separate) is also available for comparison.

Olive Groves is the key Albania-specific class absent from EuroSAT, reflecting the country's Mediterranean agricultural landscape.

### Band Configuration

All multi-temporal patches stack three seasonal composites:

```
[B2_spr, B3_spr, B4_spr, B8_spr, B11_spr, B12_spr,   ← spring
 B2_sum, B3_sum, B4_sum, B8_sum, B11_sum, B12_sum,   ← summer
 B2_win, B3_win, B4_win, B8_win, B11_win, B12_win]   ← winter
```

18 channels total. Computed indices (NDVI amplitude, BSI) were tested and found redundant — the CNN extracts these patterns from raw bands internally (+0.26pp, not worth the added complexity).

### Dataset Versions

| Version | Classes | Patches | Bands | Labels | Notes |
|---|---|---|---|---|---|
| `processed/` | 8 | 4,000 | 4 (RGB+NIR) | CORINE | v1 only |
| `processed_v2/` | 8 | 8,000 | 4 (RGB+NIR) | CORINE | v1 + v2, spatially disjoint |
| `processed_v3/` | 8 | 8,000 | 6 (+SWIR) | CORINE | B2,B3,B4,B8,B11,B12 |
| `processed_v4/` | 7 | 8,000 | 6 (+SWIR) | CORINE | Shrubland merged, summer only |
| `processed_phase2/` | 8 | 8,000 | 6 (+SWIR) | WorldCover+CORINE | Summer only |
| `processed_phase3/` | 8 | ~7,800 | 18 (3 seasons) | WorldCover+CORINE | Multi-temporal |
| `processed_phase3_7class/` | **7** | ~7,800 | 18 (3 seasons) | WorldCover+CORINE | **Primary dataset** |
| `processed_phase3_ndvi/` | 8 | ~7,800 | 22 (+indices) | WorldCover+CORINE | NDVI amplitude + BSI |

---

## Results

### Full Progression

| Phase | Labels | Bands | Seasons | Classes | Test Accuracy |
|---|---|---|---|---|---|
| EuroSAT zero-shot | — | RGB | — | 8 | 19.08% |
| Baseline fine-tuned | CORINE | RGB | Summer | 8 | 63.50% |
| Baseline fine-tuned | CORINE | 4-band | Summer | 8 | 64.58% |
| Baseline fine-tuned | CORINE | 6-band | Summer | 8 | 65.33% |
| Phase 2 | WorldCover+CORINE | 6-band | Summer | 8 | 70.50% |
| Phase 3 | WorldCover+CORINE | 18-band | Spring+Summer+Winter | 8 | 77.35% |
| Phase 3 + indices | WorldCover+CORINE | 22-band | Spring+Summer+Winter | 8 | 77.61% |
| **Phase 3 — 7-class** | **WorldCover+CORINE** | **18-band** | **Spring+Summer+Winter** | **7** | **80.94%** |

### Per-Class Accuracy — Best Model (Phase 3, 18-band, 7-class)

| Class | Correct | Total | Accuracy |
|---|---|---|---|
| Water | 145 | 151 | 96.0% |
| Urban | 126 | 137 | 91.9% |
| Agricultural | 137 | 161 | 85.1% |
| Coniferous Forest | 117 | 141 | 83.0% |
| Olive Groves | 131 | 167 | 78.4% |
| Broad-leaved Forest | 91 | 126 | 72.2% |
| Natural Vegetation | 200 | 287 | 69.7% |

Every class exceeds 69%. No single class is a structural outlier.

### Key Findings

1. **Domain gap is real and large**: EuroSAT drops from 98.57% on Europe to 19.08% on Albania — 80pp — proving the need for a country-specific dataset.

2. **Label quality is the primary bottleneck, not model capacity**: Switching from CORINE (100m) to WorldCover (10m) gave +5pp overall and +29.7pp on Agricultural alone. Adding SWIR bands gave only +0.75pp. More data (4K→8K patches) gave only +0.25pp.

3. **Multi-temporal data is the largest single improvement**: Adding spring and winter composites alongside summer delivered +6.85pp (70.50% → 77.35%) — more than any other intervention.

4. **Computed indices are redundant**: NDVI amplitude and BSI added to the 18-band stack gave +0.26pp. The CNN extracts equivalent information from raw multi-temporal bands internally.

5. **Grassland/Shrubland is not separable at 10m with Sentinel-2**: The class pair confused each other in every configuration — summer, multi-temporal, 4-band, 18-band, CORINE labels, WorldCover labels. This is a structural property of the classification problem, not a data or model failure. Merging into Natural Vegetation is the correct scientific conclusion.

6. **Fine-tuning strategy matters**: Discriminative fine-tuning (gradual 3-stage unfreezing) consistently outperformed full fine-tuning by preserving low-level satellite features in early ResNet layers.

---

## Training Strategy

**SoftCon ResNet-50** (pretrained on SSL4EO-S12 via self-supervised learning) with **discriminative fine-tuning** across 3 stages:

| Stage | Trainable Layers | Learning Rate | Epochs |
|---|---|---|---|
| 1 | fc only | 1e-3 | 5 |
| 2 | fc + layer4 | 1e-4 | 10 |
| 3 | fc + layer4 + layer3 | 1e-5 | 10 |

Early layers (layer1, layer2) remain frozen to preserve universal satellite features. `conv1` adapted from 13 channels to 18 channels — pretrained weights copied for the 6 summer bands (B2, B3, B4, B8, B11, B12 at their SoftCon indices), remaining channels initialized with Kaiming normal. Data augmentation: random horizontal/vertical flips and 90° rotations (training only). Optimizer: Adam. Scheduler: CosineAnnealingLR.

---

## Repository Structure

```
AlbaniaSAT/
│
├── data/                                 # NOT tracked by git
│   ├── AlbaniaSAT/
│   │   ├── raw/                          # v1 GeoJSON from GEE (500 per class, summer)
│   │   ├── raw_v2/                       # v2 GeoJSON (500 new per class, summer)
│   │   ├── raw_v3/                       # 6-band GeoJSON split v1/v2
│   │   ├── raw_phase2/                   # WorldCover-labeled GeoJSON
│   │   ├── raw_phase3/                   # Multi-temporal GeoJSON (3 seasons)
│   │   ├── processed/                    # 4,000 patches — 4-band, CORINE
│   │   ├── processed_v2/                 # 8,000 patches — 4-band, CORINE
│   │   ├── processed_v3/                 # 8,000 patches — 6-band, CORINE
│   │   ├── processed_v4/                 # 8,000 patches — 6-band, 7-class, CORINE
│   │   ├── processed_phase2/             # 8,000 patches — 6-band, WorldCover+CORINE
│   │   ├── processed_phase3/             # ~7,800 patches — 18-band, 8-class
│   │   ├── processed_phase3_7class/      # ~7,800 patches — 18-band, 7-class [PRIMARY]
│   │   ├── processed_phase3_ndvi/        # ~7,800 patches — 22-band, 8-class
│   │   └── images_v2/                    # JPG exports for visual inspection
│
├── notebooks/
│   ├── data_pipeline.ipynb               # GEE collection — composites, sampling, export
│   ├── albaniasat.ipynb                  # Dataset building, QC, multi-temporal stacking
│   ├── training.ipynb                    # All training runs — all phases
│   ├── evaluation_v2.ipynb               # Evaluation, confusion matrices, comparison
│   └── eurostat_reproduction.ipynb       # EuroSAT zero-shot on AlbaniaSAT
│
├── results/
│   ├── models/                           # Saved model weights (not tracked by git)
│   │   ├── resnet50_softcon_phase3_7class.pth   # Best model — 80.94%
│   │   ├── resnet50_softcon_phase3.pth          # 8-class multi-temporal — 77.35%
│   │   ├── resnet50_softcon_phase2.pth          # WorldCover labels — 70.50%
│   │   └── ...                                  # Earlier baseline models
│   └── figures/                          # Confusion matrices, band distributions, samples
│
├── IMPROVEMENT_PLAN.md                   # Strategy document and phase log
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

Data collection runs in Google Earth Engine (`notebooks/data_pipeline.ipynb`):

1. **GEE initialization** — project `albaniasat`
2. **Albania boundary** — `USDOS/LSIB_SIMPLE/2017`
3. **Three seasonal composites** — Sentinel-2 L2A, SCL snow/cloud masking, median composite
4. **Hybrid labels** — WorldCover 2021 (10m) for Vegetation, Agricultural, Urban, Water; CORINE 2018 for Forest types and Olive Groves; 200m purity erosion filter applied before sampling
5. **Stratified sampling** — ~1,000 points per class; v2 points ≥640m from v1
6. **Patch export** — 64×64 neighborhoods to Google Drive as GeoJSON

Run `notebooks/albaniasat.ipynb` to build and stack the NumPy arrays into 18-channel patches.

---

## Known Limitations

- **Natural Vegetation ceiling at ~70%**: Grassland and Shrubland share spectral space at 10m resolution across every season and band combination tested. The merged class is the correct scientific response.
- **WorldCover Grassland noise**: WorldCover improved Agricultural (+29.7pp) but degraded the original Grassland class (−13.6pp), suggesting its Grassland boundary is noisier than CORINE for Albanian terrain specifically.
- **Winter patch count reduced**: Some classes have fewer than 1,000 patches in the multi-temporal dataset (~938–1,000) due to snow/cloud masking removing valid summer points from the winter composite.
- **3 duplicate patches**: Detected in quality control. Negligible (0.04% of dataset).

---

## Roadmap

| Phase | Configuration | Result | Status |
|---|---|---|---|
| 1 | 7-class imbalance fix + band order fix | 73.33% (7-class, summer) | ✓ Done |
| 2 | WorldCover labels + purity filter | 70.50% (8-class, summer) | ✓ Done |
| 3 | 18-band multi-temporal, 8-class | 77.35% | ✓ Done |
| 3+ | 22-band + computed indices | 77.61% — redundant | ✓ Done |
| 3 | 18-band multi-temporal, 7-class | **80.94%** | ✓ Done |
| 4 | Prithvi foundation model | — | Pending |

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

@misc{jakubik2023prithvi,
  title={Foundation Models for Generalist Geospatial Artificial Intelligence},
  author={Jakubik, Johannes and Roy, Sujit and Phillips, C E and Fraccaro, Paolo and Godwin, Denys and Zadrozny, Bianca and Szwarcman, Daniela and Gomes, Carlos and Nyirjesy, Gabby and Edwards, Blair and others},
  journal={arXiv preprint arXiv:2310.18660},
  year={2023}
}
```

---

## License

Code: MIT License
Dataset: MIT License
