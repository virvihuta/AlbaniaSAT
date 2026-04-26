# AlbaniaSAT

**Land Use and Land Cover Classification for Albania using Sentinel-2 Satellite Imagery and Deep Learning**

HTL Diplomarbeit — Conference Paper Submission

---

## Overview

EuroSAT (Helber et al., 2019) established a benchmark for land use and land cover (LULC) classification across Europe using Sentinel-2 satellite imagery, achieving 98.57% accuracy with a ResNet-50 across 10 classes. **AlbaniaSAT** extends this work to Albania — a country with active land use changes, unique Mediterranean and Alpine geography, and no existing ML-ready satellite dataset.

We demonstrate a significant domain gap: a ResNet-50 model trained on EuroSAT achieves only **19.08% accuracy** on Albanian imagery — marginally above the 12.5% random baseline — motivating the need for a country-specific dataset. Fine-tuning on AlbaniaSAT raises accuracy to **65.33%** (8-class) and **73.33%** (7-class, Shrubland merged into Grassland), with ongoing work targeting 90% through multi-temporal data and improved labeling.

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
- **Source**: Sentinel-2 L2A, summer 2021, <10% cloud cover, median composite
- **Labels**: CORINE Land Cover 2018
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

| Version | Classes | Patches | Bands | Notes |
|---|---|---|---|---|
| `processed/` | 8 | 4,000 | 4 (RGB+NIR) | v1 only (500 per class) |
| `processed_v2/` | 8 | 8,000 | 4 (RGB+NIR) | v1 + v2, spatially disjoint |
| `processed_v3/` | 8 | 8,000 | 6 (RGB+NIR+SWIR) | B2,B3,B4,B8,B11,B12 |
| `processed_v4/` | 7 | 8,000 | 6 (RGB+NIR+SWIR) | Shrubland merged into Grassland |

---

## Results

### 8-Class Models

| Model | Pretraining | Bands | Patches | Test Accuracy |
|---|---|---|---|---|
| Random baseline | — | — | — | 12.50% |
| ResNet-50 zero-shot | EuroSAT | RGB | — | 19.08% |
| ResNet-50 fine-tuned | EuroSAT | RGB | 4,000 | 63.50% |
| SoftCon fine-tuned | SSL4EO-S12 | 4-band | 4,000 | 64.33% |
| ResNet-50 fine-tuned | EuroSAT | RGB | 8,000 | 64.58% |
| SoftCon fine-tuned | SSL4EO-S12 | 4-band | 8,000 | 64.58% |
| SoftCon fine-tuned | SSL4EO-S12 | 6-band | 8,000 | **65.33%** |

### 7-Class Model (Shrubland merged into Grassland)

| Model | Pretraining | Bands | Patches | Test Accuracy |
|---|---|---|---|---|
| SoftCon fine-tuned | SSL4EO-S12 | 6-band | 8,000 | **73.33%** |

### Per-Class Accuracy (Best 8-Class Model — SoftCon 4-band)

| Class | Accuracy | Notes |
|---|---|---|
| Water | 94.8% | Strong NIR absorption — unique signature |
| Coniferous Forest | 82.3% | SoftCon pretraining captures needle reflectance |
| Urban | 83.1% | Distinct from vegetated classes |
| Broad-leaved Forest | 76.3% | Dense canopy → high NIR |
| Olive Groves | 71.1% | Mediterranean-specific signature |
| Agricultural | 54.8% | Confused with dry Grassland in summer |
| Grassland | 51.9% | Overlaps Agricultural and Shrubland |
| Shrubland | 22.7% | Primary bottleneck — label noise + spectral ambiguity |

### Key Findings

1. **Domain gap is real and large**: EuroSAT drops from 98.57% on Europe to 19.08% on Albania — 80 percentage points — motivating a country-specific dataset.
2. **Fine-tuning recovers most performance**: +44pp over zero-shot transfer from AlbaniaSAT data alone.
3. **Dataset size has diminishing returns**: Doubling from 4,000 to 8,000 patches gives only +0.25pp, suggesting spectral ambiguity — not sample count — is the bottleneck.
4. **SWIR bands help marginally at class level**: Adding B11, B12 improves Grassland (+6.5pp) but hurts Coniferous Forest (−7.5pp), indicating a band-weighting problem rather than a simple gain.
5. **Shrubland is the structural bottleneck**: CORINE 323/324 are transition classes at 100m resolution — label noise at 10m is the primary cause of the 22.7% accuracy.

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
│   │   ├── processed/                # 4,000 patches — 4-band
│   │   ├── processed_v2/             # 8,000 patches — 4-band (main training set)
│   │   ├── processed_v3/             # 8,000 patches — 6-band
│   │   ├── processed_v4/             # 8,000 patches — 6-band, 7-class
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
3. **Sentinel-2 composite** — L2A, summer 2021, <10% cloud cover, median composite
4. **CORINE analysis** — identify land cover classes present in Albania
5. **Stratified sampling** — 1,000 CORINE-verified points per class with 640m exclusion zones for v2
6. **Patch export** — 64×64 pixel neighborhoods exported to Google Drive as GeoJSON

Download exports from Google Drive and place in `data/AlbaniaSAT/raw/` (v1) and `data/AlbaniaSAT/raw_v2/` (v2). Then run `notebooks/albaniasat.ipynb` to build the NumPy arrays.

---

## Known Issues

- **7-class class imbalance**: Merging Shrubland into Grassland creates a 2,000-patch Grassland class vs 1,000 for all others. The current model uses unweighted CrossEntropyLoss, biasing predictions toward Grassland. Fix: add inverse-frequency class weights.
- **Band order inconsistency**: 4-band data is ordered `[B4, B3, B2, B8]` but 6-band data is `[B2, B3, B4, B8, B11, B12]`. New exports should standardize to Sentinel-2 natural order.
- **3 duplicate patches**: Detected via mean fingerprinting in `albaniasat.ipynb`. Negligible impact (0.04% of dataset).

---

## Roadmap

See [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) for the full strategy. Summary:

| Phase | What | Expected Gain |
|---|---|---|
| 1 | 9-band training + imbalance fix + band order fix | +3–5pp |
| 2 | ESA WorldCover 2021 labels (10m) + purity filter | +5–10pp |
| 3 | Multi-temporal: spring + summer + winter composites, 18-band | +10–15pp |
| 4 | Computed indices or Prithvi foundation model (if Phase 3 plateaus) | +3–5pp |

Realistic target: **82–87%** on 8 classes. 90% is a stretch goal dependent on label quality.

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
