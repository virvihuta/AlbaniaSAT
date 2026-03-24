# AlbaniaSAT

**Land Use and Land Cover Classification for Albania using Sentinel-2 Satellite Imagery and Deep Learning**

HTL Diplomarbeit — Conference Paper Submission

---

## Overview

EuroSAT (Helber et al., 2019) established a benchmark for land use and land cover (LULC) classification across Europe using Sentinel-2 satellite imagery, achieving 98.57% accuracy with CNNs across 10 classes. **AlbaniaSAT** extends this work to Albania — a country with active land use changes, unique Mediterranean and Alpine geography, and no existing ML-ready satellite dataset.

We demonstrate a significant domain gap: a ResNet-50 model trained on EuroSAT achieves only **19.08% accuracy** on Albanian imagery — marginally above the random baseline of 12.5% — motivating the need for a country-specific dataset. Fine-tuning on AlbaniaSAT raises accuracy to **64.58%**, with further improvements expected through multispectral band expansion (SWIR).

This repository contains:
- The full GEE-based data pipeline for building the AlbaniaSAT dataset from Sentinel-2 L2A imagery
- Model training and evaluation code (ResNet-50 and SoftCon fine-tuned baselines)
- Ablation experiments across dataset sizes, band configurations, and pretraining strategies
- Results, figures, and confusion matrices

---

## Team

| Name | Role |
|------|------|
| Virvi Huta | Dataset, model, training, evaluation, paper |
| Suisa Shehi | Frontend, map UI, design |
| Regis Balla | Infrastructure, deployment |
| Reuf Kozlica | Supervisor |

---

## Dataset

### AlbaniaSAT v1
- **8 land cover classes** (CORINE Land Cover 2018 verified)
- **1,000 patches per class** (500 original + 500 spatially non-overlapping)
- **8,000 total patches**
- **64×64 pixels** at 10m resolution (640m × 640m per patch)
- **4 spectral bands**: B4 (Red), B3 (Green), B2 (Blue), B8 (NIR)
- **Source**: Sentinel-2 L2A, summer 2021, <10% cloud cover, median composite

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

**Note**: Olive Groves is the key Albania-specific class absent from EuroSAT, reflecting the country's Mediterranean agricultural landscape.

### Why These Classes

Broad-leaved and Coniferous Forest are kept separate — they are spectrally distinct in the NIR band and merging them would discard real signal. Olive Groves is retained as the single most scientifically distinctive Albanian class. Classes were selected based on CORINE pixel coverage and spectral separability at 10m resolution.

---

## Results

### Main Benchmark

| Model | Pretraining | Bands | Patches | Test Accuracy |
|---|---|---|---|---|
| Random baseline | — | — | — | 12.50% |
| ResNet-50 zero-shot | EuroSAT | RGB | — | 19.08% |
| ResNet-50 fine-tuned | EuroSAT | RGB | 4,000 | 63.50% |
| SoftCon fine-tuned | SSL4EO-S12 | 4-band | 4,000 | 64.33% |
| ResNet-50 fine-tuned | EuroSAT | RGB | 8,000 | 64.58% |
| SoftCon fine-tuned | SSL4EO-S12 | 4-band | 8,000 | **64.58%** |

### Key Findings

1. **Domain gap is real and significant**: EuroSAT model drops from 98.57% on European imagery to 19.08% on Albania — an 80 percentage point degradation that motivates AlbaniaSAT.

2. **Fine-tuning recovers most performance**: A 44 percentage point improvement over zero-shot transfer by fine-tuning on AlbaniaSAT data.

3. **Dataset size is the primary bottleneck**: Both ResNet-50 and SoftCon converge to ~64.5% regardless of pretraining domain or band count at 1,000 patches per class, suggesting spectral ambiguity among vegetation classes (Shrubland, Grassland, Agricultural) as the limiting factor.

4. **SWIR bands are the recommended next step**: Bands B11 and B12 are specifically designed to distinguish vegetation types and are expected to resolve class confusion. This is identified as the primary future work direction.

---

## Training Strategy

We use **discriminative fine-tuning** (gradual unfreezing) across 3 stages:

| Stage | Trainable Layers | Learning Rate | Epochs |
|---|---|---|---|
| 1 | fc only | 1e-3 | 5 |
| 2 | fc + layer4 | 1e-4 | 10 |
| 3 | fc + layer4 + layer3 | 1e-5 | 10 |

Early layers (layer1, layer2) remain frozen throughout to preserve universal satellite features and prevent catastrophic forgetting. Data augmentation (random flips, 90° rotations) is applied to training data only.

---

## Repository Structure

```
albaniasat/
│
├── data/                         # NOT tracked by git
│   ├── EuroSAT_RGB/              # Original EuroSAT dataset
│   ├── AlbaniaSAT/
│   │   ├── raw/                  # v1 GeoJSON exports (500 per class)
│   │   ├── raw_v2/               # v2 GeoJSON exports (500 new per class)
│   │   ├── processed/            # Numpy arrays — 4000 patches
│   │   ├── processed_v2/         # Numpy arrays — 8000 patches
│   │   ├── images/               # JPG patches — 500 per class
│   │   └── images_v2/            # JPG patches — 1000 per class
│
├── notebooks/
│   ├── data_pipeline.ipynb       # GEE data collection pipeline
│   ├── albaniasat.ipynb          # Dataset building and visualization
│   ├── training.ipynb            # Model training — all versions
│   └── evaluation_v2.ipynb       # Final evaluation and comparison
│
├── results/
│   ├── models/                   # Saved model weights (not tracked)
│   │   ├── resnet50_eurosat.pth
│   │   ├── resnet50_albaniasat_v2.pth
│   │   ├── resnet50_albaniasat_v3.pth
│   │   ├── resnet50_softcon_albaniasat.pth
│   │   └── resnet50_softcon_albaniasat_v3.pth
│   └── figures/
│       ├── sample_patches.png
│       ├── confusion_matrix.png
│       └── training_curves.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/albaniasat
cd albaniasat

# Create virtual environment
conda create -n albaniasat python=3.11
conda activate albaniasat

# Install dependencies
pip install -r requirements.txt

# Authenticate Google Earth Engine
earthengine authenticate
```

---

## Data Pipeline

Data collection runs entirely in Google Earth Engine (GEE). Open `notebooks/data_pipeline.ipynb` and run cells in order:

1. **GEE initialization** — connect with project `albaniasat`
2. **Albania boundary** — loaded from USDOS/LSIB_SIMPLE/2017
3. **Sentinel-2 composite** — L2A, summer 2021, <10% cloud, median
4. **CORINE analysis** — identify land cover classes present in Albania
5. **Stratified sampling** — 1000 CORINE-verified points per class
6. **Patch export** — 64×64 neighborhoods exported to Google Drive

Exports land in `AlbaniaSAT_v3` and `AlbaniaSAT_v4` on Google Drive. Download and place in `data/AlbaniaSAT/raw/` and `data/AlbaniaSAT/raw_v2/` respectively.

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
pandas
scikit-learn
huggingface_hub
timm
```

---

## Future Work

- **SWIR bands (B11, B12)**: Primary recommendation — expected to resolve spectral ambiguity among vegetation classes
- **2000 patches per class**: Expand dataset with spatially non-overlapping sampling
- **Full 13-band SoftCon**: Evaluate SoftCon with complete Sentinel-2 band set
- **Temporal analysis**: Multi-season composites to improve seasonal class separation
- **Geographic block splitting**: Address spatial autocorrelation in train/test splits

---

## References

```
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

@misc{wang2024multilabel,
  title={Multi-Label Guided Soft Contrastive Learning for Efficient Earth Observation Pretraining},
  author={Wang, Yi and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2405.20462},
  year={2024}
}
```

---

## License

Dataset: MIT License

Code: MIT License
│
├── src/
│   ├── dataset.py          # Dataset class, data loading, transforms
│   ├── model.py            # ResNet-50 setup and fine-tuning
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation, confusion matrix, accuracy
│   └── inference.py        # Run model on new patches
│
├── api/
│   └── main.py             # FastAPI inference backend
│
├── results/
│   ├── models/             # Saved model weights (not tracked by git)
│   └── figures/            # Confusion matrices, accuracy plots
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/albaniasat
cd albaniasat

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download EuroSAT RGB dataset and place in data/EuroSAT_RGB/
# https://www.kaggle.com/datasets/waseemalastal/eurosat-rgb-dataset
```

---

## Training

```bash
# Reproduce EuroSAT baseline
python src/train.py --dataset eurosat --epochs 30

# Train on AlbaniaSAT
python src/train.py --dataset albaniasat --epochs 30
```

---

## Dataset

The EuroSAT RGB dataset contains 27,000 labeled 64×64 pixel satellite image patches across 10 land use and land cover classes sourced from Sentinel-2 imagery of 34 European countries.

AlbaniaSAT follows the same structure but covers Albanian terrain specifically — download instructions and pipeline scripts coming soon.

---

## References

```
@article{helber2019eurosat,
  title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
  author={Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2019}
}
```

---

## License

Dataset: MIT License

Code: MIT License
