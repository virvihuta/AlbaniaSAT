# AlbaniaSAT

**Land Use and Land Cover Classification for Albania using Sentinel-2 Satellite Imagery and Deep Learning**

HTL Diplomarbeit

---

## Overview

EuroSAT (Helber et al., 2019) established a benchmark for land use and land cover classification across Europe using Sentinel-2 satellite imagery. **AlbaniaSAT** extends this work to Albania — a country with active land use changes and no existing ML-ready satellite dataset.

This repository contains:
- The data pipeline for building the AlbaniaSAT dataset from Sentinel-2 imagery
- Model training and evaluation code (ResNet-50 fine-tuned baseline + experiments)
- FastAPI inference backend
- Results and figures

The accompanying web application (AlbaniaSAT Explorer) is in a separate repository.

---

## Team

| Name | Role |
|------|------|
| Virvi Huta | Dataset, model, backend, paper |
| Suisa | Frontend, map UI, design |
| Regis | Infrastructure, deployment |
| Reuf Kozlica | Supervisor |

---

## Results

| Model | Dataset | Bands | Accuracy |
|-------|---------|-------|----------|
| ResNet-50 (fine-tuned) | EuroSAT RGB | RGB | 98.57% (target) |
| ResNet-50 (fine-tuned) | AlbaniaSAT | RGB | TBD |

---

## Repository Structure

```
albaniasat/
│
├── data/                   # NOT tracked by git — download separately
│   ├── EuroSAT_RGB/        # Original EuroSAT dataset
│   ├── raw/                # Downloaded Sentinel-2 imagery for Albania
│   ├── patches/            # Cropped 64x64 patches
│   └── labeled/            # Final labeled AlbaniaSAT dataset
│
├── notebooks/
│   ├── exploration.ipynb   # Data exploration and visualization
│   └── experiments.ipynb   # Model experiments
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
