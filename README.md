# NIH Chest X-ray Multi-Label Classification Pipeline

An end-to-end deep learning pipeline for classifying 14 thoracic diseases from chest X-ray images using the [NIH Chest X-ray Dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community). Built on a **CheXNet-style DenseNet121** architecture with memory-efficient data handling, mixed-precision training, and multi-GPU support — optimized to run within Kaggle's resource constraints.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Stages](#pipeline-stages)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [Training Strategy](#training-strategy)
- [Evaluation Metrics](#evaluation-metrics)
- [Inference](#inference)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

This pipeline takes raw NIH Chest X-ray images and metadata CSV, processes them into an efficient Parquet format, trains a DenseNet121-based classifier, evaluates on a held-out test set, and provides single-image and batch inference capabilities — all in a single script.

**Diseases classified (14 labels):**

| | | | |
|---|---|---|---|
| Atelectasis | Cardiomegaly | Effusion | Infiltration |
| Mass | Nodule | Pneumonia | Pneumothorax |
| Consolidation | Edema | Emphysema | Fibrosis |
| Pleural Thickening | Hernia | | |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIGH-LEVEL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Raw Data │───>│  Dask    │───>│ Parquet  │───>│ PyTorch  │ │
│  │  CSV +    │    │  Chunk   │    │ Storage  │    │ Dataset  │ │
│  │  Images   │    │ Process  │    │          │    │          │ │
│  └───────────┘    └──────────┘    └──────────┘    └────┬─────┘ │
│                                                        │       │
│                                                        v       │
│  ┌───────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │  Results  │<───│  Test    │<───│ Training │<───│  Data    │ │
│  │  AUROC,   │    │  Eval    │    │  Loop    │    │ Loaders  │ │
│  │  F1, CSV  │    │          │    │ (AMP+GA) │    │          │ │
│  └───────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    CheXNet Model                          │  │
│  │  ┌─────────────┐   ┌─────────┐   ┌───────┐   ┌───────┐  │  │
│  │  │ DenseNet121 │──>│ FC(512) │──>│ ReLU  │──>│ FC(14)│  │  │
│  │  │ (backbone)  │   │         │   │+Drop  │   │       │  │  │
│  │  └─────────────┘   └─────────┘   └───────┘   └───────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

The script is organized into **9 sequential parts**:

### Part 1 — Installations
Installs required packages (`dask`, `pyarrow`, `fastparquet`, `scikit-learn`). Designed for Kaggle notebook environments.

### Part 2 — Imports & Configuration
- Centralizes all hyperparameters and paths in a `CFG` class
- Sets deterministic seeds for full reproducibility (`seed_everything`)
- Provides memory monitoring (`mem_report`) and garbage collection (`aggressive_gc`) utilities

### Part 3 — Data Loading & Parquet Conversion
The most memory-critical stage:
1. **Discovers** image directories (handles both flat and nested NIH Kaggle structures like `images_001/images/`, `images_002/images/`, etc.)
2. **Builds** a filename-to-path mapping for all `.png` images
3. **Reads** the metadata CSV lazily via **Dask** (no full-file memory spike)
4. **Processes** each Dask partition: normalizes column names, multi-hot encodes the 14 disease labels, maps image paths
5. **Saves** the combined DataFrame as a **Parquet** file for fast reloading
6. **Loads splits** using NIH-provided `train_val_list.txt` / `test_list.txt` with a patient-level 90/10 train/val split (falls back to random 70/15/15 if split files are absent)

### Part 4 — Dataset & DataLoaders
- `ChestXrayDataset`: stores only file paths + labels as numpy arrays; loads and transforms images on-the-fly
- **Train augmentations**: random crop, horizontal flip, rotation (±10°), color jitter, ImageNet normalization
- **Val/Test transforms**: deterministic resize + ImageNet normalization
- DataLoaders with pinned memory, prefetching, and persistent workers

### Part 5 — Model Architecture
- **CheXNet**: DenseNet121 backbone (ImageNet-pretrained) with a custom multi-label classification head
- Head: `Linear(1024→512) → ReLU → Dropout(0.3) → Linear(512→14)`
- Auto-wraps with `nn.DataParallel` when multiple GPUs are detected

### Part 6 — Training Loop
- **Mixed-precision training** via `torch.cuda.amp` (FP16 forward/backward, FP32 optimizer)
- **Gradient accumulation** (default 2 steps) for a larger effective batch size
- **Gradient clipping** (max norm 1.0) for stability
- **Cosine annealing** LR schedule
- **Class-weighted BCE loss** (`pos_weight = neg_count / pos_count`, capped at 50) to handle severe label imbalance
- **Checkpointing**: saves the model with the best validation mean AUROC
- Periodic garbage collection during training to prevent OOM

### Part 7 — Test Evaluation
- Loads the best checkpoint
- Reports per-class AUROC and F1 scores (threshold = 0.5)
- Saves raw prediction probabilities to `test_predictions.csv`

### Part 8 — Inference
- `predict_single_image()`: processes one X-ray, prints top-k disease probabilities with a visual bar
- `predict_batch()`: processes a list of image paths, returns a DataFrame of probabilities

### Part 9 — Visualization (Optional)
- Plots training/validation loss curves, validation AUROC curve, and per-class test AUROC bar chart
- Saves to `training_curves.png`

---

## Project Structure

```
Graphics/
├── nih_chestxray_pipeline.py   # Main pipeline (all 9 parts)
├── _clean_pipeline.py          # Cleanup/utility script
├── README.md                   # This file
│
├── /kaggle/input/data/         # Expected input data (Kaggle)
│   ├── Data_Entry_2017_v2020.csv
│   ├── train_val_list.txt
│   ├── test_list.txt
│   ├── images_001/images/*.png
│   ├── images_002/images/*.png
│   └── ...
│
└── /kaggle/working/            # Output directory (Kaggle)
    ├── metadata.parquet        # Processed metadata cache
    ├── best_model.pth          # Best model checkpoint
    ├── test_predictions.csv    # Test set predictions
    └── training_curves.png     # Training visualization
```

---

## Configuration

All hyperparameters are centralized in the `CFG` class:

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `densenet121` | Backbone architecture |
| `IMG_SIZE` | `224` | Input image resolution |
| `NUM_CLASSES` | `14` | Number of disease labels |
| `EPOCHS` | `5` | Training epochs |
| `BATCH_SIZE` | `48` | Per-GPU batch size |
| `ACCUMULATION_STEPS` | `2` | Gradient accumulation (effective batch = 96) |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `WEIGHT_DECAY` | `1e-5` | AdamW weight decay |
| `NUM_WORKERS` | `2` | DataLoader workers |
| `CHUNK_SIZE` | `10000` | CSV processing chunk size |
| `GC_INTERVAL` | `500` | Garbage collection frequency (batches) |
| `SEED` | `42` | Random seed |

---

## Dataset

**NIH Chest X-ray Dataset** (also known as ChestX-ray14):
- ~112,120 frontal-view chest X-ray images from 30,805 unique patients
- 14 disease labels extracted from radiology reports using NLP
- Multi-label: each image can have zero or more diseases
- Official train/test split provided via text files
- Significant class imbalance (e.g., Hernia is rare; Infiltration is common)

---

## Model Details

The model follows the **CheXNet** architecture (Rajpurkar et al., 2017):

```
Input Image (224×224×3)
        │
        v
┌─────────────────────┐
│    DenseNet121       │   ← ImageNet-pretrained backbone
│  (feature extractor) │     121 layers, dense connections
│                     │     Output: 1024-dim feature vector
└────────┬────────────┘
         │
         v
┌─────────────────────┐
│   Linear(1024, 512) │
│   ReLU              │
│   Dropout(0.3)      │
│   Linear(512, 14)   │   ← Multi-label classification head
└────────┬────────────┘
         │
         v
   14 logits → Sigmoid → Probabilities (one per disease)
```

**Loss function**: `BCEWithLogitsLoss` with per-class positive weights to combat label imbalance.

---

## Training Strategy

| Technique | Purpose |
|---|---|
| **Mixed-Precision (AMP)** | 2x memory savings, faster forward/backward passes |
| **Gradient Accumulation** | Larger effective batch size without more GPU memory |
| **Gradient Clipping** | Prevents exploding gradients (max norm = 1.0) |
| **Cosine Annealing LR** | Smooth decay from `1e-4` to `1e-6` over training |
| **Class-Weighted Loss** | Handles severe imbalance (pos_weight = neg/pos, capped at 50) |
| **Patient-Level Splits** | Prevents data leakage (same patient never in train + val/test) |
| **Aggressive GC** | Periodic garbage collection + CUDA cache clearing to prevent OOM |
| **DataParallel** | Automatic multi-GPU when available |
| **Persistent Workers** | DataLoader workers stay alive between epochs (lower overhead) |

---

## Evaluation Metrics

- **AUROC** (Area Under ROC Curve): primary metric, computed per-class and averaged — threshold-independent measure of discriminative ability
- **F1 Score** (at threshold 0.5): per-class and mean — balances precision and recall
- **Classification Report**: full precision/recall/F1 breakdown

---

## Inference

### Single Image
```python
results = predict_single_image(model, "path/to/xray.png", device, top_k=5)
# Prints top-5 disease probabilities with visual bars
```

### Batch
```python
df = predict_batch(model, list_of_image_paths, device)
# Returns DataFrame: [image_path, Atelectasis, Cardiomegaly, ..., Hernia]
```

---

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- dask[dataframe]
- pyarrow / fastparquet
- pandas, numpy
- scikit-learn
- Pillow
- psutil
- matplotlib (optional, for visualization)

---

## Usage

### On Kaggle
1. Add the **NIH Chest X-ray** dataset to your notebook
2. Run all cells in `nih_chestxray_pipeline.py` sequentially
3. Outputs appear in `/kaggle/working/`

### Locally
1. Update paths in the `CFG` class:
   ```python
   CFG.BASE_DIR = "/path/to/nih-chest-xray-data"
   CFG.OUTPUT_DIR = "/path/to/output"
   CFG.PARQUET_PATH = "/path/to/output/metadata.parquet"
   ```
2. Run:
   ```bash
   python nih_chestxray_pipeline.py
   ```

### Outputs
| File | Description |
|---|---|
| `metadata.parquet` | Cached processed metadata (skips reprocessing on reruns) |
| `best_model.pth` | Best checkpoint (model weights, optimizer state, AUROC) |
| `test_predictions.csv` | Raw sigmoid probabilities for every test image |
| `training_curves.png` | Loss, AUROC, and per-class performance plots |
