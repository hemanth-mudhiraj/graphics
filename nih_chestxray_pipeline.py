###############################################################################
# ==================== PART 1: INSTALLATIONS ====================
# Run this cell first. Restart runtime after if prompted.
###############################################################################

# !pip install dask[dataframe] pyarrow fastparquet scikit-learn --quiet

###############################################################################
# ==================== PART 2: IMPORTS & CONFIGURATION ====================
# Core imports, seed setting, memory utilities, and global config.
###############################################################################

import os
import gc
import sys
import glob
import time
import random
import warnings
import numpy as np
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image, ImageFile
from sklearn.metrics import roc_auc_score, f1_score, classification_report

warnings.filterwarnings("ignore")
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images gracefully

# --------------- Global Configuration ---------------
class CFG:
    # Paths — adjust if your Kaggle input path differs
    BASE_DIR = "/kaggle/input/data"
    OUTPUT_DIR = "/kaggle/working"
    PARQUET_PATH = "/kaggle/working/metadata.parquet"

    # Model
    MODEL_NAME = "densenet121"  # CheXNet-style architecture
    IMG_SIZE = 224
    NUM_CLASSES = 14
    PRETRAINED = True

    # Training
    EPOCHS = 5
    BATCH_SIZE = 48          # per-GPU batch; effective = 48 * num_gpus
    ACCUMULATION_STEPS = 2   # gradient accumulation for larger effective batch
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2          # Kaggle has limited CPU cores
    PIN_MEMORY = True

    # Memory management
    CHUNK_SIZE = 10000       # rows per chunk when processing CSV
    GC_INTERVAL = 500        # garbage collect every N batches during training
    PREFETCH_FACTOR = 2

    # Reproducibility
    SEED = 42

    # Disease labels (in order)
    DISEASE_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]

def seed_everything(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

def mem_report(tag=""):
    """Print current memory usage."""
    import psutil
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1e9
    gpu_alloc = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
    print(f"[MEM {tag}] RAM: {rss:.2f}GB | GPU alloc: {gpu_alloc:.2f}GB | GPU reserved: {gpu_reserved:.2f}GB")

def aggressive_gc():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
mem_report("init")


###############################################################################
# ==================== PART 3: DATA LOADING & PARQUET CONVERSION ====================
# Use Dask to process the large CSV in chunks, convert to Parquet,
# and build train/val/test splits. This is the most memory-critical step.
###############################################################################

def discover_image_dirs(base_dir):
    """Find all image directories (handles both flat and nested structures)."""
    # NIH dataset on Kaggle: images in images_001/images/, images_002/images/, etc.
    pattern_nested = os.path.join(base_dir, "images_*/images")
    nested_dirs = sorted(glob.glob(pattern_nested))

    if nested_dirs:
        print(f"Found {len(nested_dirs)} nested image directories")
        return nested_dirs

    # Flat structure: all images directly in base_dir or base_dir/images
    flat_dir = os.path.join(base_dir, "images")
    if os.path.isdir(flat_dir):
        return [flat_dir]

    # Images directly in base_dir
    return [base_dir]

def build_image_path_map(image_dirs):
    """Build a dictionary mapping filename -> full path (chunked to save memory)."""
    path_map = {}
    for img_dir in image_dirs:
        for fname in os.listdir(img_dir):
            if fname.lower().endswith(".png"):
                path_map[fname] = os.path.join(img_dir, fname)
        gc.collect()
        print(f"  Indexed {img_dir}: total files so far = {len(path_map)}")
    return path_map

def process_metadata_to_parquet():
    """
    Load the CSV metadata using Dask, process in chunks, and save as Parquet.
    This avoids loading the entire CSV + image paths into RAM at once.
    """
    if os.path.exists(CFG.PARQUET_PATH):
        print(f"Parquet already exists at {CFG.PARQUET_PATH}, skipping conversion.")
        return

    print("=== Step 1: Discovering image directories ===")
    image_dirs = discover_image_dirs(CFG.BASE_DIR)
    print(f"Image directories: {image_dirs}")

    print("\n=== Step 2: Building image path index ===")
    path_map = build_image_path_map(image_dirs)
    print(f"Total images indexed: {len(path_map)}")

    print("\n=== Step 3: Loading CSV with Dask ===")
    # Find the CSV file
    csv_candidates = [
        os.path.join(CFG.BASE_DIR, "Data_Entry_2017_v2020.csv"),
        os.path.join(CFG.BASE_DIR, "Data_Entry_2017.csv"),
    ]
    csv_path = None
    for c in csv_candidates:
        if os.path.exists(c):
            csv_path = c
            break

    if csv_path is None:
        # Search for any CSV
        csvs = glob.glob(os.path.join(CFG.BASE_DIR, "*.csv"))
        csv_path = csvs[0] if csvs else None
        if csv_path is None:
            raise FileNotFoundError(f"No CSV found in {CFG.BASE_DIR}")

    print(f"Using CSV: {csv_path}")

    # Read with Dask — lazy, no memory spike
    ddf = dd.read_csv(csv_path)
    print(f"Columns: {list(ddf.columns)}")

    # Process in pandas chunks via Dask partitions
    print("\n=== Step 4: Processing labels into multi-hot vectors ===")
    all_chunks = []
    for i, partition in enumerate(ddf.to_delayed()):
        chunk = partition.compute()

        # Rename columns for consistency
        col_map = {}
        for col in chunk.columns:
            cl = col.strip().lower().replace(" ", "_")
            if "image" in cl and "index" in cl:
                col_map[col] = "image_index"
            elif "finding" in cl and "label" in cl:
                col_map[col] = "finding_labels"
            elif "patient" in cl and "id" in cl:
                col_map[col] = "patient_id"
            elif "patient" in cl and "age" in cl:
                col_map[col] = "patient_age"
            elif "patient" in cl and "gender" in cl:
                col_map[col] = "patient_gender"
        chunk = chunk.rename(columns=col_map)

        # Multi-hot encode disease labels
        for disease in CFG.DISEASE_LABELS:
            chunk[disease] = chunk["finding_labels"].str.contains(disease, na=False).astype(np.int8)

        # Map image paths
        chunk["image_path"] = chunk["image_index"].map(path_map)
        # Drop rows without valid image paths
        chunk = chunk.dropna(subset=["image_path"])

        # Keep only needed columns to save memory
        keep_cols = ["image_index", "image_path", "finding_labels", "patient_id"] + CFG.DISEASE_LABELS
        keep_cols = [c for c in keep_cols if c in chunk.columns]
        chunk = chunk[keep_cols]

        all_chunks.append(chunk)

        if (i + 1) % 5 == 0:
            gc.collect()
            print(f"  Processed partition {i+1}")

    print("\n=== Step 5: Combining and saving to Parquet ===")
    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    aggressive_gc()

    print(f"Total samples: {len(df)}")
    print(f"Disease distribution:")
    for disease in CFG.DISEASE_LABELS:
        print(f"  {disease}: {df[disease].sum()}")

    # Save as Parquet — much faster to reload and lower memory footprint
    df.to_parquet(CFG.PARQUET_PATH, engine="pyarrow", index=False)
    print(f"\nParquet saved to {CFG.PARQUET_PATH} ({os.path.getsize(CFG.PARQUET_PATH)/1e6:.1f} MB)")

    del df, path_map
    aggressive_gc()
    mem_report("after_parquet")


def load_splits():
    """
    Load train/val/test splits from the provided text files.
    Returns three DataFrames read efficiently from Parquet.
    """
    print("=== Loading data from Parquet ===")
    df = pd.read_parquet(CFG.PARQUET_PATH, engine="pyarrow")
    print(f"Loaded {len(df)} rows from Parquet")

    # Load split files
    train_val_path = os.path.join(CFG.BASE_DIR, "train_val_list.txt")
    test_path = os.path.join(CFG.BASE_DIR, "test_list.txt")

    train_val_list = set()
    test_list = set()

    if os.path.exists(train_val_path):
        with open(train_val_path, "r") as f:
            train_val_list = set(line.strip() for line in f if line.strip())
        print(f"Train/Val list: {len(train_val_list)} images")

    if os.path.exists(test_path):
        with open(test_path, "r") as f:
            test_list = set(line.strip() for line in f if line.strip())
        print(f"Test list: {len(test_list)} images")

    if train_val_list and test_list:
        train_val_mask = df["image_index"].isin(train_val_list)
        test_mask = df["image_index"].isin(test_list)

        df_train_val = df[train_val_mask].reset_index(drop=True)
        df_test = df[test_mask].reset_index(drop=True)

        # Split train_val into train (90%) and val (10%) — patient-level split
        patients = df_train_val["patient_id"].unique()
        np.random.shuffle(patients)
        val_patients = set(patients[:len(patients) // 10])

        val_mask = df_train_val["patient_id"].isin(val_patients)
        df_val = df_train_val[val_mask].reset_index(drop=True)
        df_train = df_train_val[~val_mask].reset_index(drop=True)
    else:
        # Fallback: random 70/15/15 split by patient
        print("Split files not found, using random patient-level split")
        patients = df["patient_id"].unique()
        np.random.shuffle(patients)
        n = len(patients)
        train_patients = set(patients[:int(0.7 * n)])
        val_patients = set(patients[int(0.7 * n):int(0.85 * n)])

        df_train = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
        df_val = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)
        df_test = df[~df["patient_id"].isin(train_patients | val_patients)].reset_index(drop=True)

    del df
    aggressive_gc()

    print(f"\nFinal splits — Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    return df_train, df_val, df_test


# ---- RUN DATA PROCESSING ----
process_metadata_to_parquet()
df_train, df_val, df_test = load_splits()
mem_report("after_splits")


###############################################################################
# ==================== PART 4: DATASET & DATALOADERS ====================
# Memory-efficient PyTorch Dataset that loads images on-the-fly.
# Images are resized to 224x224 and normalized for DenseNet.
###############################################################################

class ChestXrayDataset(Dataset):
    """
    Memory-efficient dataset: stores only file paths and labels as numpy arrays.
    Images are loaded and transformed on-the-fly.
    """
    def __init__(self, df, transform=None, label_cols=None):
        self.image_paths = df["image_path"].values
        self.label_cols = label_cols or CFG.DISEASE_LABELS
        # Store labels as int8 numpy array — minimal memory
        self.labels = df[self.label_cols].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = torch.from_numpy(self.labels[idx])

        try:
            # Load as RGB, resize immediately to save memory
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            else:
                # Fallback: basic resize + tensor
                img = transforms.functional.resize(img, (CFG.IMG_SIZE, CFG.IMG_SIZE))
                img = transforms.functional.to_tensor(img)
        except Exception as e:
            # Return a black image if loading fails (rare but handles corruption)
            print(f"Warning: failed to load {img_path}: {e}")
            img = torch.zeros(3, CFG.IMG_SIZE, CFG.IMG_SIZE)

        return img, label


# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32)),  # Slight upscale for crop
    transforms.RandomCrop(CFG.IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Create datasets
train_dataset = ChestXrayDataset(df_train, transform=train_transforms)
val_dataset = ChestXrayDataset(df_val, transform=val_transforms)
test_dataset = ChestXrayDataset(df_test, transform=val_transforms)

# Free the DataFrames — dataset holds only numpy arrays now
del df_train, df_val, df_test
aggressive_gc()

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.BATCH_SIZE,
    shuffle=True,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY,
    prefetch_factor=CFG.PREFETCH_FACTOR,
    drop_last=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CFG.BATCH_SIZE * 2,  # larger batch for validation (no grads)
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY,
    prefetch_factor=CFG.PREFETCH_FACTOR,
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.BATCH_SIZE * 2,
    shuffle=False,
    num_workers=CFG.NUM_WORKERS,
    pin_memory=CFG.PIN_MEMORY,
    prefetch_factor=CFG.PREFETCH_FACTOR,
    persistent_workers=True,
)

print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
mem_report("after_dataloaders")


###############################################################################
# ==================== PART 5: MODEL ARCHITECTURE ====================
# DenseNet121 (CheXNet-style) with multi-label sigmoid output.
# Supports multi-GPU via DataParallel.
###############################################################################

class CheXNet(nn.Module):
    """
    DenseNet121-based model for 14-class multi-label chest X-ray classification.
    This mirrors the original CheXNet architecture.
    """
    def __init__(self, num_classes=CFG.NUM_CLASSES, pretrained=CFG.PRETRAINED):
        super(CheXNet, self).__init__()
        self.densenet = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT if pretrained else None
        )
        num_features = self.densenet.classifier.in_features
        # Replace classifier with our multi-label head
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.densenet(x)


def build_model():
    """Build model, move to GPU, and wrap with DataParallel if multiple GPUs."""
    model = CheXNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model, device

model, device = build_model()
mem_report("after_model")


###############################################################################
# ==================== PART 6: TRAINING LOOP ====================
# Mixed-precision training with gradient accumulation, periodic GC,
# and learning rate scheduling. Saves best model by val AUROC.
###############################################################################

def compute_class_weights(dataset):
    """Compute positive class weights for imbalanced multi-label data."""
    labels = dataset.labels  # numpy array (N, 14)
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    # pos_weight = num_neg / num_pos (capped to avoid extreme values)
    weights = np.clip(neg_counts / (pos_counts + 1e-5), a_min=1.0, a_max=50.0)
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixed precision forward pass
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CFG.ACCUMULATION_STEPS  # Scale for accumulation

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % CFG.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * CFG.ACCUMULATION_STEPS
        num_batches += 1

        # Periodic garbage collection
        if (batch_idx + 1) % CFG.GC_INTERVAL == 0:
            aggressive_gc()

        # Progress logging
        if (batch_idx + 1) % 200 == 0:
            avg_loss = running_loss / num_batches
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(loader)} | Loss: {avg_loss:.4f}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()

        # Sigmoid to get probabilities
        preds = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())

        # Periodic GC during validation too
        if (batch_idx + 1) % (CFG.GC_INTERVAL // 2) == 0:
            aggressive_gc()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_loss = running_loss / max(len(loader), 1)

    # Compute per-class AUROC
    aurocs = []
    for i, disease in enumerate(CFG.DISEASE_LABELS):
        try:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aurocs.append(auc)
        except ValueError:
            aurocs.append(0.5)  # If only one class present

    mean_auroc = np.mean(aurocs)
    return avg_loss, mean_auroc, aurocs, all_preds, all_labels


def train(model, train_loader, val_loader, device):
    """Full training loop with mixed precision, LR scheduling, and checkpointing."""
    # Class weights for imbalanced data
    pos_weight = compute_class_weights(train_dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG.LEARNING_RATE,
        weight_decay=CFG.WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-6
    )

    scaler = GradScaler()

    best_auroc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(CFG.EPOCHS):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)

        # Validate
        val_loss, val_auroc, per_class_aurocs, _, _ = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Logging
        print(f"\nEpoch {epoch+1}/{CFG.EPOCHS} ({epoch_time:.0f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val AUROC:  {val_auroc:.4f} (mean)")
        print(f"  Per-class AUROCs:")
        for disease, auc in zip(CFG.DISEASE_LABELS, per_class_aurocs):
            print(f"    {disease:22s}: {auc:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)

        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch + 1
            save_path = os.path.join(CFG.OUTPUT_DIR, "best_model.pth")
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_auroc,
                "per_class_aurocs": per_class_aurocs,
            }, save_path)
            print(f"  >>> New best model saved (AUROC: {val_auroc:.4f})")

        aggressive_gc()
        mem_report(f"epoch_{epoch+1}")

    print(f"\nTraining complete! Best AUROC: {best_auroc:.4f} at epoch {best_epoch}")
    return history

# ---- RUN TRAINING ----
history = train(model, train_loader, val_loader, device)


###############################################################################
# ==================== PART 7: EVALUATION ON TEST SET ====================
# Load best model, compute AUROC, F1, and classification report.
###############################################################################

def evaluate_test(model, test_loader, device):
    """Full evaluation on the held-out test set."""
    # Load best checkpoint
    ckpt_path = os.path.join(CFG.OUTPUT_DIR, "best_model.pth")
    if os.path.exists(ckpt_path):
        print("Loading best model checkpoint...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        if hasattr(model, "module"):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from epoch {checkpoint['epoch']} (val AUROC: {checkpoint['val_auroc']:.4f})")

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_auroc, per_class_aurocs, all_preds, all_labels = validate(
        model, test_loader, criterion, device
    )

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Mean AUROC: {test_auroc:.4f}")
    print(f"\nPer-class AUROCs:")
    for disease, auc in zip(CFG.DISEASE_LABELS, per_class_aurocs):
        print(f"  {disease:22s}: {auc:.4f}")

    # Compute F1 at threshold 0.5
    binary_preds = (all_preds >= 0.5).astype(int)

    print(f"\nPer-class F1 scores (threshold=0.5):")
    f1_scores = []
    for i, disease in enumerate(CFG.DISEASE_LABELS):
        f1 = f1_score(all_labels[:, i], binary_preds[:, i], zero_division=0)
        f1_scores.append(f1)
        print(f"  {disease:22s}: {f1:.4f}")
    print(f"  {'Mean F1':22s}: {np.mean(f1_scores):.4f}")

    # Save predictions
    pred_df = pd.DataFrame(all_preds, columns=CFG.DISEASE_LABELS)
    pred_df.to_csv(os.path.join(CFG.OUTPUT_DIR, "test_predictions.csv"), index=False)
    print(f"\nPredictions saved to {CFG.OUTPUT_DIR}/test_predictions.csv")

    return test_auroc, per_class_aurocs, all_preds, all_labels

# ---- RUN EVALUATION ----
test_auroc, per_class_aurocs, test_preds, test_labels = evaluate_test(model, test_loader, device)
aggressive_gc()
mem_report("after_eval")


###############################################################################
# ==================== PART 8: INFERENCE / PREDICTION ====================
# Predict on a single image or batch of images.
###############################################################################

def predict_single_image(model, image_path, device, top_k=5):
    """
    Run inference on a single chest X-ray image.
    Returns disease probabilities and top-k predictions.
    """
    model.eval()

    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    transform = val_transforms
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad(), autocast():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Format results
    results = []
    for i, disease in enumerate(CFG.DISEASE_LABELS):
        results.append({"disease": disease, "probability": float(probs[i])})

    results.sort(key=lambda x: x["probability"], reverse=True)

    print(f"\nPrediction for: {os.path.basename(image_path)}")
    print("-" * 40)
    for r in results[:top_k]:
        bar = "#" * int(r["probability"] * 30)
        print(f"  {r['disease']:22s}: {r['probability']:.4f} {bar}")

    return results


def predict_batch(model, image_paths, device):
    """
    Run inference on a batch of images. Returns a DataFrame of probabilities.
    """
    model.eval()
    transform = val_transforms

    all_probs = []
    batch_size = CFG.BATCH_SIZE * 2

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        tensors = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
            except Exception:
                tensors.append(torch.zeros(3, CFG.IMG_SIZE, CFG.IMG_SIZE))

        batch_tensor = torch.stack(tensors).to(device)

        with torch.no_grad(), autocast():
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

        all_probs.append(probs)

        del batch_tensor, logits, probs
        if start % (batch_size * 10) == 0:
            aggressive_gc()

    all_probs = np.concatenate(all_probs, axis=0)
    result_df = pd.DataFrame(all_probs, columns=CFG.DISEASE_LABELS)
    result_df.insert(0, "image_path", image_paths[:len(result_df)])

    return result_df


# ---- EXAMPLE: Predict on a sample test image ----
sample_path = test_dataset.image_paths[0]
if os.path.exists(sample_path):
    predict_single_image(model, sample_path, device)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
mem_report("final")


###############################################################################
# ==================== PART 9: TRAINING VISUALIZATION (OPTIONAL) ====================
# Plot training curves. Run this cell only if matplotlib is available.
###############################################################################

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(history["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # AUROC curve
    axes[1].plot(history["val_auroc"], label="Val AUROC", marker="o", color="green")
    axes[1].set_title("Validation AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Per-class AUROC bar chart (final)
    axes[2].barh(CFG.DISEASE_LABELS, per_class_aurocs, color="steelblue")
    axes[2].set_title("Test AUROC per Disease")
    axes[2].set_xlabel("AUROC")
    axes[2].axvline(x=test_auroc, color="red", linestyle="--", label=f"Mean: {test_auroc:.3f}")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Training curves saved.")
except ImportError:
    print("matplotlib not available — skipping visualization.")
