###############################################################################
# ==================== CELL 1: IMPORTS & CONFIGURATION ====================
# Run this cell first. All dependencies for visualization.
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    average_precision_score, classification_report, f1_score,
    multilabel_confusion_matrix, roc_auc_score
)
from itertools import cycle
import warnings

warnings.filterwarnings("ignore")

# ---- Project Configuration ----
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

OUTPUT_DIR = "."  # Change if needed
THRESHOLD = 0.5   # Classification threshold for binary predictions

# Style settings
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")

print("Cell 1 complete: Imports loaded.")


###############################################################################
# ==================== CELL 2: LOAD DATA ====================
# Load test predictions, ground-truth labels, and training history.
# Adjust paths to match your output directory.
###############################################################################

# --- Option A: Load from CSV files produced by the pipeline ---
# If you have test_predictions.csv and metadata.parquet:
#
# test_preds = pd.read_csv(f"{OUTPUT_DIR}/test_predictions.csv").values
# df = pd.read_parquet(f"{OUTPUT_DIR}/metadata.parquet")
# test_labels = df[DISEASE_LABELS].values[-len(test_preds):]

# --- Option B: Use variables already in memory from the pipeline ---
# test_preds, test_labels, history, per_class_aurocs should already exist
# after running nih_chestxray_pipeline.py

# --- Option C: Generate realistic synthetic data for demonstration ---
# This lets you preview all graphs without running the full pipeline.

np.random.seed(42)
N_TEST = 5000   # number of test samples
N_EPOCHS = 5    # number of training epochs

# Simulate ground-truth labels (matches NIH dataset class imbalance)
prevalence = [0.103, 0.025, 0.119, 0.177, 0.051, 0.056,
              0.013, 0.047, 0.042, 0.021, 0.022, 0.015, 0.030, 0.002]
test_labels = np.zeros((N_TEST, 14), dtype=np.int32)
for i, p in enumerate(prevalence):
    test_labels[:, i] = np.random.binomial(1, p, N_TEST)

# Simulate model prediction probabilities (correlated with true labels)
test_preds = np.zeros((N_TEST, 14), dtype=np.float32)
for i in range(14):
    noise = np.random.beta(2, 5, N_TEST)
    test_preds[:, i] = np.where(
        test_labels[:, i] == 1,
        np.clip(0.5 + 0.3 * np.random.randn(N_TEST), 0.1, 0.99),
        np.clip(noise, 0.01, 0.9)
    )

# Simulate training history
history = {
    "train_loss": [0.35, 0.22, 0.17, 0.14, 0.12],
    "val_loss":   [0.30, 0.23, 0.20, 0.19, 0.18],
    "val_auroc":  [0.72, 0.78, 0.81, 0.82, 0.83],
}

# Simulate per-class AUROC scores
per_class_aurocs = []
for i in range(14):
    try:
        per_class_aurocs.append(roc_auc_score(test_labels[:, i], test_preds[:, i]))
    except ValueError:
        per_class_aurocs.append(0.5)

test_auroc = np.mean(per_class_aurocs)
binary_preds = (test_preds >= THRESHOLD).astype(int)

print(f"Cell 2 complete: {N_TEST} test samples, {14} disease classes loaded.")
print(f"Mean Test AUROC: {test_auroc:.4f}")


###############################################################################
# ==================== CELL 3: TRAINING CURVES ====================
# Loss curves and AUROC progression over epochs.
###############################################################################

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (a) Train vs Val Loss
epochs_range = range(1, len(history["train_loss"]) + 1)
axes[0].plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#e74c3c", linewidth=2)
axes[0].plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#3498db", linewidth=2)
axes[0].fill_between(epochs_range, history["train_loss"], history["val_loss"], alpha=0.1, color="gray")
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("BCE Loss")
axes[0].legend(frameon=True, fancybox=True)
axes[0].grid(True, alpha=0.3)

# (b) Val AUROC over epochs
axes[1].plot(epochs_range, history["val_auroc"], "D-", color="#2ecc71", linewidth=2, markersize=8)
axes[1].axhline(y=max(history["val_auroc"]), color="red", linestyle="--", alpha=0.5,
                label=f'Best: {max(history["val_auroc"]):.4f}')
axes[1].set_title("Validation AUROC over Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean AUROC")
axes[1].set_ylim(0.5, 1.0)
axes[1].legend(frameon=True, fancybox=True)
axes[1].grid(True, alpha=0.3)

# (c) Overfitting gap
gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gap]
axes[2].bar(epochs_range, gap, color=colors, edgecolor="black", alpha=0.7)
axes[2].axhline(y=0, color="black", linewidth=0.8)
axes[2].set_title("Overfitting Gap (Val Loss − Train Loss)")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss Difference")
axes[2].grid(True, alpha=0.3)

plt.suptitle("Training Progress", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_training_curves.png")
plt.show()
print("Cell 3 complete: Training curves saved.")


###############################################################################
# ==================== CELL 4: PER-CLASS AUROC BAR CHART ====================
# Horizontal bar chart of AUROC for each of the 14 diseases.
###############################################################################

fig, ax = plt.subplots(figsize=(10, 7))

sorted_indices = np.argsort(per_class_aurocs)
sorted_labels = [DISEASE_LABELS[i] for i in sorted_indices]
sorted_aurocs = [per_class_aurocs[i] for i in sorted_indices]

# Color gradient: red(poor) -> yellow(fair) -> green(good)
colors = plt.cm.RdYlGn([(a - 0.5) / 0.5 for a in sorted_aurocs])

bars = ax.barh(sorted_labels, sorted_aurocs, color=colors, edgecolor="black", linewidth=0.5)
ax.axvline(x=test_auroc, color="red", linestyle="--", linewidth=1.5,
           label=f"Mean AUROC: {test_auroc:.4f}")
ax.axvline(x=0.8, color="gray", linestyle=":", alpha=0.5, label="Good threshold (0.80)")

# Value labels on bars
for bar, val in zip(bars, sorted_aurocs):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9, fontweight="bold")

ax.set_xlim(0.4, 1.0)
ax.set_xlabel("AUROC Score")
ax.set_title("Test AUROC per Disease Class", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", frameon=True, fancybox=True)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_per_class_auroc.png")
plt.show()
print("Cell 4 complete: Per-class AUROC chart saved.")


###############################################################################
# ==================== CELL 5: MULTI-LABEL CONFUSION MATRICES ====================
# One confusion matrix per disease (binary: present vs absent).
###############################################################################

fig, axes = plt.subplots(3, 5, figsize=(22, 13))
axes = axes.flatten()

mcm = multilabel_confusion_matrix(test_labels, binary_preds)

for i, (disease, cm) in enumerate(zip(DISEASE_LABELS, mcm)):
    ax = axes[i]
    # Normalize by row (true label) to show rates
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm_norm / row_sums * 100, 0)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
                cbar=False, linewidths=0.5, linecolor="gray")

    # Add percentage annotations
    for row in range(2):
        for col in range(2):
            ax.text(col + 0.5, row + 0.78, f"({cm_pct[row, col]:.1f}%)",
                    ha="center", va="center", fontsize=7, color="gray")

    ax.set_title(disease, fontsize=10, fontweight="bold")
    ax.set_ylabel("True" if i % 5 == 0 else "")
    ax.set_xlabel("Predicted" if i >= 10 else "")

# Hide the 15th subplot
axes[14].axis("off")

plt.suptitle("Confusion Matrices per Disease (Counts & Row %)",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_confusion_matrices.png")
plt.show()
print("Cell 5 complete: Confusion matrices saved.")


###############################################################################
# ==================== CELL 6: ROC CURVES (ALL CLASSES) ====================
# One ROC curve per disease overlaid on the same plot.
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.tab20(np.linspace(0, 1, 14))

for i, (disease, color) in enumerate(zip(DISEASE_LABELS, colors)):
    fpr, tpr, _ = roc_curve(test_labels[:, i], test_preds[:, i])
    roc_auc_val = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, linewidth=1.5,
            label=f"{disease} ({roc_auc_val:.3f})")

# Diagonal reference
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (0.500)")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All 14 Disease Classes", fontsize=14, fontweight="bold")
ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=True, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_roc_curves.png")
plt.show()
print("Cell 6 complete: ROC curves saved.")


###############################################################################
# ==================== CELL 7: PRECISION-RECALL CURVES ====================
# Precision-Recall curves per disease (better for imbalanced datasets).
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.tab20(np.linspace(0, 1, 14))

for i, (disease, color) in enumerate(zip(DISEASE_LABELS, colors)):
    precision, recall, _ = precision_recall_curve(test_labels[:, i], test_preds[:, i])
    ap = average_precision_score(test_labels[:, i], test_preds[:, i])
    ax.plot(recall, precision, color=color, linewidth=1.5,
            label=f"{disease} (AP={ap:.3f})")

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All 14 Disease Classes",
             fontsize=14, fontweight="bold")
ax.legend(loc="upper right", fontsize=8, frameon=True, fancybox=True, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_precision_recall.png")
plt.show()
print("Cell 7 complete: Precision-Recall curves saved.")


###############################################################################
# ==================== CELL 8: DISEASE PREVALENCE (LABEL DISTRIBUTION) ========
# Shows the class imbalance in the dataset.
###############################################################################

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) Bar chart: count of positive samples per disease
pos_counts = test_labels.sum(axis=0)
sorted_idx = np.argsort(pos_counts)[::-1]

bars = axes[0].bar(
    [DISEASE_LABELS[i] for i in sorted_idx],
    [pos_counts[i] for i in sorted_idx],
    color=sns.color_palette("viridis", 14),
    edgecolor="black", linewidth=0.5
)
axes[0].set_title("Disease Prevalence (Positive Samples)", fontweight="bold")
axes[0].set_xlabel("Disease")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)

# Value labels
for bar, val in zip(bars, [pos_counts[i] for i in sorted_idx]):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(int(val)), ha="center", fontsize=8)

# (b) Pie chart: healthy vs any-disease
n_healthy = np.sum(test_labels.sum(axis=1) == 0)
n_diseased = len(test_labels) - n_healthy
axes[1].pie(
    [n_healthy, n_diseased],
    labels=["No Finding", "Disease Present"],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    startangle=90, textprops={"fontsize": 12},
    explode=(0.03, 0.03), shadow=True
)
axes[1].set_title("Healthy vs Disease Ratio", fontweight="bold")

plt.suptitle("Dataset Label Distribution", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_label_distribution.png")
plt.show()
print("Cell 8 complete: Label distribution chart saved.")


###############################################################################
# ==================== CELL 9: F1 SCORE COMPARISON ====================
# Per-class F1 scores at different thresholds.
###############################################################################

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (a) F1 at default threshold (0.5)
f1_scores = []
for i in range(14):
    f1_scores.append(f1_score(test_labels[:, i], binary_preds[:, i], zero_division=0))

sorted_idx = np.argsort(f1_scores)
sorted_f1 = [f1_scores[i] for i in sorted_idx]
sorted_names = [DISEASE_LABELS[i] for i in sorted_idx]

colors = plt.cm.RdYlGn([f / max(max(sorted_f1), 0.01) for f in sorted_f1])
bars = axes[0].barh(sorted_names, sorted_f1, color=colors, edgecolor="black", linewidth=0.5)

for bar, val in zip(bars, sorted_f1):
    axes[0].text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", fontsize=9)

axes[0].set_title(f"F1 Scores per Disease (threshold={THRESHOLD})", fontweight="bold")
axes[0].set_xlabel("F1 Score")
axes[0].set_xlim(0, 1.0)
axes[0].axvline(x=np.mean(f1_scores), color="red", linestyle="--",
                label=f"Mean: {np.mean(f1_scores):.3f}")
axes[0].legend(frameon=True)
axes[0].grid(axis="x", alpha=0.3)

# (b) F1 vs threshold sweep
thresholds = np.arange(0.1, 0.9, 0.05)
mean_f1_per_thresh = []
for t in thresholds:
    bp = (test_preds >= t).astype(int)
    f1s = [f1_score(test_labels[:, i], bp[:, i], zero_division=0) for i in range(14)]
    mean_f1_per_thresh.append(np.mean(f1s))

best_thresh = thresholds[np.argmax(mean_f1_per_thresh)]
axes[1].plot(thresholds, mean_f1_per_thresh, "o-", color="#9b59b6", linewidth=2)
axes[1].axvline(x=best_thresh, color="red", linestyle="--",
                label=f"Optimal: {best_thresh:.2f} (F1={max(mean_f1_per_thresh):.3f})")
axes[1].axvline(x=0.5, color="gray", linestyle=":", alpha=0.5, label="Default: 0.50")
axes[1].set_title("Mean F1 vs Classification Threshold", fontweight="bold")
axes[1].set_xlabel("Threshold")
axes[1].set_ylabel("Mean F1 Score")
axes[1].legend(frameon=True, fancybox=True)
axes[1].grid(True, alpha=0.3)

plt.suptitle("F1 Score Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_f1_analysis.png")
plt.show()
print("Cell 9 complete: F1 analysis saved.")


###############################################################################
# ==================== CELL 10: PREDICTION CONFIDENCE DISTRIBUTION ==========
# Histogram of predicted probabilities for positive vs negative cases.
###############################################################################

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

# Show 8 diseases (most prevalent)
show_diseases = [3, 2, 0, 7, 4, 5, 8, 6]  # Infiltration, Effusion, Atelectasis, ...

for plot_idx, disease_idx in enumerate(show_diseases):
    ax = axes[plot_idx]
    disease = DISEASE_LABELS[disease_idx]

    pos_mask = test_labels[:, disease_idx] == 1
    neg_mask = ~pos_mask

    ax.hist(test_preds[neg_mask, disease_idx], bins=50, alpha=0.6,
            color="#3498db", label="Negative", density=True)
    ax.hist(test_preds[pos_mask, disease_idx], bins=50, alpha=0.6,
            color="#e74c3c", label="Positive", density=True)
    ax.axvline(x=THRESHOLD, color="black", linestyle="--", linewidth=1, label="Threshold")
    ax.set_title(f"{disease}\n(n+={pos_mask.sum()}, n-={neg_mask.sum()})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7, frameon=True)

plt.suptitle("Prediction Confidence Distribution (Positive vs Negative)",
             fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_confidence_distribution.png")
plt.show()
print("Cell 10 complete: Confidence distributions saved.")


###############################################################################
# ==================== CELL 11: LABEL CO-OCCURRENCE HEATMAP ================
# Shows which diseases commonly appear together.
###############################################################################

fig, ax = plt.subplots(figsize=(10, 8))

co_occurrence = np.zeros((14, 14), dtype=int)
for i in range(14):
    for j in range(14):
        co_occurrence[i, j] = np.sum((test_labels[:, i] == 1) & (test_labels[:, j] == 1))

mask = np.triu(np.ones_like(co_occurrence, dtype=bool), k=1)
sns.heatmap(co_occurrence, mask=mask, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=DISEASE_LABELS, yticklabels=DISEASE_LABELS,
            linewidths=0.5, linecolor="white", ax=ax,
            cbar_kws={"label": "Co-occurrence Count"})

ax.set_title("Disease Label Co-occurrence Matrix", fontsize=14, fontweight="bold")
ax.tick_params(axis="x", rotation=45)
ax.tick_params(axis="y", rotation=0)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_co_occurrence.png")
plt.show()
print("Cell 11 complete: Co-occurrence heatmap saved.")


###############################################################################
# ==================== CELL 12: MODEL PERFORMANCE SUMMARY DASHBOARD =========
# Combined dashboard with key metrics in one figure.
###############################################################################

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

# --- (a) AUROC vs F1 grouped bar chart ---
ax1 = fig.add_subplot(gs[0, 0:2])
x = np.arange(14)
width = 0.35

bars1 = ax1.bar(x - width / 2, per_class_aurocs, width, label="AUROC",
                color="#3498db", edgecolor="black", linewidth=0.5)
bars2 = ax1.bar(x + width / 2, f1_scores, width, label="F1 Score",
                color="#e74c3c", edgecolor="black", linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(DISEASE_LABELS, rotation=45, ha="right", fontsize=8)
ax1.set_ylabel("Score")
ax1.set_title("AUROC vs F1 Score per Disease", fontweight="bold")
ax1.legend(frameon=True, fancybox=True)
ax1.set_ylim(0, 1.05)
ax1.grid(axis="y", alpha=0.3)

# --- (b) Radar chart of AUROC ---
ax2 = fig.add_subplot(gs[0, 2], polar=True)
angles = np.linspace(0, 2 * np.pi, 14, endpoint=False).tolist()
aurocs_plot = per_class_aurocs + [per_class_aurocs[0]]  # close the polygon
angles += angles[:1]

ax2.plot(angles, aurocs_plot, "o-", linewidth=2, color="#2ecc71")
ax2.fill(angles, aurocs_plot, alpha=0.2, color="#2ecc71")
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels([d[:6] for d in DISEASE_LABELS], fontsize=7)
ax2.set_ylim(0.4, 1.0)
ax2.set_title("AUROC Radar Chart", fontweight="bold", pad=20)

# --- (c) Sensitivity vs Specificity ---
ax3 = fig.add_subplot(gs[1, 0])
sensitivity = []
specificity = []
for i in range(14):
    cm = confusion_matrix(test_labels[:, i], binary_preds[:, i], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
    specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)

ax3.scatter(specificity, sensitivity, c=per_class_aurocs, cmap="RdYlGn",
            s=100, edgecolors="black", linewidth=0.5, zorder=5)
for i, disease in enumerate(DISEASE_LABELS):
    ax3.annotate(disease[:8], (specificity[i], sensitivity[i]),
                 fontsize=6, ha="center", va="bottom")
ax3.set_xlabel("Specificity (TNR)")
ax3.set_ylabel("Sensitivity (TPR)")
ax3.set_title("Sensitivity vs Specificity", fontweight="bold")
ax3.set_xlim(0.5, 1.02)
ax3.set_ylim(0, 1.02)
ax3.grid(True, alpha=0.3)

# --- (d) Error analysis: false positives vs false negatives ---
ax4 = fig.add_subplot(gs[1, 1])
fp_counts = []
fn_counts = []
for i in range(14):
    cm = confusion_matrix(test_labels[:, i], binary_preds[:, i], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fp_counts.append(fp)
    fn_counts.append(fn)

x = np.arange(14)
ax4.barh(x - 0.2, fp_counts, 0.4, label="False Positives", color="#f39c12", edgecolor="black")
ax4.barh(x + 0.2, fn_counts, 0.4, label="False Negatives", color="#8e44ad", edgecolor="black")
ax4.set_yticks(x)
ax4.set_yticklabels(DISEASE_LABELS, fontsize=8)
ax4.set_xlabel("Count")
ax4.set_title("Error Analysis: FP vs FN", fontweight="bold")
ax4.legend(fontsize=8, frameon=True)
ax4.grid(axis="x", alpha=0.3)

# --- (e) Summary stats table ---
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")

summary_data = [
    ["Metric", "Value"],
    ["Mean AUROC", f"{test_auroc:.4f}"],
    ["Mean F1", f"{np.mean(f1_scores):.4f}"],
    ["Best Disease", f"{DISEASE_LABELS[np.argmax(per_class_aurocs)]}"],
    ["Best AUROC", f"{max(per_class_aurocs):.4f}"],
    ["Worst Disease", f"{DISEASE_LABELS[np.argmin(per_class_aurocs)]}"],
    ["Worst AUROC", f"{min(per_class_aurocs):.4f}"],
    ["Optimal Thresh", f"{best_thresh:.2f}"],
    ["Test Samples", f"{len(test_labels):,}"],
]

table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0],
                  cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)

# Style header row
for j in range(2):
    table[0, j].set_facecolor("#34495e")
    table[0, j].set_text_props(color="white", fontweight="bold")

ax5.set_title("Performance Summary", fontweight="bold", pad=20)

plt.suptitle("CheXNet Model Performance Dashboard",
             fontsize=16, fontweight="bold", y=1.01)
plt.savefig(f"{OUTPUT_DIR}/graph_dashboard.png")
plt.show()
print("Cell 12 complete: Dashboard saved.")


###############################################################################
# ==================== CELL 13: MULTI-LABEL STATISTICS ====================
# How many diseases per image, predicted vs actual.
###############################################################################

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Distribution of number of diseases per image (ground truth)
n_diseases_true = test_labels.sum(axis=1)
n_diseases_pred = binary_preds.sum(axis=1)

max_d = max(n_diseases_true.max(), n_diseases_pred.max())
bins = np.arange(0, max_d + 2) - 0.5

axes[0].hist(n_diseases_true, bins=bins, alpha=0.6, color="#3498db",
             label="Ground Truth", edgecolor="black")
axes[0].hist(n_diseases_pred, bins=bins, alpha=0.6, color="#e74c3c",
             label="Predicted", edgecolor="black")
axes[0].set_xlabel("Number of Diseases per Image")
axes[0].set_ylabel("Count")
axes[0].set_title("Disease Count Distribution per Image", fontweight="bold")
axes[0].legend(frameon=True, fancybox=True)
axes[0].grid(axis="y", alpha=0.3)

# (b) Exact match ratio by number of diseases
match_by_count = {}
for n_true in range(int(max_d) + 1):
    mask = n_diseases_true == n_true
    if mask.sum() > 0:
        exact_match = np.all(test_labels[mask] == binary_preds[mask], axis=1).mean()
        match_by_count[n_true] = exact_match

counts = list(match_by_count.keys())
matches = list(match_by_count.values())

axes[1].bar(counts, matches, color=sns.color_palette("coolwarm", len(counts)),
            edgecolor="black", linewidth=0.5)
axes[1].set_xlabel("Number of True Diseases")
axes[1].set_ylabel("Exact Match Ratio")
axes[1].set_title("Prediction Accuracy by Disease Count", fontweight="bold")
axes[1].set_ylim(0, 1.05)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("Multi-Label Analysis", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph_multilabel_stats.png")
plt.show()
print("Cell 13 complete: Multi-label statistics saved.")


###############################################################################
# ==================== CELL 14: SUMMARY ====================
# Print all generated graph files.
###############################################################################

graph_files = [
    "graph_training_curves.png",
    "graph_per_class_auroc.png",
    "graph_confusion_matrices.png",
    "graph_roc_curves.png",
    "graph_precision_recall.png",
    "graph_label_distribution.png",
    "graph_f1_analysis.png",
    "graph_confidence_distribution.png",
    "graph_co_occurrence.png",
    "graph_dashboard.png",
    "graph_multilabel_stats.png",
]

print("=" * 60)
print("ALL GRAPHS GENERATED SUCCESSFULLY")
print("=" * 60)
for f in graph_files:
    print(f"  - {f}")
print(f"\nTotal: {len(graph_files)} visualization files")
print("\nTo use with real data, replace Cell 2 (Option C) with")
print("Option A (CSV files) or Option B (pipeline variables).")
