###############################################################################
# ==================== DATASET INFO PLOT (STANDALONE CELL) ====================
# Independent cell — no dependency on the pipeline or other cells.
# Visualizes the NIH Chest X-ray dataset characteristics.
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ---- NIH Chest X-ray Dataset Statistics ----
# Source: NIH Clinical Center — 112,120 frontal-view chest X-rays from 30,805 patients
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

TOTAL_IMAGES = 112_120
TOTAL_PATIENTS = 30_805
IMG_SIZE = 1024  # original resolution

# Approximate positive counts per disease (from NIH data entry CSV)
POSITIVE_COUNTS = {
    "Atelectasis":        11559,
    "Cardiomegaly":        2776,
    "Effusion":           13317,
    "Infiltration":       19894,
    "Mass":                5782,
    "Nodule":              6331,
    "Pneumonia":           1431,
    "Pneumothorax":        5302,
    "Consolidation":       4667,
    "Edema":               2303,
    "Emphysema":           2516,
    "Fibrosis":            1686,
    "Pleural_Thickening":  3385,
    "Hernia":               227,
}

NO_FINDING_COUNT = 60_361  # images with "No Finding" label

counts = np.array([POSITIVE_COUNTS[d] for d in DISEASE_LABELS])
prevalence = counts / TOTAL_IMAGES * 100  # percentage

# ---- Split info (official NIH split) ----
TRAIN_VAL_COUNT = 86_524
TEST_COUNT = 25_596

# ========================= FIGURE =========================
fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35)

# --- (1) Disease Prevalence Bar Chart ---
ax1 = fig.add_subplot(gs[0, 0:2])
sorted_idx = np.argsort(counts)[::-1]
sorted_labels = [DISEASE_LABELS[i] for i in sorted_idx]
sorted_counts = counts[sorted_idx]
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.9, 14))

bars = ax1.bar(sorted_labels, sorted_counts, color=colors_bar,
               edgecolor="black", linewidth=0.5)
for bar, val in zip(bars, sorted_counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
             f"{val:,}", ha="center", fontsize=8, fontweight="bold")
ax1.set_title("Disease Prevalence (Positive Sample Count)", fontweight="bold")
ax1.set_xlabel("Disease")
ax1.set_ylabel("Number of Images")
ax1.tick_params(axis="x", rotation=45)
ax1.grid(axis="y", alpha=0.3)

# --- (2) Healthy vs Disease Pie Chart ---
ax2 = fig.add_subplot(gs[0, 2])
disease_any = TOTAL_IMAGES - NO_FINDING_COUNT
wedges, texts, autotexts = ax2.pie(
    [NO_FINDING_COUNT, disease_any],
    labels=["No Finding", "Disease Present"],
    autopct="%1.1f%%",
    colors=["#2ecc71", "#e74c3c"],
    startangle=90,
    textprops={"fontsize": 11},
    explode=(0.03, 0.03),
    shadow=True,
)
for t in autotexts:
    t.set_fontweight("bold")
ax2.set_title("Healthy vs Disease Distribution", fontweight="bold")

# --- (3) Prevalence Rate (%) Horizontal Bar ---
ax3 = fig.add_subplot(gs[1, 0:2])
sorted_prev = prevalence[sorted_idx]
colors_rg = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, 14))

bars3 = ax3.barh(sorted_labels[::-1], sorted_prev[::-1], color=colors_rg,
                  edgecolor="black", linewidth=0.5)
for bar, val in zip(bars3, sorted_prev[::-1]):
    ax3.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")
ax3.set_title("Disease Prevalence Rate (%)", fontweight="bold")
ax3.set_xlabel("Prevalence (%)")
ax3.axvline(x=np.mean(prevalence), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean: {np.mean(prevalence):.1f}%")
ax3.legend(frameon=True)
ax3.grid(axis="x", alpha=0.3)

# --- (4) Train / Test Split ---
ax4 = fig.add_subplot(gs[1, 2])
split_labels = ["Train + Val", "Test"]
split_counts = [TRAIN_VAL_COUNT, TEST_COUNT]
split_colors = ["#3498db", "#f39c12"]

wedges4, texts4, auto4 = ax4.pie(
    split_counts,
    labels=split_labels,
    autopct=lambda p: f"{p:.1f}%\n({int(p * TOTAL_IMAGES / 100):,})",
    colors=split_colors,
    startangle=90,
    textprops={"fontsize": 10},
    explode=(0.02, 0.02),
    shadow=True,
)
for t in auto4:
    t.set_fontsize(9)
    t.set_fontweight("bold")
ax4.set_title("Official Train/Test Split", fontweight="bold")

# --- (5) Class Imbalance Ratio (Negative : Positive) ---
ax5 = fig.add_subplot(gs[2, 0:2])
neg_pos_ratio = (TOTAL_IMAGES - counts) / counts
sorted_ratio_idx = np.argsort(neg_pos_ratio)[::-1]
sorted_ratio_labels = [DISEASE_LABELS[i] for i in sorted_ratio_idx]
sorted_ratios = neg_pos_ratio[sorted_ratio_idx]

colors_imb = plt.cm.magma(np.linspace(0.2, 0.85, 14))
bars5 = ax5.barh(sorted_ratio_labels[::-1], sorted_ratios[::-1],
                  color=colors_imb, edgecolor="black", linewidth=0.5)
for bar, val in zip(bars5, sorted_ratios[::-1]):
    ax5.text(val + 2, bar.get_y() + bar.get_height() / 2,
             f"{val:.0f}:1", va="center", fontsize=9, fontweight="bold")
ax5.set_title("Class Imbalance Ratio (Neg:Pos)", fontweight="bold")
ax5.set_xlabel("Ratio")
ax5.axvline(x=np.median(neg_pos_ratio), color="red", linestyle="--",
            label=f"Median: {np.median(neg_pos_ratio):.0f}:1")
ax5.legend(frameon=True)
ax5.grid(axis="x", alpha=0.3)

# --- (6) Summary Stats Table ---
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis("off")

summary = [
    ["Metric", "Value"],
    ["Total Images", f"{TOTAL_IMAGES:,}"],
    ["Total Patients", f"{TOTAL_PATIENTS:,}"],
    ["Disease Classes", "14"],
    ["No Finding Images", f"{NO_FINDING_COUNT:,}"],
    ["Multi-label", "Yes"],
    ["Image Format", "PNG (1024x1024)"],
    ["Train+Val / Test", f"{TRAIN_VAL_COUNT:,} / {TEST_COUNT:,}"],
    ["Most Common", "Infiltration"],
    ["Rarest", "Hernia"],
    ["Mean Prevalence", f"{np.mean(prevalence):.1f}%"],
]

table = ax6.table(cellText=summary[1:], colLabels=summary[0],
                  cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.6)

for j in range(2):
    table[0, j].set_facecolor("#2c3e50")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(summary)):
    table[i, 0].set_facecolor("#ecf0f1")

ax6.set_title("Dataset Summary", fontweight="bold", pad=20)

plt.suptitle("NIH Chest X-ray Dataset Overview\n(112,120 images | 30,805 patients | 14 diseases)",
             fontsize=16, fontweight="bold", y=1.02)
plt.savefig("graph_dataset_info.png")
plt.show()
print("Dataset info plot saved to graph_dataset_info.png")
