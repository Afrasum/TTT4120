import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis("off")

# Title
ax.text(
    7,
    3.7,
    "Problem 2a: Sampling Rate Conversion 8 kHz → 6 kHz",
    ha="center",
    fontsize=14,
    fontweight="bold",
)

# Input signal
ax.add_patch(
    FancyBboxPatch(
        (0.5, 1.5),
        1.5,
        1,
        boxstyle="round,pad=0.1",
        edgecolor="blue",
        facecolor="lightblue",
        linewidth=2,
    )
)
ax.text(1.25, 2, "x[n]", ha="center", va="center", fontsize=12, fontweight="bold")
ax.text(1.25, 1.3, "Fsx = 8 kHz", ha="center", fontsize=9)

# Arrow 1
arrow1 = FancyArrowPatch(
    (2, 2), (3, 2), arrowstyle="->", mutation_scale=20, linewidth=2, color="black"
)
ax.add_patch(arrow1)

# Upsampler
ax.add_patch(
    FancyBboxPatch(
        (3, 1.5),
        1.5,
        1,
        boxstyle="round,pad=0.1",
        edgecolor="green",
        facecolor="lightgreen",
        linewidth=2,
    )
)
ax.text(3.75, 2.2, "↑ L", ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(3.75, 1.8, "L = 3", ha="center", fontsize=10)
ax.text(3.75, 1.3, "24 kHz", ha="center", fontsize=9, style="italic")

# Arrow 2
arrow2 = FancyArrowPatch(
    (4.5, 2), (5.5, 2), arrowstyle="->", mutation_scale=20, linewidth=2, color="black"
)
ax.add_patch(arrow2)
ax.text(5, 2.3, "w[m]", ha="center", fontsize=9, style="italic")

# Filter
ax.add_patch(
    FancyBboxPatch(
        (5.5, 1.5),
        1.5,
        1,
        boxstyle="round,pad=0.1",
        edgecolor="red",
        facecolor="lightyellow",
        linewidth=2,
    )
)
ax.text(6.25, 2.2, "h[m]", ha="center", va="center", fontsize=12, fontweight="bold")
ax.text(6.25, 1.8, "Lowpass", ha="center", fontsize=9)
ax.text(6.25, 1.3, "24 kHz", ha="center", fontsize=9, style="italic")

# Arrow 3
arrow3 = FancyArrowPatch(
    (7, 2), (8, 2), arrowstyle="->", mutation_scale=20, linewidth=2, color="black"
)
ax.add_patch(arrow3)
ax.text(7.5, 2.3, "v[m]", ha="center", fontsize=9, style="italic")

# Downsampler
ax.add_patch(
    FancyBboxPatch(
        (8, 1.5),
        1.5,
        1,
        boxstyle="round,pad=0.1",
        edgecolor="orange",
        facecolor="lightyellow",
        linewidth=2,
    )
)
ax.text(8.75, 2.2, "↓ M", ha="center", va="center", fontsize=14, fontweight="bold")
ax.text(8.75, 1.8, "M = 4", ha="center", fontsize=10)
ax.text(8.75, 1.3, "6 kHz", ha="center", fontsize=9, style="italic")

# Arrow 4
arrow4 = FancyArrowPatch(
    (9.5, 2), (10.5, 2), arrowstyle="->", mutation_scale=20, linewidth=2, color="black"
)
ax.add_patch(arrow4)

# Output signal
ax.add_patch(
    FancyBboxPatch(
        (10.5, 1.5),
        1.5,
        1,
        boxstyle="round,pad=0.1",
        edgecolor="purple",
        facecolor="lavender",
        linewidth=2,
    )
)
ax.text(11.25, 2, "y[k]", ha="center", va="center", fontsize=12, fontweight="bold")
ax.text(11.25, 1.3, "Fsy = 6 kHz", ha="center", fontsize=9)

# Annotations below
ax.text(
    1.25,
    0.7,
    "Input:\n80 samples\nper 10ms",
    ha="center",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3),
)
ax.text(
    3.75,
    0.7,
    "Insert 2 zeros\nbetween samples",
    ha="center",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.3),
)
ax.text(
    6.25,
    0.5,
    "Anti-aliasing filter\nRemove images\nPrevent aliasing",
    ha="center",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
)
ax.text(
    8.75,
    0.7,
    "Keep every\n4th sample",
    ha="center",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.3),
)
ax.text(
    11.25,
    0.7,
    "Output:\n60 samples\nper 10ms",
    ha="center",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.3),
)

plt.tight_layout()
plt.savefig("problem2a_block_diagram.png", dpi=150, bbox_inches="tight")
plt.show()

print("=" * 80)
print("PROBLEM 2a: BLOCK DIAGRAM AND COMPONENT FUNCTIONS")
print("=" * 80)
print("\nConversion: Fsx = 8 kHz → Fsy = 6 kHz")
print("Ratio: 6/8 = 3/4 (L/M where L=3, M=4)")
print("\n" + "-" * 80)
print("BLOCK 1: Upsampler ↑L (L=3)")
print("-" * 80)
print("Function: Increase sampling rate by factor L = 3")
print("  Input:  x[n] at Fsx = 8 kHz")
print("  Output: w[m] at L×Fsx = 24 kHz")
print("  Operation: w[m] = x[m/L] if m is multiple of L, else 0")
print("  Effect: Inserts 2 zeros between each sample")
print("\nWhy? Creates 'headroom' in frequency domain for safe downsampling")

print("\n" + "-" * 80)
print("BLOCK 2: Lowpass Filter h[m]")
print("-" * 80)
print("Function: Anti-aliasing filter")
print("  Input:  w[m] at 24 kHz")
print("  Output: v[m] at 24 kHz (filtered)")
print("  Purpose:")
print("    1. Remove spectral images from upsampling")
print("    2. Prevent aliasing in subsequent downsampling")
print("  Specifications: (determined in part b)")
print("    - Cutoff frequency: TBD")
print("    - Gain: L = 3")

print("\n" + "-" * 80)
print("BLOCK 3: Downsampler ↓M (M=4)")
print("-" * 80)
print("Function: Decrease sampling rate by factor M = 4")
print("  Input:  v[m] at 24 kHz")
print("  Output: y[k] at 24/M = 6 kHz")
print("  Operation: y[k] = v[kM] (keep every 4th sample)")
print("  Effect: Reduces sample rate from 24 kHz to 6 kHz")

print("\n" + "-" * 80)
print("WHY THIS ORDER?")
print("-" * 80)
print("✓ Upsample FIRST (↑3): Creates room in frequency domain")
print("✓ Filter: Removes unwanted content")
print("✓ Downsample LAST (↓4): Safely reduces rate")
print("\n✗ WRONG order (downsample first):")
print("  8 kHz → ↓4 → 2 kHz would cause MASSIVE aliasing!")
print("  (Nyquist = 1 kHz, but signal has content up to 4 kHz)")

print("\n" + "=" * 80)
