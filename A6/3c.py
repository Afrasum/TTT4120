import numpy as np
from matplotlib import pyplot as plt

# Signal parameters
f1 = 7 / 40  # 0.175
f2 = 9 / 40  # 0.225

# Fixed segment length
N = 100

# Generate signal segment
n_vec = np.arange(N)
xn = np.sin(2 * np.pi * f1 * n_vec) + np.sin(2 * np.pi * f2 * n_vec)

# DFT lengths to test
N_dft_vec = [1024, 256, 128]
figure_num = 15

# Create figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

for i, N_dft in enumerate(N_dft_vec):
    print(f"\n{'='*60}")
    print(f"DFT Length N_DFT = {N_dft}")
    print(f"{'='*60}")

    # Compute DFT
    xk = np.fft.fft(xn, n=N_dft)
    xk_mag = np.abs(xk)

    # Create frequency axis
    freq_axis = np.linspace(0, 1, N_dft)

    # Frequency spacing
    freq_spacing = 1 / N_dft
    print(f"Segment length N: {N}")
    print(f"Frequency resolution (from segment): Δf ≈ 1/{N} = {1/N:.6f}")
    print(f"DFT frequency spacing: 1/{N_dft} = {freq_spacing:.6f}")
    print(f"Number of points in [0, 0.5]: {N_dft//2}")

    # Plot magnitude spectrum (only [0, 0.5])
    half_idx = N_dft // 2
    axes[i].plot(
        freq_axis[:half_idx],
        xk_mag[:half_idx],
        "b-",
        linewidth=1.5,
        marker="o",
        markersize=3,
        markevery=max(1, N_dft // 50),
    )

    # Mark true frequencies
    axes[i].axvline(
        x=f1,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"$f_1$={f1:.3f}",
    )
    axes[i].axvline(
        x=f2,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label=f"$f_2$={f2:.3f}",
    )

    # Formatting
    figure_num += 1
    axes[i].set_title(
        f"Figure {figure_num}: N = {N}, DFT length = {N_dft}",
        fontsize=14,
        fontweight="bold",
    )
    axes[i].set_xlabel("$F$ [Hz] (Normalized Frequency)", fontsize=12)
    axes[i].set_ylabel("$|X(F/F_s)|$", fontsize=12)
    axes[i].legend(fontsize=11)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([0, 0.5])

    # Add text annotation
    axes[i].text(
        0.4,
        np.max(xk_mag[:half_idx]) * 0.9,
        f"Freq spacing: 1/{N_dft} = {freq_spacing:.6f}\n"
        f"Points in [0, 0.5]: {half_idx}",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

plt.tight_layout()
plt.show()

# Detailed comparison: Overlay all three
fig, ax = plt.subplots(figsize=(14, 8))

colors = ["blue", "green", "red"]
markers = ["o", "s", "^"]
labels = [f"N_DFT = {N_dft}" for N_dft in N_dft_vec]

for N_dft, color, marker, label in zip(N_dft_vec, colors, markers, labels):
    xk = np.fft.fft(xn, n=N_dft)
    xk_mag = np.abs(xk)
    freq_axis = np.linspace(0, 1, N_dft)

    half_idx = N_dft // 2
    ax.plot(
        freq_axis[:half_idx],
        xk_mag[:half_idx],
        color=color,
        linewidth=2,
        marker=marker,
        markersize=4,
        markevery=max(1, N_dft // 30),
        label=label,
        alpha=0.7,
    )

# Mark true frequencies
ax.axvline(
    x=f1,
    color="black",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"True $f_1$={f1:.3f}",
)
ax.axvline(
    x=f2,
    color="black",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"True $f_2$={f2:.3f}",
)

ax.set_title(
    f"Comparison: DFT Length Effect (Segment Length N = {N} fixed)",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Normalized Frequency $f$", fontsize=13)
ax.set_ylabel("$|X(f)|$", fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 0.5])

plt.tight_layout()
plt.show()

# Zoomed comparison around peaks
fig, ax = plt.subplots(figsize=(14, 8))

for N_dft, color, marker, label in zip(N_dft_vec, colors, markers, labels):
    xk = np.fft.fft(xn, n=N_dft)
    xk_mag = np.abs(xk)
    freq_axis = np.linspace(0, 1, N_dft)

    # Zoom around f1 and f2
    zoom_mask = (freq_axis >= 0.1) & (freq_axis <= 0.35)
    ax.plot(
        freq_axis[zoom_mask],
        xk_mag[zoom_mask],
        color=color,
        linewidth=2,
        marker=marker,
        markersize=5,
        label=label,
        alpha=0.7,
    )

# Mark true frequencies
ax.axvline(
    x=f1,
    color="black",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"True $f_1$={f1:.3f}",
)
ax.axvline(
    x=f2,
    color="black",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"True $f_2$={f2:.3f}",
)

ax.set_title(
    f"Zoomed View: DFT Length Effect (N = {N})", fontsize=15, fontweight="bold"
)
ax.set_xlabel("Normalized Frequency $f$", fontsize=13)
ax.set_ylabel("$|X(f)|$", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0.1, 0.35])

# Annotate mainlobe width (same for all)
mainlobe_width = 1 / N
ax.annotate(
    "",
    xy=(f1 - mainlobe_width, np.max(xk_mag[zoom_mask]) * 0.8),
    xytext=(f1 + mainlobe_width, np.max(xk_mag[zoom_mask]) * 0.8),
    arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
)
ax.text(
    f1,
    np.max(xk_mag[zoom_mask]) * 0.85,
    f"Mainlobe width ≈ 2/N = {2/N:.3f}\n(Same for all DFT lengths!)",
    ha="center",
    fontsize=11,
    color="purple",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
)

plt.tight_layout()
plt.show()

# Summary table
print("\n" + "=" * 80)
print("SUMMARY: Effect of DFT Length (with fixed Segment Length N=100)")
print("=" * 80)
print(
    f"{'DFT Length':<12} {'Freq Spacing':<15} {'Points [0,0.5]':<15} {'Resolution':<20}"
)
print("=" * 80)

resolution = 1 / N
for N_dft in N_dft_vec:
    spacing = 1 / N_dft
    points = N_dft // 2
    print(f"{N_dft:<12} {spacing:<15.6f} {points:<15} {resolution:.6f} (unchanged)")

print("=" * 80)
print(f"\nKey Insight:")
print(f"  - Frequency RESOLUTION: Δf ≈ 1/N = 1/{N} = {resolution:.6f}")
print(f"    This is determined by SEGMENT length, NOT DFT length!")
print(f"  - Frequency SPACING between DFT samples: 1/N_DFT")
print(f"    This changes with DFT length (just interpolation)")
print(f"\n  Conclusion: Longer DFT gives smoother curve, but NOT better resolution!")

# Demonstration: Zero-padding doesn't add information
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

N = 100
n_vec = np.arange(N)
xn = np.sin(2 * np.pi * f1 * n_vec) + np.sin(2 * np.pi * f2 * n_vec)

# Show time domain with different zero-padding
for i, (N_dft, title) in enumerate(
    [
        (100, "No Zero-Padding (N_DFT=100)"),
        (256, "Zero-Padded to 256"),
        (512, "Zero-Padded to 512"),
        (1024, "Zero-Padded to 1024"),
    ]
):
    row = i // 2
    col = i % 2

    # Time domain
    xn_padded = np.zeros(N_dft)
    xn_padded[:N] = xn

    n_padded = np.arange(N_dft)

    axes[row, col].plot(n_padded, xn_padded, "b-", linewidth=1.5)
    axes[row, col].axvline(
        x=N - 1, color="red", linestyle="--", linewidth=2, label="Original signal end"
    )
    axes[row, col].axvspan(N, N_dft, alpha=0.2, color="gray", label="Zero-padding")

    axes[row, col].set_title(title, fontsize=13, fontweight="bold")
    axes[row, col].set_xlabel("$n$", fontsize=11)
    axes[row, col].set_ylabel("$x[n]$", fontsize=11)
    axes[row, col].legend(fontsize=9)
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_xlim([0, N_dft])

    # Add text
    axes[row, col].text(
        N_dft * 0.7,
        np.max(xn) * 0.8,
        f"Data: {N} points\nZeros: {N_dft-N} points",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

plt.suptitle(
    "Zero-Padding in Time Domain: Adds zeros, NOT information!",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("COMPARISON: DFT Length Effect")
print("=" * 80)
print(f"{'Parameter':<25} {'Value':<20} {'Effect'}")
print("=" * 80)
print(f"{'Segment Length (N)':<25} {'100 (FIXED)':<20} {'Determines resolution'}")
print(f"{'Frequency Resolution':<25} {'~0.01 (FIXED)':<20} {'Can resolve f₁, f₂'}")
print(f"{'Mainlobe Width':<25} {'~0.02 (FIXED)':<20} {'Same for all DFTs'}")
print("-" * 80)
print(f"{'DFT Length (N_DFT=1024)':<25} {'0.001':<20} {'512 samples, smooth'}")
print(f"{'DFT Length (N_DFT=256)':<25} {'0.004':<20} {'128 samples, OK'}")
print(f"{'DFT Length (N_DFT=128)':<25} {'0.008':<20} {'64 samples, coarse'}")
print("=" * 80)
print("\nConclusion:")
print("  • Longer DFT → Smoother curve (better interpolation)")
print("  • But same resolution (determined by segment length)")
print("  • Zero-padding ≠ More information!")
