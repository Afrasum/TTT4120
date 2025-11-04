import numpy as np
from matplotlib import pyplot as plt

# Signal parameters
f1 = 7 / 40  # 0.175
f2 = 9 / 40  # 0.225
N_dft = 1024  # DFT length (high resolution)

# Segment lengths to test
N_vec = [100, 1000, 30, 10]
figure_num = 11

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, N in enumerate(N_vec):
    print(f"\n{'='*60}")
    print(f"Segment Length N = {N}")
    print(f"{'='*60}")

    # Generate signal segment
    n_vec = np.arange(N)
    xn = np.sin(2 * np.pi * f1 * n_vec) + np.sin(2 * np.pi * f2 * n_vec)

    # Compute DFT
    xk = np.fft.fft(xn, n=N_dft)
    xk_mag = np.abs(xk)

    # Create frequency axis
    freq_axis = np.linspace(0, 1, N_dft)

    # Plot magnitude spectrum (only [0, 0.5])
    half_idx = N_dft // 2
    axes[i].plot(freq_axis[:half_idx], xk_mag[:half_idx], "b-", linewidth=1.5)

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
        fontsize=13,
        fontweight="bold",
    )
    axes[i].set_xlabel("$F$ [Hz] (Normalized Frequency)", fontsize=11)
    axes[i].set_ylabel("$|X(F/F_s)|$", fontsize=11)
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([0, 0.5])

    # Calculate frequency resolution
    freq_res = 1 / N
    print(f"Frequency resolution: Δf ≈ 1/{N} = {freq_res:.4f}")
    print(f"Frequency separation: f₂ - f₁ = {f2-f1:.4f}")
    print(f"Can resolve? {freq_res < (f2-f1)}")

    # Find peaks
    peaks_idx = []
    threshold = np.max(xk_mag[:half_idx]) * 0.5
    for k in range(1, half_idx - 1):
        if (
            xk_mag[k] > threshold
            and xk_mag[k] > xk_mag[k - 1]
            and xk_mag[k] > xk_mag[k + 1]
        ):
            peaks_idx.append(k)

    print(f"Number of distinct peaks found: {len(peaks_idx)}")
    if len(peaks_idx) >= 2:
        print(f"Peak frequencies: {[freq_axis[k] for k in peaks_idx[:2]]}")

plt.tight_layout()
plt.show()

# Additional detailed plot for N=100
print("\n" + "=" * 70)
print("DETAILED ANALYSIS: N = 100")
print("=" * 70)

N = 100
n_vec = np.arange(N)
xn = np.sin(2 * np.pi * f1 * n_vec) + np.sin(2 * np.pi * f2 * n_vec)
xk = np.fft.fft(xn, n=N_dft)
xk_mag = np.abs(xk)
freq_axis = np.linspace(0, 1, N_dft)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Time domain signal
axes[0].plot(n_vec, xn, "b-", linewidth=1.5)
axes[0].set_title(
    f"Time Domain Signal: N = {N} samples", fontsize=14, fontweight="bold"
)
axes[0].set_xlabel("$n$", fontsize=12)
axes[0].set_ylabel("$x[n]$", fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, N - 1])

# Plot 2: Frequency domain - zoomed around peaks
axes[1].plot(freq_axis[: N_dft // 2], xk_mag[: N_dft // 2], "b-", linewidth=1.5)
axes[1].axvline(
    x=f1, color="red", linestyle="--", linewidth=2, alpha=0.5, label=f"$f_1$={f1:.3f}"
)
axes[1].axvline(
    x=f2, color="green", linestyle="--", linewidth=2, alpha=0.5, label=f"$f_2$={f2:.3f}"
)
axes[1].set_title(
    "Frequency Domain: Full View [0, 0.5]", fontsize=14, fontweight="bold"
)
axes[1].set_xlabel("Normalized Frequency $f$", fontsize=12)
axes[1].set_ylabel("$|X(f)|$", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 0.5])

# Plot 3: Zoomed around the two frequencies
zoom_range = 0.1
f_center = (f1 + f2) / 2
zoom_mask = (freq_axis >= f_center - zoom_range) & (freq_axis <= f_center + zoom_range)
axes[2].plot(freq_axis[zoom_mask], xk_mag[zoom_mask], "b-", linewidth=2)
axes[2].axvline(
    x=f1, color="red", linestyle="--", linewidth=2, alpha=0.7, label=f"$f_1$={f1:.3f}"
)
axes[2].axvline(
    x=f2, color="green", linestyle="--", linewidth=2, alpha=0.7, label=f"$f_2$={f2:.3f}"
)
axes[2].set_title(
    f"Zoomed View: [{f_center-zoom_range:.2f}, {f_center+zoom_range:.2f}]",
    fontsize=14,
    fontweight="bold",
)
axes[2].set_xlabel("Normalized Frequency $f$", fontsize=12)
axes[2].set_ylabel("$|X(f)|$", fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

# Annotate mainlobe width
mainlobe_width = 1 / N
axes[2].annotate(
    "",
    xy=(f1 - mainlobe_width / 2, np.max(xk_mag[zoom_mask]) * 0.8),
    xytext=(f1 + mainlobe_width / 2, np.max(xk_mag[zoom_mask]) * 0.8),
    arrowprops=dict(arrowstyle="<->", color="purple", lw=2),
)
axes[2].text(
    f1,
    np.max(xk_mag[zoom_mask]) * 0.85,
    f"Mainlobe ≈ 2/N = {2/N:.3f}",
    ha="center",
    fontsize=10,
    color="purple",
)

plt.tight_layout()
plt.show()

# Summary comparison
print("\n" + "=" * 70)
print("SUMMARY: Effect of Segment Length on Spectral Estimation")
print("=" * 70)
print(f"{'Segment N':<12} {'Δf_res':<12} {'Can Resolve?':<15} {'Quality'}")
print("=" * 70)

for N in [1000, 100, 30, 10]:
    freq_res = 1 / N
    can_resolve = freq_res < (f2 - f1)
    quality = "Excellent" if N >= 100 else ("Good" if N >= 30 else "Poor")
    resolve_str = "✅ YES" if can_resolve else "❌ NO"
    print(f"{N:<12} {freq_res:<12.4f} {resolve_str:<15} {quality}")

print("=" * 70)
print(f"\nNote: Two frequencies are separated by Δf = {f2-f1:.4f}")
print(f"To resolve them, we need: Segment length N > 1/Δf = {1/(f2-f1):.1f}")
print(f"Minimum acceptable: N ≈ 20 samples")


# Create comparison plot
fig, ax = plt.subplots(figsize=(14, 8))

colors = ["blue", "green", "orange", "red"]
labels = ["N=1000 (Excellent)", "N=100 (Good)", "N=30 (Marginal)", "N=10 (Poor)"]
N_vals = [1000, 100, 30, 10]

for N, color, label in zip(N_vals, colors, labels):
    n_vec = np.arange(N)
    xn = np.sin(2 * np.pi * f1 * n_vec) + np.sin(2 * np.pi * f2 * n_vec)
    xk = np.fft.fft(xn, n=N_dft)
    xk_mag = np.abs(xk)
    freq_axis = np.linspace(0, 1, N_dft)

    # Normalize for comparison
    xk_mag_norm = xk_mag / np.max(xk_mag)

    ax.plot(
        freq_axis[: N_dft // 2],
        xk_mag_norm[: N_dft // 2],
        color=color,
        linewidth=2,
        label=label,
        alpha=0.7,
    )

# Mark true frequencies
ax.axvline(
    x=f1, color="black", linestyle="--", linewidth=2, label=f"True $f_1$={f1:.3f}"
)
ax.axvline(
    x=f2, color="black", linestyle=":", linewidth=2, label=f"True $f_2$={f2:.3f}"
)

ax.set_title(
    "Comparison: Effect of Segment Length on Spectral Resolution",
    fontsize=15,
    fontweight="bold",
)
ax.set_xlabel("Normalized Frequency $f$", fontsize=13)
ax.set_ylabel("Normalized $|X(f)|$", fontsize=13)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim([0.1, 0.35])  # Zoom around frequencies of interest
ax.set_ylim([0, 1.1])

plt.tight_layout()
plt.show()
