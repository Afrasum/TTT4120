import numpy as np
from matplotlib import pyplot as plt

# Frequencies
f1 = 7 / 40  # 0.175
f2 = 9 / 40  # 0.225

# Create figure with two subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# ===== Plot 1: Ideal spectrum [0, 0.5] =====
axes[0].stem([f1], linefmt="r-", markerfmt="r^", basefmt=" ", label=f"$f_1$ = {f1:.3f}")
axes[0].stem([f2], linefmt="g-", markerfmt="g^", basefmt=" ", label=f"$f_2$ = {f2:.3f}")

# Zero line
freq_axis = np.linspace(0, 0.5, 1000)
axes[0].plot(freq_axis, np.zeros(1000), "b-", linewidth=1, alpha=0.3)

axes[0].set_title(
    "Ideal Magnitude Spectrum: $f \\in [0, 0.5]$", fontsize=14, fontweight="bold"
)
axes[0].set_xlabel("Normalized Frequency $f$", fontsize=12)
axes[0].set_ylabel("$|X(f)|$", fontsize=12)
axes[0].set_xlim([0, 0.5])
axes[0].set_ylim([0, 1.2])
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Add frequency labels
axes[0].axvline(x=f1, color="red", linestyle="--", alpha=0.3)
axes[0].axvline(x=f2, color="green", linestyle="--", alpha=0.3)
axes[0].text(f1, -0.1, f"{f1:.3f}", ha="center", fontsize=10, color="red")
axes[0].text(f2, -0.1, f"{f2:.3f}", ha="center", fontsize=10, color="green")

# ===== Plot 2: Full spectrum [0, 1] showing symmetry =====
axes[1].stem(
    [f1, f2],
    [1, 1],
    linefmt="r-",
    markerfmt="r^",
    basefmt=" ",
    label="Positive frequencies",
)
axes[1].stem(
    [1 - f2, 1 - f1],
    [1, 1],
    linefmt="b-",
    markerfmt="bo",
    basefmt=" ",
    label="Negative frequencies (mirrored)",
)

# Zero line
freq_axis_full = np.linspace(0, 1, 1000)
axes[1].plot(freq_axis_full, np.zeros(1000), "k-", linewidth=1, alpha=0.3)

# Nyquist line
axes[1].axvline(
    x=0.5, color="orange", linestyle="--", linewidth=2, label="Nyquist (f=0.5)"
)

axes[1].set_title(
    "Full Magnitude Spectrum: $f \\in [0, 1]$ (Shows Symmetry)",
    fontsize=14,
    fontweight="bold",
)
axes[1].set_xlabel("Normalized Frequency $f$", fontsize=12)
axes[1].set_ylabel("$|X(f)|$", fontsize=12)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.2])
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

# Add labels for all four frequencies
for freq, color, label in [
    (f1, "red", f"$f_1$={f1:.3f}"),
    (f2, "green", f"$f_2$={f2:.3f}"),
    (1 - f2, "blue", f"$1-f_2$={1-f2:.3f}"),
    (1 - f1, "blue", f"$1-f_1$={1-f1:.3f}"),
]:
    axes[1].axvline(x=freq, color=color, linestyle="--", alpha=0.3)
    axes[1].text(freq, -0.1, label, ha="center", fontsize=9, color=color)

plt.tight_layout()
plt.show()

# Print summary
print("=" * 70)
print("SIGNAL SPECTRUM ANALYSIS")
print("=" * 70)
print(f"\nSignal: x[n] = sin(2π·f₁·n) + sin(2π·f₂·n)")
print(f"\nFrequency components:")
print(f"  f₁ = 7/40 = {f1}")
print(f"  f₂ = 9/40 = {f2}")
print(f"\nFrequency separation: Δf = {f2 - f1} = {(f2-f1):.3f}")
print(f"\nSpectrum characteristics:")
print(f"  - Type: Discrete (line spectrum)")
print(f"  - Number of impulses in [0, 0.5]: 2")
print(f"  - Number of impulses in [0, 1]: 4 (due to symmetry)")
print(f"\nAll four impulse locations:")
print(f"  1. f = {f1:.3f} (in [0, 0.5])")
print(f"  2. f = {f2:.3f} (in [0, 0.5])")
print(f"  3. f = {1-f2:.3f} (in [0.5, 1] - mirror of f₂)")
print(f"  4. f = {1-f1:.3f} (in [0.5, 1] - mirror of f₁)")
print("=" * 70)
