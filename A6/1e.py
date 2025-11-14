import numpy as np
from matplotlib import pyplot as plt

# Signal
Nx = 28
alpha = 0.9
n = np.arange(Nx)
xn = alpha**n

# Compute DFT
Xk = np.fft.fft(xn, n=Nx)
freq = np.arange(Nx) / Nx

# Plot full spectrum
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Magnitude spectrum [0, 1]
axes[0].stem(freq, np.abs(Xk), basefmt=" ")
axes[0].axvline(
    x=0.5, color="red", linestyle="--", linewidth=2, label="f=0.5 (Nyquist)"
)
axes[0].axvspan(0, 0.5, alpha=0.2, color="green", label="Unique Info [0, 0.5]")
axes[0].axvspan(0.5, 1, alpha=0.2, color="orange", label="Redundant [0.5, 1]")
axes[0].set_title("Full Magnitude Spectrum: f ∈ [0, 1]", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Normalized Frequency $f$")
axes[0].set_ylabel("$|X[k]|$")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 1])

# Plot 2: First half [0, 0.5]
half_idx = Nx // 2 + 1
axes[1].stem(freq[:half_idx], np.abs(Xk[:half_idx]), basefmt=" ")
axes[1].set_title(
    "First Half Only: f ∈ [0, 0.5] (All Unique Information)",
    fontsize=14,
    fontweight="bold",
)
axes[1].set_xlabel("Normalized Frequency $f$")
axes[1].set_ylabel("$|X[k]|$")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 0.5])

# Plot 3: Overlay to show symmetry
axes[2].stem(
    freq[:half_idx],
    np.abs(Xk[:half_idx]),
    basefmt=" ",
    linefmt="blue",
    markerfmt="bo",
    label="First Half [0, 0.5]",
)
# Flip the second half
freq_flipped = 1 - freq[half_idx:]
axes[2].stem(
    freq_flipped,
    np.abs(Xk[half_idx:]),
    basefmt=" ",
    linefmt="red",
    markerfmt="r^",
    label="Second Half [0.5, 1] (flipped)",
)
axes[2].set_title(
    "Demonstrating Symmetry: Second Half is Mirror of First Half",
    fontsize=14,
    fontweight="bold",
)
axes[2].set_xlabel("Normalized Frequency $f$")
axes[2].set_ylabel("$|X[k]|$")
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([0, 0.5])

plt.tight_layout()
plt.show()
