import numpy as np
from matplotlib import pyplot as plt

# Parameters
Nx = 28
alpha = 0.9
n = np.arange(Nx)
xn = alpha**n

# DTFT (continuous)
freq_dtft = np.linspace(0, 1, 1000)
X_dtft = (1 - (alpha * np.exp(-1j * 2 * np.pi * freq_dtft)) ** Nx) / (
    1 - alpha * np.exp(-1j * 2 * np.pi * freq_dtft)
)

# DFT lengths
N_vec = [2 * Nx, Nx, Nx // 2, Nx // 4]
titles = [
    "DFT length = 56 (Zero-padded)",
    "DFT length = 28 (Exact)",
    "DFT length = 14 (Aliased)",
    "DFT length = 7 (Severely Aliased)",
]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (N, title) in enumerate(zip(N_vec, titles)):
    # Compute DFT
    Xk = np.fft.fft(xn, n=N)
    freq_dft = np.arange(N) / N

    # Plot DTFT (red curve)
    axes[i].plot(
        freq_dtft,
        np.abs(X_dtft),
        "r-",
        linewidth=2.5,
        label="DTFT (True Spectrum)",
        alpha=0.7,
    )

    # Plot DFT (blue stems)
    markerline, stemlines, baseline = axes[i].stem(freq_dft, np.abs(Xk), basefmt=" ")
    plt.setp(markerline, "markerfacecolor", "blue", "markersize", 8)
    plt.setp(stemlines, "color", "blue", "linewidth", 1.5)
    markerline.set_label("DFT Samples")

    # Formatting
    axes[i].set_title(title, fontsize=13, fontweight="bold")
    axes[i].set_xlabel("Normalized Frequency $f$", fontsize=11)
    axes[i].set_ylabel("Magnitude $|X(f)|$", fontsize=11)
    axes[i].legend(fontsize=10)
    axes[i].grid(True, alpha=0.3, linestyle="--")
    axes[i].set_xlim([0, 1])
    axes[i].set_ylim([0, 10])

    # Add text annotation
    status = "✅ Correct" if N >= Nx else "❌ Time-Domain Aliasing"
    axes[i].text(
        0.7,
        8.5,
        status,
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

plt.tight_layout()
plt.show()

# Print numerical comparison
print("\n" + "=" * 80)
print("Numerical Comparison: X[0] (DC Component)")
print("=" * 80)
print(f"{'DFT Length':<12} {'X[0] Magnitude':<20} {'True Value':<15} {'Error (%)'}")
print("=" * 80)

true_value = np.abs(X_dtft[0])
for N in N_vec:
    Xk = np.fft.fft(xn, n=N)
    error_pct = 100 * abs(np.abs(Xk[0]) - true_value) / true_value
    status = "✓" if N >= Nx else "✗"
    print(
        f"{N:<12} {np.abs(Xk[0]):<20.6f} {true_value:<15.6f} {error_pct:>7.2f}% {status}"
    )


# Signal
Nx = 28
alpha = 0.9
n = np.arange(Nx)
xn = alpha**n

# Compute DFT
Xk = np.fft.fft(xn, n=Nx)

# Check symmetry
print("Checking Conjugate Symmetry:")
print("=" * 60)
print(f"{'k':<5} {'f=k/28':<10} {'X[k]':<30} {'|X[k]|'}")
print("=" * 60)

for k in range(0, 15):  # First half + Nyquist
    print(f"{k:<5} {k/28:<10.3f} {Xk[k]:<30} {np.abs(Xk[k]):.6f}")

print("\n" + "=" * 60)
print("Second Half (Should be conjugate symmetric):")
print("=" * 60)
print(f"{'k':<5} {'f=k/28':<10} {'X[k]':<30} {'|X[k]|'}")
print("=" * 60)

for k in range(14, 28):  # Second half
    mirror_k = 28 - k  # The mirror index
    print(
        f"{k:<5} {k/28:<10.3f} {Xk[k]:<30} {np.abs(Xk[k]):.6f} "
        f"(mirror of k={mirror_k})"
    )

# Verify symmetry numerically
print("\n" + "=" * 60)
print("Verification: X[k] vs X[N-k]*")
print("=" * 60)
print(f"{'k':<5} {'|X[k]|':<15} {'|X[28-k]|':<15} {'Match?'}")
print("=" * 60)

for k in range(1, 14):  # Skip k=0 and k=14 (they're real)
    match = "✓" if np.isclose(np.abs(Xk[k]), np.abs(Xk[28 - k])) else "✗"
    print(f"{k:<5} {np.abs(Xk[k]):<15.6f} {np.abs(Xk[28-k]):<15.6f} {match}")
