import matplotlib.pyplot as plt
import numpy as np
import scipy.io

# Load the signals
data = scipy.io.loadmat("signals.mat")
x = data["x"].flatten()  # Emitted signal
y = data["y"].flatten()  # Received signal

print("=" * 70)
print("PROBLEM 2a: RADAR SIGNAL ANALYSIS")
print("=" * 70)
print(f"\nEmitted signal x[n]:")
print(f"  Length: {len(x)}")
print(f"  Min value: {np.min(x):.4f}")
print(f"  Max value: {np.max(x):.4f}")
print(f"  Mean: {np.mean(x):.4f}")
print(f"  Std dev: {np.std(x):.4f}")

print(f"\nReceived signal y[n]:")
print(f"  Length: {len(y)}")
print(f"  Min value: {np.min(y):.4f}")
print(f"  Max value: {np.max(y):.4f}")
print(f"  Mean: {np.mean(y):.4f}")
print(f"  Std dev: {np.std(y):.4f}")

# Create the plots
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot emitted signal x[n]
n = np.arange(len(x))
axes[0].stem(n, x, linefmt="b-", markerfmt="bo", basefmt="k-", label="x[n]")
axes[0].set_xlabel("n (sample index)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("x[n]", fontsize=12, fontweight="bold")
axes[0].set_title("Emitted Signal x[n]", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, len(x)])
axes[0].legend(fontsize=11)

# Add some statistics on the plot
stats_text = f"Length: {len(x)}\nMax: {np.max(x):.2f}\nEnergy: {np.sum(x**2):.2f}"
axes[0].text(
    0.02,
    0.98,
    stats_text,
    transform=axes[0].transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)

# Plot received signal y[n]
axes[1].stem(n, y, linefmt="r-", markerfmt="ro", basefmt="k-", label="y[n]")
axes[1].set_xlabel("n (sample index)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("y[n]", fontsize=12, fontweight="bold")
axes[1].set_title("Received Signal y[n]", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, len(y)])
axes[1].legend(fontsize=11)

# Add some statistics on the plot
stats_text = f"Length: {len(y)}\nMax: {np.max(y):.2f}\nMin: {np.min(y):.2f}"
axes[1].text(
    0.02,
    0.98,
    stats_text,
    transform=axes[1].transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
)

plt.tight_layout()
plt.savefig("2a_signals.png", dpi=300, bbox_inches="tight")
print("\n✓ Plot saved!")

# Create a more detailed comparison plot
fig2, axes2 = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Emitted signal (zoomed to first 50 samples)
n_zoom = 50
axes2[0].stem(n[:n_zoom], x[:n_zoom], linefmt="b-", markerfmt="bo", basefmt="k-")
axes2[0].set_xlabel("n", fontsize=11)
axes2[0].set_ylabel("x[n]", fontsize=11)
axes2[0].set_title(
    "Emitted Signal x[n] (First 50 samples)", fontsize=13, fontweight="bold"
)
axes2[0].grid(True, alpha=0.3)
axes2[0].set_xlim([0, n_zoom])

# Plot 2: Received signal (zoomed to first 50 samples)
axes2[1].stem(n[:n_zoom], y[:n_zoom], linefmt="r-", markerfmt="ro", basefmt="k-")
axes2[1].set_xlabel("n", fontsize=11)
axes2[1].set_ylabel("y[n]", fontsize=11)
axes2[1].set_title(
    "Received Signal y[n] (First 50 samples)", fontsize=13, fontweight="bold"
)
axes2[1].grid(True, alpha=0.3)
axes2[1].set_xlim([0, n_zoom])

# Plot 3: Both signals overlaid (normalized for comparison)
x_norm = x / np.max(np.abs(x))
y_norm = y / np.max(np.abs(y))
axes2[2].plot(n, x_norm, "b-", linewidth=1.5, alpha=0.7, label="x[n] (normalized)")
axes2[2].plot(n, y_norm, "r-", linewidth=1.5, alpha=0.7, label="y[n] (normalized)")
axes2[2].set_xlabel("n", fontsize=11)
axes2[2].set_ylabel("Normalized amplitude", fontsize=11)
axes2[2].set_title("Both Signals Overlaid (Normalized)", fontsize=13, fontweight="bold")
axes2[2].grid(True, alpha=0.3)
axes2[2].legend(fontsize=11)
axes2[2].set_xlim([0, len(x)])

plt.tight_layout()
plt.savefig("2a_detailed.png", dpi=300, bbox_inches="tight")
print("✓ Detailed plot saved!")

# Analysis: Can we detect the object from visual inspection?
print("\n" + "=" * 70)
print("ANALYSIS: Can we detect if an object was hit?")
print("=" * 70)

# Look at the characteristics
print("\nCharacteristics of x[n]:")
print(f"  - Appears to be a short pulse/burst")
print(f"  - Most energy in first ~30 samples")
print(f"  - Well-defined structure")

print("\nCharacteristics of y[n]:")
print(f"  - Looks very noisy!")
print(f"  - No clear pattern visible")
print(f"  - Amplitude similar to x[n] but spread throughout")

# Try to find correlation visually
print("\nVisual inspection:")
print("  - x[n] has clear structure")
print("  - y[n] appears to be mostly noise")
print("  - DIFFICULT to see if y[n] contains a delayed copy of x[n]!")
print("  - The noise overwhelms any reflected signal")

# Calculate SNR estimate
signal_energy_x = np.sum(x**2)
noise_estimate = np.var(y)  # Rough estimate assuming mostly noise
print(f"\nRough estimates:")
print(f"  - Signal energy of x[n]: {signal_energy_x:.2f}")
print(f"  - Noise variance in y[n]: {noise_estimate:.4f}")
print(f"  - Very difficult to detect reflection visually!")

print("\n" + "=" * 70)
print("CONCLUSION FOR 2a:")
print("=" * 70)
print(
    """
From visual inspection of the two plots alone:

❌ NO, we CANNOT reliably determine if an object was hit!

Reasons:
--------
1. The received signal y[n] appears very noisy
2. No obvious delayed copy of x[n] is visible in y[n]
3. The noise level is comparable to or larger than the signal
4. Any reflected signal is buried in the noise

The emitted signal x[n] has clear structure, but the received 
signal y[n] looks like random noise. We cannot tell if there's
a delayed, attenuated copy of x[n] hidden in y[n].

This is EXACTLY why we need crosscorrelation (parts b, c, d)!
Crosscorrelation can detect signals even when they're buried 
in noise and not visible to the naked eye.
"""
)
