import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# PARAMETERS
# ============================================================
N = 20000  # Number of samples
np.random.seed(42)  # For reproducibility

# ============================================================
# GENERATE NOISE SIGNALS
# ============================================================
binary_noise = np.random.choice([-1, 1], size=N)
gaussian_noise = np.random.randn(N)
uniform_noise = np.random.uniform(-np.sqrt(3), np.sqrt(3), N)

noises = {"Binary": binary_noise, "Gaussian": gaussian_noise, "Uniform": uniform_noise}

# ============================================================
# ESTIMATE AND DISPLAY MEAN VALUES
# ============================================================
print("=" * 60)
print("MEAN ESTIMATES (Theoretical = 0):")
print("=" * 60)
for name, noise in noises.items():
    mean_est = np.mean(noise)
    print(f"{name:12s}: {mean_est:+.6f}  |  Error: {abs(mean_est):.6f}")

# ============================================================
# ESTIMATE AUTOCORRELATION FUNCTIONS
# ============================================================
lags = np.arange(-10, 11)
lag_range = 10

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle(
    "Autocorrelation Function Estimates (N=20000 samples)",
    fontsize=14,
    fontweight="bold",
)

for idx, (name, noise) in enumerate(noises.items()):
    # Compute autocorrelation using np.correlate
    autocorr_full = np.correlate(noise, noise, mode="full") / N

    # Extract center portion (lags -10 to +10)
    center_idx = len(autocorr_full) // 2
    autocorr_segment = autocorr_full[
        center_idx - lag_range : center_idx + lag_range + 1
    ]

    # Plot estimated autocorrelation
    axes[idx].stem(
        lags,
        autocorr_segment,
        basefmt=" ",
        label="Estimated",
        linefmt="C0-",
        markerfmt="C0o",
    )

    # Overlay theoretical values
    theoretical = np.zeros_like(lags, dtype=float)
    theoretical[lag_range] = 1.0  # Delta at k=0
    axes[idx].stem(
        lags,
        theoretical,
        basefmt=" ",
        label="Theoretical δ[k]",
        linefmt="r--",
        markerfmt="r^",
    )

    # Formatting
    axes[idx].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    axes[idx].set_title(f"{name} Noise")
    axes[idx].set_ylabel(r"$\hat{R}_X[k]$", fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(loc="upper right")
    axes[idx].set_ylim([-0.2, 1.2])

axes[2].set_xlabel("Lag k", fontsize=12)
plt.tight_layout()
plt.show()

# ============================================================
# COMPARE VALUES AT k=0
# ============================================================
print("\n" + "=" * 60)
print("AUTOCORRELATION AT k=0 (Should be 1.0):")
print("=" * 60)
for name, noise in noises.items():
    R_0 = np.var(noise, ddof=0)  # Same as R[0] = E[X²]
    print(f"{name:12s}: {R_0:.6f}  |  Error: {abs(R_0 - 1.0):.6f}")
