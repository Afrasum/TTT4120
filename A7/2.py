import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ============================================================
# PARAMETERS
# ============================================================
N = 20000
sigma_w = np.sqrt(3 / 4)  # Standard deviation of input noise

np.random.seed(42)

# ============================================================
# GENERATE SIGNAL x[n]
# ============================================================

import matplotlib.pyplot as plt

# Step 1: Generate white Gaussian noise w[n]
import numpy as np
from scipy import signal

# ============================================================
# PARAMETERS
# ============================================================
N = 20000
sigma_w = np.sqrt(3 / 4)  # Standard deviation of input noise

np.random.seed(42)

# ============================================================
# GENERATE SIGNAL x[n]
# ============================================================

# Step 1: Generate white Gaussian noise w[n]
w = np.random.normal(0, sigma_w, N)

# Step 2: Define filter H(z) = 1/(1 + 0.5*z^-1)
# In difference equation form: x[n] = -0.5*x[n-1] + w[n]
# Or in filter form: b = , a = [1, 0.5]

b = [1]  # Numerator coefficients
a = [1, 0.5]  # Denominator coefficients

# Step 3: Filter the noise
x = signal.lfilter(b, a, w)

# Alternative: Manual filtering using difference equation
# x = np.zeros(N)
# x[0] = w[0]
# for n in range(1, N):
#     x[n] = -0.5*x[n-1] + w[n]

print("=" * 60)
print("SIGNAL GENERATION")
print("=" * 60)
print(f"Generated {N} samples of x[n]")
print(f"Input noise variance: {sigma_w**2:.4f}")

# ============================================================
# ESTIMATE MEAN
# ============================================================

mean_estimate = np.mean(x)
mean_theoretical = 0

print("\n" + "=" * 60)
print("MEAN")
print("=" * 60)
print(f"Theoretical:  {mean_theoretical:.6f}")
print(f"Estimated:    {mean_estimate:+.6f}")
print(f"Error:        {abs(mean_estimate - mean_theoretical):.6f}")

# ============================================================
# ESTIMATE POWER
# ============================================================

power_estimate = np.mean(x**2)
power_theoretical = 1.0

print("\n" + "=" * 60)
print("POWER")
print("=" * 60)
print(f"Theoretical:  {power_theoretical:.6f}")
print(f"Estimated:    {power_estimate:.6f}")
print(f"Error:        {abs(power_estimate - power_theoretical):.6f}")

# ============================================================
# ESTIMATE AUTOCORRELATION
# ============================================================

# Method: Using np.correlate
autocorr_full = np.correlate(x, x, mode="full") / N

# Extract lags -10 to 10
center = len(autocorr_full) // 2
lags = np.arange(-10, 11)
autocorr_estimate = autocorr_full[center - 10 : center + 11]

# Theoretical autocorrelation
autocorr_theoretical = (-0.5) ** np.abs(lags)

# ============================================================
# PLOT AUTOCORRELATION COMPARISON
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Plot estimated
ax.stem(
    lags,
    autocorr_estimate,
    basefmt=" ",
    label="Estimated",
    linefmt="C0-",
    markerfmt="C0o",
)

# Plot theoretical
ax.plot(
    lags,
    autocorr_theoretical,
    "r--o",
    label="Theoretical: $(-1/2)^{|m|}$",
    markersize=8,
    linewidth=2,
)

ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax.set_xlabel("Lag m", fontsize=12)
ax.set_ylabel(r"$\gamma_{xx}[m]$", fontsize=12)
ax.set_title(
    "Autocorrelation Function: Estimated vs Theoretical", fontsize=14, fontweight="bold"
)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Print numerical comparison
print("\n" + "=" * 60)
print("AUTOCORRELATION FUNCTION")
print("=" * 60)
print(f"{'Lag':>5s} {'Theoretical':>12s} {'Estimated':>12s} {'Error':>12s}")
print("-" * 60)
for i, lag in enumerate(lags):
    theo = autocorr_theoretical[i]
    est = autocorr_estimate[i]
    err = abs(theo - est)
    print(f"{lag:5d} {theo:12.6f} {est:12.6f} {err:12.6f}")

# ============================================================
# ESTIMATE POWER SPECTRAL DENSITY (PERIODOGRAM)
# ============================================================

# Method 1: FFT of autocorrelation estimate
N_fft = 512
gamma_hat_fft = np.fft.fft(autocorr_estimate, N_fft)
freqs_fft = np.linspace(0, 2 * np.pi, N_fft)

# Method 2: Periodogram (direct method)
X_fft = np.fft.fft(x, N_fft)
periodogram = (np.abs(X_fft) ** 2) / N
freqs_periodogram = np.linspace(0, 2 * np.pi, N_fft)

# Theoretical PSD
omega = np.linspace(0, 2 * np.pi, 1000)
psd_theoretical = 3 / (5 + 4 * np.cos(omega))

# ============================================================
# PLOT POWER SPECTRAL DENSITY
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: FFT of autocorrelation
axes[0].plot(
    freqs_fft,
    np.abs(gamma_hat_fft),
    "b-",
    label="Estimated (FFT of autocorrelation)",
    linewidth=1.5,
)
axes[0].plot(
    omega, psd_theoretical, "r--", label="Theoretical: $3/(5+4\cos\omega)$", linewidth=2
)
axes[0].set_xlabel(r"Frequency $\omega$ (rad)", fontsize=12)
axes[0].set_ylabel(r"$\Gamma_{xx}(\omega)$", fontsize=12)
axes[0].set_title(
    "Power Spectral Density: FFT of Autocorrelation", fontsize=13, fontweight="bold"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_xlim([0, 2 * np.pi])
axes[0].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
axes[0].set_xticklabels(["0", r"$\pi$/2", r"$\pi$", r"3$\pi$/2", r"2$\pi$"])

# Subplot 2: Periodogram (more noisy)
axes[1].plot(
    freqs_periodogram,
    periodogram,
    "g-",
    label="Periodogram (direct method)",
    linewidth=1,
    alpha=0.7,
)
axes[1].plot(omega, psd_theoretical, "r--", label="Theoretical", linewidth=2)
axes[1].set_xlabel(r"Frequency $\omega$ (rad)", fontsize=12)
axes[1].set_ylabel(r"$\Gamma_{xx}(\omega)$", fontsize=12)
axes[1].set_title(
    "Power Spectral Density: Periodogram (Noisy!)", fontsize=13, fontweight="bold"
)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].set_xlim([0, 2 * np.pi])
axes[1].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
axes[1].set_xticklabels(["0", r"$\pi$/2", r"$\pi$", r"3$\pi$/2", r"2$\pi$"])

plt.tight_layout()
plt.show()


# Assume x is already generated from part (c)
N = 20000

# ============================================================
# BARTLETT METHOD IMPLEMENTATION
# ============================================================


def bartlett_psd(signal_data, K, N_fft=512):
    """
    Estimate PSD using Bartlett's method

    Parameters:
    -----------
    signal_data : array
        Input signal
    K : int
        Number of non-overlapping segments
    N_fft : int
        FFT length for each segment

    Returns:
    --------
    freqs : array
        Frequency axis
    psd_estimate : array
        Averaged PSD estimate
    """
    N = len(signal_data)
    segment_length = N // K

    # Initialize
    psd_sum = np.zeros(N_fft)

    # Process each segment
    for k in range(K):
        # Extract segment
        start_idx = k * segment_length
        end_idx = start_idx + segment_length
        segment = signal_data[start_idx:end_idx]

        # Compute periodogram for this segment
        X_segment = np.fft.fft(segment, N_fft)
        periodogram_segment = (np.abs(X_segment) ** 2) / segment_length

        # Accumulate
        psd_sum += periodogram_segment

    # Average
    psd_estimate = psd_sum / K

    # Frequency axis
    freqs = np.linspace(0, 2 * np.pi, N_fft)

    return freqs, psd_estimate


# ============================================================
# COMPUTE FOR DIFFERENT K VALUES
# ============================================================

K_values = [10, 100]
N_fft = 2048

# Theoretical PSD
omega_theory = np.linspace(0, 2 * np.pi, 2000)
psd_theory = 3 / (5 + 4 * np.cos(omega_theory))

# Periodogram (K=1, entire signal)
freqs_periodogram, psd_periodogram = bartlett_psd(x, K=1, N_fft=N_fft)

# ============================================================
# PLOT COMPARISON
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle("Bartlett Method: Effect of Averaging", fontsize=15, fontweight="bold")

# Plot 1: Periodogram (no averaging)
axes[0].plot(
    freqs_periodogram,
    psd_periodogram,
    "g-",
    linewidth=0.8,
    alpha=0.7,
    label="Periodogram (K=1)",
)
axes[0].plot(omega_theory, psd_theory, "r--", linewidth=2.5, label="Theoretical")
axes[0].set_ylabel(r"$\hat{\Gamma}_{xx}(\omega)$", fontsize=12)
axes[0].set_title("K = 1 (No Averaging) - Very Noisy", fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([0, 2 * np.pi])
axes[0].set_ylim([0, 4])

# Plot 2 & 3: Bartlett with different K
for idx, K in enumerate(K_values):
    freqs_bartlett, psd_bartlett = bartlett_psd(x, K=K, N_fft=N_fft)

    axes[idx + 1].plot(
        freqs_bartlett, psd_bartlett, "b-", linewidth=1.2, label=f"Bartlett (K={K})"
    )
    axes[idx + 1].plot(
        omega_theory, psd_theory, "r--", linewidth=2.5, label="Theoretical"
    )
    axes[idx + 1].set_ylabel(r"$\hat{\Gamma}_{xx}(\omega)$", fontsize=12)
    axes[idx + 1].set_title(
        f"K = {K} (Averaging {K} segments) - " f"Segment Length = {N//K}", fontsize=12
    )
    axes[idx + 1].legend(fontsize=10)
    axes[idx + 1].grid(True, alpha=0.3)
    axes[idx + 1].set_xlim([0, 2 * np.pi])
    axes[idx + 1].set_ylim([0, 4])

axes[2].set_xlabel(r"Frequency $\omega$ (rad)", fontsize=12)

# Set x-axis ticks
for ax in axes:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$/2", r"$\pi$", r"3$\pi$/2", r"2$\pi$"])

plt.tight_layout()
plt.show()

# ============================================================
# ALTERNATIVE: Using scipy.signal.welch
# ============================================================

# Welch's method (Bartlett with K=10 corresponds to welch with noverlap=0)
segment_length_10 = N // 10
freqs_welch_10, psd_welch_10 = signal.welch(
    x,
    nperseg=segment_length_10,
    noverlap=0,
    nfft=N_fft,
    fs=2 * np.pi,
    return_onesided=False,
    scaling="density",
)

segment_length_100 = N // 100
freqs_welch_100, psd_welch_100 = signal.welch(
    x,
    nperseg=segment_length_100,
    noverlap=0,
    nfft=N_fft,
    fs=2 * np.pi,
    return_onesided=False,
    scaling="density",
)

print("\n" + "=" * 60)
print("BARTLETT METHOD RESULTS")
print("=" * 60)
print(f"K = 10:  Segment length = {N//10}")
print(f"K = 100: Segment length = {N//100}")
print("\nObservations:")
print("- K=10:  Less smooth but better frequency resolution")
print("- K=100: Much smoother but coarser frequency resolution")
print("- Trade-off: Variance reduction vs. resolution")

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ============================================================
# RUN MULTIPLE REALIZATIONS
# ============================================================

num_realizations = 5
N = 20000
sigma_w = np.sqrt(3 / 4)
b = [1]
a = [1, 0.5]

# Storage
autocorr_realizations = []
psd_periodogram_realizations = []
psd_bartlett_10_realizations = []
psd_bartlett_100_realizations = []

for i in range(num_realizations):
    # Generate new realization
    np.random.seed(i)  # Different seed each time
    w = np.random.normal(0, sigma_w, N)
    x = signal.lfilter(b, a, w)

    # Autocorrelation
    autocorr_full = np.correlate(x, x, mode="full") / N
    center = len(autocorr_full) // 2
    autocorr = autocorr_full[center - 10 : center + 11]
    autocorr_realizations.append(autocorr)

    # Periodogram
    N_fft = 1024
    X_fft = np.fft.fft(x, N_fft)
    periodogram = (np.abs(X_fft) ** 2) / N
    psd_periodogram_realizations.append(periodogram)

    # Bartlett K=10
    freqs_10, psd_10 = bartlett_psd(x, K=10, N_fft=N_fft)
    psd_bartlett_10_realizations.append(psd_10)

    # Bartlett K=100
    freqs_100, psd_100 = bartlett_psd(x, K=100, N_fft=N_fft)
    psd_bartlett_100_realizations.append(psd_100)

# ============================================================
# PLOT: AUTOCORRELATION VARIABILITY
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

lags = np.arange(-10, 11)
autocorr_theoretical = (-0.5) ** np.abs(lags)

# Plot all realizations
for i, autocorr in enumerate(autocorr_realizations):
    ax.plot(lags, autocorr, "o-", alpha=0.6, label=f"Realization {i+1}")

# Plot theoretical
ax.plot(
    lags,
    autocorr_theoretical,
    "r--",
    linewidth=3,
    marker="s",
    markersize=10,
    label="Theoretical",
)

ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
ax.set_xlabel("Lag m", fontsize=12)
ax.set_ylabel(r"$\hat{\gamma}_{xx}[m]$", fontsize=12)
ax.set_title("Autocorrelation: Multiple Realizations", fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# ============================================================
# PLOT: PSD VARIABILITY
# ============================================================

omega_theory = np.linspace(0, 2 * np.pi, 1000)
psd_theory = 3 / (5 + 4 * np.cos(omega_theory))

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle(
    "PSD Estimates: Variability Across Realizations", fontsize=15, fontweight="bold"
)

freqs = np.linspace(0, 2 * np.pi, N_fft)

# Periodogram
for i, psd in enumerate(psd_periodogram_realizations):
    axes[0].plot(freqs, psd, alpha=0.5, linewidth=0.8)
axes[0].plot(omega_theory, psd_theory, "r--", linewidth=3, label="Theoretical")
axes[0].set_ylabel(r"$\hat{\Gamma}_{xx}(\omega)$", fontsize=11)
axes[0].set_title("Periodogram (K=1) - High Variability", fontsize=12)
axes[0].set_ylim([0, 5])
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bartlett K=10
for i, psd in enumerate(psd_bartlett_10_realizations):
    axes[1].plot(freqs_10, psd, alpha=0.6, linewidth=1.2)
axes[1].plot(omega_theory, psd_theory, "r--", linewidth=3, label="Theoretical")
axes[1].set_ylabel(r"$\hat{\Gamma}_{xx}(\omega)$", fontsize=11)
axes[1].set_title("Bartlett K=10 - Moderate Variability", fontsize=12)
axes[1].set_ylim([0, 4])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Bartlett K=100
for i, psd in enumerate(psd_bartlett_100_realizations):
    axes[2].plot(freqs_100, psd, alpha=0.7, linewidth=1.5)
axes[2].plot(omega_theory, psd_theory, "r--", linewidth=3, label="Theoretical")
axes[2].set_ylabel(r"$\hat{\Gamma}_{xx}(\omega)$", fontsize=11)
axes[2].set_xlabel(r"Frequency $\omega$ (rad)", fontsize=12)
axes[2].set_title("Bartlett K=100 - Low Variability", fontsize=12)
axes[2].set_ylim([0, 3])
axes[2].legend()
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$/2", r"$\pi$", r"3$\pi$/2", r"2$\pi$"])

plt.tight_layout()
plt.show()
