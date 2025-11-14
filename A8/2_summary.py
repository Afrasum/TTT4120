import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.linalg import solve, toeplitz

# ============================================================
# ASSIGNMENT 8 - PROBLEMS 1 & 2: COMPLETE SOLUTION
# ============================================================

print("=" * 80)
print("TTT4120 ASSIGNMENT 8: AR AND MA PROCESSES")
print("=" * 80)

# ============================================================
# PROBLEM 1: AR(1) PROCESS OPTIMAL PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("PROBLEM 1: AR(1) PROCESS")
print("=" * 80)

# Signal: x[n] = -0.5*x[n-1] + w[n], where w[n] ~ N(0, 3/4)
# H(z) = 1/(1 + 0.5*z^-1)
# Autocorrelation: gamma_xx[m] = (-1/2)^|m|


def gamma_xx_ar1(m):
    """Autocorrelation for AR(1): x[n] = -0.5*x[n-1] + w[n]"""
    return (-0.5) ** np.abs(m)


print("\nProcess: x[n] = -0.5*x[n-1] + w[n]")
print("Type: AR(1) - AutoRegressive order 1")
print("Transfer function: H(z) = 1/(1 + 0.5*z^-1)")
print("Autocorrelation: γ_xx[m] = (-1/2)^|m|")

# Solve for optimal predictors p=1 and p=2
print("\n--- First-Order Predictor (p=1) ---")
R1 = np.array([[gamma_xx_ar1(0)]])
r1 = np.array([gamma_xx_ar1(1)])
c1 = solve(R1, r1)
sigma_e1_sq = gamma_xx_ar1(0) - np.dot(c1, r1)

print(f"Coefficient: c_1 = {c1[0]:.6f}")
print(f"Predictor: x̂[n] = {c1[0]:.4f}*x[n-1]")
print(f"Error variance: σ²_e = {sigma_e1_sq:.6f}")
print(f"Matches AR coefficient: {np.isclose(c1[0], -0.5)}")
print(f"Achieves driving noise variance (0.75): {np.isclose(sigma_e1_sq, 0.75)}")

print("\n--- Second-Order Predictor (p=2) ---")
R2 = np.array([[gamma_xx_ar1(0), gamma_xx_ar1(1)], [gamma_xx_ar1(1), gamma_xx_ar1(0)]])
r2 = np.array([gamma_xx_ar1(1), gamma_xx_ar1(2)])
c2 = solve(R2, r2)
sigma_e2_sq = gamma_xx_ar1(0) - np.dot(c2, r2)

print(f"Coefficients: c_1 = {c2[0]:.6f}, c_2 = {c2[1]:.6f}")
print(f"Predictor: x̂[n] = {c2[0]:.4f}*x[n-1] + {c2[1]:.4f}*x[n-2]")
print(f"Error variance: σ²_e = {sigma_e2_sq:.6f}")
print(f"c_2 is zero: {np.isclose(c2[1], 0)}")
print(f"No improvement over p=1: {np.isclose(sigma_e1_sq, sigma_e2_sq)}")

print("\nConclusion: AR(1) process perfectly predicted with first-order predictor!")

# ============================================================
# PROBLEM 2: MA(1) PROCESS
# ============================================================

print("\n" + "=" * 80)
print("PROBLEM 2: MA(1) PROCESS")
print("=" * 80)

# Signal: x[n] = w[n] - 0.5*w[n-1], where w[n] ~ N(0, 1)
# H(z) = 1 - 0.5*z^-1


def gamma_xx_ma1(l):
    """Autocorrelation for MA(1): x[n] = w[n] - 0.5*w[n-1]"""
    if l == 0:
        return 5 / 4
    elif abs(l) == 1:
        return -1 / 2
    else:
        return 0.0


print("\nProcess: x[n] = w[n] - 0.5*w[n-1]")
print("Type: MA(1) - Moving Average order 1")
print("Transfer function: H(z) = 1 - 0.5*z^-1")
print("\nAutocorrelation:")
print(f"  γ_xx[0] = {gamma_xx_ma1(0)}")
print(f"  γ_xx[1] = {gamma_xx_ma1(1)}")
print(f"  γ_xx[l] = 0 for |l| ≥ 2")
print("\nPower Spectral Density: Γ_xx(f) = 5/4 - cos(2πf)")

# ============================================================
# PROBLEM 2(c): OPTIMAL PREDICTORS FOR MA(1)
# ============================================================

print("\n" + "=" * 80)
print("PROBLEM 2(c): OPTIMAL PREDICTORS FOR MA(1)")
print("=" * 80)

orders = [1, 2, 3]
ma_results = {}

for p in orders:
    # Build Yule-Walker system
    R = np.array([[gamma_xx_ma1(abs(i - j)) for j in range(p)] for i in range(p)])
    r = np.array([gamma_xx_ma1(k + 1) for k in range(p)])

    # Solve for predictor coefficients
    c = solve(R, r)

    # Prediction error variance
    sigma_e_sq = gamma_xx_ma1(0) - np.dot(c, r)

    # Store results
    ma_results[p] = {
        "c": c,
        "a": -c,  # AR coefficients are negatives of predictor coefficients
        "sigma_e_sq": sigma_e_sq,
    }

    print(f"\n--- Order p={p} ---")
    print(f"Coefficients: {', '.join([f'c_{i+1}={c[i]:.4f}' for i in range(p)])}")
    print(f"Error variance: σ²_e = {sigma_e_sq:.6f}")
    improvement = (gamma_xx_ma1(0) - sigma_e_sq) / gamma_xx_ma1(0) * 100
    print(f"Variance reduction: {improvement:.2f}%")

    if sigma_e_sq > 1.0:
        print(f"⚠ σ²_e > σ²_w (1.0) - Cannot reach driving noise variance!")

print("\nConclusion: MA(1) cannot be perfectly predicted from past outputs!")
print("Even 3rd order predictor has σ²_e ≈ 1.003 > σ²_w = 1.0")

# ============================================================
# PROBLEM 2(d): AR MODEL APPROXIMATION OF MA(1)
# ============================================================

print("\n" + "=" * 80)
print("PROBLEM 2(d): AR MODEL APPROXIMATION")
print("=" * 80)


# True MA(1) PSD
def true_psd_ma1(f):
    """True PSD of MA(1): Γ_xx(f) = 5/4 - cos(2πf)"""
    return 1.25 - np.cos(2 * np.pi * f)


# Frequency vector
N_freq = 2048
frequencies = np.linspace(0, 0.5, N_freq)
true_psd_values = true_psd_ma1(frequencies)

# Compute AR model PSDs
ar_psds = {}
mse_values = {}

for p in orders:
    # AR coefficients (scipy convention: [1, a1, a2, ...])
    a_coeffs = np.concatenate([[1], ma_results[p]["a"]])
    b_coeffs = [1]

    # Compute frequency response
    w, H = signal.freqz(b_coeffs, a_coeffs, worN=N_freq, fs=1.0)

    # PSD = σ²_e * |H(ω)|²
    psd = ma_results[p]["sigma_e_sq"] * np.abs(H) ** 2
    ar_psds[p] = psd

    # Compute MSE
    error = psd - true_psd_values
    mse = np.mean(error**2)
    mse_values[p] = mse

    print(f"\nAR[{p}] Model:")
    print(f"  Coefficients: a = {ma_results[p]['a']}")
    print(f"  Error variance: σ²_e = {ma_results[p]['sigma_e_sq']:.6f}")
    print(f"  MSE vs true PSD: {mse:.6f}")

best_model = min(mse_values, key=mse_values.get)
print(
    f"\nBest approximation: AR[{best_model}] (lowest MSE = {mse_values[best_model]:.6f})"
)

# ============================================================
# VISUALIZATION: PROBLEM 1 - AR(1) AUTOCORRELATION
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(
    "Assignment 8 Summary: AR(1) and MA(1) Processes", fontsize=16, fontweight="bold"
)

# Plot 1: AR(1) Autocorrelation
lags_ar = np.arange(-10, 11)
gamma_ar_values = [gamma_xx_ar1(l) for l in lags_ar]

axes[0, 0].stem(lags_ar, gamma_ar_values, basefmt=" ")
axes[0, 0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
axes[0, 0].set_xlabel("Lag m", fontsize=11)
axes[0, 0].set_ylabel(r"$\gamma_{xx}[m]$", fontsize=11)
axes[0, 0].set_title(
    "Problem 1: AR(1) Autocorrelation\n"
    + r"$\gamma_{xx}[m] = (-1/2)^{|m|}$ (Infinite length)",
    fontsize=12,
    fontweight="bold",
)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([-0.6, 1.2])

# Plot 2: MA(1) Autocorrelation
lags_ma = np.arange(-5, 6)
gamma_ma_values = [gamma_xx_ma1(l) for l in lags_ma]

markerline, stemlines, baseline = axes[0, 1].stem(lags_ma, gamma_ma_values, basefmt=" ")
# Highlight non-zero values
axes[0, 1].stem(
    [0], [gamma_xx_ma1(0)], markerfmt="ro", linefmt="r-", basefmt=" ", label="l=0"
)
axes[0, 1].stem(
    [-1, 1],
    [gamma_xx_ma1(-1), gamma_xx_ma1(1)],
    markerfmt="go",
    linefmt="g-",
    basefmt=" ",
    label="|l|=1",
)
axes[0, 1].axhline(y=0, color="k", linestyle="-", linewidth=0.5)
axes[0, 1].set_xlabel("Lag m", fontsize=11)
axes[0, 1].set_ylabel(r"$\gamma_{xx}[m]$", fontsize=11)
axes[0, 1].set_title(
    "Problem 2: MA(1) Autocorrelation\n" + "Non-zero only for |m|≤1 (Finite length)",
    fontsize=12,
    fontweight="bold",
)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=9)
axes[0, 1].set_ylim([-0.8, 1.5])

# Plot 3: MA(1) Prediction Error Variance
p_vals = list(ma_results.keys())
sigma_e_vals = [ma_results[p]["sigma_e_sq"] for p in p_vals]

axes[1, 0].plot(
    p_vals,
    sigma_e_vals,
    "o-",
    linewidth=2.5,
    markersize=12,
    color="steelblue",
    label="Prediction error σ²_e",
)
axes[1, 0].axhline(
    y=gamma_xx_ma1(0),
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Signal variance = {gamma_xx_ma1(0):.2f}",
)
axes[1, 0].axhline(
    y=1.0, color="green", linestyle="--", linewidth=2, label="Driving noise σ²_w = 1.00"
)
axes[1, 0].set_xlabel("Predictor Order p", fontsize=11)
axes[1, 0].set_ylabel("Error Variance", fontsize=11)
axes[1, 0].set_title(
    "Problem 2(c): MA(1) Prediction Error\n" + "Cannot reach σ²_w = 1.0",
    fontsize=12,
    fontweight="bold",
)
axes[1, 0].set_xticks(p_vals)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=9)
axes[1, 0].set_ylim([0.95, 1.30])

# Plot 4: AR Approximations of MA(1) PSD
axes[1, 1].plot(
    frequencies, true_psd_values, "k-", linewidth=3, label="True MA(1)", zorder=10
)

colors = ["blue", "red", "green"]
linestyles = ["--", "-.", ":"]
for idx, p in enumerate(orders):
    axes[1, 1].plot(
        frequencies,
        ar_psds[p],
        color=colors[idx],
        linestyle=linestyles[idx],
        linewidth=2,
        label=f"AR[{p}] (MSE={mse_values[p]:.3f})",
        alpha=0.8,
    )

axes[1, 1].set_xlabel("Normalized Frequency f", fontsize=11)
axes[1, 1].set_ylabel("PSD Γ_xx(f)", fontsize=11)
axes[1, 1].set_title(
    f"Problem 2(d): AR Approximations\nBest: AR[{best_model}]",
    fontsize=12,
    fontweight="bold",
)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=9, loc="upper left")
axes[1, 1].set_xlim([0, 0.5])
axes[1, 1].set_ylim([0, 2.5])

plt.tight_layout()
plt.show()

# ============================================================
# DETAILED PSD COMPARISON
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top: PSDs
axes[0].plot(
    frequencies,
    true_psd_values,
    "k-",
    linewidth=3,
    label="True MA(1): Γ_xx(f) = 5/4 - cos(2πf)",
    zorder=10,
)

for idx, p in enumerate(orders):
    axes[0].plot(
        frequencies,
        ar_psds[p],
        color=colors[idx],
        linestyle=linestyles[idx],
        linewidth=2.5,
        label=f"AR[{p}] approximation",
        alpha=0.8,
    )

axes[0].set_ylabel("PSD Γ_xx(f)", fontsize=12, fontweight="bold")
axes[0].set_title(
    "Power Spectral Density: True vs AR Approximations", fontsize=14, fontweight="bold"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11, loc="upper left")
axes[0].set_xlim([0, 0.5])
axes[0].set_ylim([0, 2.5])

# Bottom: Errors
for idx, p in enumerate(orders):
    error = ar_psds[p] - true_psd_values
    axes[1].plot(
        frequencies,
        error,
        color=colors[idx],
        linewidth=2,
        label=f"AR[{p}] error",
        alpha=0.8,
    )

axes[1].axhline(y=0, color="k", linestyle="-", linewidth=1)
axes[1].set_xlabel("Normalized Frequency f", fontsize=12, fontweight="bold")
axes[1].set_ylabel("Approximation Error", fontsize=12, fontweight="bold")
axes[1].set_title("AR Model Errors (AR_PSD - True_PSD)", fontsize=14, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)
axes[1].set_xlim([0, 0.5])

plt.tight_layout()
plt.show()

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY")
print("=" * 80)

print("\n--- PROBLEM 1: AR(1) PROCESS ---")
print(f"{'Property':<30s} {'Value':<30s}")
print("-" * 60)
print(f"{'Process type':<30s} {'AR(1) - AutoRegressive':<30s}")
print(f"{'Equation':<30s} {'x[n] = -0.5*x[n-1] + w[n]':<30s}")
print(f"{'Transfer function':<30s} {'H(z) = 1/(1+0.5*z^-1)':<30s}")
print(f"{'Autocorrelation':<30s} {'(-1/2)^|m| (infinite)':<30s}")
print(f"{'Optimal predictor order':<30s} {'p=1 (perfect prediction)':<30s}")
print(f"{'Prediction error (p=1)':<30s} {f'{sigma_e1_sq:.6f} = σ²_w':<30s}")

print("\n--- PROBLEM 2: MA(1) PROCESS ---")
print(f"{'Property':<30s} {'Value':<30s}")
print("-" * 60)
print(f"{'Process type':<30s} {'MA(1) - Moving Average':<30s}")
print(f"{'Equation':<30s} {'x[n] = w[n] - 0.5*w[n-1]':<30s}")
print(f"{'Transfer function':<30s} {'H(z) = 1 - 0.5*z^-1':<30s}")
print(f"{'Autocorrelation':<30s} {'Non-zero only for |m|≤1':<30s}")
print(f"{'PSD':<30s} {'Γ_xx(f) = 5/4 - cos(2πf)':<30s}")

print("\n--- PREDICTION RESULTS (MA(1)) ---")
print(
    f"{'Order':<8s} {'c_1':<12s} {'c_2':<12s} {'c_3':<12s} {'σ²_e':<12s} {'Improve%':<10s}"
)
print("-" * 80)
for p in orders:
    c = ma_results[p]["c"]
    sigma_e = ma_results[p]["sigma_e_sq"]
    improvement = (gamma_xx_ma1(0) - sigma_e) / gamma_xx_ma1(0) * 100

    c_str = [f"{c[i]:.4f}" if i < len(c) else "---" for i in range(3)]
    print(
        f"{'p='+str(p):<8s} {c_str[0]:<12s} {c_str[1]:<12s} {c_str[2]:<12s} "
        f"{sigma_e:<12.6f} {improvement:<10.2f}"
    )

print("\n--- AR APPROXIMATION RESULTS ---")
print(f"{'Model':<8s} {'MSE':<15s} {'Max Error':<15s} {'Quality':<15s}")
print("-" * 80)
for p in orders:
    error = ar_psds[p] - true_psd_values
    mse = mse_values[p]
    max_err = np.max(np.abs(error))

    if mse < 0.01:
        quality = "Excellent"
    elif mse < 0.05:
        quality = "Good"
    else:
        quality = "Poor"

    print(f"{'AR['+str(p)+']':<8s} {mse:<15.6f} {max_err:<15.6f} {quality:<15s}")

print(f"\nBest AR model: AR[{best_model}]")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print(
    """
1. AR vs MA PREDICTABILITY:
   • AR(1): Perfectly predictable with first-order predictor (σ²_e = σ²_w)
   • MA(1): Cannot achieve σ²_w even with higher-order predictors
   
2. AUTOCORRELATION STRUCTURE:
   • AR: Infinite-length autocorrelation (exponentially decaying)
   • MA: Finite-length autocorrelation (q+1 non-zero lags)
   
3. AR APPROXIMATION OF MA:
   • Cannot perfectly match with finite-order AR
   • Higher orders give better approximation
   • AR[3] best for MA(1) approximation
   
4. PRACTICAL IMPLICATIONS:
   • Use AR models for prediction tasks (better predictability)
   • Use MA models for accurate spectral representation
   • For MA processes, use higher-order AR for approximation
"""
)

print("=" * 80)
print("CODE EXECUTION COMPLETE")
print("=" * 80)
