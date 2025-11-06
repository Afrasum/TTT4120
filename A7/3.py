import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.stats import norm

# ============================================================
# GENERATE SIGNAL x[n] FROM PROBLEM 2
# ============================================================

N = 20000
sigma_w = np.sqrt(3 / 4)
np.random.seed(42)

# Generate white noise and filter
w = np.random.normal(0, sigma_w, N)
b = [1]
a = [1, 0.5]
x = signal.lfilter(b, a, w)

print("=" * 70)
print("PROBLEM 3: STATISTICAL PROPERTIES OF MEAN ESTIMATOR")
print("=" * 70)
print(f"\nGenerated signal x[n] with {N} samples")
print(f"True mean: μ = 0")
print(f"True variance of x[n]: σ² = 1")

# ============================================================
# FUNCTION TO ANALYZE MEAN ESTIMATOR
# ============================================================


def analyze_mean_estimator(signal_data, K, num_segments, K_label):
    """
    Analyze statistical properties of mean estimator

    Parameters:
    -----------
    signal_data : array
        Input signal x[n]
    K : int
        Segment length
    num_segments : int
        Number of segments to analyze
    K_label : str
        Label for plotting

    Returns:
    --------
    mean_estimates : array
        Array of mean estimates
    stats : dict
        Dictionary with statistical results
    """

    # Compute mean estimates for each segment
    mean_estimates = []

    for i in range(num_segments):
        start_idx = i * K
        end_idx = start_idx + K
        segment = signal_data[start_idx:end_idx]
        mean_est = np.mean(segment)
        mean_estimates.append(mean_est)

    mean_estimates = np.array(mean_estimates)

    # Compute statistics
    mean_of_estimates = np.mean(mean_estimates)
    var_of_estimates = np.var(mean_estimates, ddof=0)  # Population variance
    std_of_estimates = np.sqrt(var_of_estimates)

    # Theoretical variance
    theoretical_var = 1 / (3 * K)
    theoretical_std = np.sqrt(theoretical_var)

    stats = {
        "K": K,
        "num_segments": num_segments,
        "mean_of_estimates": mean_of_estimates,
        "var_of_estimates": var_of_estimates,
        "std_of_estimates": std_of_estimates,
        "theoretical_var": theoretical_var,
        "theoretical_std": theoretical_std,
        "var_error": abs(var_of_estimates - theoretical_var),
        "var_error_percent": 100
        * abs(var_of_estimates - theoretical_var)
        / theoretical_var,
    }

    return mean_estimates, stats


# ============================================================
# ANALYZE FOR K = 20, 40, 100
# ============================================================

K_values = [20, 40, 100]
all_results = {}

for K in K_values:
    num_segments = N // K
    mean_estimates, stats = analyze_mean_estimator(x, K, num_segments, f"K={K}")
    all_results[K] = {"estimates": mean_estimates, "stats": stats}

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(
    f"\n{'K':>5s} {'Segments':>10s} {'Mean(m̂)':>12s} {'Var(m̂)':>12s} "
    f"{'Theoretical':>12s} {'Error %':>10s}"
)
print("-" * 70)

for K in K_values:
    stats = all_results[K]["stats"]
    print(
        f"{stats['K']:5d} {stats['num_segments']:10d} "
        f"{stats['mean_of_estimates']:+12.6f} {stats['var_of_estimates']:12.6f} "
        f"{stats['theoretical_var']:12.6f} {stats['var_error_percent']:9.2f}%"
    )

print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

for K in K_values:
    stats = all_results[K]["stats"]
    print(f"\n--- K = {K} (Segment Length) ---")
    print(f"  Number of segments:        {stats['num_segments']}")
    print(f"  Mean of estimates:         {stats['mean_of_estimates']:+.6f}")
    print(f"  Variance of estimates:     {stats['var_of_estimates']:.6f}")
    print(f"  Std Dev of estimates:      {stats['std_of_estimates']:.6f}")
    print(f"  Theoretical variance:      {stats['theoretical_var']:.6f} = 1/(3×{K})")
    print(f"  Theoretical std dev:       {stats['theoretical_std']:.6f}")
    print(f"  Variance error:            {stats['var_error']:.6f}")
    print(f"  Error percentage:          {stats['var_error_percent']:.2f}%")

# ============================================================
# PLOT HISTOGRAMS WITH 20 BINS
# ============================================================

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle(
    "Histograms of Mean Estimates for Different Segment Lengths",
    fontsize=15,
    fontweight="bold",
)

for idx, K in enumerate(K_values):
    estimates = all_results[K]["estimates"]
    stats = all_results[K]["stats"]

    # Plot histogram
    counts, bins, patches = axes[idx].hist(
        estimates,
        bins=20,
        density=True,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label=f'Histogram ({stats["num_segments"]} estimates)',
    )

    # Overlay theoretical Gaussian
    x_range = np.linspace(estimates.min(), estimates.max(), 200)
    theoretical_pdf = norm.pdf(
        x_range, loc=0, scale=stats["theoretical_std"]  # True mean
    )
    axes[idx].plot(
        x_range,
        theoretical_pdf,
        "r--",
        linewidth=3,
        label=f'Theoretical N(0, {stats["theoretical_var"]:.4f})',
    )

    # Overlay empirical Gaussian
    empirical_pdf = norm.pdf(
        x_range, loc=stats["mean_of_estimates"], scale=stats["std_of_estimates"]
    )
    axes[idx].plot(
        x_range,
        empirical_pdf,
        "g-",
        linewidth=2.5,
        label=f'Empirical N({stats["mean_of_estimates"]:.4f}, {stats["var_of_estimates"]:.4f})',
    )

    # Vertical line at true mean
    axes[idx].axvline(
        x=0, color="red", linestyle=":", linewidth=2, alpha=0.7, label="True mean (μ=0)"
    )

    # Vertical line at sample mean
    axes[idx].axvline(
        x=stats["mean_of_estimates"],
        color="green",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label=f'Sample mean ({stats["mean_of_estimates"]:.4f})',
    )

    # Labels and formatting
    axes[idx].set_xlabel("Mean Estimate Value", fontsize=11)
    axes[idx].set_ylabel("Probability Density", fontsize=11)
    axes[idx].set_title(
        f"K = {K} (Segment Length) | "
        f'Var(m̂) ≈ {stats["var_of_estimates"]:.4f} '
        f'≈ 1/(3×{K}) = {stats["theoretical_var"]:.4f}',
        fontsize=12,
    )
    axes[idx].legend(fontsize=9, loc="upper right")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================
# PLOT: VARIANCE VS K
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

K_plot = np.array(K_values)
empirical_vars = [all_results[K]["stats"]["var_of_estimates"] for K in K_values]
theoretical_vars = [all_results[K]["stats"]["theoretical_var"] for K in K_values]

# Plot empirical variance
ax.plot(
    K_plot,
    empirical_vars,
    "o-",
    markersize=12,
    linewidth=2.5,
    color="steelblue",
    label="Empirical Var(m̂)",
)

# Plot theoretical variance
ax.plot(
    K_plot,
    theoretical_vars,
    "s--",
    markersize=12,
    linewidth=2.5,
    color="red",
    label="Theoretical Var(m̂) = 1/(3K)",
)

# Add theoretical curve
K_smooth = np.linspace(10, 110, 200)
var_smooth = 1 / (3 * K_smooth)
ax.plot(K_smooth, var_smooth, ":", linewidth=1.5, color="red", alpha=0.5)

ax.set_xlabel("Segment Length K", fontsize=13, fontweight="bold")
ax.set_ylabel("Variance of Mean Estimator", fontsize=13, fontweight="bold")
ax.set_title(
    "Variance of Mean Estimator vs Segment Length\n" "Demonstrates: Var(m̂) ∝ 1/K",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([15, 105])
ax.set_ylim([0, 0.020])

# Add text annotations
for K in K_values:
    var = all_results[K]["stats"]["var_of_estimates"]
    ax.annotate(
        f"K={K}\nVar={var:.5f}",
        xy=(K, var),
        xytext=(K - 5, var + 0.002),
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
    )

plt.tight_layout()
plt.show()

# ============================================================
# PLOT: COMPARISON OF ALL THREE HISTOGRAMS (NORMALIZED)
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))

colors = ["steelblue", "darkorange", "green"]
for idx, K in enumerate(K_values):
    estimates = all_results[K]["estimates"]
    stats = all_results[K]["stats"]

    # Normalized histogram
    ax.hist(
        estimates,
        bins=20,
        density=True,
        alpha=0.5,
        color=colors[idx],
        edgecolor="black",
        label=f'K={K} (Var={stats["var_of_estimates"]:.5f})',
    )

# Overlay theoretical distributions
x_range = np.linspace(-0.15, 0.15, 500)
for idx, K in enumerate(K_values):
    stats = all_results[K]["stats"]
    theoretical_pdf = norm.pdf(x_range, loc=0, scale=stats["theoretical_std"])
    ax.plot(x_range, theoretical_pdf, "--", linewidth=2.5, color=colors[idx], alpha=0.8)

ax.axvline(x=0, color="red", linestyle=":", linewidth=3, label="True mean (μ=0)")

ax.set_xlabel("Mean Estimate Value", fontsize=13, fontweight="bold")
ax.set_ylabel("Probability Density", fontsize=13, fontweight="bold")
ax.set_title(
    "Distribution of Mean Estimates: Effect of Segment Length K\n"
    "Larger K → Narrower distribution → More precise estimates",
    fontsize=14,
    fontweight="bold",
)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.15, 0.15])

plt.tight_layout()
plt.show()

# ============================================================
# STATISTICAL TESTS
# ============================================================

print("\n" + "=" * 70)
print("STATISTICAL VERIFICATION")
print("=" * 70)

for K in K_values:
    estimates = all_results[K]["estimates"]
    stats = all_results[K]["stats"]

    print(f"\n--- K = {K} ---")

    # Check if mean is close to 0
    mean_error = abs(stats["mean_of_estimates"])
    expected_std_error = stats["theoretical_std"] / np.sqrt(stats["num_segments"])

    print(f"  Mean of estimates: {stats['mean_of_estimates']:+.6f}")
    print(f"  Expected std error of mean: {expected_std_error:.6f}")
    print(
        f"  Is mean ≈ 0? {mean_error < 3*expected_std_error} "
        f"(|error| = {mean_error:.6f} < 3σ = {3*expected_std_error:.6f})"
    )

    # Check if variance matches theory
    var_ratio = stats["var_of_estimates"] / stats["theoretical_var"]
    print(f"  Empirical variance / Theoretical variance = {var_ratio:.4f}")
    print(f"  Matches theory? {0.90 < var_ratio < 1.10} (within 10%)")

    # Normality test (visual assessment from histogram)
    print(f"  Distribution appears Gaussian? Check histogram above")

# ============================================================
# KEY OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("KEY OBSERVATIONS")
print("=" * 70)
print(
    """
1. MEAN OF ESTIMATES:
   - All three cases: mean ≈ 0 ✓
   - Confirms unbiased estimator: E[m̂] = μ

2. VARIANCE OF ESTIMATES:
   - K=20:  Var(m̂) ≈ 0.0167 ≈ 1/60  = 1/(3×20)  ✓
   - K=40:  Var(m̂) ≈ 0.0083 ≈ 1/120 = 1/(3×40)  ✓
   - K=100: Var(m̂) ≈ 0.0033 ≈ 1/300 = 1/(3×100) ✓
   
   Theory predicts: Var(m̂) = 1/(3K)
   Empirical results match theory within statistical error!

3. DISTRIBUTION SHAPE:
   - All three histograms are approximately Gaussian ✓
   - Central Limit Theorem in action!
   - Even though x[n] is correlated, m̂ becomes Gaussian

4. EFFECT OF SEGMENT LENGTH:
   - Larger K → Smaller variance → More precise estimates
   - Doubling K approximately halves the variance
   - Trade-off: More precision requires more data

5. CORRELATION EFFECT:
   - For white noise: Var(m̂) = σ²/K = 1/K
   - For our signal: Var(m̂) = 1/(3K)
   - Factor of 3 improvement due to alternating correlation!
   - Negative correlations help cancel out noise
"""
)

print("=" * 70)
