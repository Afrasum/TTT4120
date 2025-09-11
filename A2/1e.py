import matplotlib.pyplot as plt
import numpy as np

print("=== WHY X(ω) AND c_k(ω) DON'T LOOK IDENTICAL ===\n")


def original_dtft(omega):
    """DTFT of original signal x[n]: X(ω) = 2 + 2cos(ω)"""
    return 2 + 2 * np.cos(omega)


def dtfs_coefficient(k, N):
    """DTFS coefficient: c_k = (1/N)[2 + 2cos(2πk/N)]"""
    return (1 / N) * (2 + 2 * np.cos(2 * np.pi * k / N))


print("FUNDAMENTAL DIFFERENCE:")
print("=" * 50)
print("• X(ω) is a CONTINUOUS function of frequency")
print("• {c_k} are DISCRETE values at specific frequencies")
print("• When we plot c_k vs ω, we get SAMPLES, not a continuous curve")

N = 10
omega_continuous = np.linspace(-np.pi, np.pi, 1000)
X_continuous = original_dtft(omega_continuous)

# Calculate DTFS coefficients
k_range = np.arange(N)
c_k_values = [dtfs_coefficient(k, N) for k in k_range]

# Map to [-π, π] frequency range
omega_k = 2 * np.pi * k_range / N
omega_centered = omega_k.copy()
omega_centered[omega_centered > np.pi] -= 2 * np.pi

# Sort for plotting
sort_idx = np.argsort(omega_centered)
omega_sorted = omega_centered[sort_idx]
c_k_sorted = np.array(c_k_values)[sort_idx]

print("COMPARISON FOR N = 10:")
print("=" * 50)

print("X(ω) = 2 + 2cos(ω):")
print("• Continuous function defined for ALL ω")
print("• Smooth curve from -π to π")
print("• Peak value: X(0) = 4")
print("• Minimum value: X(π) = 0")

print(f"\nc_k coefficients:")
print("• Only defined at discrete frequencies ω = 2πk/N")
print("• We have only 10 discrete values")
print("• Peak value: c_0 = 0.4 (note: 1/N scaling!)")
print("• Each c_k represents X(ω) sampled and scaled")

print(f"\nKey differences:")
print("1. CONTINUITY: X(ω) is continuous, {c_k} are discrete points")
print("2. AMPLITUDE: c_k = (1/N) × X(ω), so c_k values are much smaller")
print("3. RESOLUTION: X(ω) has infinite resolution, {c_k} has only N points")

# Create comparison plots
plt.figure(figsize=(15, 12))

# Plot 1: Both on same scale (shows the scaling difference)
plt.subplot(2, 3, 1)
plt.plot(omega_continuous, X_continuous, "b-", label="X(ω) (DTFT)")
plt.stem(omega_sorted, c_k_sorted, basefmt="r-", markerfmt="ro", label="c_k (DTFS)")
plt.xlabel("ω (radians)")
plt.ylabel("Amplitude")
plt.title("Direct Comparison (Different Scales!)")
plt.grid(True, alpha=0.3)
plt.xlim([-np.pi, np.pi])
plt.legend()
plt.text(
    0,
    3,
    "Notice: c_k values are much smaller!",
    ha="center",
    bbox=dict(boxstyle="round", facecolor="yellow"),
)

# Plot 2: Scaled comparison (shows they sample the same shape)
plt.subplot(2, 3, 2)
plt.plot(omega_continuous, X_continuous, "b-", label="X(ω)")
plt.stem(
    omega_sorted,
    c_k_sorted * N,
    basefmt="r-",
    markerfmt="ro",
    label="N×c_k (scaled DTFS)",
)
plt.xlabel("ω (radians)")
plt.ylabel("Amplitude")
plt.title("Scaled Comparison (Same Shape!)")
plt.grid(True, alpha=0.3)
plt.xlim([-np.pi, np.pi])
plt.legend()

# Show sampling points
for omega, c_val in zip(omega_sorted, c_k_sorted):
    x_val = original_dtft(omega)
    plt.plot([omega, omega], [0, x_val], "g--", alpha=0.5)
    plt.scatter([omega], [x_val], color="green", s=50, zorder=5)

plt.text(
    0,
    3,
    "c_k samples X(ω) at discrete points",
    ha="center",
    bbox=dict(boxstyle="round", facecolor="lightgreen"),
)

# Plot 3: X(ω) alone
plt.subplot(2, 3, 3)
plt.plot(omega_continuous, X_continuous, "b-")
plt.xlabel("ω (radians)")
plt.ylabel("X(ω)")
plt.title("Original DTFT: X(ω) = 2 + 2cos(ω)")
plt.grid(True, alpha=0.3)
plt.xlim([-np.pi, np.pi])
plt.ylim([0, 4.5])

# Annotate key points
plt.annotate(
    "Peak: X(0) = 4",
    xy=(0, 4),
    xytext=(0.5, 4.2),
    arrowprops=dict(arrowstyle="->", color="red"),
)
plt.annotate(
    "Zero: X(π) = 0",
    xy=(np.pi, 0),
    xytext=(2, 0.5),
    arrowprops=dict(arrowstyle="->", color="red"),
)

# Plot 4: {c_k} alone
plt.subplot(2, 3, 4)
plt.stem(omega_sorted, c_k_sorted, basefmt="r-", markerfmt="ro")
plt.xlabel("ω (radians)")
plt.ylabel("c_k")
plt.title("DTFS Coefficients: c_k")
plt.grid(True, alpha=0.3)
plt.xlim([-np.pi, np.pi])

# Annotate key points
plt.annotate(
    f"Peak: c_0 = {c_k_sorted[omega_sorted == 0][0]:.3f}",
    xy=(0, c_k_sorted[omega_sorted == 0][0]),
    xytext=(0.5, 0.35),
    arrowprops=dict(arrowstyle="->", color="red"),
)

# Plot 5: Show what "identical" would look like (WRONG!)
plt.subplot(2, 3, 5)
plt.plot(omega_continuous, X_continuous, "b-", label="X(ω)")
# This would be wrong - continuous line through discrete points
omega_interpolated = np.linspace(-np.pi, np.pi, 1000)
c_k_interpolated = np.interp(omega_interpolated, omega_sorted, c_k_sorted)
plt.plot(
    omega_interpolated,
    c_k_interpolated,
    "r--",
    label="Interpolated c_k (WRONG!)",
)
plt.xlabel("ω (radians)")
plt.ylabel("Amplitude")
plt.title('What "Identical" Would Mean (INCORRECT!)')
plt.grid(True, alpha=0.3)
plt.xlim([-np.pi, np.pi])
plt.legend()
plt.text(
    0,
    2,
    "This is NOT correct!\nc_k are discrete values only",
    ha="center",
    bbox=dict(boxstyle="round", facecolor="pink"),
)

# Plot 6: Correct interpretation
plt.subplot(2, 3, 6)
plt.text(
    0.1,
    0.9,
    "CORRECT INTERPRETATION:",
    fontsize=14,
    weight="bold",
    transform=plt.gca().transAxes,
)
plt.text(
    0.1, 0.8, "• X(ω): Continuous spectrum", fontsize=12, transform=plt.gca().transAxes
)
plt.text(
    0.1, 0.7, "• {c_k}: Discrete samples", fontsize=12, transform=plt.gca().transAxes
)
plt.text(
    0.1,
    0.6,
    "• NOT identical curves!",
    fontsize=12,
    weight="bold",
    color="red",
    transform=plt.gca().transAxes,
)

plt.text(
    0.1,
    0.45,
    "RELATIONSHIP:",
    fontsize=14,
    weight="bold",
    transform=plt.gca().transAxes,
)
plt.text(
    0.1, 0.35, "c_k = (1/N) × X(2πk/N)", fontsize=12, transform=plt.gca().transAxes
)
plt.text(
    0.1, 0.25, "• Sampling relationship", fontsize=12, transform=plt.gca().transAxes
)
plt.text(0.1, 0.15, "• Scaling by 1/N", fontsize=12, transform=plt.gca().transAxes)
plt.text(
    0.1, 0.05, "• Discrete vs continuous", fontsize=12, transform=plt.gca().transAxes
)

plt.xlim([0, 1])
plt.ylim([0, 1])
plt.axis("off")

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print("WHY THEY'RE NOT IDENTICAL")
print("=" * 60)

print("1. MATHEMATICAL NATURE:")
print("   • X(ω) is a continuous function: defined for every ω ∈ ℝ")
print("   • {c_k} are discrete coefficients: only N values")

print("\n2. AMPLITUDE SCALING:")
print("   • X(ω) ranges from 0 to 4")
print("   • c_k ranges from 0 to 0.4 (scaled by 1/N)")

print("\n3. FREQUENCY RESOLUTION:")
print("   • X(ω) has infinite frequency resolution")
print("   • {c_k} has resolution limited by N")

print("\n4. REPRESENTATION:")
print("   • X(ω): Spectrum of aperiodic signal")
print("   • {c_k}: Spectrum of periodic signal")

# Numerical demonstration
print(f"\n{'='*60}")
print("NUMERICAL DEMONSTRATION")
print("=" * 60)

print("At ω = 0:")
print(f"  X(0) = {original_dtft(0):.3f}")
print(f"  c_0 = {dtfs_coefficient(0, N):.3f}")
print(f"  Ratio: X(0)/c_0 = {original_dtft(0)/dtfs_coefficient(0, N):.1f} = N")

print("\nAt ω = π/5 ≈ 0.628:")
omega_test = np.pi / 5
k_test = 1  # corresponds to 2π×1/10 = π/5
print(f"  X(π/5) = {original_dtft(omega_test):.3f}")
print(f"  c_1 = {dtfs_coefficient(k_test, N):.3f}")
print(
    f"  Ratio: X(π/5)/c_1 = {original_dtft(omega_test)/dtfs_coefficient(k_test, N):.1f} = N"
)

print(f"\n{'='*60}")
print("CORRECT ANSWER")
print("=" * 60)

print("❌ NO, X(ω) and c_k(ω) do NOT look identical!")
print("")
print("✅ The correct relationship is:")
print("   • c_k are SAMPLES of X(ω) at discrete frequencies")
print("   • c_k = (1/N) × X(2πk/N)")
print("   • They have the same SHAPE when properly scaled")
print("   • But fundamentally different nature: continuous vs discrete")
print("")
print("🔑 Key insight:")
print("   Periodic extension in TIME → Sampling in FREQUENCY")
print("   Continuous spectrum → Discrete spectrum")
print("   This is the foundation of the DFT!")

print(f"\n{'='*60}")
print("ANALOGY")
print("=" * 60)

print("Think of it like photography:")
print("• X(ω) = continuous scene (infinite detail)")
print("• {c_k} = digital photo (finite pixels sampling the scene)")
print("• Higher N = higher resolution photo")
print("• But a photo is still discrete samples, not the continuous scene!")
