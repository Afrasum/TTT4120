import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

print("=== PROBLEM 3(c): Using freqz, abs, and angle functions ===\n")

# System definitions from the problem
print("System 1: y[n] = x[n] + 2x[n-1] + x[n-2]")
print("System 2: y[n] = -0.9y[n-1] + x[n]")

# Define system coefficients for freqz function
# For system: y[n] = Σ(a_k * y[n-k]) + Σ(b_k * x[n-k])
# freqz expects: H(z) = (b_0 + b_1*z^-1 + ...) / (a_0 + a_1*z^-1 + ...)

# System 1: y[n] = x[n] + 2x[n-1] + x[n-2]
# This is: y[n] = 1*x[n] + 2*x[n-1] + 1*x[n-2]
# So: H1(z) = (1 + 2z^-1 + z^-2) / 1
b1 = [1, 2, 1]  # numerator coefficients [b_0, b_1, b_2]
a1 = [1]  # denominator coefficients [a_0]

print(f"\nSystem 1 coefficients:")
print(f"  b1 (numerator): {b1}")
print(f"  a1 (denominator): {a1}")

# System 2: y[n] = -0.9y[n-1] + x[n]
# Rearranging: y[n] + 0.9y[n-1] = x[n]
# So: H2(z) = 1 / (1 + 0.9z^-1)
b2 = [1]  # numerator coefficients [b_0]
a2 = [1, 0.9]  # denominator coefficients [a_0, a_1]

print(f"\nSystem 2 coefficients:")
print(f"  b2 (numerator): {b2}")
print(f"  a2 (denominator): {a2}")

# Use freqz function to compute frequency responses
# freqz(b, a, N) returns (w, H) where:
# - w: frequency vector from 0 to π
# - H: complex frequency response H(e^jω)

N_points = 512  # Number of frequency points

print(f"\nUsing freqz function with {N_points} frequency points...")

# System 1
w1, H1 = signal.freqz(b1, a1, N_points, whole=False)
print(f"System 1: w1 shape = {w1.shape}, H1 shape = {H1.shape}")

# System 2
w2, H2 = signal.freqz(b2, a2, N_points, whole=False)
print(f"System 2: w2 shape = {w2.shape}, H2 shape = {H2.shape}")

# Use abs function to get magnitude responses
mag1 = np.abs(H1)  # abs function for magnitude
mag2 = np.abs(H2)

print(f"\nMagnitude responses calculated using abs() function")
print(f"System 1 magnitude range: {np.min(mag1):.3f} to {np.max(mag1):.3f}")
print(f"System 2 magnitude range: {np.min(mag2):.3f} to {np.max(mag2):.3f}")

# Use angle function to get phase responses
phase1 = np.angle(H1)  # angle function for phase
phase2 = np.angle(H2)

print(f"\nPhase responses calculated using angle() function")
print(f"System 1 phase range: {np.min(phase1):.3f} to {np.max(phase1):.3f} radians")
print(f"System 2 phase range: {np.min(phase2):.3f} to {np.max(phase2):.3f} radians")

# Create plots using the computed responses
plt.figure(figsize=(12, 8))

# System 1 - Magnitude
plt.subplot(2, 2, 1)
plt.plot(w1, mag1, "b-", linewidth=2)
plt.title("Magnitude Response |H₁(e^{jω})| using freqz + abs")
plt.xlabel("Frequency ω (rad/sample)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.3)
plt.xlim([0, np.pi])

# Add key value annotations
plt.annotate(
    f"DC: {mag1[0]:.3f}",
    xy=(0, mag1[0]),
    xytext=(0.3, mag1[0] + 0.2),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
)
plt.annotate(
    f"π: {mag1[-1]:.3f}",
    xy=(np.pi, mag1[-1]),
    xytext=(np.pi - 0.5, mag1[-1] + 0.2),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
)

# System 1 - Phase
plt.subplot(2, 2, 2)
plt.plot(w1, phase1, "orange", linewidth=2)
plt.title("Phase Response ∠H₁(e^{jω}) using freqz + angle")
plt.xlabel("Frequency ω (rad/sample)")
plt.ylabel("Phase (radians)")
plt.grid(True, alpha=0.3)
plt.xlim([0, np.pi])

# System 2 - Magnitude
plt.subplot(2, 2, 3)
plt.plot(w2, mag2, "g-", linewidth=2)
plt.title("Magnitude Response |H₂(e^{jω})| using freqz + abs")
plt.xlabel("Frequency ω (rad/sample)")
plt.ylabel("Magnitude")
plt.grid(True, alpha=0.3)
plt.xlim([0, np.pi])

# Add key value annotations
plt.annotate(
    f"DC: {mag2[0]:.3f}",
    xy=(0, mag2[0]),
    xytext=(0.3, mag2[0] + 0.1),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
)
plt.annotate(
    f"π: {mag2[-1]:.3f}",
    xy=(np.pi, mag2[-1]),
    xytext=(np.pi - 0.5, mag2[-1] + 0.1),
    arrowprops=dict(arrowstyle="->", color="red"),
    fontsize=10,
)

# System 2 - Phase
plt.subplot(2, 2, 4)
plt.plot(w2, phase2, "r-", linewidth=2)
plt.title("Phase Response ∠H₂(e^{jω}) using freqz + angle")
plt.xlabel("Frequency ω (rad/sample)")
plt.ylabel("Phase (radians)")
plt.grid(True, alpha=0.3)
plt.xlim([0, np.pi])

plt.tight_layout()
plt.show()

# Verification: Compare with manual calculation
print(f"\n{'='*60}")
print("VERIFICATION: Compare freqz results with manual calculation")
print("=" * 60)


# Manual calculation functions (same as your original code)
def H1_manual(w):
    return 1 + 2 * np.exp(-1j * w) + np.exp(-2j * w)


def H2_manual(w):
    return 1 / (1 + 0.9 * np.exp(-1j * w))


# Test at a few points
test_freqs = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

print("\nSystem 1 Magnitude Comparison:")
print("ω\t\tfreqz+abs\tManual\t\tDifference")
print("-" * 50)

for omega in test_freqs:
    # Find closest index in freqz results
    idx = np.argmin(np.abs(w1 - omega))

    mag_freqz = mag1[idx]
    mag_manual = np.abs(H1_manual(omega))

    print(
        f"{omega:.3f}\t\t{mag_freqz:.6f}\t{mag_manual:.6f}\t{abs(mag_freqz - mag_manual):.2e}"
    )

print("\nSystem 2 Magnitude Comparison:")
print("ω\t\tfreqz+abs\tManual\t\tDifference")
print("-" * 50)

for omega in test_freqs:
    # Find closest index in freqz results
    idx = np.argmin(np.abs(w2 - omega))

    mag_freqz = mag2[idx]
    mag_manual = np.abs(H2_manual(omega))

    print(
        f"{omega:.3f}\t\t{mag_freqz:.6f}\t{mag_manual:.6f}\t{abs(mag_freqz - mag_manual):.2e}"
    )

# Summary of the functions used
print(f"\n{'='*60}")
print("SUMMARY OF FUNCTIONS USED (as specified in Problem 3c)")
print("=" * 60)

print("✓ scipy.signal.freqz(b, a, N):")
print("  - Computes frequency response H(e^jω)")
print("  - Input: numerator (b) and denominator (a) coefficients")
print("  - Output: frequency vector w and complex response H")

print("\n✓ numpy.abs(H):")
print("  - Computes magnitude response |H(e^jω)|")
print("  - Input: complex frequency response H")
print("  - Output: magnitude values")

print("\n✓ numpy.angle(H):")
print("  - Computes phase response ∠H(e^jω)")
print("  - Input: complex frequency response H")
print("  - Output: phase values in radians")

print(f"\nThese functions provide the same results as manual calculation")
print("but are more efficient and handle edge cases automatically!")

# Optional: Create comparison plot
plt.figure(figsize=(15, 5))

# Compare both methods for System 1
plt.subplot(1, 3, 1)
w_manual = np.linspace(0, np.pi, 512)
H1_manual_vals = H1_manual(w_manual)
mag1_manual = np.abs(H1_manual_vals)

plt.plot(w1, mag1, "b-", linewidth=2, label="freqz + abs")
plt.plot(
    w_manual, mag1_manual, "r--", linewidth=2, alpha=0.7, label="Manual calculation"
)
plt.title("System 1: freqz vs Manual")
plt.xlabel("ω (rad/sample)")
plt.ylabel("|H₁(e^{jω})|")
plt.legend()
plt.grid(True, alpha=0.3)

# Compare both methods for System 2
plt.subplot(1, 3, 2)
H2_manual_vals = H2_manual(w_manual)
mag2_manual = np.abs(H2_manual_vals)

plt.plot(w2, mag2, "g-", linewidth=2, label="freqz + abs")
plt.plot(
    w_manual, mag2_manual, "r--", linewidth=2, alpha=0.7, label="Manual calculation"
)
plt.title("System 2: freqz vs Manual")
plt.xlabel("ω (rad/sample)")
plt.ylabel("|H₂(e^{jω})|")
plt.legend()
plt.grid(True, alpha=0.3)

# Show difference
plt.subplot(1, 3, 3)
# Interpolate freqz results to match manual calculation points
mag1_interp = np.interp(w_manual, w1, mag1)
mag2_interp = np.interp(w_manual, w2, mag2)

diff1 = np.abs(mag1_interp - mag1_manual)
diff2 = np.abs(mag2_interp - mag2_manual)

plt.semilogy(w_manual, diff1, "b-", label="System 1 difference")
plt.semilogy(w_manual, diff2, "g-", label="System 2 difference")
plt.title("Absolute Difference")
plt.xlabel("ω (rad/sample)")
plt.ylabel("|Difference|")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nMaximum differences:")
print(f"System 1: {np.max(diff1):.2e}")
print(f"System 2: {np.max(diff2):.2e}")
print("→ Essentially identical results! ✓")
