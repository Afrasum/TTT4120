import numpy as np
from scipy.linalg import solve, toeplitz

# ============================================================
# AUTOCORRELATION FUNCTION
# ============================================================


def gamma_xx(m):
    """Autocorrelation function: gamma_xx[m] = (-1/2)^|m|"""
    return (-0.5) ** np.abs(m)


print("=" * 70)
print("PROBLEM 1: AR(1) PROCESS OPTIMAL PREDICTION")
print("=" * 70)

# Signal properties
print("\nSignal properties:")
print(f"  Process type: AR(1)")
print(f"  Difference equation: x[n] = -0.5*x[n-1] + w[n]")
print(f"  Autocorrelation: γ_xx[m] = (-1/2)^|m|")
print(f"  Variance: σ²_x = {gamma_xx(0):.4f}")
print(f"  Driving noise variance: σ²_w = 0.75")

# ============================================================
# FIRST-ORDER PREDICTOR (p=1)
# ============================================================

print("\n" + "=" * 70)
print("FIRST-ORDER PREDICTOR (p=1)")
print("=" * 70)

p = 1

# Build Yule-Walker system
R1 = np.array([[gamma_xx(0)]])
r1 = np.array([gamma_xx(1)])

# Solve for coefficients
c1 = solve(R1, r1)

# Prediction error variance
sigma_e1_squared = gamma_xx(0) - np.dot(c1, r1)

print(f"\nYule-Walker equation:")
print(f"  c₁ × γ_xx[0] = γ_xx[1]")
print(f"  c₁ × {gamma_xx(0)} = {gamma_xx(1)}")

print(f"\nOptimal predictor coefficients:")
print(f"  c₁ = {c1[0]:.6f}")

print(f"\nPredictor equation:")
print(f"  x̂[n] = {c1[0]:.6f} × x[n-1]")

print(f"\nPrediction error variance:")
print(f"  σ²_e = {sigma_e1_squared:.6f}")
print(f"  Equal to driving noise variance: {np.isclose(sigma_e1_squared, 0.75)}")

# ============================================================
# SECOND-ORDER PREDICTOR (p=2)
# ============================================================

print("\n" + "=" * 70)
print("SECOND-ORDER PREDICTOR (p=2)")
print("=" * 70)

p = 2

# Build Yule-Walker system
R2 = np.array([[gamma_xx(0), gamma_xx(1)], [gamma_xx(1), gamma_xx(0)]])
r2 = np.array([gamma_xx(1), gamma_xx(2)])

# Solve for coefficients
c2 = solve(R2, r2)

# Prediction error variance
sigma_e2_squared = gamma_xx(0) - np.dot(c2, r2)

print(f"\nYule-Walker equations:")
print(f"  c₁ × γ_xx[0] + c₂ × γ_xx[1] = γ_xx[1]")
print(f"  c₁ × γ_xx[1] + c₂ × γ_xx[0] = γ_xx[2]")

print(f"\nSubstituting values:")
print(f"  c₁ × {gamma_xx(0)} + c₂ × {gamma_xx(1)} = {gamma_xx(1)}")
print(f"  c₁ × {gamma_xx(1)} + c₂ × {gamma_xx(0)} = {gamma_xx(2)}")

print(f"\nMatrix form:")
print(f"  R₂ =")
print(f"  {R2}")
print(f"\n  r₂ = {r2}")

print(f"\nOptimal predictor coefficients:")
print(f"  c₁ = {c2[0]:.6f}")
print(f"  c₂ = {c2[1]:.6f}")

print(f"\nPredictor equation:")
print(f"  x̂[n] = {c2[0]:.6f} × x[n-1] + {c2[1]:.6f} × x[n-2]")

print(f"\nPrediction error variance:")
print(f"  σ²_e = {sigma_e2_squared:.6f}")

print(f"\nComparison with first-order:")
print(f"  c₂ is effectively zero: {np.isclose(c2[1], 0)}")
print(f"  Error variance unchanged: {np.isclose(sigma_e1_squared, sigma_e2_squared)}")

# ============================================================
# SUMMARY TABLE
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'Order':>6s} {'c₁':>10s} {'c₂':>10s} {'σ²_e':>10s} {'Improvement':>15s}")
print("-" * 70)
print(f"{'p=1':>6s} {c1[0]:10.6f} {'---':>10s} {sigma_e1_squared:10.6f} {'---':>15s}")
print(
    f"{'p=2':>6s} {c2[0]:10.6f} {c2[1]:10.6f} {sigma_e2_squared:10.6f} "
    f"{'None':>15s}"
)

# ============================================================
# KEY INSIGHTS
# ============================================================

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print(
    """
1. AR(1) PROCESS IDENTIFICATION:
   ✓ Transfer function H(z) = 1/(1 + 0.5z⁻¹) is all-pole
   ✓ Difference equation: x[n] = -0.5x[n-1] + w[n]
   ✓ Classic AR(1) form with coefficient a₁ = 0.5

2. FIRST-ORDER PREDICTOR:
   ✓ Optimal coefficient: c₁ = -0.5 (matches AR coefficient!)
   ✓ Prediction error variance: σ²_e = 0.75 (equals driving noise σ²_w)
   ✓ This is the BEST possible predictor for this process

3. SECOND-ORDER PREDICTOR:
   ✓ Optimal coefficients: c₁ = -0.5, c₂ = 0
   ✓ Second coefficient is ZERO—no improvement from p=1
   ✓ Prediction error variance: σ²_e = 0.75 (unchanged)
   ✓ Confirms process is truly AR(1), not higher order

4. FUNDAMENTAL PRINCIPLE:
   For an AR(p) process, the optimal p-th order predictor:
   - Has coefficients equal to the AR coefficients
   - Achieves minimum error variance = driving noise variance
   - Cannot be improved by higher-order predictors

5. WHY c₂ = 0?
   Because x[n-2] is conditionally independent of x[n] given x[n-1]
   In other words: x[n-1] contains ALL the information about x[n]
                   that past samples can provide
"""
)

print("=" * 70)
