import numpy as np

# Given parameters
a1 = 0.9  # AR coefficient
sigma_v_sq = 0.09  # Variance of v[n]
sigma_w_sq = 1.0  # Variance of w[n]

# Calculate variance of s[n]
sigma_s_sq = sigma_v_sq / (1 - a1**2)
print(f"Variance of s[n]: σ²_s = {sigma_s_sq:.4f}")

# Calculate autocorrelation of s[n]
r_s = sigma_s_sq * np.array([a1**0, a1**1, a1**2])
print(f"\nAutocorrelation of s[n]:")
print(f"r_s[0] = {r_s[0]:.4f}")
print(f"r_s[1] = {r_s[1]:.4f}")
print(f"r_s[2] = {r_s[2]:.4f}")

# Calculate autocorrelation of x[n]
r_x = r_s.copy()
r_x[0] += sigma_w_sq  # Add noise variance at lag 0
print(f"\nAutocorrelation of x[n]:")
print(f"r_x[0] = {r_x[0]:.4f}")
print(f"r_x[1] = {r_x[1]:.4f}")
print(f"r_x[2] = {r_x[2]:.4f}")

# Build R_x matrix (Toeplitz autocorrelation matrix)
R_x = np.array(
    [[r_x[0], r_x[1], r_x[2]], [r_x[1], r_x[0], r_x[1]], [r_x[2], r_x[1], r_x[0]]]
)

print("\nAutocorrelation matrix R_x:")
print(R_x)

# Cross-correlation vector r_xs
r_xs = r_s  # Since w[n] is uncorrelated with s[n]

print("\nCross-correlation vector r_xs:")
print(r_xs)

# Solve the Wiener-Hopf equations: R_x * w = r_xs
w = np.linalg.solve(R_x, r_xs)

print("\n" + "=" * 50)
print("WIENER FILTER COEFFICIENTS:")
print("=" * 50)
print(f"w[0] = {w[0]:.6f}")
print(f"w[1] = {w[1]:.6f}")
print(f"w[2] = {w[2]:.6f}")
print(
    "\nThe filter is: ŝ[n] = {:.6f}·x[n] + {:.6f}·x[n-1] + {:.6f}·x[n-2]".format(
        w[0], w[1], w[2]
    )
)
