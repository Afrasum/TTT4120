import numpy as np
from scipy.linalg import solve


def gamma(l):
    if l == 0:
        return 1.25
    elif abs(l) == 1:
        return -0.5
    else:
        return 0.0


orders = [1, 2, 3]
results = []

for p in orders:
    R = np.array([[gamma(abs(i - j)) for j in range(p)] for i in range(p)], dtype=float)
    r = np.array([gamma(i + 1) for i in range(p)], dtype=float)

    c = solve(R, r)
    sigma_e2 = gamma(0) - np.dot(c, r)

    results.append((p, c, sigma_e2))

for p, c, sigma_e2 in results:
    print(f"p={p}")
    print("Coefficients:", np.round(c, 6))
    print("Prediction error variance:", round(sigma_e2, 6))
    print()
