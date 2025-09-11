import matplotlib.pyplot as plt
import numpy as np


def x(n):
    return 2 if n == 0 else (1 if abs(n) == 1 else 0)


def y(n, M):
    return 1 if (-M <= n < M) else 0


def X(w):
    return 2 + 2 * np.cos(w)


def Y(w, M):
    return np.sin(w * (M + 1 / 2)) / np.sin(w / 2)


def z(n, N):
    return np.array([sum([x(ni - l * N) for l in range(-N, N + 1)]) for ni in n])


# plot x ∈ [-pi, pi]
w = np.linspace(-np.pi, np.pi, 1000)
plt.plot(w, X(w))
plt.title("DTFT of x[n]")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.grid()
plt.axhline(0, color="black", lw=0.5)
plt.axvline(0, color="black", lw=0.5)

# plot Y ∈ [-pi, pi] for M = 10

M = 10
plt.figure()
plt.plot(w, Y(w, M))
plt.title(f"DTFT of y[n] for M={M}")
plt.xlabel("Frequency (rad/sample)")
plt.ylabel("Magnitude")
plt.grid()
plt.axhline(0, color="black", lw=0.5)
plt.axvline(0, color="black", lw=0.5)

# plot n ∈ [-10, 10] for N = 6
N = 6
n = np.arange(-10, 11)
plt.figure()
plt.stem(n, z(n, N))
plt.title(f"z[n] for N={N}")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.grid()


def c_k(k, N):
    """Correct DTFS coefficient for Problem 1(d)"""
    return (1 / N) * (2 + 2 * np.cos(2 * np.pi * k / N))


# Plot c_k as function of ω = 2πk/N for N = 10 ∈ [-π, π]
N = 40
k_range = np.arange(N)  # k = 0, 1, 2, ..., N-1

# Calculate coefficients
coefficients = [c_k(k, N) for k in k_range]

# Calculate frequencies and map to [-π, π]
w = 2 * np.pi * k_range / N
w_centered = np.copy(w)
w_centered[w_centered > np.pi] -= 2 * np.pi

# Sort for proper plotting
sort_idx = np.argsort(w_centered)
w_sorted = w_centered[sort_idx]
c_sorted = np.array(coefficients)[sort_idx]

# Plot
plt.figure(figsize=(10, 6))
plt.stem(w_sorted, c_sorted, basefmt="b-")
plt.title(f"DTFS Coefficients as function of ω ∈ [-π, π] for N={N}")
plt.xlabel("ω (radians)")
plt.ylabel("c_k")
plt.grid(True)
plt.xlim([-np.pi, np.pi])
plt.show()

plt.show()
