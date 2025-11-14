import numpy as np
from matplotlib import pyplot as plt

# Signal parameters
Nx = 28
Nh = 9
alpha = 0.9

# Generate input signal x[n]
xn_x_axis = np.linspace(0, Nx - 1, Nx)
xn = alpha**xn_x_axis

# Generate filter h[n] (all ones)
hn = np.ones(Nh)

# Compute convolution: y[n] = x[n] * h[n]
yn = np.convolve(xn, hn)

# Create time axis for y[n]
yn_x_axis = np.arange(len(yn))

# Plot
plt.figure(figsize=(12, 6))
plt.stem(yn_x_axis, yn, basefmt=" ")
plt.title("Figure 6: Filter output $y(n)$", fontsize=15)
plt.xlabel("$n$")
plt.ylabel("$y(n)$")
plt.grid(True, alpha=0.3)
plt.xlim([-1, 37])
plt.show()

# Print info
print(f"Input signal length (Nx): {Nx}")
print(f"Filter length (Nh): {Nh}")
print(f"Output signal length (Ny): {len(yn)}")
print(f"Expected length (Nx + Nh - 1): {Nx + Nh - 1}")
print(f"\nFirst 10 values of y[n]:")
for i in range(10):
    print(f"y[{i}] = {yn[i]:.6f}")
