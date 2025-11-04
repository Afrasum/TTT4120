import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

# Parameters
Nx = 28
alpha = 0.9

# Create frequency axis with 1000 points from 0 to 1
freq_axis = np.linspace(0, 1, 1000)

# Compute DTFT X(f) using the formula:
# X(f) = (1 - (0.9*e^(-j2πf))^28) / (1 - 0.9*e^(-j2πf))
X_f = (1 - (alpha * np.exp(-1j * 2 * np.pi * freq_axis)) ** Nx) / (
    1 - alpha * np.exp(-1j * 2 * np.pi * freq_axis)
)

# Plot magnitude of X(f)
plt.figure(figsize=(10, 6))
plt.plot(freq_axis, np.abs(X_f))
plt.title("Figure 1: Magnitude of $X(f)$", fontsize=15)
plt.ylabel("$|X(f)|$")
plt.xlabel("$f$")
plt.grid(True)
plt.show()
