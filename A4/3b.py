import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Parameters
alpha = 0.9
K = 1

# A(z) = (alpha - z^-1) / (1 - alpha*z^-1)
num_A = [alpha, -1]
den_A = [1, -alpha]

# Branch 1: H1(z) = 1/2 * (1 + A(z))
# After algebra: H1(z) = (1+alpha)/2 * (1 - z^-1) / (1 - alpha*z^-1)
num_H1 = [0.5 * (1 + alpha), -0.5 * (1 + alpha)]
den_H1 = [1, -alpha]

# Branch 2: H2(z) = K/2 * (1 - A(z))
# After algebra: H2(z) = K*(1-alpha)/2 * (1 + z^-1) / (1 - alpha*z^-1)
num_H2 = [K / 2 * (1 - alpha), K / 2 * (1 - alpha)]
den_H2 = [1, -alpha]

# Calculate frequency responses
w, H1 = signal.freqz(num_H1, den_H1, worN=1024)
w, H2 = signal.freqz(num_H2, den_H2, worN=1024)

# Convert to normalized frequency [0, 0.5]
f = w / (2 * np.pi)

# Simple plots
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(f, np.abs(H1), "r-", linewidth=2, label="Branch 1")
plt.title("Branch 1: H1(z) = 1/2 * (1 + A(z))")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(f, np.abs(H2), "g-", linewidth=2, label="Branch 2")
plt.title("Branch 2: H2(z) = K/2 * (1 - A(z))")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude")
plt.grid(True)

plt.show()

# Print key values
print(f"Branch 1 at DC: {np.abs(H1[0]):.3f}")
print(f"Branch 1 at Nyquist: {np.abs(H1[-1]):.3f}")
print(f"Branch 2 at DC: {np.abs(H2[0]):.3f}")
print(f"Branch 2 at Nyquist: {np.abs(H2[-1]):.3f}")

# Answer the question
print("\nFilter types:")
if np.abs(H1[-1]) > np.abs(H1[0]):
    print("Branch 1: HIGHPASS (higher at high frequencies)")
else:
    print("Branch 1: LOWPASS (higher at low frequencies)")

if np.abs(H2[0]) > np.abs(H2[-1]):
    print("Branch 2: LOWPASS (higher at low frequencies)")
else:
    print("Branch 2: HIGHPASS (higher at high frequencies)")
