import matplotlib.pyplot as plt
import numpy as np

N = 100

# Generate the three types of white noise
binary_noise = np.random.choice([-1, 1], size=N)
gaussian_noise = np.random.randn(N)
uniform_noise = np.random.uniform(-np.sqrt(3), np.sqrt(3), N)

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

axes[0].stem(binary_noise)
axes[0].set_title("White Binary Noise")
axes[0].set_ylabel("Amplitude")
axes[0].grid(True)

axes[1].stem(gaussian_noise)
axes[1].set_title("White Gaussian Noise")
axes[1].set_ylabel("Amplitude")
axes[1].grid(True)

axes[2].stem(uniform_noise)
axes[2].set_title("White Uniform Noise")
axes[2].set_ylabel("Amplitude")
axes[2].set_xlabel("Sample Index n")
axes[2].grid(True)

plt.tight_layout()
plt.show()
