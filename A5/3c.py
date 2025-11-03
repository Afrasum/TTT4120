import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Parameters
R = 10  # delay in samples
alpha = 0.9  # echo strength
fs = 22050  # sample frequency

# Create filter coefficients
# For H(z) = 1 + α*z^(-R), we need:
# - numerator (b): [1, 0, 0, ..., 0, α] where α is at position R
# - denominator (a):

b = np.zeros(R + 1)  # numerator coefficients
a = np.ones(1)  # denominator coefficient

b[0] = 1  # coefficient for z^0
b[R] = alpha  # coefficient for z^(-R)

print(f"Filter coefficients b: {b}")
print(f"Filter coefficients a: {a}")

# Plot impulse response
plt.figure(figsize=(12, 5))

# Impulse response
plt.subplot(1, 2, 1)
impulse_length = 50
impulse = np.zeros(impulse_length)
impulse[0] = 1
h = signal.lfilter(b, a, impulse)

plt.stem(h)
plt.title(f"Impulse Response (R={R}, α={alpha})")
plt.xlabel("Sample number (n)")
plt.ylabel("Amplitude")
plt.grid(True)

# Frequency response
plt.subplot(1, 2, 2)
w, H = signal.freqz(b, a, worN=2048)
f = w * fs / (2 * np.pi)  # Convert to Hz

plt.plot(f, np.abs(H))
plt.title(f"Frequency Response Magnitude (R={R}, α={alpha})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()
