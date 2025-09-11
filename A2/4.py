import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd


def xa(t):
    return np.cos(2000 * np.pi * t)


def x(n, Fs):
    return xa(n / Fs)


F = 1000
Fs1, Fs2 = 4000, 1500
duration = 1  # second

# Generate the signal arrays
n1 = np.arange(0, Fs1 * duration)  # 4000 samples
n2 = np.arange(0, Fs2 * duration)  # 1500 samples

x1 = x(n1, Fs1)
x2 = x(n2, Fs2)

# Plot spectra
f1 = F / Fs1  # 0.25
f2 = 1 - F / Fs2  # 0.333 (aliased)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem([f1, -f1], [1, 1], basefmt=" ")
plt.title("X₁(f), Fs=4000Hz")
plt.xlabel("f")
plt.xlim([-0.5, 0.5])
plt.xticks(np.arange(-0.5, 0.55, 0.05))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem([f2, -f2], [1, 1], basefmt=" ")
plt.title("X₂(f), Fs=1500Hz")
plt.xlabel("f")
plt.xlim([-0.5, 0.5])
plt.xticks(np.arange(-0.5, 0.55, 0.05))
plt.grid(True)

plt.show()

# Play the sounds
print("Playing x1 (Fs=4000Hz)...")
sd.play(x1, Fs1)
sd.wait()

print("Playing x2 (Fs=1500Hz)...")
sd.play(x2, Fs2)
sd.wait()

print(f"x1 frequency: {F} Hz")
print(f"x2 apparent frequency: {Fs2 - F} = {Fs2 - F} Hz")
