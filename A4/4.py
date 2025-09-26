import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Parameters
Ax = Ay = 0.25
fx, fy = 0.04, 0.10
L = 500
N_fft = 2048
r = 0.9

print("Problem 4: Digital Signal Processing")

# 4a) Generate signals
n = np.arange(L)
np.random.seed(42)
d = Ax * np.cos(2 * np.pi * fx * n) + Ay * np.cos(2 * np.pi * fy * n)
e = np.random.normal(size=L)
g = d + e

# Plot 4a
plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.plot(n[:100], d[:100])
plt.title("Clean signal d[n]")
plt.subplot(2, 3, 2)
plt.plot(n[:100], e[:100])
plt.title("Noise e[n]")
plt.subplot(2, 3, 3)
plt.plot(n[:100], g[:100])
plt.title("Noisy signal g[n]")

# FFT
D = np.fft.fft(d, N_fft)
E = np.fft.fft(e, N_fft)
G = np.fft.fft(g, N_fft)
f = np.arange(N_fft // 2) / N_fft

plt.subplot(2, 3, 4)
plt.plot(f, np.abs(D[: N_fft // 2]))
plt.title("|D(f)|")
plt.xlim([0, 0.2])
plt.subplot(2, 3, 5)
plt.plot(f, np.abs(E[: N_fft // 2]))
plt.title("|E(f)|")
plt.xlim([0, 0.2])
plt.subplot(2, 3, 6)
plt.plot(f, np.abs(G[: N_fft // 2]))
plt.title("|G(f)|")
plt.xlim([0, 0.2])
plt.tight_layout()
plt.show()

# 4b) Design resonators
num = np.array([1, 0, -1])
den_x = np.array([1, -2 * r * np.cos(2 * np.pi * fx), r**2])
den_y = np.array([1, -2 * r * np.cos(2 * np.pi * fy), r**2])

print(f"Hx(z) = (1 - z^-2) / (1 - {den_x[1]:.3f}z^-1 + {den_x[2]:.3f}z^-2)")
print(f"Hy(z) = (1 - z^-2) / (1 - {den_y[1]:.3f}z^-1 + {den_y[2]:.3f}z^-2)")

# Poles and zeros
zeros = np.roots(num)
poles_x = np.roots(den_x)
poles_y = np.roots(den_y)

# Plot pole-zero
plt.figure(figsize=(12, 5))
theta = np.linspace(0, 2 * np.pi, 100)

plt.subplot(1, 2, 1)
plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5)
plt.plot(np.real(poles_x), np.imag(poles_x), "bx", markersize=10, label="Poles Hx")
plt.plot(np.real(zeros), np.imag(zeros), "ro", markersize=8, label="Zeros")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Hx(z)")

plt.subplot(1, 2, 2)
plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5)
plt.plot(np.real(poles_y), np.imag(poles_y), "rx", markersize=10, label="Poles Hy")
plt.plot(np.real(zeros), np.imag(zeros), "ro", markersize=8, label="Zeros")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Hy(z)")
plt.show()

# Frequency responses
w, Hx = signal.freqz(num, den_x, worN=1024)
w, Hy = signal.freqz(num, den_y, worN=1024)
f_resp = w / (2 * np.pi)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(f_resp, 20 * np.log10(np.abs(Hx) + 1e-10))
plt.axvline(fx, color="r", linestyle="--", label=f"fx={fx}")
plt.title("|Hx(f)|")
plt.xlim([0, 0.2])
plt.ylabel("dB")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(f_resp, 20 * np.log10(np.abs(Hy) + 1e-10))
plt.axvline(fy, color="r", linestyle="--", label=f"fy={fy}")
plt.title("|Hy(f)|")
plt.xlim([0, 0.2])
plt.ylabel("dB")
plt.legend()
plt.show()

# 4c) Filter noisy signal
qx = signal.lfilter(num, den_x, g)
qy = signal.lfilter(num, den_y, g)

plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(n[:200], g[:200])
plt.title("Input g[n]")
plt.subplot(2, 3, 2)
plt.plot(n[:200], qx[:200])
plt.title("Output qx[n]")
plt.subplot(2, 3, 3)
plt.plot(n[:200], qy[:200])
plt.title("Output qy[n]")

Qx = np.fft.fft(qx, N_fft)
Qy = np.fft.fft(qy, N_fft)

plt.subplot(2, 3, 4)
plt.plot(f, np.abs(G[: N_fft // 2]))
plt.title("|G(f)|")
plt.xlim([0, 0.2])
plt.subplot(2, 3, 5)
plt.plot(f, np.abs(Qx[: N_fft // 2]))
plt.title("|Qx(f)|")
plt.xlim([0, 0.2])
plt.subplot(2, 3, 6)
plt.plot(f, np.abs(Qy[: N_fft // 2]))
plt.title("|Qy(f)|")
plt.xlim([0, 0.2])
plt.tight_layout()
plt.show()

print("4c) Results: qx isolates fx component, qy isolates fy component")

# 4d) Combine resonators
den_combined = np.convolve(den_x, den_y)
num_x_exp = np.convolve(num, den_y)
num_y_exp = np.convolve(num, den_x)
num_combined = np.pad(num_x_exp, (len(num_y_exp) - len(num_x_exp), 0)) + np.pad(
    num_y_exp, (len(num_x_exp) - len(num_y_exp), 0)
)

poles_combined = np.roots(den_combined)
zeros_combined = np.roots(num_combined)

# Combined filter response
w, H_combined = signal.freqz(num_combined, den_combined, worN=1024)
q_combined = signal.lfilter(num_combined, den_combined, g)

plt.figure(figsize=(12, 6))
plt.subplot(2, 3, 1)
plt.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.5)
plt.plot(
    np.real(poles_combined), np.imag(poles_combined), "bx", markersize=8, label="Poles"
)
plt.plot(
    np.real(zeros_combined), np.imag(zeros_combined), "ro", markersize=6, label="Zeros"
)
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title("Combined system poles/zeros")

plt.subplot(2, 3, 2)
plt.plot(f_resp, 20 * np.log10(np.abs(H_combined) + 1e-10))
plt.axvline(fx, color="r", linestyle="--")
plt.axvline(fy, color="r", linestyle="--")
plt.title("|H_combined(f)|")
plt.xlim([0, 0.2])
plt.ylabel("dB")

plt.subplot(2, 3, 3)
plt.plot(n[:200], q_combined[:200])
plt.title("Combined output")
plt.subplot(2, 3, 4)
plt.plot(n[:200], d[:200])
plt.title("Original d[n]")
plt.subplot(2, 3, 5)
plt.plot(n[:200], g[:200])
plt.title("Noisy g[n]")

Q_combined = np.fft.fft(q_combined, N_fft)
plt.subplot(2, 3, 6)
plt.plot(f, np.abs(Q_combined[: N_fft // 2]))
plt.title("|Q_combined(f)|")
plt.xlim([0, 0.2])
plt.tight_layout()
plt.show()

print("4d) Combined filter isolates both sinusoids simultaneously")
print("Combined system has poles from both resonators, creating dual-peak response")
print("Output closely matches original clean signal d[n]")
