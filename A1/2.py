import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

# --- Parameters (edit these if needed) ---
Fs = 6000  # sampling frequency [Hz]
T = 4  # duration [s]
F1 = 1000  # physical frequency [Hz]
A = 1.0  # amplitude

# --- Derived ---
N = Fs * T
n = np.arange(N)
t = n / Fs
f1 = F1 / Fs  # normalized frequency f = F1/Fs
x_n = A * np.cos(2 * np.pi * f1 * n)  # x[n] = A cos(2π f n)

# --- Plot 1: zoomed stem (first 200 samples) ---
plt.figure(figsize=(10, 3.6))
markerline, stemlines, baseline = plt.stem(t[:200], x_n[:200])
plt.setp(stemlines, "color", "b")
plt.setp(markerline, "markerfacecolor", "b")
plt.xlabel("time (seconds)")
plt.ylabel("Amplitude")
plt.title(
    f"x[n] = A cos(2π f n), F1={F1} Hz, Fs={Fs} Hz, f={f1:.3f}  (first 200 samples)"
)
plt.grid(True)

# top axis in samples
ax1 = plt.gca()
ax2 = ax1.secondary_xaxis("top", functions=(lambda tt: tt * Fs, lambda nn: nn / Fs))
ax2.set_xlabel("n (sample index)")

# --- Plot 2: overview of the full 4 seconds (use a line so it’s readable) ---
plt.figure(figsize=(10, 3.6))
plt.plot(t, x_n, linewidth=0.8)
plt.xlabel("time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"x[n] over full duration (0–{T} s)")
plt.grid(True)
plt.xlim(0, T)
plt.xticks([0, 1, 2, 3, 4])

# plt.show()


# ------------ c) Playback -----------


# # --- Parameters ---
# Fs = 1000  # try 1000, 3000, 12000
# T = 2  # play 2 seconds
# f1 = 0.3  # normalized frequency
# A = 0.8
#
# # --- Derived ---
# N = int(Fs * T)
# n = np.arange(N)
# t = n / Fs
# x_n = A * np.cos(2 * np.pi * f1 * n)
#
# sd.wait()  # wait if needed
#
# # --- Playback ---
# print(f"Playing signal: Fs={Fs}, physical frequency={f1*Fs} Hz")
# sd.play(x_n.astype(np.float32), Fs)
# sd.wait()
# sd.play(x_n.astype(np.float32), Fs)
# sd.wait()
#
# Fs = 3000  # try 1000, 3000, 12000
# # --- Derived ---
# N = int(Fs * T)
# n = np.arange(N)
# t = n / Fs
# x_n = A * np.cos(2 * np.pi * f1 * n)
#
# # --- Playback ---
# print(f"Playing signal: Fs={Fs}, physical frequency={f1*Fs} Hz")
# sd.play(x_n.astype(np.float32), Fs)
# sd.wait()
#
# Fs = 12000  # try 1000, 3000, 12000
#
# # --- Derived ---
# N = int(Fs * T)
# n = np.arange(N)
# t = n / Fs
# x_n = A * np.cos(2 * np.pi * f1 * n)
#
# # --- Playback ---
# print(f"Playing signal: Fs={Fs}, physical frequency={f1*Fs} Hz")
# sd.play(x_n.astype(np.float32), Fs)
# sd.wait()
# print("Done")
#


# Fs = 8000, F1 = 1000,3000, 12000

# --- Derived ---
Fs = 8000
F1 = 3000
f1 = F1 / Fs  # normalized frequency f = F1/Fs
N = int(Fs * T)
n = np.arange(N)
t = n / Fs
x_n = A * np.cos(2 * np.pi * f1 * n)

# --- Playback ---
print(
    f"Playing signal: Fs={Fs}, physical frequency={f1*Fs} Hz, normalized frequency={f1}"
)
sd.play(x_n.astype(np.float32), Fs)
sd.wait()

F1 = 12000
f1 = F1 / Fs  # normalized frequency f = F1/Fs
N = int(Fs * T)
n = np.arange(N)
t = n / Fs
x_n = A * np.cos(2 * np.pi * f1 * n)
# --- Playback ---
print(
    f"Playing signal: Fs={Fs}, physical frequency={f1*Fs} Hz, normalized frequency={f1}"
)
sd.play(x_n.astype(np.float32), Fs)
sd.wait()

Fs = F1 = 1000
f1 = F1 / Fs  # normalized frequency f = F1/Fs
N = int(Fs * T)
n = np.arange(N)
t = n / Fs
x_n = A * np.cos(2 * np.pi * f1 * n)


# --- Playback ---
print(
    f"Playing signal: Fs={Fs}, physical frequency={f1*Fs} Hz, normalized frequency={f1}"
)
sd.play(x_n.astype(np.float32), Fs)
sd.wait()
print("Done")
