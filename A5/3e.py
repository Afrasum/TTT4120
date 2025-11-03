import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io.wavfile import read, write

# Parameters for multiple echo filter
R = 16  # spacing between echoes (in samples)
alpha = 0.8  # decay factor for each echo
N = 6  # number of echoes
fs = 22050  # sample rate

# Create filter coefficients
# Numerator: 1 - α^N * z^(-NR)
b = np.zeros(N * R + 1)
b[0] = 1
b[N * R] = -(alpha**N)

# Denominator: 1 - α * z^(-R)
a = np.zeros(R + 1)
a[0] = 1
a[R] = -alpha

print(f"Filter parameters:")
print(f"  R (echo spacing) = {R} samples = {R/fs*1000:.2f} ms")
print(f"  α (decay factor) = {alpha}")
print(f"  N (number of echoes) = {N}")
print(f"  Total echo duration = {N*R/fs*1000:.2f} ms")
print(f"\nNumerator coefficients (b): length = {len(b)}")
print(f"  b[0] = {b[0]}")
print(f"  b[{N*R}] = {b[N*R]}")
print(f"\nDenominator coefficients (a): length = {len(a)}")
print(f"  a[0] = {a[0]}")
print(f"  a[{R}] = {a[R]}")

# Plot impulse response
plt.figure(figsize=(15, 10))

# Impulse response
plt.subplot(3, 2, 1)
impulse_length = N * R + 50
impulse = np.zeros(impulse_length)
impulse[0] = 1
h = signal.lfilter(b, a, impulse)

plt.stem(h)
plt.title(f"Impulse Response (R={R}, α={alpha}, N={N})")
plt.xlabel("Sample number (n)")
plt.ylabel("Amplitude")
plt.grid(True)

# Add annotations for each echo
for i in range(N + 1):
    if i * R < len(h):
        plt.axvline(x=i * R, color="r", linestyle="--", alpha=0.3)
        plt.text(
            i * R,
            plt.ylim()[1] * 0.9,
            f"Echo {i}",
            rotation=90,
            verticalalignment="bottom",
            fontsize=8,
        )

# Frequency response (magnitude)
plt.subplot(3, 2, 2)
w, H = signal.freqz(b, a, worN=4096)
f = w * fs / (2 * np.pi)

plt.plot(f, np.abs(H))
plt.title(f"Magnitude Response (R={R}, α={alpha}, N={N})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.xlim([0, fs / 2])

# Frequency response in dB
plt.subplot(3, 2, 3)
plt.plot(f, 20 * np.log10(np.abs(H) + 1e-10))
plt.title(f"Magnitude Response in dB")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.xlim([0, fs / 2])
plt.ylim([-40, 20])

# Phase response
plt.subplot(3, 2, 4)
plt.plot(f, np.angle(H))
plt.title(f"Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.grid(True)
plt.xlim([0, fs / 2])

# Zoomed impulse response showing echo decay
plt.subplot(3, 2, 5)
plt.stem(
    h[: min(len(h), N * R + 10)], linefmt="b-", markerfmt="bo"
)
plt.title("Zoomed Impulse Response (showing echo decay)")
plt.xlabel("Sample number (n)")
plt.ylabel("Amplitude")
plt.grid(True)

# Theoretical echo amplitudes
echo_positions = [i * R for i in range(N + 1)]
echo_amplitudes = [alpha**i for i in range(N + 1)]
plt.plot(
    echo_positions,
    echo_amplitudes,
    "ro-",
    linewidth=2,
    markersize=8,
    label="Theoretical decay",
)
plt.legend()

# Information panel
plt.subplot(3, 2, 6)
plt.axis("off")

# Calculate actual echo amplitudes from impulse response
actual_echoes = []
for i in range(min(N + 1, len(h) // R + 1)):
    if i * R < len(h):
        actual_echoes.append(h[i * R])

info_text = f"""
MULTIPLE ECHO FILTER CHARACTERISTICS

Parameters:
  • R = {R} samples ({R/fs*1000:.2f} ms)
  • α = {alpha}
  • N = {N} echoes
  • Sample rate = {fs} Hz

Echo timings:
"""
for i in range(min(N + 1, len(actual_echoes))):
    info_text += (
        f"  Echo {i}: {i*R/fs*1000:.1f} ms, amplitude ≈ {actual_echoes[i]:.3f}\n"
    )

info_text += f"""
Frequency characteristics:
  • Comb filter with {N} teeth
  • Notch spacing: {fs/R:.1f} Hz
  • Creates rich, spacious sound
"""

plt.text(
    0.05,
    0.95,
    info_text,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    transform=plt.gca().transAxes,
)

plt.tight_layout()
plt.show()

print("\nFilter created successfully!")
