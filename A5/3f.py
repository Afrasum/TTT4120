import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

fs = 22050  # Sample rate

# Compare single echo vs multiple echo filters
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Parameters
R = 16
alpha = 0.8
N = 6

# SINGLE ECHO FILTER
b_single = np.zeros(R + 1)
a_single = np.ones(1)
b_single[0] = 1
b_single[R] = alpha

# MULTIPLE ECHO FILTER
b_multi = np.zeros(N * R + 1)
b_multi[0] = 1
b_multi[N * R] = -(alpha**N)

a_multi = np.zeros(R + 1)
a_multi[0] = 1
a_multi[R] = -alpha

# Create impulse responses
impulse_length = max(R + 50, N * R + 50)
impulse = np.zeros(impulse_length)
impulse[0] = 1

h_single = signal.lfilter(b_single, a_single, impulse)
h_multi = signal.lfilter(b_multi, a_multi, impulse)

# Plot 1: Single echo impulse response
axes[0, 0].stem(h_single)
axes[0, 0].set_title("Single Echo - Impulse Response")
axes[0, 0].set_xlabel("Sample number (n)")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True)
axes[0, 0].axvline(x=R, color="r", linestyle="--", alpha=0.5, label=f"Echo at R={R}")
axes[0, 0].legend()

# Plot 2: Multiple echo impulse response
axes[1, 0].stem(h_multi)
axes[1, 0].set_title(f"Multiple Echo - Impulse Response (N={N})")
axes[1, 0].set_xlabel("Sample number (n)")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].grid(True)
for i in range(N + 1):
    axes[1, 0].axvline(x=i * R, color="r", linestyle="--", alpha=0.3)

# Plot 3: Single echo frequency response
w, H_single = signal.freqz(b_single, a_single, worN=4096)
f = w * fs / (2 * np.pi)
axes[0, 1].plot(f, np.abs(H_single))
axes[0, 1].set_title("Single Echo - Magnitude Response")
axes[0, 1].set_xlabel("Frequency (Hz)")
axes[0, 1].set_ylabel("Magnitude")
axes[0, 1].grid(True)
axes[0, 1].set_xlim([0, 5000])

# Plot 4: Multiple echo frequency response
w, H_multi = signal.freqz(b_multi, a_multi, worN=4096)
axes[1, 1].plot(f, np.abs(H_multi))
axes[1, 1].set_title("Multiple Echo - Magnitude Response")
axes[1, 1].set_xlabel("Frequency (Hz)")
axes[1, 1].set_ylabel("Magnitude")
axes[1, 1].grid(True)
axes[1, 1].set_xlim([0, 5000])

# Plot 5: Comparison text for single echo
axes[0, 2].axis("off")
single_text = f"""
SINGLE ECHO FILTER

Structure:
  y[n] = x[n] + α·x[n-R]

Characteristics:
  • Only 2 impulses
  • Original + 1 echo
  • Simple comb filter
  • Broader frequency peaks
  
Sound quality:
  • Clear, distinct echo
  • Less natural
  • "Ping-pong" effect
  • Good for special effects
  
Parameters:
  R = {R} ({R/fs*1000:.2f} ms)
  α = {alpha}
"""
axes[0, 2].text(
    0.1,
    0.9,
    single_text,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    transform=axes[0, 2].transAxes,
)

# Plot 6: Comparison text for multiple echo
axes[1, 2].axis("off")
multi_text = f"""
MULTIPLE ECHO FILTER

Structure:
  Recursive feedback creates
  multiple decaying echoes

Characteristics:
  • {N+1} impulses total
  • Original + {N} echoes
  • Complex comb filter
  • Sharper frequency peaks
  
Sound quality:
  • Gradual reverb decay
  • Natural room sound
  • Realistic space simulation
  • Professional audio effect
  
Parameters:
  R = {R} ({R/fs*1000:.2f} ms)
  α = {alpha}
  N = {N} echoes
  Duration = {N*R/fs*1000:.1f} ms
"""
axes[1, 2].text(
    0.1,
    0.9,
    multi_text,
    fontsize=10,
    verticalalignment="top",
    family="monospace",
    transform=axes[1, 2].transAxes,
)

plt.suptitle(
    "Comparison: Single Echo vs Multiple Echo Filter", fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.show()

# Key differences summary
print("\n" + "=" * 60)
print("KEY DIFFERENCES: Single vs Multiple Echo")
print("=" * 60)
print("\n1. IMPULSE RESPONSE:")
print("   Single echo:   2 spikes (original + 1 echo)")
print(f"   Multiple echo: {N+1} spikes (original + {N} echoes)")

print("\n2. ECHO PATTERN:")
print(f"   Single echo:   One echo at {R/fs*1000:.2f} ms")
print(
    f"   Multiple echo: Echoes at {R/fs*1000:.2f}, {2*R/fs*1000:.2f}, {3*R/fs*1000:.2f}... ms"
)

print("\n3. DECAY:")
print("   Single echo:   Abrupt (no decay)")
print(f"   Multiple echo: Gradual (each echo is {alpha} times the previous)")

print("\n4. FREQUENCY RESPONSE:")
print("   Single echo:   Simple comb pattern")
print("   Multiple echo: More complex, sharper peaks/notches")

print("\n5. SOUND CHARACTER:")
print("   Single echo:   Artificial 'slap-back' delay")
print("   Multiple echo: Natural room/hall reverb")
print("=" * 60)
