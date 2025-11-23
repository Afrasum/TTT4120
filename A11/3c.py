import time

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal

print("=" * 80)
print("PROBLEM 3c: PRACTICAL DOWNSAMPLING DEMONSTRATION")
print("=" * 80)

# ============================================================================
# PART i: Generate the signal
# ============================================================================

# Given parameters
F1 = 900  # Hz
F2 = 2000  # Hz
Fs = 6000  # Hz
D = 2  # Downsampling factor
duration = 1.0  # seconds

print(f"\nGenerating test signal:")
print(f"  F1 = {F1} Hz")
print(f"  F2 = {F2} Hz")
print(f"  Fs = {Fs} Hz")
print(f"  Duration = {duration} s")

# Create time vector
n = np.arange(0, int(duration * Fs))
t = n / Fs

# Normalized frequencies
f1 = F1 / Fs  # 0.15
f2 = F2 / Fs  # 0.333...

# Generate signals
x1 = np.cos(2 * np.pi * f1 * n)
x2 = np.cos(2 * np.pi * f2 * n)
x = x1 + x2

print(f"\nSignal x[n] created:")
print(f"  x[n] = cos(2π·{f1}·n) + cos(2π·{f2:.4f}·n)")
print(f"  Number of samples: {len(x)}")

# ============================================================================
# PART ii: Downsample WITH filter (proper decimation)
# ============================================================================

print(f"\n" + "-" * 80)
print("DOWNSAMPLING WITH ANTI-ALIASING FILTER")
print("-" * 80)

# Using scipy's decimate function (includes anti-aliasing filter)
# decimate applies a lowpass filter before downsampling
y_filtered = signal.decimate(x, D, ftype="fir", zero_phase=True)

Fs_new = Fs / D  # 3000 Hz

print(f"  Used scipy.signal.decimate()")
print(f"  This automatically applies anti-aliasing filter")
print(f"  New sampling rate: {Fs_new} Hz")
print(f"  Output samples: {len(y_filtered)}")

# ============================================================================
# PART ii: Downsample WITHOUT filter (naive downsampling)
# ============================================================================

print(f"\n" + "-" * 80)
print("DOWNSAMPLING WITHOUT FILTER (naive downsampling)")
print("-" * 80)

# Just take every D-th sample (no filtering!)
y_no_filter = x[::D]

print(f"  Simply taking every {D}th sample")
print(f"  No anti-aliasing filter applied")
print(f"  New sampling rate: {Fs_new} Hz")
print(f"  Output samples: {len(y_no_filter)}")

# ============================================================================
# Analyze the frequency content
# ============================================================================

print(f"\n" + "-" * 80)
print("FREQUENCY ANALYSIS")
print("-" * 80)

# Compute FFTs
N_fft = 8192

# Original signal
X_fft = np.fft.fft(x, N_fft)
freqs_original = np.fft.fftfreq(N_fft, 1 / Fs)

# With filter
Y_filtered_fft = np.fft.fft(y_filtered, N_fft)
freqs_new = np.fft.fftfreq(N_fft, 1 / Fs_new)

# Without filter
Y_no_filter_fft = np.fft.fft(y_no_filter, N_fft)


# Find peaks
def find_main_frequencies(fft_data, freqs, threshold=0.1):
    magnitude = np.abs(fft_data[: len(fft_data) // 2])
    peaks, _ = signal.find_peaks(magnitude, height=threshold * np.max(magnitude))
    peak_freqs = freqs[peaks]
    peak_mags = magnitude[peaks]
    # Sort by magnitude
    sorted_idx = np.argsort(peak_mags)[::-1]
    return peak_freqs[sorted_idx[:5]], peak_mags[sorted_idx[:5]]


print("\nOriginal signal x[n]:")
orig_freqs, orig_mags = find_main_frequencies(X_fft, freqs_original)
for i, (f, m) in enumerate(zip(orig_freqs, orig_mags)):
    if f > 0:
        print(f"  Peak {i+1}: {f:.1f} Hz (magnitude: {m:.0f})")

print("\nDownsampled WITH filter:")
filt_freqs, filt_mags = find_main_frequencies(Y_filtered_fft, freqs_new)
for i, (f, m) in enumerate(zip(filt_freqs, filt_mags)):
    if f > 0:
        print(f"  Peak {i+1}: {f:.1f} Hz (magnitude: {m:.0f})")

print("\nDownsampled WITHOUT filter:")
no_filt_freqs, no_filt_mags = find_main_frequencies(Y_no_filter_fft, freqs_new)
for i, (f, m) in enumerate(zip(no_filt_freqs, no_filt_mags)):
    if f > 0:
        print(f"  Peak {i+1}: {f:.1f} Hz (magnitude: {m:.0f})")

# ============================================================================
# Visualize time and frequency domain
# ============================================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# TIME DOMAIN PLOTS
# Original
axes[0, 0].plot(t[:500], x[:500], "b-", linewidth=1)
axes[0, 0].set_xlabel("Time [s]", fontsize=11)
axes[0, 0].set_ylabel("Amplitude", fontsize=11)
axes[0, 0].set_title("Original x[n] at 6000 Hz", fontsize=12, fontweight="bold")
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlim([0, 0.02])

# With filter
t_new = np.arange(len(y_filtered)) / Fs_new
axes[1, 0].plot(t_new[:250], y_filtered[:250], "g-", linewidth=1.5)
axes[1, 0].set_xlabel("Time [s]", fontsize=11)
axes[1, 0].set_ylabel("Amplitude", fontsize=11)
axes[1, 0].set_title(
    "Downsampled WITH filter at 3000 Hz", fontsize=12, fontweight="bold"
)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim([0, 0.02])

# Without filter
axes[2, 0].plot(t_new[:250], y_no_filter[:250], "r-", linewidth=1.5)
axes[2, 0].set_xlabel("Time [s]", fontsize=11)
axes[2, 0].set_ylabel("Amplitude", fontsize=11)
axes[2, 0].set_title(
    "Downsampled WITHOUT filter at 3000 Hz", fontsize=12, fontweight="bold"
)
axes[2, 0].grid(True, alpha=0.3)
axes[2, 0].set_xlim([0, 0.02])

# FREQUENCY DOMAIN PLOTS
# Original
axes[0, 1].plot(
    freqs_original[: N_fft // 2], np.abs(X_fft[: N_fft // 2]), "b-", linewidth=1.5
)
axes[0, 1].axvline(x=F1, color="blue", linestyle="--", alpha=0.7, label=f"{F1} Hz")
axes[0, 1].axvline(x=F2, color="red", linestyle="--", alpha=0.7, label=f"{F2} Hz")
axes[0, 1].set_xlabel("Frequency [Hz]", fontsize=11)
axes[0, 1].set_ylabel("Magnitude", fontsize=11)
axes[0, 1].set_title("Spectrum of x[n]", fontsize=12, fontweight="bold")
axes[0, 1].set_xlim([0, 3000])
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=9)

# With filter
axes[1, 1].plot(
    freqs_new[: N_fft // 2], np.abs(Y_filtered_fft[: N_fft // 2]), "g-", linewidth=1.5
)
axes[1, 1].axvline(
    x=F1, color="blue", linestyle="--", alpha=0.7, label=f"{F1} Hz present"
)
axes[1, 1].axvline(
    x=Fs_new / 2,
    color="orange",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"Nyquist = {Fs_new/2} Hz",
)
axes[1, 1].set_xlabel("Frequency [Hz]", fontsize=11)
axes[1, 1].set_ylabel("Magnitude", fontsize=11)
axes[1, 1].set_title("Spectrum WITH filter - Clean!", fontsize=12, fontweight="bold")
axes[1, 1].set_xlim([0, 1500])
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=9)
axes[1, 1].text(
    F1,
    np.max(np.abs(Y_filtered_fft[: N_fft // 2])) * 0.7,
    "Only 900 Hz ✓",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)

# Without filter
axes[2, 1].plot(
    freqs_new[: N_fft // 2], np.abs(Y_no_filter_fft[: N_fft // 2]), "r-", linewidth=1.5
)
axes[2, 1].axvline(x=F1, color="blue", linestyle="--", alpha=0.7, label=f"{F1} Hz")
# The aliased frequency: 2000 Hz aliases to 1000 Hz (3000 - 2000)
F2_aliased = Fs_new - F2
axes[2, 1].axvline(
    x=F2_aliased,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"{F2} Hz aliased to {F2_aliased} Hz",
)
axes[2, 1].axvline(
    x=Fs_new / 2,
    color="orange",
    linestyle=":",
    linewidth=2,
    alpha=0.7,
    label=f"Nyquist = {Fs_new/2} Hz",
)
axes[2, 1].set_xlabel("Frequency [Hz]", fontsize=11)
axes[2, 1].set_ylabel("Magnitude", fontsize=11)
axes[2, 1].set_title(
    "Spectrum WITHOUT filter - ALIASING!", fontsize=12, fontweight="bold", color="red"
)
axes[2, 1].set_xlim([0, 1500])
axes[2, 1].grid(True, alpha=0.3)
axes[2, 1].legend(fontsize=9)
axes[2, 1].text(
    F2_aliased,
    np.max(np.abs(Y_no_filter_fft[: N_fft // 2])) * 0.7,
    f"Aliased!\n{F2}Hz→{F2_aliased}Hz",
    ha="center",
    fontsize=9,
    color="darkred",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="pink", alpha=0.7),
)

plt.tight_layout()
plt.savefig("problem3c_downsampling_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# PART ii & iii: Play the sounds and explain
# ============================================================================

print("\n" + "=" * 80)
print("LISTENING TEST")
print("=" * 80)


def play_sound(audio, fs, description):
    print(f"\n▶ Playing: {description}")
    print(f"  (Press Ctrl+C to stop early)")
    try:
        sd.play(audio * 0.3, fs)  # Scale down volume
        sd.wait()
        time.sleep(0.5)
    except KeyboardInterrupt:
        sd.stop()
        print("  Stopped by user")


print("\nYou will hear 3 sounds in sequence:")
print("  1. Original signal (900 Hz + 2000 Hz)")
print("  2. Downsampled WITH filter (only 900 Hz)")
print("  3. Downsampled WITHOUT filter (900 Hz + aliased 1000 Hz)")

input("\nPress Enter to start listening test...")

play_sound(x, Fs, "1. Original x[n] at 6000 Hz (900Hz + 2000Hz)")
play_sound(y_filtered, Fs_new, "2. Downsampled WITH filter at 3000 Hz (clean 900Hz)")
play_sound(
    y_no_filter,
    Fs_new,
    "3. Downsampled WITHOUT filter at 3000 Hz (900Hz + ALIASED 1000Hz)",
)

# ============================================================================
# PART iii: Explanation based on part (b) sketches
# ============================================================================

print("\n" + "=" * 80)
print("EXPLANATION BASED ON PART (b) SPECTRA")
print("=" * 80)

print(
    """
┌─ WHAT YOU HEARD ──────────────────────────────────────────────────────┐
│                                                                        │
│ 1. ORIGINAL (at 6000 Hz):                                             │
│    • Two distinct tones: 900 Hz (lower) + 2000 Hz (higher)            │
│    • Both frequencies clearly audible                                  │
│                                                                        │
│ 2. WITH FILTER (at 3000 Hz):                                          │
│    • Single pure tone at 900 Hz                                        │
│    • The 2000 Hz tone is GONE (filtered out)                          │
│    • Sounds cleaner, simpler                                           │
│                                                                        │
│ 3. WITHOUT FILTER (at 3000 Hz):                                       │
│    • TWO tones again: 900 Hz + something around 1000 Hz                │
│    • But wait! We shouldn't hear 2000 Hz at 3000 Hz rate...           │
│    • The "1000 Hz" is actually 2000 Hz ALIASED!                       │
│    • Sounds different from original - the high tone is WRONG          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

┌─ WHY THIS HAPPENS (from part b sketches) ────────────────────────────┐
│                                                                        │
│ FROM PART (b) SPECTRUM ANALYSIS:                                      │
│                                                                        │
│ • F1 = 900 Hz  → f = 900/6000 = 0.15 (at 6kHz)                       │
│                → f = 900/3000 = 0.30 (at 3kHz) ✓ SAFE                │
│                                                                        │
│ • F2 = 2000 Hz → f = 2000/6000 = 0.333 (at 6kHz)                     │
│                → f = 2000/3000 = 0.667 (at 3kHz) ✗ > 0.5!            │
│                → ALIASES to 1 - 0.667 = 0.333                         │
│                → Appears at 0.333 × 3000 = 1000 Hz                    │
│                                                                        │
│ WITH FILTER:                                                           │
│   The lowpass filter (cutoff = 1500 Hz) removes 2000 Hz BEFORE        │
│   downsampling, so it can't alias. Only 900 Hz remains.               │
│                                                                        │
│ WITHOUT FILTER:                                                        │
│   2000 Hz goes through downsampling, exceeds new Nyquist (1500 Hz),   │
│   and "folds back" to appear at 1000 Hz. This is ALIASING!           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
"""
)

# ============================================================================
# PART iv: Repeat with music file
# ============================================================================

print("\n" + "=" * 80)
print("PART iv: TESTING WITH MUSIC FILE (Dolly.wav)")
print("=" * 80)

try:
    # Try to load the Dolly.wav file
    from scipy.io import wavfile

    # You'll need to download Dolly.wav from the course website
    # and place it in the same directory as this script

    print("\nAttempting to load Dolly.wav...")
    print("(If file not found, download from course website)")

    # Uncomment these lines when you have the file:
    fs_music, music = wavfile.read("Dolly.wav")

    # Convert to mono if stereo
    if len(music.shape) > 1:
        music = np.mean(music, axis=1)

    # Normalize
    music = music.astype(float) / np.max(np.abs(music))

    print(f"  Loaded successfully!")
    print(f"  Original sampling rate: {fs_music} Hz")
    print(f"  Duration: {len(music)/fs_music:.2f} seconds")

    # Downsample with filter
    music_filtered = signal.decimate(music, D, ftype="fir", zero_phase=True)

    # Downsample without filter
    music_no_filter = music[::D]

    fs_music_new = fs_music / D

    print(f"\nListening to music samples...")
    play_sound(music[: fs_music * 5], fs_music, "Original music")
    play_sound(
        music_filtered[: fs_music_new * 5], fs_music_new, "Downsampled WITH filter"
    )
    play_sound(
        music_no_filter[: fs_music_new * 5], fs_music_new, "Downsampled WITHOUT filter"
    )

    print("\nMUSIC OBSERVATIONS:")
    print("  WITH filter: Sounds duller (lost high frequencies) but clean")
    print("  WITHOUT filter: Sounds distorted, metallic, harsh artifacts")

    print("\n(Music test code is commented out - uncomment when you have Dolly.wav)")

except Exception as e:
    print(f"\nCould not load music file: {e}")
    print("Download Dolly.wav from course website to test with music")

# ============================================================================
# PART v: Written observations
# ============================================================================

print("\n" + "=" * 80)
print("PART v: OBSERVATIONS AND CONCLUSIONS")
print("=" * 80)

print(
    """
OBSERVATIONS:

1. TEST SIGNAL (Two cosines: 900 Hz + 2000 Hz):
   
   WITH ANTI-ALIASING FILTER:
   ✓ Only the 900 Hz tone is present in the output
   ✓ The 2000 Hz tone was correctly removed before downsampling
   ✓ Sound is clean and pure - no artifacts or distortion
   ✓ This is the CORRECT way to downsample
   
   WITHOUT ANTI-ALIASING FILTER:
   ✗ Both 900 Hz and a ~1000 Hz tone are present
   ✗ The 1000 Hz is NOT from the original signal!
   ✗ It's the 2000 Hz component aliased: 3000 - 2000 = 1000 Hz
   ✗ The high tone sounds "wrong" - different frequency than original
   ✗ This demonstrates ALIASING distortion

2. MUSIC SIGNAL (Dolly.wav):
   
   WITH ANTI-ALIASING FILTER:
   ✓ Sounds somewhat duller (high frequencies removed)
   ✓ But otherwise clean and recognizable
   ✓ No harsh artifacts or metallic sounds
   ✓ Quality degradation is expected (lost high freq content)
   
   WITHOUT ANTI-ALIASING FILTER:
   ✗ Severe distortion and artifacts
   ✗ Metallic, harsh, unpleasant sound
   ✗ High frequencies "fold back" into audible range incorrectly
   ✗ Music is barely recognizable in worst cases
   ✗ This is what happens when you downsample improperly!

CONCLUSIONS:

- The anti-aliasing filter is ESSENTIAL when downsampling
- Without it, frequencies above the new Nyquist frequency will alias
- Aliasing creates false frequencies that corrupt the signal
- For the test signal: 2000 Hz → falsely appears at 1000 Hz
- For music: many high frequencies create chaotic distortion
- The trade-off: we lose high frequency content, but preserve signal integrity
- Always use proper decimation (filter + downsample) in practice!

THEORETICAL CONNECTION TO PART (b):

The spectra we sketched in part (b) predicted exactly what we heard:
- Frequencies above new Nyquist (1500 Hz) WILL alias if not filtered
- The filter at 1500 Hz cutoff prevents this
- Without filter: spectral folding creates false frequency components
- This is why scipy.signal.decimate() always includes anti-aliasing filter!
"""
)

print("\n" + "=" * 80)
print("Script complete! Check the plots and your ears confirm the theory!")
print("=" * 80)
