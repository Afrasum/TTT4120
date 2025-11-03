import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io.wavfile import read, write

print("=" * 70)
print("PROBLEM 3g: MULTIPLE ECHO FILTER WITH VARIED PARAMETERS")
print("=" * 70)

fs_audio, piano = read("piano.wav")
piano_normalized = piano / np.max(np.abs(piano))

# Load and process piano with both filters
fs_audio, piano = read("piano.wav")
piano_normalized = piano / np.max(np.abs(piano))

print(f"Piano audio: {len(piano)} samples, {len(piano)/fs_audio:.2f} seconds")

# Test different parameter combinations
test_configurations = [
    {"R": 1000, "alpha": 0.6, "N": 4, "desc": "Small room"},
    {"R": 2205, "alpha": 0.7, "N": 6, "desc": "Medium hall"},
    {"R": 4410, "alpha": 0.8, "N": 8, "desc": "Large cathedral"},
]

plt.figure(figsize=(15, 12))

for idx, config in enumerate(test_configurations):
    R = config["R"]
    alpha = config["alpha"]
    N = config["N"]

    # Create filter
    b = np.zeros(N * R + 1)
    b[0] = 1
    b[N * R] = -(alpha**N)

    a = np.zeros(R + 1)
    a[0] = 1
    a[R] = -alpha

    # Apply filter
    piano_filtered = signal.lfilter(b, a, piano_normalized)
    piano_filtered = piano_filtered / np.max(np.abs(piano_filtered))

    # Plot waveform comparison
    plt.subplot(len(test_configurations), 3, idx * 3 + 1)
    plot_samples = 20000
    time = np.arange(plot_samples) / fs_audio
    plt.plot(time, piano_normalized[:plot_samples], alpha=0.5, label="Original")
    plt.plot(time, piano_filtered[:plot_samples], alpha=0.5, label="Filtered")
    plt.title(f'{config["desc"]}: Waveform')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Plot frequency spectrum
    plt.subplot(len(test_configurations), 3, idx * 3 + 2)
    w, H = signal.freqz(b, a, worN=4096)
    f = w * fs_audio / (2 * np.pi)
    plt.plot(f, 20 * np.log10(np.abs(H) + 1e-10))
    plt.title(f'{config["desc"]}: Frequency Response')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.xlim([0, 5000])
    plt.ylim([-30, 15])

    # Information
    plt.subplot(len(test_configurations), 3, idx * 3 + 3)
    plt.axis("off")

    info = f"""
{config['desc'].upper()}

Parameters:
  R = {R} samples
  Delay = {R/fs_audio*1000:.1f} ms
  α = {alpha}
  N = {N} echoes
  Total reverb = {N*R/fs_audio*1000:.0f} ms

Echo amplitudes:
"""
    for i in range(min(N + 1, 5)):
        info += f"  Echo {i}: {alpha**i:.3f}\n"
    if N > 4:
        info += f"  ...\n"

    info += f"""
Expected sound:
  • Reverb time: {N*R/fs_audio:.2f}s
  • Spaciousness: {"low" if R < 2000 else "medium" if R < 4000 else "high"}
  • Echo density: {N} echoes
"""

    plt.text(
        0.05,
        0.95,
        info,
        fontsize=9,
        verticalalignment="top",
        family="monospace",
        transform=plt.gca().transAxes,
    )

    # Save filtered audio
    filename = f'piano_multi_echo_{config["desc"].replace(" ", "_")}.wav'
    write(filename, fs_audio, (piano_filtered * 32767).astype(np.int16))
    print(f"\nSaved: {filename}")

    # Play audio
    print(f"\n{'='*50}")
    print(f"Configuration {idx+1}: {config['desc']}")
    print(f"{'='*50}")
    print("Playing ORIGINAL...")
    sd.play(piano_normalized, fs_audio)
    sd.wait()

    print(f"Playing FILTERED ({config['desc']})...")
    sd.play(piano_filtered, fs_audio)
    sd.wait()

plt.tight_layout()
plt.show()

# Detailed sound description
print("\n" + "=" * 70)
print("SOUND DESCRIPTION: How the filtered signal sounds")
print("=" * 70)

descriptions = """
COMPARED TO THE ORIGINAL:

1. SPATIAL DIMENSION:
   - Original: Dry, close, "in your face"
   - Filtered: Spacious, distant, "in a room/hall"
   - Effect: Adds depth and three-dimensionality

2. SUSTAIN:
   - Original: Notes die out quickly
   - Filtered: Notes linger, blend together
   - Effect: More "legato" feeling, smoother

3. RICHNESS:
   - Original: Clean but thin
   - Filtered: Fuller, more complex timbre
   - Effect: Frequency reinforcement from echoes

4. REALISM:
   - Original: Studio recording (dead room)
   - Filtered: Concert hall / church
   - Effect: Mimics real acoustic spaces

HOW PARAMETERS AFFECT SOUND:

- Small R (short delay):
  - Dense, thick echo pattern
  - "Room" character
  - Quick echo buildup
  
- Large R (long delay):
  - Sparse, discrete echoes
  - "Cathedral" character
  - Slow echo decay
  
- Small α (weak echoes):
  - Subtle ambience
  - Natural sound
  - Professional mix
  
- Large α (strong echoes):
  - Dramatic effect
  - Can sound "washy"
  - Special effect territory
  
- Small N (few echoes):
  - Quick decay
  - Small room
  - Tight sound
  
- Large N (many echoes):
  - Long decay
  - Large space
  - Ethereal quality
"""

print(descriptions)
print("=" * 70)
