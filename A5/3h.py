# Direct comparison and analysis

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io.wavfile import read

print("\n" + "=" * 70)
print("PART (h): WHICH FILTER SOUNDS MORE NATURAL?")
print("=" * 70)

# Create both filters with same R and alpha for fair comparison
R = 2205  # 100ms delay
alpha = 0.7
N = 8

# Single echo
b_single = np.zeros(R + 1)
a_single = np.ones(1)
b_single[0] = 1
b_single[R] = alpha

# Multiple echo
b_multi = np.zeros(N * R + 1)
b_multi[0] = 1
b_multi[N * R] = -(alpha**N)

a_multi = np.zeros(R + 1)
a_multi[0] = 1
a_multi[R] = -alpha

# Load piano
fs_audio, piano = read("piano.wav")
piano_normalized = piano / np.max(np.abs(piano))

# Filter with both
piano_single = signal.lfilter(b_single, a_single, piano_normalized)
piano_single = piano_single / np.max(np.abs(piano_single))

piano_multi = signal.lfilter(b_multi, a_multi, piano_normalized)
piano_multi = piano_multi / np.max(np.abs(piano_multi))

# Play comparison
print("\n1. Playing ORIGINAL (no processing)...")
sd.play(piano_normalized, fs_audio)
sd.wait()

print("\n2. Playing SINGLE ECHO filter...")
sd.play(piano_single, fs_audio)
sd.wait()

print("\n3. Playing MULTIPLE ECHO filter...")
sd.play(piano_multi, fs_audio)
sd.wait()

# Analysis
analysis = """

ANSWER: THE MULTIPLE ECHO FILTER SOUNDS MORE NATURAL

WHY? Physical and Perceptual Reasons:

1. REAL-WORLD ACOUSTICS:
   
   In actual rooms/halls:
   ✓ Sound reflects off MANY surfaces (walls, ceiling, floor, objects)
   ✓ Each reflection creates an echo
   ✓ Echoes arrive continuously, not just once
   ✓ Each bounce loses energy (absorption)
   
   Single echo filter:
   ✗ Only ONE reflection
   ✗ Unrealistic in nature
   ✗ Sounds like a single wall bounce
   
   Multiple echo filter:
   ✓ MANY reflections
   ✓ Mimics real acoustic spaces
   ✓ Sounds like actual room acoustics

2. ECHO DENSITY:
   
   Real spaces:
   ✓ Dense echo pattern (hundreds per second)
   ✓ Echoes merge into smooth reverb tail
   
   Single echo:
   ✗ Only 2 events: original + echo
   ✗ Obvious gap between them
   ✗ "Ping-pong" effect
   
   Multiple echo:
   ✓ Many echoes fill the time gap
   ✓ Smoother transition
   ✓ More like real reverb

3. FREQUENCY RESPONSE:
   
   Real rooms:
   ✓ Complex frequency coloration
   ✓ Multiple resonances
   
   Single echo:
   ✗ Simple comb filtering
   ✗ Obvious frequency artifacts
   
   Multiple echo:
   ✓ More complex, natural filtering
   ✓ Closer to real room response

4. PERCEPTUAL NATURALNESS:
   
   Our brains expect:
   ✓ Gradual reverb decay
   ✓ Rich, complex reflections
   ✓ Spatial cues from multiple directions
   
   Single echo sounds:
   ✗ Artificial
   ✗ Like a digital effect
   ✗ "Cheap" delay pedal
   
   Multiple echo sounds:
   ✓ Organic
   ✓ Like a real space
   ✓ Professional reverb

5. TEMPORAL CHARACTERISTICS:
   
   Real acoustic decay:
   ✓ Smooth, exponential
   ✓ Continuous
   
   Single echo:
   ✗ Step function (on/off)
   ✗ Abrupt
   
   Multiple echo:
   ✓ Approximates exponential decay
   ✓ More continuous

MATHEMATICAL INSIGHT:

Real room impulse response:
  h(t) = Σ(many reflections) ≈ exponential decay

Single echo:
  h[n] = δ[n] + α·δ[n-R]  (only 2 terms)

Multiple echo:
  h[n] = Σ(α^i · δ[n-iR]) for i=0 to N  (N+1 terms)
  
The multiple echo is closer to the real continuous sum!

PRACTICAL APPLICATIONS:

Single echo good for:
- Special effects (slapback delay)
- Vocal doubling
- Rhythmic effects
- Creative production

Multiple echo good for:
- Realistic room simulation
- Natural reverb
- Professional mixing
- Classical music recording

CONCLUSION:
The multiple echo filter is MORE NATURAL because it better
approximates the complex reflection patterns found in real
acoustic spaces. Real rooms don't have just one reflection—
they have thousands, and the multiple echo filter (with its
recursive structure creating many decaying repetitions) is
much closer to this reality.
"""

print(analysis)
print("=" * 70)

# Visual comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Impulse responses
axes[0, 0].stem(
    signal.lfilter(b_single, a_single, np.concatenate([[1], np.zeros(200)]))
)
axes[0, 0].set_title("Single Echo: Impulse Response\n(Unnatural - only 2 spikes)")
axes[0, 0].set_xlabel("Samples")
axes[0, 0].set_ylabel("Amplitude")
axes[0, 0].grid(True)

axes[1, 0].stem(signal.lfilter(b_multi, a_multi, np.concatenate([[1], np.zeros(2000)])))
axes[1, 0].set_title(
    f"Multiple Echo: Impulse Response\n(Natural - {N+1} decaying spikes)"
)
axes[1, 0].set_xlabel("Samples")
axes[1, 0].set_ylabel("Amplitude")
axes[1, 0].grid(True)

# Real room comparison (conceptual)
axes[0, 1].axis("off")
axes[0, 1].text(
    0.1,
    0.5,
    """
REAL ROOM ACOUSTICS:

🏛️ Concert Hall:
   • 1000s of reflections
   • Smooth decay (2-3 seconds)
   • Complex frequency response
   • Diffuse sound field

📊 Impulse Response:
   ┌─┐
   │ │╲
   │ │ ╲___
   │ │     ╲____
   └─┴──────────╲___
   
   Exponential decay
   Many small echoes
""",
    fontsize=11,
    family="monospace",
    verticalalignment="center",
)

axes[1, 1].axis("off")
comparison_text = f"""
FILTER COMPARISON:

Single Echo:
   ┌─┐    ┌─┐
   │ │    │ │
   └─┘    └─┘
   
   Only 2 events
   Unnatural gap
   ⭐ Naturalness: 3/10

Multiple Echo (N={N}):
   ┌─┐┌┐┌┐┌┐┌┐┌┐
   │ ││││││││││
   └─┘└┘└┘└┘└┘└┘
   
   {N+1} decaying events
   Smoother transition
   ⭐ Naturalness: 8/10

Winner: MULTIPLE ECHO ✓
Reason: Mimics real acoustics
"""
axes[1, 1].text(
    0.1,
    0.5,
    comparison_text,
    fontsize=11,
    family="monospace",
    verticalalignment="center",
)

plt.suptitle("Why Multiple Echo Sounds More Natural", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()
