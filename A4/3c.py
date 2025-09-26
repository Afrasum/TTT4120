import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read
from scipy.signal import freqz, lfilter

# Load audio file
fs, x = read("pluto.wav")
print(f"Sampling frequency: {fs} Hz")
print(f"Audio length: {len(x)} samples = {len(x)/fs:.2f} seconds")

# Play original sound first
print("\n" + "=" * 60)
print("PLAYING *ORIGINAL* SOUND CLIP")
print("=" * 60)
xscaled = x / np.max(np.abs(x))
sd.play(xscaled, fs)
input("Press Enter after listening to original...")

print("\n" + "=" * 60)
print("PART 1: VARYING α WITH K = 3")
print("=" * 60)

# PART 1: K = 3, varying alpha
K = 3
alpha_values = [0.5, 0.7, 0.9]

plt.figure(figsize=(15, 12))

for i, alpha in enumerate(alpha_values):
    print(f"\n--- Testing α = {alpha}, K = {K} ---")

    # Calculate coefficients
    b = [
        (K / 2) * (1 - alpha) + (1 / 2) * (1 + alpha),
        (K / 2) * (1 - alpha) - (1 / 2) * (1 + alpha),
    ]
    a = [1, -alpha]

    print(f"Coefficients: b = [{b[0]:.3f}, {b[1]:.3f}], a = {a}")

    # Filter audio
    y = lfilter(b, a, x)

    # Calculate frequency response
    w, h = freqz(b, a, worN=1024)
    normFreq = w / np.pi

    # Plot
    plt.subplot(2, 3, i + 1)
    plt.plot(normFreq, 20 * np.log10(np.abs(h)), "b-", linewidth=2)
    plt.title(f"α = {alpha}, K = {K}")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True, alpha=0.3)

    # Add analysis
    dc_gain = 20 * np.log10(np.abs(h[0]))
    nyquist_gain = 20 * np.log10(np.abs(h[-1]))
    shelf_amount = dc_gain - nyquist_gain
    plt.text(
        0.02,
        max(20 * np.log10(np.abs(h))) - 2,
        f"Shelf: {shelf_amount:.1f} dB",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    # Play filtered sound
    print(f"Playing filtered sound with α = {alpha}...")
    yscaled = y / np.max(np.abs(y))
    sd.play(yscaled, fs)

    if i < len(alpha_values) - 1:
        input("Press Enter for next α value...")

print("\n" + "=" * 60)
print("PART 2: VARYING K WITH α = 0.7")
print("=" * 60)

# PART 2: α = 0.7, varying K
alpha = 0.7
K_values = [0.5, 1, 4]

for i, K in enumerate(K_values):
    print(f"\n--- Testing α = {alpha}, K = {K} ---")

    # Calculate coefficients
    b = [
        (K / 2) * (1 - alpha) + (1 / 2) * (1 + alpha),
        (K / 2) * (1 - alpha) - (1 / 2) * (1 + alpha),
    ]
    a = [1, -alpha]

    print(f"Coefficients: b = [{b[0]:.3f}, {b[1]:.3f}], a = {a}")

    # Filter audio
    y = lfilter(b, a, x)

    # Calculate frequency response
    w, h = freqz(b, a, worN=1024)
    normFreq = w / np.pi

    # Plot
    plt.subplot(2, 3, i + 4)
    plt.plot(normFreq, 20 * np.log10(np.abs(h)), "r-", linewidth=2)
    plt.title(f"α = {alpha}, K = {K}")
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Magnitude [dB]")
    plt.grid(True, alpha=0.3)

    # Add analysis
    dc_gain = 20 * np.log10(np.abs(h[0]))
    nyquist_gain = 20 * np.log10(np.abs(h[-1]))
    shelf_amount = dc_gain - nyquist_gain
    plt.text(
        0.02,
        max(20 * np.log10(np.abs(h))) - 2,
        f"Shelf: {shelf_amount:.1f} dB",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    # Play filtered sound
    print(f"Playing filtered sound with K = {K}...")
    yscaled = y / np.max(np.abs(y))
    sd.play(yscaled, fs)

    if i < len(K_values) - 1:
        input("Press Enter for next K value...")

plt.tight_layout()
plt.show()

# COMPARISON PLOTS
plt.figure(figsize=(12, 5))

# Alpha comparison
plt.subplot(1, 2, 1)
K = 3
for alpha in alpha_values:
    b = [
        (K / 2) * (1 - alpha) + (1 / 2) * (1 + alpha),
        (K / 2) * (1 - alpha) - (1 / 2) * (1 + alpha),
    ]
    a = [1, -alpha]
    w, h = freqz(b, a, worN=1024)
    normFreq = w / np.pi
    plt.plot(normFreq, 20 * np.log10(np.abs(h)), linewidth=2, label=f"α = {alpha}")
plt.title(f"α Comparison (K = {K})")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude [dB]")
plt.legend()
plt.grid(True)

# K comparison
plt.subplot(1, 2, 2)
alpha = 0.7
for K in K_values:
    b = [
        (K / 2) * (1 - alpha) + (1 / 2) * (1 + alpha),
        (K / 2) * (1 - alpha) - (1 / 2) * (1 + alpha),
    ]
    a = [1, -alpha]
    w, h = freqz(b, a, worN=1024)
    normFreq = w / np.pi
    plt.plot(normFreq, 20 * np.log10(np.abs(h)), linewidth=2, label=f"K = {K}")
plt.title(f"K Comparison (α = {alpha})")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude [dB]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# FINAL ANALYSIS
print("\n" + "=" * 60)
print("COMPLETE ANALYSIS - PROBLEM 3c SUMMARY")
print("=" * 60)

print("\nPART 1 RESULTS (K = 3, varying α):")
K = 3
for alpha in alpha_values:
    b = [
        (K / 2) * (1 - alpha) + (1 / 2) * (1 + alpha),
        (K / 2) * (1 - alpha) - (1 / 2) * (1 + alpha),
    ]
    a = [1, -alpha]
    w, h = freqz(b, a, worN=1024)

    dc_gain = 20 * np.log10(np.abs(h[0]))
    nyquist_gain = 20 * np.log10(np.abs(h[-1]))
    shelf_amount = dc_gain - nyquist_gain

    print(f"α = {alpha}: Bass boost = {shelf_amount:.1f} dB")

print("\nPART 2 RESULTS (α = 0.7, varying K):")
alpha = 0.7
for K in K_values:
    b = [
        (K / 2) * (1 - alpha) + (1 / 2) * (1 + alpha),
        (K / 2) * (1 - alpha) - (1 / 2) * (1 + alpha),
    ]
    a = [1, -alpha]
    w, h = freqz(b, a, worN=1024)

    dc_gain = 20 * np.log10(np.abs(h[0]))
    nyquist_gain = 20 * np.log10(np.abs(h[-1]))
    shelf_amount = dc_gain - nyquist_gain

    print(f"K = {K}: Bass boost = {shelf_amount:.1f} dB")

print("\nWHAT THE PARAMETERS CONTROL:")
print("α (Alpha): Controls the SHARPNESS of the shelf transition")
print("  - Higher α → sharper transition, more selective")
print("  - Lower α → gentler transition, broader effect")
print("\nK: Controls the AMOUNT of shelving (boost/cut)")
print("  - Higher K → more dramatic bass boost")
print("  - Lower K → gentler bass boost")
print("  - K = 1 → moderate effect")

print("\nAUDIO EFFECTS YOU SHOULD HEAR:")
print("Part 1 (α variation):")
print("  α = 0.5: Gentle, wide bass boost")
print("  α = 0.7: Moderate, focused bass boost")
print("  α = 0.9: Sharp, very focused bass boost")
print("\nPart 2 (K variation):")
print("  K = 0.5: Subtle bass warmth")
print("  K = 1.0: Noticeable bass enhancement")
print("  K = 4.0: Strong, dramatic bass boost")
