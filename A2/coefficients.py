import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

print("=== HVA KAN DU BRUKE DTFS-KOEFFISIENTER TIL? ===\n")


# Our signal from Problem 1(d)
def c_k(k, N):
    """DTFS coefficients for our problem"""
    return (1 / N) * (2 + 2 * np.cos(2 * np.pi * k / N))


def periodic_signal(n, N):
    """Our periodic signal z[n]"""
    n_mod = n % N
    if n_mod == 0:
        return 2
    elif n_mod == 1:
        return 1
    elif n_mod == N - 1:
        return 1
    else:
        return 0


N = 10

print("ANVENDELSE 1: REKONSTRUERE SIGNALET")
print("=" * 50)

print("Du kan rekonstruere det originale signalet fra koeffisientene!")
print("Synthesis equation: z[n] = Σ(k=0 to N-1) c_k * e^(j2πkn/N)")

# Calculate coefficients
coefficients = [c_k(k, N) for k in range(N)]

# Reconstruct signal
n_range = np.arange(2 * N)  # Two periods for visualization
z_original = [periodic_signal(n, N) for n in n_range]
z_reconstructed = np.zeros(len(n_range), dtype=complex)

for n in n_range:
    for k in range(N):
        z_reconstructed[n] += coefficients[k] * np.exp(1j * 2 * np.pi * k * n / N)

print(f"Reconstruction error: {np.max(np.abs(z_reconstructed.real - z_original)):.2e}")

# Plot reconstruction
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
plt.stem(n_range, z_original, basefmt="b-", label="Original", markerfmt="bo")
plt.plot(n_range, z_reconstructed.real, "r--", linewidth=2, label="Reconstructed")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.title("Signal Reconstruction")
plt.legend()
plt.grid(True)

print(f"\nANVENDELSE 2: FREKVENSANALYSE")
print("=" * 50)

print("Koeffisientene forteller deg hvilke frekvenser som er i signalet!")
print("Størrelsen |c_k| = hvor mye av hver frekvens")
print("Fasen ∠c_k = tidsforsinkelse for hver frekvens")

# Frequency analysis
k_vals = np.arange(N)
magnitudes = [abs(c_k(k, N)) for k in k_vals]
frequencies_hz = k_vals / N  # Normalized frequency

plt.subplot(3, 3, 2)
plt.stem(frequencies_hz, magnitudes, basefmt="g-")
plt.xlabel("Normalized Frequency (cycles/sample)")
plt.ylabel("|c_k|")
plt.title("Frequency Content Analysis")
plt.grid(True)

# Identify dominant frequencies
dominant_freqs = []
for k, mag in enumerate(magnitudes):
    if mag > 0.1:  # Threshold for "significant" frequency
        dominant_freqs.append((k, frequencies_hz[k], mag))

print(f"Dominante frekvenser i signalet:")
for k, freq, mag in dominant_freqs:
    print(f"  k={k}: f={freq:.3f} cycles/sample, magnitude={mag:.3f}")

print(f"\nANVENDELSE 3: FILTRERING")
print("=" * 50)

print("Du kan endre signalet ved å modifisere koeffisientene!")

# Example: Low-pass filtering (remove high frequencies)
coefficients_lowpass = coefficients.copy()
cutoff_k = 3  # Keep only k=0,1,2
for k in range(cutoff_k, N):
    coefficients_lowpass[k] = 0

# Reconstruct filtered signal
z_filtered = np.zeros(len(n_range), dtype=complex)
for n in n_range:
    for k in range(cutoff_k):
        z_filtered[n] += coefficients_lowpass[k] * np.exp(1j * 2 * np.pi * k * n / N)

plt.subplot(3, 3, 3)
plt.stem(n_range, z_original, basefmt="b-", alpha=0.5, label="Original")
plt.plot(n_range, z_filtered.real, "r-", linewidth=2, label="Low-pass filtered")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.title("Low-pass Filtering")
plt.legend()
plt.grid(True)

print(f"Lavpass-filtrering: Beholder kun k=0,1,2 (lave frekvenser)")

print(f"\nANVENDELSE 4: KOMPRESJON")
print("=" * 50)

print("Mange koeffisienter er små - du kan 'kaste' dem for å spare plass!")

# Find coefficients above threshold
threshold = 0.05
significant_coeffs = [(k, c) for k, c in enumerate(coefficients) if abs(c) > threshold]

print(f"Originale koeffisienter: {N}")
print(f"Betydningsfulle koeffisienter (> {threshold}): {len(significant_coeffs)}")
print(f"Kompresjon ratio: {N/len(significant_coeffs):.1f}:1")

# Reconstruct with only significant coefficients
z_compressed = np.zeros(len(n_range), dtype=complex)
for n in n_range:
    for k, c in significant_coeffs:
        z_compressed[n] += c * np.exp(1j * 2 * np.pi * k * n / N)

plt.subplot(3, 3, 4)
plt.stem(n_range, z_original, basefmt="b-", alpha=0.5, label="Original")
plt.plot(n_range, z_compressed.real, "g-", linewidth=2, label="Compressed")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.title("Signal Compression")
plt.legend()
plt.grid(True)

compression_error = np.max(np.abs(z_compressed.real - z_original))
print(f"Kompresjonsfeil: {compression_error:.4f}")

print(f"\nANVENDELSE 5: SPEKTRAL FORMNING")
print("=" * 50)

print("Du kan endre 'klangen' av signalet ved å modifisere koeffisientene!")

# Example: Emphasis on mid frequencies
coefficients_shaped = coefficients.copy()
for k in range(N):
    if 2 <= k <= 4:  # Emphasize mid frequencies
        coefficients_shaped[k] *= 2

# Reconstruct shaped signal
z_shaped = np.zeros(len(n_range), dtype=complex)
for n in n_range:
    for k in range(N):
        z_shaped[n] += coefficients_shaped[k] * np.exp(1j * 2 * np.pi * k * n / N)

plt.subplot(3, 3, 5)
plt.stem(k_vals, magnitudes, basefmt="b-", alpha=0.5, label="Original spectrum")
shaped_magnitudes = [abs(coefficients_shaped[k]) for k in k_vals]
plt.stem(k_vals, shaped_magnitudes, basefmt="r-", label="Shaped spectrum")
plt.xlabel("k")
plt.ylabel("|c_k|")
plt.title("Spectral Shaping")
plt.legend()
plt.grid(True)

plt.subplot(3, 3, 6)
plt.stem(n_range, z_original, basefmt="b-", alpha=0.5, label="Original")
plt.plot(n_range, z_shaped.real, "r-", linewidth=2, label="Spectrally shaped")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.title("Shaped Signal")
plt.legend()
plt.grid(True)

print(f"\nANVENDELSE 6: STØYFJERNING")
print("=" * 50)

print("Hvis du vet at støy er i visse frekvenser, kan du fjerne dem!")

# Add noise at specific frequency
noise_k = 7  # Add noise at k=7
coefficients_noisy = coefficients.copy()
coefficients_noisy[noise_k] += 0.1  # Add noise

# Create noisy signal
z_noisy = np.zeros(len(n_range), dtype=complex)
for n in n_range:
    for k in range(N):
        z_noisy[n] += coefficients_noisy[k] * np.exp(1j * 2 * np.pi * k * n / N)

# Remove noise by zeroing the noisy coefficient
coefficients_denoised = coefficients_noisy.copy()
coefficients_denoised[noise_k] = 0

# Create denoised signal
z_denoised = np.zeros(len(n_range), dtype=complex)
for n in n_range:
    for k in range(N):
        z_denoised[n] += coefficients_denoised[k] * np.exp(1j * 2 * np.pi * k * n / N)

plt.subplot(3, 3, 7)
plt.stem(n_range, z_original, basefmt="b-", alpha=0.3, label="Original")
plt.plot(n_range, z_noisy.real, "r-", alpha=0.7, label="Noisy")
plt.plot(n_range, z_denoised.real, "g-", linewidth=2, label="Denoised")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.title("Noise Removal")
plt.legend()
plt.grid(True)

print(f"\nANVENDELSE 7: AUDIO EQUALIZER")
print("=" * 50)

print("Som en audio equalizer - boost/cut forskjellige frekvenser!")

# Create equalizer settings (like on a stereo)
eq_gains = [
    1.0,
    0.5,
    1.5,
    0.8,
    1.2,
    1.0,
    0.7,
    1.3,
    0.9,
    1.1,
]  # Different gains for each band

coefficients_eq = [c * gain for c, gain in zip(coefficients, eq_gains)]

plt.subplot(3, 3, 8)
plt.stem(k_vals, eq_gains, basefmt="m-")
plt.xlabel("Frequency Band (k)")
plt.ylabel("EQ Gain")
plt.title("Equalizer Settings")
plt.grid(True)
plt.ylim([0, 2])

# Reconstruct equalized signal
z_equalized = np.zeros(len(n_range), dtype=complex)
for n in n_range:
    for k in range(N):
        z_equalized[n] += coefficients_eq[k] * np.exp(1j * 2 * np.pi * k * n / N)

plt.subplot(3, 3, 9)
plt.stem(n_range, z_original, basefmt="b-", alpha=0.5, label="Original")
plt.plot(n_range, z_equalized.real, "purple", linewidth=2, label="Equalized")
plt.xlabel("n")
plt.ylabel("z[n]")
plt.title("Equalized Signal")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\n{'='*60}")
print("PRAKTISKE ANVENDELSER I VIRKELIGHETEN")
print("=" * 60)

applications = [
    (
        "AUDIO PROCESSING",
        [
            "MP3 compression (DCT coefficients)",
            "Spotify's audio equalizer",
            "Noise cancellation i hodetelefoner",
            "Auto-tune i musikk",
        ],
    ),
    (
        "BILDEBEHANDLING",
        [
            "JPEG compression",
            "Instagram filters",
            "Medical imaging (MRI, CT scans)",
            "Satellittbilder",
        ],
    ),
    (
        "KOMMUNIKASJON",
        ["WiFi/4G/5G signaler", "Digital TV (DVB)", "Bluetooth", "Radar systemer"],
    ),
    (
        "MASKINLÆRING",
        [
            "Feature extraction",
            "Signal preprocessing",
            "Pattern recognition",
            "Speech recognition",
        ],
    ),
    (
        "MEDISIN",
        ["EKG/EEG analyse", "Ultralyd imaging", "DNA sekvensering", "Drug discovery"],
    ),
    (
        "FINANS",
        ["Markedsanalyse", "Trend detection", "Risk assessment", "Algorithmic trading"],
    ),
]

for category, examples in applications:
    print(f"\n{category}:")
    for example in examples:
        print(f"  • {example}")

print(f"\n{'='*60}")
print("HVORFOR ER DETTE SÅ KRAFTFULLT?")
print("=" * 60)

print(
    """
1. FREKVENSDOMENET AVSLØRER MØNSTRE
   • Ting som er vanskelig å se i tiden blir tydelige i frekvens
   • Eksempel: Rytme i musikk, periodicitet i data

2. EFFEKTIV PROSESSERING
   • Mange operasjoner er enklere i frekvensdomenet
   • Filtrering: multiplikasjon i stedet for konvolusjon

3. KOMPRESJON
   • Mange signaler har energi konsentrert i få frekvenser
   • Kan 'kaste' ubetydelige koeffisienter

4. STØYFJERNING
   • Støy og signal har ofte forskjellige frekvensegenskaper
   • Kan separere dem i frekvensdomenet

5. ANALYSE OG DIAGNOSE
   • Frekvensinnhold kan avsløre problemer eller egenskaper
   • Eksempel: Motorvibrasjoner, hjertearytmi
"""
)

print(f"\nKONKLUSJON:")
print("DTFS-koeffisienter er som en 'oppskrift' for signalet ditt.")
print("Ved å endre oppskriften kan du:")
print("• Fjerne ingredienser du ikke vil ha (støy)")
print("• Forsterke smaker du liker (viktige frekvenser)")
print("• Lage en lettere versjon (kompresjon)")
print("• Analysere hva som er i retten (frekvensinnhold)")

print(f"\nDette er grunnen til at Fourier-analyse er fundamentet")
print("for nesten all moderne digital teknologi!")
