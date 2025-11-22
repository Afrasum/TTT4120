import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


def problem_4a():
    """
    Problem 4(a): Design FIR bandstop filter using windowing method with Hamming window.
    Remove bandlimited noise centered at 5000Hz with bandwidth 700Hz from piano audio.

    Specifications:
    - Sample frequency: 22050 Hz
    - Noise: ~700Hz bandwidth centered at ~5000Hz
    - Transition band: max 200Hz
    - Passband ripple: max 1dB peak-to-peak
    - Stopband attenuation: min 50dB
    """

    # Load audio file
    fs, audio = wavfile.read("pianoise.wav")

    print("=" * 70)
    print("PROBLEM 4(a): FIR Bandstop Filter Design (Hamming Window)")
    print("=" * 70)

    print(f"\nAudio file loaded:")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Audio length: {len(audio)} samples ({len(audio)/fs:.2f} seconds)")
    print(f"  Audio dtype: {audio.dtype}")

    # Design parameters
    noise_center = 5000  # Hz
    noise_bandwidth = 700  # Hz
    transition_band = 200  # Hz

    # Calculate stopband edges
    f_stop_low = noise_center - noise_bandwidth / 2
    f_stop_high = noise_center + noise_bandwidth / 2

    # Calculate passband edges (add transition band)
    f_pass_low = f_stop_low - transition_band
    f_pass_high = f_stop_high + transition_band

    print(f"\nFilter specifications:")
    print(f"  Noise center: {noise_center} Hz")
    print(f"  Noise bandwidth: {noise_bandwidth} Hz")
    print(f"  Stopband: [{f_stop_low}, {f_stop_high}] Hz")
    print(f"  Passband: [0, {f_pass_low}] Hz and [{f_pass_high}, {fs/2}] Hz")
    print(f"  Transition band: {transition_band} Hz")
    print(f"  Min stopband attenuation: 50 dB")
    print(f"  Max passband ripple: 1 dB")

    # Normalize frequencies
    f_pass_low_norm = f_pass_low / (fs / 2)
    f_pass_high_norm = f_pass_high / (fs / 2)
    f_stop_low_norm = f_stop_low / (fs / 2)
    f_stop_high_norm = f_stop_high / (fs / 2)

    # Estimate filter order for Hamming window
    # Hamming window: Δf ≈ 3.3/N, Attenuation ≈ 53 dB
    transition_width_norm = transition_band / (fs / 2)
    N_estimated = int(np.ceil(3.3 / transition_width_norm))

    # Make N odd
    if N_estimated % 2 == 0:
        N_estimated += 1

    print(f"\nEstimated filter order: N = {N_estimated}")

    # Try different filter orders to find minimum that meets specs
    print(f"\nFinding minimum filter length...")

    N_values = range(N_estimated - 20 if N_estimated > 20 else 1, N_estimated + 100, 2)

    for N in N_values:
        # Design bandstop filter
        # firwin requires cutoff frequencies for bandstop
        h = signal.firwin(
            N,
            [f_pass_low_norm, f_pass_high_norm],
            pass_zero="bandstop",
            window="hamming",
        )

        # Compute frequency response
        w, H = signal.freqz(h, worN=8192)
        f = w * fs / (2 * np.pi)
        mag_dB = 20 * np.log10(np.abs(H))

        # Check stopband attenuation
        stopband_mask = (f >= f_stop_low) & (f <= f_stop_high)
        stopband_attenuation = -np.max(mag_dB[stopband_mask])

        # Check passband ripple
        passband_mask = ((f <= f_pass_low) | (f >= f_pass_high)) & (f <= fs / 2)
        passband_mag = mag_dB[passband_mask]
        passband_ripple = np.max(passband_mag) - np.min(passband_mag)

        if stopband_attenuation >= 50 and passband_ripple <= 1:
            N_min = N
            h_final = h
            break

    print(f"Minimum filter length that meets specs: N = {N_min}")

    # Compute final frequency response
    w, H = signal.freqz(h_final, worN=8192)
    f = w * fs / (2 * np.pi)
    mag_dB = 20 * np.log10(np.abs(H))
    mag_linear = np.abs(H)
    phase = np.angle(H)

    # Verify specifications
    stopband_mask = (f >= f_stop_low) & (f <= f_stop_high)
    stopband_attenuation = -np.max(mag_dB[stopband_mask])

    passband_mask = ((f <= f_pass_low) | (f >= f_pass_high)) & (f <= fs / 2)
    passband_mag = mag_dB[passband_mask]
    passband_ripple = np.max(passband_mag) - np.min(passband_mag)

    print(f"\nFinal filter verification:")
    print(f"  Stopband attenuation: {stopband_attenuation:.2f} dB (required: ≥50 dB)")
    print(f"  Passband ripple: {passband_ripple:.2f} dB (required: ≤1 dB)")

    # Plot filter characteristics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Magnitude response (full scale)
    ax = axes[0, 0]
    ax.plot(f, mag_dB, "b", linewidth=1.5)
    ax.axvline(f_stop_low, color="r", linestyle="--", alpha=0.7, label="Stopband")
    ax.axvline(f_stop_high, color="r", linestyle="--", alpha=0.7)
    ax.axvline(f_pass_low, color="g", linestyle=":", alpha=0.7, label="Passband edge")
    ax.axvline(f_pass_high, color="g", linestyle=":", alpha=0.7)
    ax.axhline(-50, color="gray", linestyle=":", alpha=0.5, label="-50 dB")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Magnitude Response (Full Range)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, fs / 2])
    ax.set_ylim([-100, 5])

    # Magnitude response (zoomed to stopband)
    ax = axes[0, 1]
    zoom_mask = (f >= f_stop_low - 1000) & (f <= f_stop_high + 1000)
    ax.plot(f[zoom_mask], mag_dB[zoom_mask], "b", linewidth=1.5)
    ax.axvline(f_stop_low, color="r", linestyle="--", alpha=0.7, label="Stopband")
    ax.axvline(f_stop_high, color="r", linestyle="--", alpha=0.7)
    ax.axvline(f_pass_low, color="g", linestyle=":", alpha=0.7, label="Passband edge")
    ax.axvline(f_pass_high, color="g", linestyle=":", alpha=0.7)
    ax.axhline(-50, color="gray", linestyle=":", alpha=0.5, label="-50 dB")
    ax.fill_between([f_stop_low, f_stop_high], -100, 5, alpha=0.2, color="red")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Magnitude Response (Stopband Detail)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-100, 5])

    # Magnitude response in passband
    ax = axes[1, 0]
    passband_low = f <= f_pass_low
    passband_high = (f >= f_pass_high) & (f <= fs / 2)
    ax.plot(f[passband_low], mag_dB[passband_low], "b", linewidth=1.5, label="Passband")
    ax.plot(f[passband_high], mag_dB[passband_high], "b", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(-1, color="r", linestyle=":", alpha=0.5, label="±1 dB")
    ax.axhline(1, color="r", linestyle=":", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Magnitude Response (Passband Detail)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, fs / 2])
    ax.set_ylim([-2, 2])

    # Phase response
    ax = axes[1, 1]
    ax.plot(f, np.unwrap(phase), "b", linewidth=1.5)
    ax.axvline(f_stop_low, color="r", linestyle="--", alpha=0.5)
    ax.axvline(f_stop_high, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (radians)")
    ax.set_title("Phase Response")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, fs / 2])

    plt.tight_layout()
    plt.show()

    print(f"\n{'='*70}")
    print(f"Filter design complete: N = {N_min}")
    print(f"{'='*70}\n")

    return h_final, fs, audio


if __name__ == "__main__":
    h, fs, audio = problem_4a()
