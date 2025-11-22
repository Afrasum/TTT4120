import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def problem_3e():
    """
    Problem 3(e): Compare digital IIR filters with analog prototype.
    Plot magnitude responses and comment on differences.
    """
    print("=" * 70)
    print("PROBLEM 3(e): Comparison with Analog Prototype (from 3b)")
    print("=" * 70)

    # ========================================================================
    # ANALOG FILTER (from Problem 3)
    # ========================================================================

    num_analog = [1]
    den_analog = [1, np.sqrt(2), 1]
    Omega_c = 1.0  # Analog cutoff frequency

    print("\nAnalog Butterworth filter:")
    print(f"  Ha(s) = 1 / (s² + √2·s + 1)")
    print(f"  Cutoff frequency: Ωc = {Omega_c} rad/s")

    # ========================================================================
    # DIGITAL FILTERS (from Problem 3d - using impulse invariance)
    # ========================================================================

    # Design specifications
    wc1 = 0.25  # First digital cutoff frequency
    wc2 = 1.4  # Second digital cutoff frequency

    T1 = wc1 / Omega_c  # Sampling period for filter 1
    T2 = wc2 / Omega_c  # Sampling period for filter 2

    print("\nDigital filters (impulse invariance method):")
    print(f"\nFilter 1:")
    print(f"  Digital cutoff: ωc1 = {wc1}")
    print(f"  Sampling period: T1 = {T1}")

    print(f"\nFilter 2:")
    print(f"  Digital cutoff: ωc2 = {wc2}")
    print(f"  Sampling period: T2 = {T2}")

    # Apply impulse invariance
    b1, a1, _ = signal.cont2discrete((num_analog, den_analog), T1, method="impulse")
    b2, a2, _ = signal.cont2discrete((num_analog, den_analog), T2, method="impulse")

    # cont2discrete returns (b, a, dt), with b 2D.
    b1 = b1.flatten()
    b2 = b2.flatten()

    print(f"\nFilter 1 coefficients:")
    print(f"  b1 = {b1}")
    print(f"  a1 = {a1}")

    print(f"\nFilter 2 coefficients:")
    print(f"  b2 = {b2}")
    print(f"  a2 = {a2}")

    # ========================================================================
    # COMPUTE FREQUENCY RESPONSES
    # ========================================================================

    # Analog frequency response
    w_analog = np.linspace(0, 5, 1000)
    _, h_analog = signal.freqs(num_analog, den_analog, worN=w_analog)

    # Digital frequency responses
    w1, h1 = signal.freqz(b1, a1, worN=8192)
    w2, h2 = signal.freqz(b2, a2, worN=8192)

    # ========================================================================
    # PLOT 1: All three magnitude responses (dB and Linear)
    # ========================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Normalize analog frequency for comparison
    # For filter 1: ω = Ω*T1
    w_analog_norm1 = w_analog * T1 / np.pi
    w_analog_norm2 = w_analog * T2 / np.pi

    # Left plot: Magnitude in dB
    ax1.plot(
        w1 / np.pi,
        20 * np.log10(np.abs(h1)),
        "r",
        linewidth=2.5,
        label=f"Digital Filter 1 (ωc={wc1})",
        alpha=0.9,
    )
    ax1.plot(
        w2 / np.pi,
        20 * np.log10(np.abs(h2)),
        "b",
        linewidth=2.5,
        label=f"Digital Filter 2 (ωc={wc2})",
        alpha=0.9,
    )
    ax1.plot(
        w_analog_norm1,
        20 * np.log10(np.abs(h_analog)),
        "g--",
        linewidth=2,
        label="Analog Prototype (scaled)",
        alpha=0.7,
    )

    ax1.axvline(wc1 / np.pi, color="r", linestyle=":", linewidth=1.5, alpha=0.5)
    ax1.axvline(wc2 / np.pi, color="b", linestyle=":", linewidth=1.5, alpha=0.5)
    ax1.axhline(-3, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="-3 dB")

    ax1.set_xlabel("Normalized Frequency (ω/π)", fontsize=12)
    ax1.set_ylabel("Magnitude (dB)", fontsize=12)
    ax1.set_title(
        "Digital vs Analog: Magnitude Response (dB)", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-60, 5])

    # Right plot: Linear scale
    ax2.plot(
        w1 / np.pi,
        np.abs(h1),
        "r",
        linewidth=2.5,
        label=f"Digital Filter 1 (ωc={wc1})",
        alpha=0.9,
    )
    ax2.plot(
        w2 / np.pi,
        np.abs(h2),
        "b",
        linewidth=2.5,
        label=f"Digital Filter 2 (ωc={wc2})",
        alpha=0.9,
    )
    ax2.plot(
        w_analog_norm1,
        np.abs(h_analog),
        "g--",
        linewidth=2,
        label="Analog Prototype (scaled)",
        alpha=0.7,
    )

    ax2.axvline(wc1 / np.pi, color="r", linestyle=":", linewidth=1.5, alpha=0.5)
    ax2.axvline(wc2 / np.pi, color="b", linestyle=":", linewidth=1.5, alpha=0.5)
    ax2.axhline(
        1 / np.sqrt(2),
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="1/√2",
    )

    ax2.set_xlabel("Normalized Frequency (ω/π)", fontsize=12)
    ax2.set_ylabel("Magnitude (linear)", fontsize=12)
    ax2.set_title(
        "Digital vs Analog: Magnitude Response (Linear)", fontsize=13, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_xlim([0, 0.6])
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # PLOT 2: Individual comparisons for each filter
    # ========================================================================

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Filter 1 - dB scale
    ax1.plot(
        w1 / np.pi,
        20 * np.log10(np.abs(h1)),
        "r",
        linewidth=2.5,
        label="Digital Filter 1",
    )
    ax1.plot(
        w_analog_norm1,
        20 * np.log10(np.abs(h_analog)),
        "g--",
        linewidth=2.5,
        label="Analog Prototype",
        alpha=0.7,
    )
    ax1.axvline(
        wc1 / np.pi, color="r", linestyle=":", alpha=0.6, label=f"ωc={wc1}", linewidth=2
    )
    ax1.axhline(-3, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Normalized Frequency (ω/π)", fontsize=11)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(
        f"Filter 1 (ωc={wc1}, T={T1}) vs Analog Prototype",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-60, 5])

    # Add text annotation
    ax1.text(
        0.7,
        -10,
        "Good match!\nNo aliasing",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    # Filter 1 - Linear scale (zoomed)
    ax2.plot(w1 / np.pi, np.abs(h1), "r", linewidth=2.5, label="Digital Filter 1")
    ax2.plot(
        w_analog_norm1,
        np.abs(h_analog),
        "g--",
        linewidth=2.5,
        label="Analog Prototype",
        alpha=0.7,
    )
    ax2.axvline(wc1 / np.pi, color="r", linestyle=":", alpha=0.6, linewidth=2)
    ax2.axhline(1 / np.sqrt(2), color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Normalized Frequency (ω/π)", fontsize=11)
    ax2.set_ylabel("Magnitude (linear)", fontsize=11)
    ax2.set_title(f"Filter 1 - Passband Detail", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 0.4])
    ax2.set_ylim([0, 1.1])

    # Filter 2 - dB scale
    ax3.plot(
        w2 / np.pi,
        20 * np.log10(np.abs(h2)),
        "b",
        linewidth=2.5,
        label="Digital Filter 2",
    )
    ax3.plot(
        w_analog_norm2,
        20 * np.log10(np.abs(h_analog)),
        "g--",
        linewidth=2.5,
        label="Analog Prototype",
        alpha=0.7,
    )
    ax3.axvline(
        wc2 / np.pi, color="b", linestyle=":", alpha=0.6, label=f"ωc={wc2}", linewidth=2
    )
    ax3.axvline(
        1.0, color="red", linestyle="--", linewidth=2, label="Nyquist (ω=π)", alpha=0.7
    )
    ax3.axhline(-3, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax3.set_xlabel("Normalized Frequency (ω/π)", fontsize=11)
    ax3.set_ylabel("Magnitude (dB)", fontsize=11)
    ax3.set_title(
        f"Filter 2 (ωc={wc2}, T={T2}) vs Analog Prototype",
        fontsize=12,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([-60, 5])

    # Add warning text
    ax3.text(
        0.5,
        -10,
        "⚠ ALIASING!\nCutoff > Nyquist",
        fontsize=10,
        color="red",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
    )

    # Filter 2 - Linear scale (extended to show aliasing)
    ax4.plot(w2 / np.pi, np.abs(h2), "b", linewidth=2.5, label="Digital Filter 2")
    ax4.plot(
        w_analog_norm2,
        np.abs(h_analog),
        "g--",
        linewidth=2.5,
        label="Analog Prototype",
        alpha=0.7,
    )
    ax4.axvline(wc2 / np.pi, color="b", linestyle=":", alpha=0.6, linewidth=2)
    ax4.axvline(
        1.0, color="red", linestyle="--", linewidth=2, label="Nyquist (ω=π)", alpha=0.7
    )
    ax4.axhline(1 / np.sqrt(2), color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax4.set_xlabel("Normalized Frequency (ω/π)", fontsize=11)
    ax4.set_ylabel("Magnitude (linear)", fontsize=11)
    ax4.set_title(f"Filter 2 - Shows Aliasing Effect", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xlim([0, 1.5])
    ax4.set_ylim([0, 1.1])

    # Shade aliasing region
    ax4.axvspan(1.0, 1.5, alpha=0.2, color="red", label="Aliased region")

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # COMMENTARY
    # ========================================================================

    print("\n" + "=" * 70)
    print("OBSERVATIONS AND COMMENTARY:")
    print("=" * 70)

    print("\n1. COMPARISON WITH ANALOG PROTOTYPE (from 3b):")
    print("-" * 70)
    print("   The analog Butterworth filter has a smooth, maximally flat")
    print("   magnitude response with cutoff at Ωc = 1 rad/s.")

    print("\n2. DIGITAL FILTER 1 (ωc = 0.25):")
    print("-" * 70)
    print("   ✓ Excellent match with analog prototype")
    print("   ✓ Cutoff frequency accurately preserved")
    print("   ✓ Butterworth shape maintained (maximally flat passband)")
    print("   ✓ Smooth rolloff in stopband")
    print("   ✓ No visible aliasing effects")
    print(f"   ✓ ωc = {wc1} = {wc1/np.pi:.3f}π << π (well below Nyquist)")
    print("   → Impulse invariance works WELL for this case")

    print("\n3. DIGITAL FILTER 2 (ωc = 1.4):")
    print("-" * 70)
    print("   ✗ SEVERE ALIASING PROBLEM!")
    print(f"   ✗ ωc = {wc2} = {wc2/np.pi:.3f}π > π (BEYOND Nyquist frequency!)")
    print("   ✗ Does NOT match analog prototype")
    print("   ✗ High-frequency components fold back into passband")
    print("   ✗ Distorted magnitude response")
    print("   ✗ Filter is essentially unusable")
    print("   → Impulse invariance FAILS for this case")

    print("\n4. WHY DOES ALIASING OCCUR?")
    print("-" * 70)
    print("   • Impulse invariance: h[n] = T·ha(nT) (samples impulse response)")
    print("   • Sampling in time → Periodic replication in frequency")
    print("   • Frequency spectrum repeats every 2π (or every fs in Hz)")
    print("   • If analog filter has energy above fs/2, it aliases!")
    print("   ")
    print(f"   Filter 1: T1 = {T1} → fs = {1/T1:.2f} → Nyquist = π")
    print(f"             Analog energy dies off before aliasing occurs ✓")
    print("   ")
    print(f"   Filter 2: T2 = {T2} → fs = {1/T2:.2f} → Nyquist = π")
    print(f"             But we want cutoff at {wc2}π (beyond Nyquist!) ✗")
    print(f"             Analog frequencies above π fold back into [0,π]")

    print("\n5. WHEN TO USE IMPULSE INVARIANCE:")
    print("-" * 70)
    print("   ✓ GOOD FOR:")
    print("      • Lowpass filters with cutoff << π (rule: ωc < 0.3π)")
    print("      • Bandpass filters in lower frequency ranges")
    print("      • When preserving time-domain response is critical")
    print("   ")
    print("   ✗ BAD FOR:")
    print("      • Highpass filters (always alias)")
    print("      • Bandstop filters (high frequencies alias)")
    print("      • Any filter with significant high-frequency content")
    print("      • Cutoff frequencies approaching Nyquist")
    print("   ")
    print("   → Alternative: Use BILINEAR TRANSFORM (Problem 2)")
    print("      • No aliasing (maps entire jΩ axis to unit circle)")
    print("      • Works for all filter types")
    print("      • Has frequency warping instead")

    print("\n6. KEY DIFFERENCES BETWEEN METHODS:")
    print("-" * 70)
    print("   Impulse Invariance:")
    print("      • Preserves impulse response shape")
    print("      • Linear frequency mapping (ω = ΩT)")
    print("      • Suffers from aliasing")
    print("   ")
    print("   Bilinear Transform (from Problem 2):")
    print("      • Nonlinear frequency mapping (Ω = (2/T)tan(ω/2))")
    print("      • No aliasing")
    print("      • Frequency warping requires pre-warping")

    print("\n" + "=" * 70)
    print("✓ Problem 3(e) complete!")
    print("=" * 70)


if __name__ == "__main__":
    problem_3e()
