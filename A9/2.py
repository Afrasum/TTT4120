import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ============================================================================
# PROBLEM 2(a): Show that cut-off frequency is Ωc = 1/RC
# ============================================================================


def problem_2a():
    """
    Verify that the analog filter has cutoff frequency Ωc = 1/RC.
    This is a theoretical derivation - see writeup.
    """
    print("=" * 70)
    print("PROBLEM 2(a): Cut-off Frequency of Analog Filter")
    print("=" * 70)

    print("\nGiven: Ha(s) = (1/RC) / (s + 1/RC)")
    print("\nTo find cutoff frequency, we need: |Ha(jΩc)| = 1/√2")
    print("\nDERIVATION:")
    print("  |Ha(jΩ)| = (1/RC) / √((1/RC)² + Ω²)")
    print("  At cutoff: (1/RC) / √((1/RC)² + Ωc²) = 1/√2")
    print("  Squaring: 2(1/RC)² = (1/RC)² + Ωc²")
    print("  Solving: Ωc² = (1/RC)²")
    print(f"\n  ✓ Therefore: Ωc = 1/RC")
    print("\nThis is a first-order lowpass filter with time constant τ = RC")

    # Visualize the derivation with a generic RC filter
    RC = 1.0  # Normalized value
    Omega_c = 1 / RC

    # Create frequency response
    num = [1 / RC]
    den = [1, 1 / RC]
    w = np.linspace(0, 5, 1000)
    w_resp, h = signal.freqs(num, den, worN=w)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Magnitude response (dB)
    ax1.plot(w, 20 * np.log10(np.abs(h)), "b", linewidth=2)
    ax1.axvline(
        Omega_c, color="r", linestyle="--", label=f"Ωc = 1/RC = {Omega_c}", linewidth=2
    )
    ax1.axhline(-3, color="gray", linestyle=":", label="-3 dB", linewidth=1.5)
    ax1.plot(Omega_c, -3, "ro", markersize=10, label="Cutoff point")
    ax1.set_xlabel("Analog Frequency Ω (rad/s)", fontsize=11)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(
        "Analog RC Filter - Magnitude Response (dB)", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 5])
    ax1.set_ylim([-30, 2])

    # Plot 2: Linear magnitude
    ax2.plot(w, np.abs(h), "b", linewidth=2)
    ax2.axvline(
        Omega_c, color="r", linestyle="--", label=f"Ωc = 1/RC = {Omega_c}", linewidth=2
    )
    ax2.axhline(
        1 / np.sqrt(2), color="gray", linestyle=":", label="1/√2 ≈ 0.707", linewidth=1.5
    )
    ax2.plot(Omega_c, 1 / np.sqrt(2), "ro", markersize=10, label="Cutoff point")
    ax2.set_xlabel("Analog Frequency Ω (rad/s)", fontsize=11)
    ax2.set_ylabel("Magnitude (linear)", fontsize=11)
    ax2.set_title(
        "Analog RC Filter - Magnitude Response (Linear)", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 5])
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()


# ============================================================================
# PROBLEM 2(b): Derive frequency transformation from bilinear transform
# ============================================================================


def problem_2b():
    """
    Show that bilinear transform gives: Ω = (2/T)tan(ω/2)
    This is a theoretical derivation - see writeup.
    """
    print("\n" + "=" * 70)
    print("PROBLEM 2(b): Frequency Transformation")
    print("=" * 70)

    print("\nGiven bilinear transform: s = (2/T)(1 - z⁻¹)/(1 + z⁻¹)")
    print("\nDERIVATION:")
    print("  On unit circle: z = e^(jω)")
    print("  On imaginary axis: s = jΩ")
    print("\n  jΩ = (2/T) · (1 - e^(-jω))/(1 + e^(-jω))")
    print(
        "  Multiply by e^(jω/2): = (2/T) · (e^(jω/2) - e^(-jω/2))/(e^(jω/2) + e^(-jω/2))"
    )
    print("  Using Euler: = (2/T) · (2j·sin(ω/2))/(2·cos(ω/2))")
    print("  Simplify: = (2/T) · j·tan(ω/2)")
    print(f"\n  ✓ Therefore: Ω = (2/T)·tan(ω/2)")
    print("\nThis is the frequency warping relationship of the bilinear transform.")

    # Visualize the frequency warping
    T = 1  # Normalized
    w_digital = np.linspace(0, np.pi, 500)
    Omega_analog = (2 / T) * np.tan(w_digital / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frequency warping curve
    ax1.plot(
        w_digital / np.pi, Omega_analog, "purple", linewidth=3, label="Ω = 2·tan(ω/2)"
    )
    ax1.plot(
        [0, 1],
        [0, 2 * np.tan(np.pi / 2)],
        "k--",
        alpha=0.4,
        linewidth=2,
        label="Linear (no warping)",
    )

    # Mark specific points
    test_points = [0, 0.25, 0.5, 0.75, 1.0]
    for pt in test_points:
        w_pt = pt * np.pi
        Omega_pt = (2 / T) * np.tan(w_pt / 2)
        if not np.isinf(Omega_pt):
            ax1.plot(pt, Omega_pt, "ro", markersize=8)
            ax1.text(pt, Omega_pt + 0.3, f"{Omega_pt:.2f}", ha="center", fontsize=9)

    ax1.set_xlabel("Digital Frequency ω/π", fontsize=11)
    ax1.set_ylabel("Analog Frequency Ω (rad/s)", fontsize=11)
    ax1.set_title(
        "Bilinear Transform: Frequency Warping", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 10])

    # Plot 2: Zoom in on low frequencies
    w_zoom = np.linspace(0, 0.5 * np.pi, 500)
    Omega_zoom = (2 / T) * np.tan(w_zoom / 2)

    ax2.plot(w_zoom / np.pi, Omega_zoom, "purple", linewidth=3, label="Ω = 2·tan(ω/2)")
    ax2.plot(
        [0, 0.5],
        [0, 2 * np.tan(0.5 * np.pi / 2)],
        "k--",
        alpha=0.4,
        linewidth=2,
        label="Linear",
    )

    # Highlight ωc = 0.2π
    wc = 0.2 * np.pi
    Omega_c = (2 / T) * np.tan(wc / 2)
    ax2.plot(0.2, Omega_c, "ro", markersize=10, label=f"ωc=0.2π → Ωc={Omega_c:.3f}")
    ax2.axvline(0.2, color="r", linestyle=":", alpha=0.5)
    ax2.axhline(Omega_c, color="r", linestyle=":", alpha=0.5)

    ax2.set_xlabel("Digital Frequency ω/π", fontsize=11)
    ax2.set_ylabel("Analog Frequency Ω (rad/s)", fontsize=11)
    ax2.set_title(
        "Frequency Warping - Low Frequency Detail", fontsize=12, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 0.5])
    ax2.set_ylim([0, 2])

    plt.tight_layout()
    plt.show()

    # Print warping table
    print("\nFrequency Warping Table:")
    print("  ω (digital) → Ω (analog)")
    print("  " + "-" * 30)
    for wd_frac in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        wd = wd_frac * np.pi
        Omega = (2 / T) * np.tan(wd / 2)
        print(f"  {wd_frac:.1f}π ({wd:.3f})  → {Omega:.3f}")


# ============================================================================
# PROBLEM 2(c): Design digital filter using bilinear transform
# ============================================================================


def problem_2c():
    """
    Design a discrete lowpass filter with cutoff ωc = 0.2π by applying
    bilinear transform to the analog RC filter.
    """
    print("\n" + "=" * 70)
    print("PROBLEM 2(c): Digital Filter Design Using Bilinear Transform")
    print("=" * 70)

    # Parameters
    wc_digital = 0.2 * np.pi  # Desired digital cutoff frequency
    T = 1  # Normalized sampling period

    # Step 1: Pre-warp the digital frequency
    Omega_c = (2 / T) * np.tan(wc_digital / 2)

    print(f"\nDesired digital cutoff: ωc = 0.2π = {wc_digital:.4f} rad")
    print(f"Pre-warped analog cutoff: Ωc = 2·tan(0.1π) = {Omega_c:.5f} rad/s")

    # From Ωc = 1/RC
    RC = 1 / Omega_c
    print(f"Time constant: RC = {RC:.5f}")

    # Step 2: Design analog filter
    # Ha(s) = (1/RC) / (s + 1/RC)
    num_analog = [1 / RC]
    den_analog = [1, 1 / RC]

    print(f"\nAnalog filter: Ha(s) = {1/RC:.5f} / (s + {1/RC:.5f})")

    # Step 3: Apply bilinear transform
    # Using scipy's bilinear function
    num_digital, den_digital = signal.bilinear(num_analog, den_analog, fs=1 / T)

    print(f"\nDigital filter coefficients:")
    print(f"  Numerator:   b = {num_digital}")
    print(f"  Denominator: a = {den_digital}")

    # Normalize and round to match expected result
    b = num_digital
    a = den_digital

    print(f"\nTransfer function:")
    print(f"  H(z) = {b[0]:.3f}(1 + z⁻¹) / (1 - {-a[1]:.2f}z⁻¹)")
    print(f"  H(z) = 0.245(1 + z⁻¹) / (1 - 0.51z⁻¹)")

    # ========================================================================
    # VERIFICATION: Check magnitude at cutoff
    # ========================================================================

    print("\n" + "-" * 70)
    print("VERIFICATION:")
    print("-" * 70)

    # Digital filter frequency response
    w_digital, h_digital = signal.freqz(b, a, worN=8192)

    # Find magnitude at cutoff
    idx_cutoff = np.argmin(np.abs(w_digital - wc_digital))
    mag_at_cutoff_linear = np.abs(h_digital[idx_cutoff])
    mag_at_cutoff_dB = 20 * np.log10(mag_at_cutoff_linear)

    print(f"\nAt ωc = 0.2π:")
    print(f"  |H(ωc)| = {mag_at_cutoff_linear:.4f}")
    print(f"  1/√2    = {1/np.sqrt(2):.4f}")
    print(f"  Magnitude (dB): {mag_at_cutoff_dB:.2f} dB")
    print(f"  Expected: -3.01 dB")
    print(f"  ✓ Error: {abs(mag_at_cutoff_dB + 3.01):.3f} dB")

    # ========================================================================
    # PLOT 1: Digital filter response
    # ========================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Magnitude in dB
    ax1.plot(w_digital / np.pi, 20 * np.log10(np.abs(h_digital)), "b", linewidth=2)
    ax1.axvline(
        0.2, color="r", linestyle="--", label="ωc = 0.2π", alpha=0.7, linewidth=2
    )
    ax1.axhline(
        -3, color="gray", linestyle=":", label="-3 dB", alpha=0.5, linewidth=1.5
    )
    ax1.plot(
        0.2,
        mag_at_cutoff_dB,
        "ro",
        markersize=10,
        label=f"Cutoff: {mag_at_cutoff_dB:.2f} dB",
    )
    ax1.set_xlabel("Normalized Digital Frequency (ω/π)", fontsize=11)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(
        "Digital Filter H(z) - Magnitude Response (dB)", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-40, 5])

    # Linear magnitude
    ax2.plot(w_digital / np.pi, np.abs(h_digital), "b", linewidth=2)
    ax2.axvline(
        0.2, color="r", linestyle="--", label="ωc = 0.2π", alpha=0.7, linewidth=2
    )
    ax2.axhline(
        1 / np.sqrt(2),
        color="gray",
        linestyle=":",
        label="1/√2",
        alpha=0.5,
        linewidth=1.5,
    )
    ax2.plot(
        0.2,
        mag_at_cutoff_linear,
        "ro",
        markersize=10,
        label=f"|H(ωc)| = {mag_at_cutoff_linear:.4f}",
    )
    ax2.set_xlabel("Normalized Digital Frequency (ω/π)", fontsize=11)
    ax2.set_ylabel("Magnitude (linear)", fontsize=11)
    ax2.set_title(
        "Digital Filter H(z) - Magnitude Response (Linear)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # OPTIONAL: Compare with analog prototype
    # ========================================================================

    print("\n" + "-" * 70)
    print("OPTIONAL: Analog vs Digital Comparison")
    print("-" * 70)

    # Analog filter frequency response
    w_analog = np.linspace(0, 3, 1000)
    w_plot, h_analog = signal.freqs(num_analog, den_analog, worN=w_analog)

    # ========================================================================
    # PLOT 2: Analog prototype response
    # ========================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Analog magnitude in dB
    ax1.plot(w_analog, 20 * np.log10(np.abs(h_analog)), "g", linewidth=2)
    ax1.axvline(
        Omega_c,
        color="r",
        linestyle="--",
        label=f"Ωc = {Omega_c:.3f}",
        alpha=0.7,
        linewidth=2,
    )
    ax1.axhline(
        -3, color="gray", linestyle=":", label="-3 dB", alpha=0.5, linewidth=1.5
    )
    ax1.set_xlabel("Analog Frequency Ω (rad/s)", fontsize=11)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(
        "Analog Prototype Ha(s) - Magnitude Response (dB)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 3])
    ax1.set_ylim([-40, 5])

    # Analog linear magnitude
    ax2.plot(w_analog, np.abs(h_analog), "g", linewidth=2)
    ax2.axvline(
        Omega_c,
        color="r",
        linestyle="--",
        label=f"Ωc = {Omega_c:.3f}",
        alpha=0.7,
        linewidth=2,
    )
    ax2.axhline(
        1 / np.sqrt(2),
        color="gray",
        linestyle=":",
        label="1/√2",
        alpha=0.5,
        linewidth=1.5,
    )
    ax2.set_xlabel("Analog Frequency Ω (rad/s)", fontsize=11)
    ax2.set_ylabel("Magnitude (linear)", fontsize=11)
    ax2.set_title(
        "Analog Prototype Ha(s) - Magnitude Response (Linear)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 3])
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # PLOT 3: Side-by-side comparison
    # ========================================================================

    # Map digital frequencies to analog using warping formula
    Omega_warped = (2 / T) * np.tan(w_digital / 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Overlay both responses on frequency axis
    ax1.plot(
        w_digital / np.pi,
        20 * np.log10(np.abs(h_digital)),
        "b",
        linewidth=2,
        label="Digital H(z)",
    )
    ax1.plot(
        w_analog / np.pi,
        20 * np.log10(np.abs(h_analog)),
        "g--",
        linewidth=2,
        label="Analog Ha(s)",
        alpha=0.7,
    )
    ax1.axvline(
        0.2, color="r", linestyle=":", alpha=0.7, linewidth=2, label="ωc = 0.2π"
    )
    ax1.axvline(
        Omega_c / np.pi,
        color="orange",
        linestyle=":",
        alpha=0.7,
        linewidth=2,
        label=f"Ωc/π = {Omega_c/np.pi:.3f}",
    )
    ax1.set_xlabel("Normalized Frequency (ω/π or Ω/π)", fontsize=11)
    ax1.set_ylabel("Magnitude (dB)", fontsize=11)
    ax1.set_title(
        "Digital vs Analog - Direct Comparison", fontsize=12, fontweight="bold"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([-40, 5])

    # Frequency warping effect
    ax2.plot(
        w_digital / np.pi,
        20 * np.log10(np.abs(h_digital)),
        "b",
        linewidth=2,
        label="Digital H(z) vs ω",
    )
    # Plot digital filter vs warped frequency to match analog
    ax2.plot(
        Omega_warped / np.pi,
        20 * np.log10(np.abs(h_digital)),
        "purple",
        linewidth=2,
        linestyle="--",
        label="Digital H(z) vs warped Ω",
        alpha=0.7,
    )
    ax2.set_xlabel("Normalized Frequency (ω/π or Ω/π)", fontsize=11)
    ax2.set_ylabel("Magnitude (dB)", fontsize=11)
    ax2.set_title("Effect of Frequency Warping", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([-40, 5])

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("✓ Digital filter design complete!")
    print("=" * 70)

    return b, a


# ============================================================================
# MAIN: Run all parts of Problem 2
# ============================================================================


def main():
    """Run all parts of Problem 2."""

    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  PROBLEM 2: BILINEAR TRANSFORM".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    # Part (a): Theoretical - show cutoff frequency
    problem_2a()

    # Part (b): Theoretical - derive frequency transformation
    problem_2b()

    # Part (c): Design digital filter and compare
    b, a = problem_2c()

    print("\n" + "#" * 70)
    print("#" + "  PROBLEM 2 COMPLETE!".center(68) + "#")
    print("#" * 70 + "\n")

    return b, a


if __name__ == "__main__":
    b, a = main()
