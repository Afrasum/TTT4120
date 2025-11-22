import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def problem_3b():
    """
    Plot the magnitude response of the 2nd order Butterworth filter.
    """
    print("=" * 70)
    print("PROBLEM 3(b): Magnitude Response of Butterworth Filter")
    print("=" * 70)

    # Transfer function: Ha(s) = 1 / (s^2 + sqrt(2)*s + 1)
    num = [1]
    den = [1, np.sqrt(2), 1]

    print(f"\nTransfer function:")
    print(f"  Ha(s) = 1 / (s² + √2·s + 1)")
    print(f"  Ha(s) = 1 / (s² + {np.sqrt(2):.4f}·s + 1)")

    # Compute frequency response using freqs
    w = np.logspace(-1, 1, 1000)  # From 0.1 to 10 rad/s (logarithmic)
    w_resp, h = signal.freqs(num, den, worN=w)

    # Find magnitude at cutoff frequency Ωc = 1
    idx_cutoff = np.argmin(np.abs(w - 1.0))
    mag_at_cutoff = np.abs(h[idx_cutoff])
    mag_at_cutoff_dB = 20 * np.log10(mag_at_cutoff)

    print(f"\nVerification at Ωc = 1:")
    print(f"  |Ha(1)|      = {mag_at_cutoff:.4f}")
    print(f"  1/√2         = {1/np.sqrt(2):.4f}")
    print(f"  Magnitude    = {mag_at_cutoff_dB:.2f} dB")
    print(f"  Expected     = -3.01 dB")
    print(f"  ✓ Confirmed: This is a Butterworth filter with Ωc = 1")

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========================================================================
    # Plot 1: Magnitude in dB (logarithmic frequency scale)
    # ========================================================================

    ax1.semilogx(w, 20 * np.log10(np.abs(h)), "b", linewidth=2.5, label="|Ha(jΩ)|")
    ax1.axvline(1.0, color="r", linestyle="--", linewidth=2, label="Ωc = 1")
    ax1.axhline(-3, color="gray", linestyle=":", linewidth=1.5, label="-3 dB")
    ax1.plot(
        1.0,
        mag_at_cutoff_dB,
        "ro",
        markersize=10,
        label=f"Cutoff: {mag_at_cutoff_dB:.2f} dB",
    )

    # Add annotation
    ax1.annotate(
        f"Ωc = 1\n{mag_at_cutoff_dB:.2f} dB",
        xy=(1.0, mag_at_cutoff_dB),
        xytext=(2, -10),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    # Mark -40 dB/decade slope
    w_slope = np.array([1, 10])
    h_slope = np.array([mag_at_cutoff_dB, mag_at_cutoff_dB - 40])
    ax1.plot(w_slope, h_slope, "g--", linewidth=1.5, label="-40 dB/decade")

    ax1.set_xlabel("Frequency Ω (rad/s)", fontsize=12)
    ax1.set_ylabel("Magnitude (dB)", fontsize=12)
    ax1.set_title(
        "2nd Order Butterworth Filter - Magnitude Response (dB)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, which="both")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_xlim([0.1, 10])
    ax1.set_ylim([-60, 5])

    # ========================================================================
    # Plot 2: Magnitude in linear scale
    # ========================================================================

    w_linear = np.linspace(0, 5, 1000)
    w_lin_resp, h_linear = signal.freqs(num, den, worN=w_linear)

    ax2.plot(w_linear, np.abs(h_linear), "b", linewidth=2.5, label="|Ha(jΩ)|")
    ax2.axvline(1.0, color="r", linestyle="--", linewidth=2, label="Ωc = 1")
    ax2.axhline(
        1 / np.sqrt(2), color="gray", linestyle=":", linewidth=1.5, label="1/√2 ≈ 0.707"
    )
    ax2.plot(
        1.0, mag_at_cutoff, "ro", markersize=10, label=f"|Ha(1)| = {mag_at_cutoff:.4f}"
    )

    ax2.annotate(
        f"Cutoff point\nΩc = 1\n|Ha| = {mag_at_cutoff:.3f}",
        xy=(1.0, mag_at_cutoff),
        xytext=(2.5, 0.5),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
    )

    ax2.text(
        0.3,
        0.95,
        "Maximally flat\npassband",
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    ax2.set_xlabel("Frequency Ω (rad/s)", fontsize=12)
    ax2.set_ylabel("Magnitude (linear)", fontsize=12)
    ax2.set_title(
        "2nd Order Butterworth Filter - Magnitude Response (Linear)",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_xlim([0, 5])
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # Additional plot: Show Butterworth characteristic |H|^2 = 1/(1+Ω^4)
    # ========================================================================

    fig, ax = plt.subplots(figsize=(10, 6))

    Omega = np.linspace(0, 3, 1000)
    H_squared_theory = 1 / (1 + Omega**4)
    H_theory = np.sqrt(H_squared_theory)

    w_comp, h_comp = signal.freqs(num, den, worN=Omega)

    ax.plot(Omega, H_theory, "b", linewidth=2.5, label="Theoretical: |H|² = 1/(1+Ω⁴)")
    ax.plot(Omega, np.abs(h_comp), "r--", linewidth=2, label="Computed from Ha(s)")
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Ωc = 1")
    ax.axhline(1 / np.sqrt(2), color="gray", linestyle=":", linewidth=1.5, label="1/√2")
    ax.plot(1.0, 1 / np.sqrt(2), "ko", markersize=10, label="Cutoff point")

    ax.set_xlabel("Frequency Ω (rad/s)", fontsize=12)
    ax.set_ylabel("Magnitude |Ha(jΩ)|", fontsize=12)
    ax.set_title(
        "Verification: Butterworth Characteristic", fontsize=13, fontweight="bold"
    )
    ax.grid(True)
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 1.1])

    textstr = "For 2nd order Butterworth:\n|Ha(jΩ)|² = 1/(1 + Ω⁴)\nΩc = 1 rad/s"
    props = dict(boxstyle="round", facecolor="wheat")
    ax.text(
        0.65,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
    )

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("✓ Magnitude response plotted successfully!")
    print("=" * 70)


if __name__ == "__main__":
    problem_3b()
