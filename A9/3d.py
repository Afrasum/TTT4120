import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def demonstrate_impulse_invariance_transform():
    """
    Demonstrate the step-by-step transformation from analog to digital
    using impulse invariance.
    """
    print("=" * 70)
    print("IMPULSE INVARIANCE: STEP-BY-STEP DEMONSTRATION")
    print("=" * 70)

    # Analog filter
    num_analog = [1]
    den_analog = [1, np.sqrt(2), 1]

    # Find poles
    poles_analog = np.roots(den_analog)
    s1 = poles_analog[0]
    s2 = poles_analog[1]

    print(f"\nSTEP 1: Analog filter poles")
    print(f"  s1 = {s1.real:.4f} + j{s1.imag:.4f}")
    print(f"  s2 = {s2.real:.4f} + j{s2.imag:.4f}")

    # Partial fraction expansion to find residues A and A*
    residues, poles, k = signal.residue(num_analog, den_analog)
    A = residues[0]
    A_conj = residues[1]

    print(f"\nSTEP 2: Partial fraction residues")
    print(f"  A  = {A.real:.4f} + j{A.imag:.4f}")
    print(f"  A* = {A_conj.real:.4f} + j{A_conj.imag:.4f}")
    print(f"\n  Ha(s) = A/(s-s1) + A*/(s-s2)")

    # Sample the impulse response
    T = 0.25  # Sampling period
    n = np.arange(0, 50)

    print(f"\nSTEP 3: Sample analog impulse response with T = {T}")
    print(f"  h[n] = T·ha(nT) = T(A·e^(s1·nT) + A*·e^(s2·nT))")

    # Calculate sampled impulse response
    h_analog_sampled = T * (A * np.exp(s1 * n * T) + A_conj * np.exp(s2 * n * T))

    print(f"  First few samples of h[n]:")
    for i in range(5):
        print(f"    h[{i}] = {h_analog_sampled[i].real:.6f}")

    # Map poles to z-plane
    z1 = np.exp(s1 * T)
    z2 = np.exp(s2 * T)

    print(f"\nSTEP 4: Map analog poles to digital poles")
    print(f"  z1 = e^(s1·T) = e^({s1.real:.4f}+j{s1.imag:.4f})·{T}")
    print(f"     = {z1.real:.4f} + j{z1.imag:.4f}")
    print(f"  z2 = e^(s2·T) = e^({s2.real:.4f}+j{s2.imag:.4f})·{T}")
    print(f"     = {z2.real:.4f} + j{z2.imag:.4f}")
    print(f"\n  |z1| = {np.abs(z1):.4f}  (inside unit circle = stable)")
    print(f"  |z2| = {np.abs(z2):.4f}  (inside unit circle = stable)")

    # Build digital transfer function
    print(f"\nSTEP 5: Construct digital transfer function")
    print(f"  H(z) = (A·T)/(1 - z1·z^-1) + (A*·T)/(1 - z2·z^-1)")

    # Convert to standard form using scipy
    system_d = signal.cont2discrete((num_analog, den_analog), T, method="impulse")
    b_digital = system_d[0].flatten()
    a_digital = system_d[1].flatten()

    print(f"\n  After combining terms:")
    print(f"  Numerator:   b = {b_digital}")
    print(f"  Denominator: a = {a_digital}")

    # Verify by computing digital impulse response
    _, h_digital = signal.dimpulse((b_digital, a_digital, T), n=50)
    h_digital = h_digital[0].flatten()

    print(f"\nSTEP 6: Verify - compare sampled analog vs digital impulse response")
    print(f"  Sample  |  Analog h[n]  |  Digital h[n]  |  Difference")
    print(f"  " + "-" * 60)
    for i in range(5):
        diff = np.abs(h_analog_sampled[i].real - h_digital[i])
        print(
            f"    {i}    |  {h_analog_sampled[i].real:11.6f}  |  {h_digital[i]:11.6f}  |  {diff:.2e}"
        )

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Analog impulse response (continuous)
    t = np.linspace(0, 10, 1000)
    h_analog_continuous = (A * np.exp(s1 * t) + A_conj * np.exp(s2 * t)).real

    axes[0, 0].plot(t, h_analog_continuous, "b-", linewidth=2, label="ha(t)")
    axes[0, 0].stem(
        n * T,
        h_analog_sampled.real,
        linefmt="r-",
        markerfmt="ro",
        basefmt=" ",
        label="Sampled: h[n]=T·ha(nT)",
    )
    axes[0, 0].set_xlabel("Time", fontsize=11)
    axes[0, 0].set_ylabel("Amplitude", fontsize=11)
    axes[0, 0].set_title(
        "Step 3: Sampling Analog Impulse Response", fontsize=12, fontweight="bold"
    )
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, 10])

    # Plot 2: Pole mapping
    theta = np.linspace(0, 2 * np.pi, 100)

    axes[0, 1].plot(
        np.cos(theta),
        np.sin(theta),
        "k--",
        linewidth=1.5,
        alpha=0.4,
        label="Unit circle",
    )
    axes[0, 1].plot(
        poles_analog.real,
        poles_analog.imag,
        "bs",
        markersize=12,
        label="Analog poles (s-plane)",
    )
    axes[0, 1].plot(
        [z1.real, z2.real],
        [z1.imag, z2.imag],
        "ro",
        markersize=12,
        label="Digital poles (z-plane)",
    )

    # Draw arrows showing mapping
    for s_pole, z_pole in zip(poles_analog, [z1, z2]):
        axes[0, 1].annotate(
            "",
            xy=(z_pole.real, z_pole.imag),
            xytext=(s_pole.real, s_pole.imag),
            arrowprops=dict(arrowstyle="->", lw=2, color="green"),
        )

    axes[0, 1].text(
        0,
        1.3,
        "z = e^(sT)",
        fontsize=12,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    axes[0, 1].axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    axes[0, 1].axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    axes[0, 1].set_xlabel("Real Part", fontsize=11)
    axes[0, 1].set_ylabel("Imaginary Part", fontsize=11)
    axes[0, 1].set_title("Step 4: Pole Mapping (s → z)", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([-1.5, 1.5])
    axes[0, 1].set_ylim([-1.5, 1.5])
    axes[0, 1].set_aspect("equal")

    # Plot 3: Comparison of impulse responses
    axes[1, 0].stem(
        n,
        h_analog_sampled.real,
        linefmt="b-",
        markerfmt="bo",
        basefmt=" ",
        label="Sampled analog: T·ha(nT)",
    )
    axes[1, 0].stem(
        n,
        h_digital,
        linefmt="r--",
        markerfmt="rx",
        basefmt=" ",
        label="Digital filter: h[n]",
    )
    axes[1, 0].set_xlabel("Sample n", fontsize=11)
    axes[1, 0].set_ylabel("Amplitude", fontsize=11)
    axes[1, 0].set_title(
        "Step 6: Verification (Should Match!)", fontsize=12, fontweight="bold"
    )
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, 30])

    # Plot 4: Frequency response comparison
    w_analog = np.linspace(0, np.pi / T, 1000)
    _, h_analog_freq = signal.freqs(num_analog, den_analog, worN=w_analog)

    w_digital, h_digital_freq = signal.freqz(b_digital, a_digital, worN=8192)

    axes[1, 1].plot(
        w_analog * T / np.pi,
        20 * np.log10(np.abs(h_analog_freq)),
        "b-",
        linewidth=2,
        label="Analog Ha(jΩ)",
    )
    axes[1, 1].plot(
        w_digital / np.pi,
        20 * np.log10(np.abs(h_digital_freq)),
        "r--",
        linewidth=2,
        label="Digital H(e^jω)",
    )
    axes[1, 1].set_xlabel("Normalized Frequency (ω/π)", fontsize=11)
    axes[1, 1].set_ylabel("Magnitude (dB)", fontsize=11)
    axes[1, 1].set_title(
        "Frequency Response Comparison", fontsize=12, fontweight="bold"
    )
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([-60, 5])

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("✓ Impulse invariance transformation demonstrated!")
    print("=" * 70)


if __name__ == "__main__":
    demonstrate_impulse_invariance_transform()
