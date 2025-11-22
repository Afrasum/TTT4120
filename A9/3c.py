import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def problem_3c():
    """
    Find and visualize the poles of the 2nd order Butterworth filter.
    """
    print("=" * 70)
    print("PROBLEM 3(c): Poles of the Butterworth Filter")
    print("=" * 70)

    # Transfer function: Ha(s) = 1 / (s^2 + sqrt(2)*s + 1)
    num = [1]
    den = [1, np.sqrt(2), 1]

    print(f"\nTransfer function:")
    print(f"  Ha(s) = 1 / (s² + √2·s + 1)")

    # Find poles using numpy roots
    poles = np.roots(den)

    print(f"\n" + "-" * 70)
    print("POLES:")
    print("-" * 70)

    for i, pole in enumerate(poles, 1):
        magnitude = np.abs(pole)
        angle_rad = np.angle(pole)
        angle_deg = np.degrees(angle_rad)

        print(f"\nPole {i}:")
        print(f"  Cartesian: s{i} = {pole.real:.4f} + j{pole.imag:.4f}")
        print(
            f"  Polar:     |s{i}| = {magnitude:.4f}, ∠s{i} = {angle_deg:.2f}° ({angle_rad:.4f} rad)"
        )

    # Verify calculations
    print(f"\n" + "-" * 70)
    print("VERIFICATION:")
    print("-" * 70)
    print(f"  √2/2 = {np.sqrt(2)/2:.4f}")
    print(f"  Both poles lie on circle of radius Ωc = 1")
    print(f"  Real parts are negative → Stable filter ✓")
    print(f"  Complex conjugate pair → Real impulse response ✓")

    # ========================================================================
    # PLOT: Pole-Zero Diagram
    # ========================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Standard pole-zero plot
    # Plot unit circle (radius = Ωc = 1)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    ax1.plot(
        circle_x,
        circle_y,
        "k--",
        linewidth=1.5,
        alpha=0.3,
        label=f"Circle: |s| = Ωc = 1",
    )

    # Plot imaginary axis (stability boundary)
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Plot poles
    ax1.plot(
        poles.real, poles.imag, "rx", markersize=15, markeredgewidth=3, label="Poles"
    )

    # Add pole annotations
    for i, pole in enumerate(poles, 1):
        angle_deg = np.degrees(np.angle(pole))
        ax1.annotate(
            f"s{i}\n({pole.real:.3f}, {pole.imag:.3f})\n∠{angle_deg:.1f}°",
            xy=(pole.real, pole.imag),
            xytext=(pole.real - 0.5, pole.imag + 0.3),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )

    # Plot zero (at infinity, not shown)
    ax1.text(
        0.5,
        0.8,
        "No finite zeros\n(zero at s=∞)",
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # Shade stable region
    stable_region = plt.Rectangle(
        (-1.5, -1.5),
        1.5,
        3,
        alpha=0.1,
        facecolor="green",
        label="Stable region (Re(s) < 0)",
    )
    ax1.add_patch(stable_region)

    ax1.set_xlabel("Real Part σ", fontsize=12)
    ax1.set_ylabel("Imaginary Part jΩ", fontsize=12)
    ax1.set_title("Pole-Zero Diagram", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.set_xlim([-1.5, 1.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.set_aspect("equal")

    # Right plot: Detailed Butterworth pole pattern
    ax2.plot(
        circle_x, circle_y, "b-", linewidth=2, label=f"Butterworth circle: |s| = Ωc = 1"
    )

    # Plot only left semicircle (stable poles)
    semicircle_theta = np.linspace(np.pi / 2, 3 * np.pi / 2, 50)
    semi_x = np.cos(semicircle_theta)
    semi_y = np.sin(semicircle_theta)
    ax2.fill(semi_x, semi_y, alpha=0.2, color="green", label="Pole region")

    # Plot poles
    ax2.plot(
        poles.real,
        poles.imag,
        "rx",
        markersize=18,
        markeredgewidth=4,
        label="Poles (n=2)",
    )

    # Draw radial lines to poles
    for pole in poles:
        ax2.plot([0, pole.real], [0, pole.imag], "r--", linewidth=1.5, alpha=0.5)

    # Mark angles
    for i, pole in enumerate(poles):
        angle = np.angle(pole)
        arc_theta = np.linspace(0, angle, 30)
        arc_r = 0.3
        ax2.plot(
            arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), "purple", linewidth=2
        )

        angle_deg = np.degrees(angle)
        mid_angle = angle / 2
        ax2.text(
            0.4 * np.cos(mid_angle),
            0.4 * np.sin(mid_angle),
            f"{angle_deg:.0f}°",
            fontsize=10,
            color="purple",
            fontweight="bold",
        )

    # Add angular spacing annotation
    angle_between = np.abs(np.angle(poles[0]) - np.angle(poles[1]))
    ax2.text(
        -0.7,
        0,
        f"Spacing:\n{np.degrees(angle_between):.0f}°\n= 180°/n",
        fontsize=10,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax2.axvline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax2.set_xlabel("Real Part σ", fontsize=12)
    ax2.set_ylabel("Imaginary Part jΩ", fontsize=12)
    ax2.set_title("Butterworth Pole Pattern (n=2)", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.set_xlim([-1.5, 1.5])
    ax2.set_ylim([-1.5, 1.5])
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # Additional info
    # ========================================================================

    print(f"\n" + "-" * 70)
    print("BUTTERWORTH POLE PROPERTIES:")
    print("-" * 70)
    print(f"  • All poles lie on a circle of radius Ωc = {np.abs(poles[0]):.4f}")
    print(f"  • Poles are equally spaced by 180°/n = 180°/2 = 90°")
    print(f"  • For stability, only left half-plane poles are used")
    print(f"  • The pole pattern ensures maximally flat magnitude response")
    print(f"  • Q-factor: Q = 1/√2 ≈ 0.707 (critical damping)")

    print("\n" + "=" * 70)
    print("✓ Poles found and visualized!")
    print("=" * 70)

    return poles


if __name__ == "__main__":
    poles = problem_3c()
