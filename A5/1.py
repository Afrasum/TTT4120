import matplotlib.pyplot as plt
import numpy as np

# Define the ranges
n_range = np.arange(0, 51)  # n from 0 to 50
l_range = np.arange(-50, 51)  # l from -50 to 50
f_range = np.linspace(-0.5, 0.5, 1000)  # f from -0.5 to 0.5

# Values of a to test
a_values = [0.5, 0.9, -0.9]

# Create figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(
    "Signal x[n], Autocorrelation rxx(l), and Energy Density Spectrum Sxx(f)",
    fontsize=16,
    fontweight="bold",
)

for idx, a in enumerate(a_values):
    # Calculate x[n]
    x_n = a**n_range

    # Calculate rxx(l)
    rxx_l = (a ** np.abs(l_range)) / (1 - a**2)

    # Calculate Sxx(f)
    Sxx_f = 1 / (1 + a**2 - 2 * a * np.cos(2 * np.pi * f_range))

    # Plot x[n]
    axes[idx, 0].stem(n_range, x_n, basefmt=" ")
    axes[idx, 0].set_xlabel("n")
    axes[idx, 0].set_ylabel("x[n]")
    axes[idx, 0].set_title(f"Signal x[n] for a = {a}")
    axes[idx, 0].grid(True, alpha=0.3)
    axes[idx, 0].set_xlim([0, 50])

    # Plot rxx(l)
    axes[idx, 1].stem(l_range, rxx_l, basefmt=" ")
    axes[idx, 1].set_xlabel("l (lag)")
    axes[idx, 1].set_ylabel("rxx(l)")
    axes[idx, 1].set_title(f"Autocorrelation rxx(l) for a = {a}")
    axes[idx, 1].grid(True, alpha=0.3)
    axes[idx, 1].set_xlim([-50, 50])

    # Plot Sxx(f)
    axes[idx, 2].plot(f_range, Sxx_f, linewidth=2)
    axes[idx, 2].set_xlabel("f (normalized frequency)")
    axes[idx, 2].set_ylabel("Sxx(f)")
    axes[idx, 2].set_title(f"Energy Density Spectrum Sxx(f) for a = {a}")
    axes[idx, 2].grid(True, alpha=0.3)
    axes[idx, 2].set_xlim([-0.5, 0.5])

plt.tight_layout()
plt.savefig("problem_1c_plots.png", dpi=300, bbox_inches="tight")
print("Plots saved successfully!")

# Print some key values for analysis
print("\n" + "=" * 60)
print("KEY VALUES FOR ANALYSIS")
print("=" * 60)

for a in a_values:
    print(f"\nFor a = {a}:")
    print(f"  Energy (rxx(0)): {1/(1-a**2):.4f}")
    print(f"  rxx(10): {(a**10)/(1-a**2):.4f}")
    print(f"  Sxx(0) [DC component]: {1/(1+a**2-2*a):.4f}")
    print(f"  Sxx(0.5) [Nyquist freq]: {1/(1+a**2+2*a):.4f}")
