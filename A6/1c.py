import matplotlib.pyplot as plt
import numpy as np

# Signal parameters
Nx = 28
alpha = 0.9
n = np.arange(0, Nx)
xn = alpha**n

# Define DFT lengths
Nx_vec = [Nx // 4, Nx // 2, Nx, 2 * Nx]  # [7, 14, 28, 56]

# Create table
print("=" * 70)
print(f"{'DFT Length (N)':<15} {'k':<10} {'f = k/N':<15} {'Frequency Resolution (Δf)'}")
print("=" * 70)

for N_dft in Nx_vec:
    k = 1  # We want k=1
    f = k / N_dft
    delta_f = 1 / N_dft
    print(f"{N_dft:<15} {k:<10} {f:<15.6f} {delta_f:.6f}")

print("=" * 70)

# Visualize frequency sampling
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

for i, N_dft in enumerate(Nx_vec):
    # Compute DFT
    Xk = np.fft.fft(xn, n=N_dft)

    # Create frequency axis: f = k/N
    k_indices = np.arange(N_dft)
    freq = k_indices / N_dft

    # Plot DFT samples
    axes[i].stem(freq, np.abs(Xk), basefmt=" ", linefmt="blue", markerfmt="bo")

    # Highlight k=1
    axes[i].stem(
        [freq[1]],
        [np.abs(Xk[1])],
        basefmt=" ",
        linefmt="red",
        markerfmt="ro",
        label=f"k=1 → f={freq[1]:.3f}",
    )

    axes[i].set_title(
        f"N = {N_dft}: Frequency resolution Δf = {1/N_dft:.3f}", fontsize=12
    )
    axes[i].set_xlabel("f (Normalized Frequency)")
    axes[i].set_ylabel("|X[k]|")
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([0, 0.5])  # Only show first half

plt.tight_layout()
plt.show()
