import numpy as np
from matplotlib import pyplot as plt

# Signal parameters
Nx = 28
alpha = 0.9

# Generate the signal x[n] = 0.9^n for n = 0 to 27
n = np.arange(0, Nx)
xn = alpha**n

# Define the four DFT lengths
N1 = Nx // 4  # 7 points
N2 = Nx // 2  # 14 points
N3 = Nx  # 28 points
N4 = 2 * Nx  # 56 points

# Compute DFT for each length using FFT algorithm
Xf1 = np.fft.fft(xn, n=N1)
Xf2 = np.fft.fft(xn, n=N2)
Xf3 = np.fft.fft(xn, n=N3)
Xf4 = np.fft.fft(xn, n=N4)

# Print the DFT lengths and first few values
print(f"DFT Length = {N1}: {len(Xf1)} points")
print(f"First 3 values: {Xf1[:3]}\n")

print(f"DFT Length = {N2}: {len(Xf2)} points")
print(f"First 3 values: {Xf2[:3]}\n")

print(f"DFT Length = {N3}: {len(Xf3)} points")
print(f"First 3 values: {Xf3[:3]}\n")

print(f"DFT Length = {N4}: {len(Xf4)} points")
print(f"First 3 values: {Xf4[:3]}\n")


# Signal parameters
Nx = 28
alpha = 0.9
n = np.arange(0, Nx)
xn = alpha**n

# Define DFT lengths
Nx_vec = [Nx // 4, Nx // 2, Nx, 2 * Nx]  # [7, 14, 28, 56]
labels = ["N=7 (Aliased)", "N=14 (Aliased)", "N=28 (Correct)", "N=56 (High Res)"]

# Compute and compare
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (N_dft, label) in enumerate(zip(Nx_vec, labels)):
    Xk = np.fft.fft(xn, n=N_dft)

    # Create frequency axis
    freq = np.arange(N_dft) / N_dft

    # Plot magnitude
    axes[i].stem(freq, np.abs(Xk), basefmt=" ")
    axes[i].set_title(f"{label}: X[0] = {Xk[0].real:.3f}", fontsize=12)
    axes[i].set_xlabel("Normalized Frequency (f)")
    axes[i].set_ylabel("|X[k]|")
    axes[i].grid(True, alpha=0.3)
    axes[i].set_xlim([0, 0.5])  # Only show 0 to 0.5

plt.tight_layout()
plt.show()

# Print comparison table
print("\nComparison Table:")
print("=" * 60)
print(f"{'DFT Length':<12} {'X[0]':<15} {'|X[1]|':<15} {'Status'}")
print("=" * 60)
for N_dft in Nx_vec:
    Xk = np.fft.fft(xn, n=N_dft)
    status = "✅ Correct" if N_dft >= Nx else "❌ Aliased"
    print(f"{N_dft:<12} {Xk[0].real:<15.3f} {np.abs(Xk[1]):<15.3f} {status}")
