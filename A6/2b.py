import numpy as np
from matplotlib import pyplot as plt

# Signal parameters
Nx = 28
Nh = 9
alpha = 0.9

# Generate signals
xn = alpha ** np.arange(Nx)
hn = np.ones(Nh)

# Expected output length
Ny = Nx + Nh - 1  # 36

# Direct convolution (reference)
yn_direct = np.convolve(xn, hn)

# Define DFT lengths to test
Nys = [2 * Ny, Ny, Ny // 2, Ny // 4]  # [72, 36, 18, 9]

# Create figure with 2x2 subplots for DFT comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, N in enumerate(Nys):
    print(f"\n{'='*60}")
    print(f"Testing DFT length N = {N}")
    print(f"{'='*60}")

    # Compute DFT
    X = np.fft.fft(xn, n=N)
    H = np.fft.fft(hn, n=N)

    # Multiply in frequency domain
    Y = X * H

    # Inverse DFT
    yn = np.fft.ifft(Y)
    yn = np.real(yn)  # Take real part

    # Check if correct
    is_correct = N >= Ny
    max_error = np.max(np.abs(yn[:Ny] - yn_direct)) if N >= Ny else float("inf")

    print(f"DFT length: {N}")
    print(f"Required minimum: {Ny}")
    print(f"Status: {'✓ Correct' if is_correct else '✗ Time-domain aliasing!'}")
    if is_correct:
        print(f"Max error vs direct convolution: {max_error:.2e}")

    # Plot
    axes[i].stem(np.arange(len(yn)), yn, basefmt=" ")

    # Overlay direct convolution if N >= Ny
    if N >= Ny:
        axes[i].plot(
            np.arange(len(yn_direct)),
            yn_direct,
            "r--",
            linewidth=2,
            label="Direct conv (reference)",
            alpha=0.7,
        )

    # Title with status
    status = "✓ Correct" if is_correct else "✗ Circular Convolution!"
    color = "green" if is_correct else "red"
    axes[i].set_title(
        f"DFT length = {N}  {status}", fontsize=13, fontweight="bold", color=color
    )
    axes[i].set_xlabel("$n$")
    axes[i].set_ylabel("$y[n]$")
    axes[i].grid(True, alpha=0.3)
    axes[i].legend()
    axes[i].set_xlim([-1, max(40, N) if N > Ny else 40])

plt.tight_layout()
plt.show()

# Additional visualization: Show circular convolution effect
N_short = 18
X_short = np.fft.fft(xn, n=N_short)
H_short = np.fft.fft(hn, n=N_short)
Y_short = X_short * H_short
yn_circular = np.real(np.fft.ifft(Y_short))

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Correct linear convolution
axes[0].stem(np.arange(Ny), yn_direct, basefmt=" ", linefmt="green", markerfmt="go")
axes[0].set_title(
    "Correct: Linear Convolution (N ≥ 36)",
    fontsize=14,
    fontweight="bold",
    color="green",
)
axes[0].set_xlabel("$n$")
axes[0].set_ylabel("$y[n]$")
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim([-1, 37])
axes[0].axvline(
    x=17.5, color="red", linestyle="--", linewidth=2, label="Wrapping point (N=18)"
)
axes[0].legend()

# Plot 2: Show how signal wraps
axes[1].stem(
    np.arange(18),
    yn_direct[:18],
    basefmt=" ",
    linefmt="blue",
    markerfmt="bo",
    label="First period [0:18]",
)
axes[1].stem(
    np.arange(18),
    yn_direct[18:36],
    basefmt=" ",
    linefmt="red",
    markerfmt="r^",
    label="Second period [18:36]",
)
axes[1].set_title(
    "How Signal Wraps with N=18 (Causes Overlap)", fontsize=14, fontweight="bold"
)
axes[1].set_xlabel("$n$")
axes[1].set_ylabel("$y[n]$")
axes[1].grid(True, alpha=0.3)
axes[1].legend()
axes[1].set_xlim([-1, 19])

# Plot 3: Circular convolution result
axes[2].stem(
    np.arange(N_short), yn_circular, basefmt=" ", linefmt="red", markerfmt="ro"
)
axes[2].set_title(
    "Wrong: Circular Convolution Result (N=18 < 36)",
    fontsize=14,
    fontweight="bold",
    color="red",
)
axes[2].set_xlabel("$n$")
axes[2].set_ylabel("$y[n]$")
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([-1, 19])

plt.tight_layout()
plt.show()

# Print numerical comparison
print("\n" + "=" * 60)
print("CIRCULAR CONVOLUTION: How Overlapping Happens (N=18)")
print("=" * 60)
print(
    f"{'n':<5} {'Correct y[n]':<15} {'y[n+18]':<15} {'Sum (Circular)':<15} {'Actual Circular':<15}"
)
print("=" * 60)
for n in range(min(18, Ny - 18)):
    sum_val = yn_direct[n] + (yn_direct[n + 18] if n + 18 < len(yn_direct) else 0)
    print(
        f"{n:<5} {yn_direct[n]:<15.6f} {yn_direct[n+18]:<15.6f} {sum_val:<15.6f} {yn_circular[n]:<15.6f}"
    )

# Summary table
print("\n" + "=" * 70)
print("SUMMARY: DFT-Based Convolution Results")
print("=" * 70)
print(
    f"{'DFT Length (N)':<15} {'Status':<10} {'Output Len':<12} {'Match Direct?':<15} {'Type'}"
)
print("=" * 70)
for N in Nys:
    X = np.fft.fft(xn, n=N)
    H = np.fft.fft(hn, n=N)
    Y = X * H
    yn = np.real(np.fft.ifft(Y))

    is_correct = N >= Ny
    status = "✅" if is_correct else "❌"
    match = "YES" if is_correct else "NO"
    conv_type = "Linear" if is_correct else "Circular"

    print(f"{N:<15} {status:<10} {len(yn):<12} {match:<15} {conv_type}")

print("=" * 70)
print(f"\nGolden Rule: N_DFT must be >= Nx + Nh - 1 = {Ny}")
