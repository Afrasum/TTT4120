import time

import matplotlib.pyplot as plt
import numpy as np

# ===== Part 1: Complexity Comparison =====
print("=" * 70)
print("PART 1: COMPUTATIONAL COMPLEXITY")
print("=" * 70)

N_values = [2**i for i in range(2, 13)]  # 4, 8, 16, ..., 4096

dft_ops = [N**2 for N in N_values]
fft_ops = [N * np.log2(N) for N in N_values]
speedup = [dft / fft for dft, fft in zip(dft_ops, fft_ops)]

print(f"\n{'N':<8} {'DFT (N²)':<15} {'FFT (N log N)':<15} {'Speedup'}")
print("-" * 70)
for N, dft, fft, speed in zip(N_values, dft_ops, fft_ops, speedup):
    print(f"{N:<8} {dft:<15.0f} {fft:<15.0f} {speed:.1f}x")

# Plot complexity
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear scale
axes[0].plot(N_values, dft_ops, "r-o", linewidth=2, markersize=8, label="DFT: O(N²)")
axes[0].plot(
    N_values, fft_ops, "b-s", linewidth=2, markersize=8, label="FFT: O(N log N)"
)
axes[0].set_title(
    "Computational Complexity: Linear Scale", fontsize=14, fontweight="bold"
)
axes[0].set_xlabel("N (DFT Length)", fontsize=12)
axes[0].set_ylabel("Number of Operations", fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Log scale
axes[1].loglog(N_values, dft_ops, "r-o", linewidth=2, markersize=8, label="DFT: O(N²)")
axes[1].loglog(
    N_values, fft_ops, "b-s", linewidth=2, markersize=8, label="FFT: O(N log N)"
)
axes[1].set_title(
    "Computational Complexity: Log-Log Scale", fontsize=14, fontweight="bold"
)
axes[1].set_xlabel("N (DFT Length)", fontsize=12)
axes[1].set_ylabel("Number of Operations", fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.show()

# ===== Part 2: Speedup Visualization =====
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(N_values, speedup, "g-^", linewidth=3, markersize=10)
ax.set_title("FFT Speedup vs. Direct DFT", fontsize=15, fontweight="bold")
ax.set_xlabel("N (DFT Length)", fontsize=13)
ax.set_ylabel("Speedup Factor (times faster)", fontsize=13)
ax.grid(True, alpha=0.3)
ax.set_xscale("log", base=2)

# Annotate some points
for i in [0, 3, 6, 9, 10]:
    ax.annotate(
        f"{speedup[i]:.1f}x",
        xy=(N_values[i], speedup[i]),
        xytext=(N_values[i] * 0.8, speedup[i] * 1.1),
        fontsize=10,
        color="red",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

plt.tight_layout()
plt.show()

# ===== Part 3: Radix-2 FFT Structure =====
print("\n" + "=" * 70)
print("PART 2: RADIX-2 FFT STRUCTURE")
print("=" * 70)

N = 8
print(f"\nExample: N = {N} (requires log₂({N}) = {int(np.log2(N))} stages)")
print("\nSignal decomposition:")

x_indices = list(range(N))
print(f"Level 0: {x_indices}")


# Show bit-reversal order
def bit_reverse(n, width):
    """Reverse bits of n with given width"""
    result = 0
    for i in range(width):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


stages = int(np.log2(N))
print(f"\nLevel 1 (split even/odd):")
even = [x_indices[i] for i in range(0, N, 2)]
odd = [x_indices[i] for i in range(1, N, 2)]
print(f"  Even indices: {even}")
print(f"  Odd indices:  {odd}")

print(f"\nLevel 2 (split again):")
print(f"  {[even[i] for i in range(0, len(even), 2)]}")
print(f"  {[even[i] for i in range(1, len(even), 2)]}")
print(f"  {[odd[i] for i in range(0, len(odd), 2)]}")
print(f"  {[odd[i] for i in range(1, len(odd), 2)]}")

print(f"\nLevel 3 (individual points - bit-reversed order):")
bit_reversed = [bit_reverse(i, stages) for i in range(N)]
print(f"  {bit_reversed}")
print(f"  This is called 'bit-reversal permutation'")

# ===== Part 4: Butterfly Diagram =====
fig, ax = plt.subplots(figsize=(12, 8))

# Draw butterfly operation
y_pos = [2, 1]
x_pos = [0, 1, 2]

# Input
ax.text(
    x_pos[0] - 0.3,
    y_pos[0],
    "$F_1[k]$",
    fontsize=14,
    ha="right",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
)
ax.text(
    x_pos[0] - 0.3,
    y_pos[1],
    "$F_2[k]$",
    fontsize=14,
    ha="right",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
)

# Twiddle factor
ax.plot([x_pos[0], x_pos[1]], [y_pos[1], y_pos[1]], "b-", linewidth=2)
ax.text(
    x_pos[0] + 0.5,
    y_pos[1] - 0.15,
    "$W_N^k$",
    fontsize=12,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# Butterfly connections
ax.plot([x_pos[0], x_pos[1]], [y_pos[0], (y_pos[0] + y_pos[1]) / 2], "g-", linewidth=2)
ax.plot([x_pos[1], x_pos[1]], [y_pos[1], (y_pos[0] + y_pos[1]) / 2], "g-", linewidth=2)
ax.plot([x_pos[1], x_pos[2]], [(y_pos[0] + y_pos[1]) / 2, y_pos[0]], "g-", linewidth=2)

ax.plot(
    [x_pos[0], x_pos[1]], [y_pos[0], (y_pos[0] + y_pos[1]) / 2 + 0.5], "r-", linewidth=2
)
ax.plot(
    [x_pos[1], x_pos[1]], [y_pos[1], (y_pos[0] + y_pos[1]) / 2 + 0.5], "r-", linewidth=2
)
ax.plot(
    [x_pos[1], x_pos[2]], [(y_pos[0] + y_pos[1]) / 2 + 0.5, y_pos[1]], "r-", linewidth=2
)

# Addition/subtraction symbols
ax.text(
    x_pos[1],
    (y_pos[0] + y_pos[1]) / 2,
    "+",
    fontsize=20,
    ha="center",
    va="center",
    bbox=dict(boxstyle="circle", facecolor="lightgreen", alpha=0.8),
)
ax.text(
    x_pos[1],
    (y_pos[0] + y_pos[1]) / 2 + 0.5,
    "−",
    fontsize=20,
    ha="center",
    va="center",
    bbox=dict(boxstyle="circle", facecolor="lightcoral", alpha=0.8),
)

# Output
ax.text(
    x_pos[2] + 0.3,
    y_pos[0],
    "$X[k]$",
    fontsize=14,
    ha="left",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)
ax.text(
    x_pos[2] + 0.3,
    y_pos[1],
    "$X[k+N/2]$",
    fontsize=14,
    ha="left",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)

ax.set_xlim([-0.8, 3])
ax.set_ylim([0.5, 2.5])
ax.axis("off")
ax.set_title("Radix-2 FFT: Butterfly Operation", fontsize=16, fontweight="bold")

# Add explanation
explanation = (
    "Butterfly Operation:\n"
    "1. Multiply F₂[k] by twiddle factor $W_N^k$\n"
    "2. Add to F₁[k] → X[k]\n"
    "3. Subtract from F₁[k] → X[k+N/2]\n"
    "Only 1 complex multiplication per butterfly!"
)
ax.text(
    1,
    0.1,
    explanation,
    fontsize=11,
    ha="center",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

plt.tight_layout()
plt.show()

# ===== Part 5: Actual Timing Comparison =====
print("\n" + "=" * 70)
print("PART 3: ACTUAL TIMING COMPARISON")
print("=" * 70)

timing_N = [2**i for i in range(8, 15)]  # 256 to 16384
dft_times = []
fft_times = []

print(f"\n{'N':<8} {'DFT Time (s)':<15} {'FFT Time (s)':<15} {'Speedup'}")
print("-" * 70)

for N in timing_N:
    # Generate random signal
    x = np.random.randn(N) + 1j * np.random.randn(N)

    # Time naive DFT (using matrix multiplication)
    start = time.time()
    W = np.exp(-2j * np.pi * np.arange(N)[:, None] * np.arange(N) / N)
    X_dft = W @ x
    dft_time = time.time() - start

    # Time FFT
    start = time.time()
    X_fft = np.fft.fft(x)
    fft_time = time.time() - start

    dft_times.append(dft_time)
    fft_times.append(fft_time)

    speedup = dft_time / fft_time
    print(f"{N:<8} {dft_time:<15.6f} {fft_time:<15.6f} {speedup:.1f}x")

# Plot timing
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(timing_N, dft_times, "r-o", linewidth=2, markersize=8, label="Direct DFT")
ax.semilogy(
    timing_N, fft_times, "b-s", linewidth=2, markersize=8, label="FFT (np.fft.fft)"
)
ax.set_title("Actual Computation Time: DFT vs FFT", fontsize=15, fontweight="bold")
ax.set_xlabel("N (DFT Length)", fontsize=13)
ax.set_ylabel("Time (seconds)", fontsize=13)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)
print("1. FFT is an ALGORITHM for computing DFT efficiently")
print("2. Radix-2 FFT works when N = 2^ν (power of 2)")
print("3. Strategy: Divide into even/odd, compute smaller DFTs, combine")
print("4. Complexity: O(N²) → O(N log N)")
print("5. Real-world speedup: 100x-1000x for typical N")
print("=" * 70)
