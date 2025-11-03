"""
Problem 2c: Crosscorrelation using Convolution

This script computes the crosscorrelation function ryx(l) using
the convolution approach with the flipped signal.

Key concept: ryx(l) = y(l) * x(-l)
Where * denotes convolution

Instructions:
-------------
1. Make sure you have the signals.mat file in the same directory
2. Run this script: python problem_2c.py
3. The plot will be saved as 'problem_2c_crosscorr_conv.png'
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

print("=" * 70)
print("PROBLEM 2c: CROSSCORRELATION USING CONVOLUTION")
print("=" * 70)

# Load the signals
print("\n1. Loading signals from signals.mat...")
data = scipy.io.loadmat("signals.mat")
x = data["x"].flatten()  # Emitted signal
y = data["y"].flatten()  # Received signal
print(f"   ✓ Loaded x[n]: length {len(x)}")
print(f"   ✓ Loaded y[n]: length {len(y)}")

# METHOD 1: Using convolution with flipped x
print("\n2. Computing crosscorrelation using CONVOLUTION...")
print("   Formula: ryx(l) = y(l) * x(-l)")
print("   Step 1: Flip x[n] to get x[-n]")

# Flip the signal x (reverse it)
x_flipped = np.flip(x)
print(f"   ✓ Flipped x[n] → x_flipped = x[-n]")

print("   Step 2: Convolve y[n] with x_flipped")
# Convolve y with flipped x
ryx_conv = np.convolve(y, x_flipped, mode="full")
print(f"   ✓ Computed ryx(l) using convolution: length {len(ryx_conv)}")

# Create the lag array
N = len(x)
lags = np.arange(-N + 1, N)
print(f"   ✓ Lag range: [{lags[0]}, {lags[-1]}]")

# Find the maximum
max_idx = np.argmax(ryx_conv)
max_lag = lags[max_idx]
max_value = ryx_conv[max_idx]

print(f"\n3. Analysis of crosscorrelation (convolution method):")
print(f"   Maximum value: {max_value:.4f}")
print(f"   Occurs at lag: l = {max_lag}")

# METHOD 2: For verification, also compute using scipy.signal.correlate
from scipy import signal

ryx_direct = signal.correlate(y, x, mode="full")
max_direct = np.max(ryx_direct)
max_lag_direct = lags[np.argmax(ryx_direct)]

print(f"\n4. Verification with scipy.signal.correlate (from 2b):")
print(f"   Maximum value: {max_direct:.4f}")
print(f"   Occurs at lag: l = {max_lag_direct}")

# Check if they match
print(f"\n5. Comparison:")
if np.allclose(ryx_conv, ryx_direct):
    print("   ✓ PERFECT MATCH! Both methods give identical results!")
else:
    print("   ✗ Results differ (this shouldn't happen)")

difference = np.max(np.abs(ryx_conv - ryx_direct))
print(f"   Maximum difference: {difference:.2e}")

# Create comprehensive plots
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Original signal x[n]
ax1 = fig.add_subplot(gs[0, 0])
n = np.arange(len(x))
ax1.stem(n, x, linefmt="b-", markerfmt="bo", basefmt="k-")
ax1.set_xlabel("n", fontsize=11)
ax1.set_ylabel("x[n]", fontsize=11)
ax1.set_title("Original Signal x[n]", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Plot 2: Flipped signal x[-n]
ax2 = fig.add_subplot(gs[0, 1])
ax2.stem(n, x_flipped, linefmt="r-", markerfmt="ro", basefmt="k-")
ax2.set_xlabel("n", fontsize=11)
ax2.set_ylabel("x[-n] (flipped)", fontsize=11)
ax2.set_title("Flipped Signal x[-n]", fontsize=12, fontweight="bold")
ax2.grid(True, alpha=0.3)
ax2.text(
    0.5,
    0.95,
    "Notice: Time-reversed version of x[n]",
    transform=ax2.transAxes,
    ha="center",
    va="top",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    fontsize=9,
)

# Plot 3: Full crosscorrelation (convolution method)
ax3 = fig.add_subplot(gs[1, :])
ax3.stem(lags, ryx_conv, linefmt="g-", markerfmt="go", basefmt="k-")
ax3.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax3.axvline(
    x=max_lag,
    color="r",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Peak at l={max_lag}",
)
ax3.set_xlabel("l (lag)", fontsize=12, fontweight="bold")
ax3.set_ylabel("ryx(l)", fontsize=12, fontweight="bold")
ax3.set_title(
    "Crosscorrelation ryx(l) using Convolution Method", fontsize=13, fontweight="bold"
)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

# Add annotation at peak
ax3.annotate(
    f"Peak: {max_value:.2f}\nat l={max_lag}",
    xy=(max_lag, max_value),
    xytext=(max_lag + 30, max_value * 0.8),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# Plot 4: Comparison of both methods
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(lags, ryx_conv, "g-", linewidth=2, alpha=0.7, label="Convolution method")
ax4.plot(lags, ryx_direct, "b--", linewidth=2, alpha=0.7, label="Direct method (2b)")
ax4.axvline(x=max_lag, color="r", linestyle="--", linewidth=1.5, alpha=0.5)
ax4.set_xlabel("l (lag)", fontsize=11)
ax4.set_ylabel("ryx(l)", fontsize=11)
ax4.set_title(
    "Comparison: Both Methods Overlay Perfectly", fontsize=12, fontweight="bold"
)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.text(
    0.5,
    0.95,
    "Both lines overlap → Methods are equivalent!",
    transform=ax4.transAxes,
    ha="center",
    va="top",
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    fontsize=9,
    fontweight="bold",
)

# Plot 5: Difference between methods
ax5 = fig.add_subplot(gs[2, 1])
difference_array = ryx_conv - ryx_direct
ax5.plot(lags, difference_array, "k-", linewidth=1)
ax5.axhline(y=0, color="r", linestyle="--", linewidth=1)
ax5.set_xlabel("l (lag)", fontsize=11)
ax5.set_ylabel("Difference", fontsize=11)
ax5.set_title("Difference Between Methods", fontsize=12, fontweight="bold")
ax5.grid(True, alpha=0.3)
max_diff = np.max(np.abs(difference_array))
ax5.text(
    0.5,
    0.95,
    f"Max difference: {max_diff:.2e} (essentially zero!)",
    transform=ax5.transAxes,
    ha="center",
    va="top",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    fontsize=9,
    fontweight="bold",
)

plt.suptitle(
    "Problem 2c: Crosscorrelation via Convolution",
    fontsize=16,
    fontweight="bold",
    y=0.995,
)
plt.savefig("problem_2c_crosscorr_conv.png", dpi=300, bbox_inches="tight")
print(f"\n6. Plot saved as 'problem_2c_crosscorr_conv.png'")

# Show some values around the peak
print("\n" + "=" * 70)
print("VALUES AROUND THE PEAK")
print("=" * 70)
print("\nConvolution Method:")
print("-" * 40)
for offset in range(-5, 6):
    lag_val = max_lag + offset
    if 0 <= max_idx + offset < len(ryx_conv):
        val = ryx_conv[max_idx + offset]
        marker = " <-- PEAK" if offset == 0 else ""
        print(f"  l = {lag_val:4d}: ryx(l) = {val:8.4f}{marker}")

print("\n" + "=" * 70)
print("WHY THIS WORKS: MATHEMATICAL EXPLANATION")
print("=" * 70)
print(
    """
Crosscorrelation definition:
  ryx(l) = Σ y[n] · x[n-l]

Convolution definition:
  (y * h)[l] = Σ y[n] · h[l-n]

If we set h[n] = x[-n] (flipped x), then:
  (y * h)[l] = Σ y[n] · x[-(l-n)]
             = Σ y[n] · x[n-l]
             = ryx(l)

So: Crosscorrelation = Convolution with flipped signal!

Steps:
------
1. Flip x[n] to get x[-n]
2. Convolve: y[n] * x[-n]
3. Result is ryx(l)

This is why numpy.convolve(y, np.flip(x)) gives the same
result as scipy.signal.correlate(y, x)!
"""
)

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print(
    f"""
✓ Crosscorrelation computed using BOTH methods
✓ Both methods give IDENTICAL results (max diff: {difference:.2e})
✓ Peak at l = {max_lag} with value {max_value:.4f}

This verifies that:
  ryx(l) = y(l) * x(-l)

where * denotes convolution.

The two methods are mathematically equivalent:
• Method 1 (2b): scipy.signal.correlate(y, x)
• Method 2 (2c): np.convolve(y, np.flip(x))

Both reveal the same hidden signal at delay D = {max_lag}!
"""
)

print("=" * 70)
print("✓ Problem 2c complete!")
print("=" * 70)
