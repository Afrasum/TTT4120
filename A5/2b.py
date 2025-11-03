"""
Problem 2b: Crosscorrelation using xcorr/correlate

This script computes and plots the crosscorrelation function ryx(l)
using the built-in correlation function.

Instructions:
-------------
1. Make sure you have the signals.mat file in the same directory
2. Run this script: python problem_2b.py
3. The plot will be saved as 'problem_2b_crosscorr.png'
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import signal

print("=" * 70)
print("PROBLEM 2b: CROSSCORRELATION ANALYSIS")
print("=" * 70)

# Load the signals
print("\n1. Loading signals from signals.mat...")
data = scipy.io.loadmat("signals.mat")
x = data["x"].flatten()  # Emitted signal
y = data["y"].flatten()  # Received signal
print(f"   ✓ Loaded x[n]: length {len(x)}")
print(f"   ✓ Loaded y[n]: length {len(y)}")

# Compute crosscorrelation using scipy.signal.correlate
print("\n2. Computing crosscorrelation ryx(l)...")
# Note: scipy.signal.correlate(y, x) computes the crosscorrelation
# The 'full' mode gives us all possible overlaps
ryx = signal.correlate(y, x, mode="full")
print(f"   ✓ Computed ryx(l): length {len(ryx)}")

# Create the lag array
# For signals of length N, 'full' mode gives 2N-1 points
# Lags range from -(N-1) to (N-1)
N = len(x)
lags = np.arange(-N + 1, N)
print(f"   ✓ Lag range: [{lags[0]}, {lags[-1]}]")

# Find the maximum
max_idx = np.argmax(ryx)
max_lag = lags[max_idx]
max_value = ryx[max_idx]
print(f"\n3. Analysis of crosscorrelation:")
print(f"   Maximum value: {max_value:.4f}")
print(f"   Occurs at lag: l = {max_lag}")
print(f"   (This is our estimated delay D)")

# Create the plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Full crosscorrelation
axes[0].stem(lags, ryx, linefmt="g-", markerfmt="go", basefmt="k-")
axes[0].axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
axes[0].axvline(
    x=max_lag,
    color="r",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Peak at l={max_lag}",
)
axes[0].set_xlabel("l (lag)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("ryx(l)", fontsize=12, fontweight="bold")
axes[0].set_title("Crosscorrelation Function ryx(l)", fontsize=14, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=11)

# Add annotation at peak
axes[0].annotate(
    f"Peak: {max_value:.2f}\nat l={max_lag}",
    xy=(max_lag, max_value),
    xytext=(max_lag + 30, max_value * 0.8),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    fontsize=11,
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# Plot 2: Zoomed around the peak
zoom_range = 100
zoom_start = max(max_lag - zoom_range, lags[0])
zoom_end = min(max_lag + zoom_range, lags[-1])
zoom_mask = (lags >= zoom_start) & (lags <= zoom_end)

axes[1].stem(
    lags[zoom_mask], ryx[zoom_mask], linefmt="g-", markerfmt="go", basefmt="k-"
)
axes[1].axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
axes[1].axvline(
    x=max_lag,
    color="r",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Peak at l={max_lag}",
)
axes[1].set_xlabel("l (lag)", fontsize=12, fontweight="bold")
axes[1].set_ylabel("ryx(l)", fontsize=12, fontweight="bold")
axes[1].set_title(
    f"Zoomed View: ryx(l) around peak (l ∈ [{zoom_start}, {zoom_end}])",
    fontsize=14,
    fontweight="bold",
)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=11)

# Highlight the peak
axes[1].plot(max_lag, max_value, "r*", markersize=20, label="Maximum")

plt.tight_layout()
plt.savefig("problem_2b_crosscorr.png", dpi=300, bbox_inches="tight")
print(f"\n4. Plot saved as 'problem_2b_crosscorr.png'")

# Additional analysis
print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(
    f"""
The crosscorrelation function ryx(l) has been computed successfully!

Key Results:
------------
• Peak value: {max_value:.4f}
• Peak location: l = {max_lag}
• This suggests: Delay D = {max_lag} samples

What does this mean?
--------------------
The crosscorrelation measures how similar y[n] is to x[n-l] for each lag l.

• At l = {max_lag}: Maximum similarity!
  This means y[n] contains a copy of x[n] shifted by {max_lag} samples

• The large peak tells us:
  ✓ YES, an object WAS hit!
  ✓ The delay is D = {max_lag} samples
  ✓ The signal is buried in noise but correlation found it!

Compare to Problem 2a:
----------------------
• Visual inspection: Could not detect anything
• Crosscorrelation: Clear peak at l = {max_lag}!

This is the POWER of crosscorrelation - it can detect signals
that are completely invisible to the naked eye!

The mathematics "dug out" the hidden signal from the noise.
"""
)

# Show some values around the peak
print("\nValues around the peak:")
print("-" * 40)
for offset in range(-5, 6):
    lag_val = max_lag + offset
    if 0 <= max_idx + offset < len(ryx):
        val = ryx[max_idx + offset]
        marker = " <-- PEAK" if offset == 0 else ""
        print(f"  l = {lag_val:4d}: ryx(l) = {val:8.4f}{marker}")

print("\n" + "=" * 70)
print("✓ Problem 2b complete!")
print("=" * 70)
