import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("PROBLEM 2c: INFORMATION LOSS ANALYSIS")
print("=" * 80)

# Parameters
Fsx = 8000
Fsy = 6000
F_cutoff = 3000

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

F = np.linspace(0, 8000, 1000)

# Original analog signal spectrum
xa_spec = np.zeros_like(F)
mask1 = F <= 4000
xa_spec[mask1] = 1 - F[mask1] / 4000

# Reconstructed signal spectrum (after rate conversion)
xhat_spec = np.zeros_like(F)
mask2 = F <= 3000
xhat_spec[mask2] = 1 - F[mask2] / 4000

# Plot 1: Original signal
axes[0].fill_between(
    F / 1000, 0, xa_spec, alpha=0.3, color="blue", label="Original signal content"
)
axes[0].plot(F / 1000, xa_spec, "b", linewidth=2)
axes[0].axvline(
    x=4,
    color="r",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label="Original bandwidth (4 kHz)",
)
axes[0].axvline(
    x=3,
    color="orange",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label="New Nyquist (3 kHz)",
)
axes[0].set_xlabel("Frequency [kHz]", fontsize=11)
axes[0].set_ylabel("Magnitude", fontsize=11)
axes[0].set_title(
    "Original Analog Signal xa(t) - Bandwidth 0-4 kHz", fontsize=12, fontweight="bold"
)
axes[0].set_xlim([0, 8])
axes[0].set_ylim([0, 1.2])
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].text(
    1.5,
    0.6,
    "PRESERVED\n(0-3 kHz)",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.6),
)
axes[0].text(
    3.5,
    0.3,
    "LOST\n(3-4 kHz)",
    ha="center",
    fontsize=10,
    color="red",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="pink", alpha=0.6),
)

# Plot 2: Reconstructed signal
axes[1].fill_between(
    F / 1000, 0, xhat_spec, alpha=0.3, color="green", label="Reconstructed signal"
)
axes[1].plot(F / 1000, xhat_spec, "g", linewidth=2)
axes[1].plot(
    F / 1000, xa_spec, "b--", linewidth=1, alpha=0.5, label="Original (for comparison)"
)
axes[1].axvline(
    x=3,
    color="orange",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label="Filter cutoff = 3 kHz",
)
axes[1].set_xlabel("Frequency [kHz]", fontsize=11)
axes[1].set_ylabel("Magnitude", fontsize=11)
axes[1].set_title(
    "Reconstructed from y[m] at 6 kHz - Bandwidth 0-3 kHz Only",
    fontsize=12,
    fontweight="bold",
)
axes[1].set_xlim([0, 8])
axes[1].set_ylim([0, 1.2])
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)
axes[1].text(
    1.5,
    0.6,
    "✓ Clean signal\n0-3 kHz",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.6),
)
axes[1].text(
    5,
    0.4,
    "3-4 kHz content\npermanently removed",
    ha="center",
    fontsize=9,
    style="italic",
)

plt.tight_layout()
plt.savefig("problem2c_information_loss.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 80)
print("INFORMATION LOSS ANALYSIS")
print("=" * 80)

print("\n┌─ ORIGINAL SIGNAL ─────────────────────────────────────────┐")
print("│                                                            │")
print(f"│  Analog signal xa(t): bandwidth 0 to {4000} Hz             │")
print(f"│  Sampled at Fsx = {Fsx} Hz                                 │")
print(f"│  Nyquist frequency = {Fsx/2} Hz                            │")
print("│  Status: ✓ Properly sampled (at Nyquist rate)             │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n┌─ AFTER RATE CONVERSION ───────────────────────────────────┐")
print("│                                                            │")
print(f"│  Output y[m]: sampled at Fsy = {Fsy} Hz                    │")
print(f"│  New Nyquist frequency = {Fsy/2} Hz                        │")
print(f"│  Filter cutoff = {F_cutoff} Hz                              │")
print("│                                                            │")
print(f"│  Frequency content 0-{F_cutoff} Hz:   ✓ PRESERVED           │")
print(f"│  Frequency content {F_cutoff}-{4000} Hz: ✗ LOST              │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n┌─ CAN WE RECONSTRUCT xa(t)? ───────────────────────────────┐")
print("│                                                            │")
print("│  Answer: NO (not perfectly)                               │")
print("│                                                            │")
print("│  We can reconstruct:                                       │")
print(f"│    x̂a(t) = band-limited version (0-{F_cutoff} Hz)           │")
print("│                                                            │")
print("│  We CANNOT reconstruct:                                    │")
print(f"│    Original xa(t) with full 0-{4000} Hz content            │")
print("│                                                            │")
print("│  Information preserved: 75% (3 kHz / 4 kHz)               │")
print("│  Information lost: 25% (1 kHz / 4 kHz)                    │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n┌─ IS THIS A PROBLEM? ──────────────────────────────────────┐")
print("│                                                            │")
print("│  NO! This is the expected trade-off:                      │")
print("│                                                            │")
print("│  ✓ We chose to downsample to 6 kHz                        │")
print("│    → This limits us to 3 kHz bandwidth                    │")
print("│                                                            │")
print("│  ✓ The alternative (no filtering) would be WORSE:         │")
print("│    → Aliasing would corrupt the ENTIRE signal             │")
print("│                                                            │")
print("│  ✓ We get a clean 0-3 kHz signal without distortion       │")
print("│    → Better to lose 3-4 kHz than corrupt everything!     │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n" + "=" * 80)
