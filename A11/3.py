import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("UNDERSTANDING NORMALIZED vs ANALOG FREQUENCY")
print("=" * 80)

# Create figure
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Frequency values
F1 = 900
F2 = 2000
Fs_original = 6000
Fs_new = 3000

# Normalized frequencies at original rate
f1_original = F1 / Fs_original  # 0.15
f2_original = F2 / Fs_original  # 0.333...

# Normalized frequencies at new rate
f1_new = F1 / Fs_new  # 0.3
f2_new = F2 / Fs_new  # 0.667...

print(f"\nOriginal sampling rate: Fs = {Fs_original} Hz")
print(f"  Nyquist frequency: {Fs_original/2} Hz")
print(f"  Nyquist (normalized): 0.5")
print(f"\n  F1 = {F1} Hz → normalized: f = {f1_original}")
print(f"  F2 = {F2} Hz → normalized: f = {f2_original:.4f}")

print(f"\nAfter downsampling: Fs_new = {Fs_new} Hz")
print(f"  NEW Nyquist frequency: {Fs_new/2} Hz")
print(f"  NEW Nyquist (normalized to NEW rate): 0.5")
print(f"\n  F1 = {F1} Hz → normalized: f = {f1_new}")
print(f"  F2 = {F2} Hz → normalized: f = {f2_new:.4f}")

print(f"\nCRITICAL INSIGHT:")
print(f"  The NEW Nyquist ({Fs_new/2} Hz) expressed in the ORIGINAL")
print(f"  normalized scale is: {Fs_new/2}/{Fs_original} = {(Fs_new/2)/Fs_original}")
print(f"  This is why the filter cutoff is at f = 0.25!")

# ============================================================================
# PLOT 1: Original rate - Analog frequency axis
# ============================================================================
ax = axes[0]
F_axis = np.linspace(0, 6000, 1000)

ax.axhline(y=0.5, color="k", linewidth=0.5, alpha=0.3)
ax.axvline(x=F1, color="blue", linewidth=3, label=f"F1 = {F1} Hz")
ax.axvline(x=F2, color="red", linewidth=3, label=f"F2 = {F2} Hz")
ax.axvline(
    x=Fs_original / 2,
    color="green",
    linestyle="--",
    linewidth=2.5,
    label=f"Original Nyquist = {Fs_original/2} Hz",
)
ax.axvline(
    x=Fs_new / 2,
    color="orange",
    linestyle="--",
    linewidth=2.5,
    label=f"NEW Nyquist = {Fs_new/2} Hz",
)

ax.fill_between(
    [0, Fs_new / 2], 0, 1, alpha=0.15, color="green", label="Safe zone (no aliasing)"
)
ax.fill_between(
    [Fs_new / 2, Fs_original / 2],
    0,
    1,
    alpha=0.15,
    color="red",
    label="Danger zone (will alias)",
)

ax.set_xlabel("Analog Frequency F [Hz]", fontsize=12, fontweight="bold")
ax.set_ylabel("Amplitude", fontsize=12)
ax.set_title(
    f"ANALOG FREQUENCY VIEW at Fs = {Fs_original} Hz", fontsize=13, fontweight="bold"
)
ax.set_xlim([0, Fs_original / 2])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax.text(
    F1,
    0.8,
    f"{F1} Hz\n(safe)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)
ax.text(
    F2,
    0.8,
    f"{F2} Hz\n(DANGER!)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="pink", alpha=0.7),
)
ax.text(
    Fs_new / 2,
    0.5,
    f"Filter cutoff\n{Fs_new/2} Hz",
    ha="right",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# ============================================================================
# PLOT 2: Original rate - Normalized frequency axis
# ============================================================================
ax = axes[1]
f_axis = np.linspace(0, 0.5, 1000)

ax.axhline(y=0.5, color="k", linewidth=0.5, alpha=0.3)
ax.axvline(
    x=f1_original, color="blue", linewidth=3, label=f"f1 = {f1_original} (= {F1} Hz)"
)
ax.axvline(
    x=f2_original, color="red", linewidth=3, label=f"f2 = {f2_original:.3f} (= {F2} Hz)"
)
ax.axvline(
    x=0.5,
    color="green",
    linestyle="--",
    linewidth=2.5,
    label=f"Original Nyquist: f = 0.5 (= {Fs_original/2} Hz)",
)
ax.axvline(
    x=0.25,
    color="orange",
    linestyle="--",
    linewidth=2.5,
    label=f"NEW Nyquist: f = 0.25 (= {Fs_new/2} Hz)",
)

ax.fill_between([0, 0.25], 0, 1, alpha=0.15, color="green")
ax.fill_between([0.25, 0.5], 0, 1, alpha=0.15, color="red")

ax.set_xlabel(
    "Normalized Frequency f = F/Fs (at Fs = 6000 Hz)", fontsize=12, fontweight="bold"
)
ax.set_ylabel("Amplitude", fontsize=12)
ax.set_title(
    f"NORMALIZED FREQUENCY VIEW at Fs = {Fs_original} Hz",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax.text(
    f1_original,
    0.8,
    f"f = {f1_original}\n(safe)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)
ax.text(
    f2_original,
    0.8,
    f"f = {f2_original:.3f}\n(DANGER!)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="pink", alpha=0.7),
)
ax.text(
    0.25,
    0.5,
    f"Filter cutoff\nf = 0.25",
    ha="right",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
)

# ============================================================================
# PLOT 3: After downsampling - Normalized to NEW rate
# ============================================================================
ax = axes[2]

ax.axhline(y=0.5, color="k", linewidth=0.5, alpha=0.3)
ax.axvline(x=f1_new, color="blue", linewidth=3, label=f"f1 = {f1_new} (still {F1} Hz!)")
ax.axvline(
    x=0.5,
    color="green",
    linestyle="--",
    linewidth=2.5,
    label=f"NEW Nyquist: f = 0.5 (= {Fs_new/2} Hz)",
)

# Show where f2 would be (outside range, aliased)
ax.axvline(
    x=1 - f2_new,
    color="red",
    linewidth=3,
    linestyle=":",
    label=f"f2 aliases to {1-f2_new:.3f}",
)
ax.plot(
    [f2_new - 1],
    [0.7],
    "rx",
    markersize=15,
    markeredgewidth=3,
    label=f"f2 = {f2_new:.3f} (outside, wraps!)",
)

ax.fill_between([0, 0.5], 0, 1, alpha=0.15, color="green", label="Valid range")

ax.set_xlabel(
    "Normalized Frequency f = F/Fs_new (at Fs_new = 3000 Hz)",
    fontsize=12,
    fontweight="bold",
)
ax.set_ylabel("Amplitude", fontsize=12)
ax.set_title(
    f"AFTER DOWNSAMPLING - Normalized to NEW rate Fs = {Fs_new} Hz",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlim([0, 0.5])
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

ax.text(
    f1_new,
    0.8,
    f"f = {f1_new}\n({F1} Hz)",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
)
ax.text(
    0.25,
    0.5,
    f"Notice: {F1} Hz moved\nfrom f=0.15 to f=0.3!",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
)

plt.tight_layout()
plt.savefig("normalized_vs_analog_frequency.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 80)
print("SUMMARY: The Key Points")
print("=" * 80)
print(
    """
1. ANALOG FREQUENCY (F in Hz) - The "real" frequency
   - F1 = 900 Hz stays 900 Hz before and after downsampling
   - F2 = 2000 Hz stays 2000 Hz before and after downsampling
   
2. NORMALIZED FREQUENCY (f = F/Fs) - Depends on sampling rate!
   
   At Fs = 6000 Hz:
   - f1 = 900/6000 = 0.15
   - f2 = 2000/6000 = 0.333...
   - Nyquist = 0.5 (always!)
   
   At Fs_new = 3000 Hz:
   - f1 = 900/3000 = 0.3  ← SAME analog freq, DIFFERENT normalized!
   - f2 = 2000/3000 = 0.667... ← This is > 0.5, so it aliases!
   - Nyquist = 0.5 (always!)
   
3. FILTER CUTOFF
   - Must be at 1500 Hz (the NEW Nyquist frequency)
   - Expressed in original scale: 1500/6000 = 0.25
   - This is where "f = 0.25" comes from!
   
4. WHY NORMALIZE?
   - DTFT theory works with normalized frequency
   - Makes formulas independent of specific sampling rates
   - But can be confusing when sampling rate changes!
"""
)
print("=" * 80)
