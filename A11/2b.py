import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("PROBLEM 2b: FILTER SPECIFICATIONS")
print("=" * 80)

# System parameters
Fsx = 8000  # Original sampling rate
Fsy = 6000  # Target sampling rate
L = 3  # Upsampling factor
M = 4  # Downsampling factor
F_intermediate = L * Fsx  # 24 kHz

print(f"\nSystem parameters:")
print(f"  Fsx (input) = {Fsx} Hz")
print(f"  Fsy (output) = {Fsy} Hz")
print(f"  L (upsample) = {L}")
print(f"  M (downsample) = {M}")
print(f"  F_intermediate = L × Fsx = {F_intermediate} Hz")

# Critical frequencies
F_signal_max = Fsx / 2  # Original signal bandwidth
F_new_nyquist = Fsy / 2  # New Nyquist after downsampling
F_cutoff = min(F_signal_max, F_new_nyquist)

print(f"\nCritical frequencies:")
print(f"  Original signal bandwidth: 0 to {F_signal_max} Hz")
print(f"  New Nyquist frequency: {F_new_nyquist} Hz")
print(f"  Required cutoff frequency: {F_cutoff} Hz")

# Frequency axis
F = np.linspace(-15000, 15000, 5000)


# Create triangular base spectrum (original signal)
def triangular(F, F_max):
    spec = np.zeros_like(F)
    mask = np.abs(F) <= F_max
    spec[mask] = 1 - np.abs(F[mask]) / F_max
    return spec


# Spectrum at each stage
def periodic_spectrum(F, base_period, num_periods=5):
    result = np.zeros_like(F)
    for k in range(-num_periods, num_periods + 1):
        F_shifted = F - k * base_period
        result += triangular(F_shifted, 4000)
    return result


# Original spectrum X at 8 kHz
X_spec = periodic_spectrum(F, Fsx, num_periods=2)

# After upsampling W at 24 kHz (same pattern, different interpretation)
W_spec = periodic_spectrum(F, Fsx, num_periods=3)

# Filter response
H_spec = np.zeros_like(F)
H_spec[np.abs(F) <= F_cutoff] = L

# After filtering V
V_spec = W_spec * H_spec

# After downsampling Y at 6 kHz
# When we downsample by M=4, we keep every 4th sample
# This causes spectral replication every Fsy = 6 kHz
Y_spec = periodic_spectrum(F, Fsy, num_periods=3) * L

# Create comprehensive figure
fig = plt.figure(figsize=(16, 14))

# Create grid for subplots
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])  # Spectrum at 8 kHz
ax2 = fig.add_subplot(gs[1, :])  # Spectrum at 24 kHz (after upsample)
ax3 = fig.add_subplot(gs[2, :])  # Filter response
ax4 = fig.add_subplot(gs[3, :])  # After filtering
ax5 = fig.add_subplot(gs[4, :])  # Final at 6 kHz

# Plot 1: Original spectrum at 8 kHz
ax1.plot(F / 1000, X_spec, "b", linewidth=2)
ax1.axvline(
    x=-4,
    color="r",
    linestyle="--",
    linewidth=1.5,
    alpha=0.7,
    label="±4 kHz (original Nyquist)",
)
ax1.axvline(x=4, color="r", linestyle="--", linewidth=1.5, alpha=0.7)
ax1.axvline(
    x=-3,
    color="orange",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label="±3 kHz (NEW Nyquist)",
)
ax1.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax1.fill_between([-4, 4], 0, 1.3, alpha=0.1, color="blue")
ax1.set_xlim([-15, 15])
ax1.set_ylim([0, 1.3])
ax1.set_xlabel("F [kHz]", fontsize=11)
ax1.set_ylabel("|X(F/Fsx)|", fontsize=11)
ax1.set_title("1. Original signal x[n] at Fsx = 8 kHz", fontsize=12, fontweight="bold")
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)
ax1.text(
    0,
    1.1,
    "Signal: 0-4 kHz",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)

# Plot 2: After upsampling at 24 kHz
ax2.plot(F / 1000, W_spec, "b", linewidth=2)
ax2.axvline(
    x=-12,
    color="purple",
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
    label="±12 kHz (24kHz Nyquist)",
)
ax2.axvline(x=12, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)
ax2.axvline(x=-4, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="±4 kHz")
ax2.axvline(x=4, color="r", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.axvline(
    x=-3,
    color="orange",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label="±3 kHz (target Nyquist)",
)
ax2.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax2.axvline(x=-8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax2.axvline(x=8, color="gray", linestyle=":", linewidth=1, alpha=0.5)
ax2.fill_between([-3, 3], 0, 1.3, alpha=0.15, color="orange", label="Must keep this")
ax2.set_xlim([-15, 15])
ax2.set_ylim([0, 1.3])
ax2.set_xlabel("F [kHz]", fontsize=11)
ax2.set_ylabel("|W(F/24kHz)|", fontsize=11)
ax2.set_title(
    "2. After upsampling ↑3 (before filtering) at 24 kHz",
    fontsize=12,
    fontweight="bold",
)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)
ax2.text(
    0,
    1.15,
    "Keep 0-3 kHz",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)
ax2.text(
    8,
    0.7,
    "Remove!",
    ha="center",
    fontsize=9,
    color="red",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="pink", alpha=0.3),
)
ax2.text(
    -8,
    0.7,
    "Remove!",
    ha="center",
    fontsize=9,
    color="red",
    fontweight="bold",
    bbox=dict(boxstyle="round", facecolor="pink", alpha=0.3),
)

# Plot 3: Filter response
ax3.plot(F / 1000, H_spec, "r", linewidth=3)
ax3.axvline(
    x=-F_cutoff / 1000,
    color="k",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"±{F_cutoff/1000:.0f} kHz (cutoff)",
)
ax3.axvline(x=F_cutoff / 1000, color="k", linestyle="--", linewidth=2, alpha=0.7)
ax3.axhline(
    y=L, color="orange", linestyle=":", linewidth=2, alpha=0.7, label=f"Gain = L = {L}"
)
ax3.fill_between(
    [-F_cutoff / 1000, F_cutoff / 1000],
    0,
    3.5,
    alpha=0.15,
    color="green",
    label="Passband",
)
ax3.set_xlim([-15, 15])
ax3.set_ylim([0, 3.5])
ax3.set_xlabel("F [kHz]", fontsize=11)
ax3.set_ylabel("|H(F/24kHz)|", fontsize=11)
ax3.set_title(
    f"3. Filter Specification: Cutoff = {F_cutoff} Hz, Gain = {L}",
    fontsize=12,
    fontweight="bold",
)
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9)
ax3.text(
    0,
    1.8,
    f"Passband: 0-{F_cutoff} Hz\nGain = {L}",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
)
ax3.text(
    7,
    0.3,
    "Stopband:\nGain = 0",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5),
)

# Plot 4: After filtering
ax4.plot(F / 1000, V_spec, "g", linewidth=2)
ax4.axvline(
    x=-3,
    color="orange",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label="±3 kHz (new Nyquist)",
)
ax4.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax4.fill_between([-3, 3], 0, 3.5, alpha=0.1, color="green")
ax4.set_xlim([-15, 15])
ax4.set_ylim([0, 3.5])
ax4.set_xlabel("F [kHz]", fontsize=11)
ax4.set_ylabel("|V(F/24kHz)|", fontsize=11)
ax4.set_title(
    "4. After filtering (before downsampling) at 24 kHz", fontsize=12, fontweight="bold"
)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9)
ax4.text(
    0,
    2.0,
    "Clean signal 0-3 kHz\nReady for downsampling!",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)
ax4.text(
    7, 0.3, "Images removed", ha="center", fontsize=9, style="italic", color="green"
)

# Plot 5: Final output at 6 kHz
ax5.plot(F / 1000, Y_spec, "purple", linewidth=2)
ax5.axvline(
    x=-3,
    color="orange",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label="±3 kHz (Nyquist)",
)
ax5.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax5.axvline(
    x=-6, color="purple", linestyle=":", linewidth=1.5, alpha=0.7, label="±6 kHz (Fsy)"
)
ax5.axvline(x=6, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)
ax5.fill_between([-3, 3], 0, 3.5, alpha=0.1, color="purple")
ax5.set_xlim([-15, 15])
ax5.set_ylim([0, 3.5])
ax5.set_xlabel("F [kHz]", fontsize=11)
ax5.set_ylabel("|Y(F/Fsy)|", fontsize=11)
ax5.set_title(
    "5. Final output y[k] at Fsy = 6 kHz (after ↓4)", fontsize=12, fontweight="bold"
)
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=9)
ax5.text(
    0,
    2.0,
    "No aliasing!\nSignal preserved",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.5),
)

plt.savefig("problem2b_filter_specs.png", dpi=150, bbox_inches="tight")
plt.show()

# Print detailed specifications
print("\n" + "=" * 80)
print("FILTER SPECIFICATIONS (DETAILED)")
print("=" * 80)

print("\n┌─ REQUIREMENT 1: Remove Upsampling Images ─────────────────┐")
print("│                                                            │")
print(
    f"│  After ↑{L}: Images appear at ±{Fsx/1000}kHz, ±{2*Fsx/1000}kHz, ...           │"
)
print(f"│  Must preserve: 0 to {F_signal_max/1000} kHz                           │")
print(f"│  Suggested cutoff: ≤ {F_signal_max} Hz                          │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n┌─ REQUIREMENT 2: Prevent Downsampling Aliasing ────────────┐")
print("│                                                            │")
print(
    f"│  After ↓{M}: New Nyquist = {F_new_nyquist/1000} kHz                          │"
)
print(f"│  Must block: Everything above {F_new_nyquist} Hz                │")
print(f"│  Required cutoff: ≤ {F_new_nyquist} Hz                          │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n┌─ FINAL SPECIFICATION ──────────────────────────────────────┐")
print("│                                                            │")
print(f"│  Cutoff frequency: fc = {F_cutoff} Hz                       │")
print(f"│                   (most restrictive requirement)           │")
print("│                                                            │")
print(f"│  In normalized frequency (at {F_intermediate} Hz):              │")
print(
    f"│    fc_normalized = {F_cutoff}/{F_intermediate} = {F_cutoff/F_intermediate}                  │"
)
print("│                                                            │")
print(f"│  Passband: 0 ≤ F ≤ {F_cutoff} Hz                            │")
print(f"│  Gain in passband: {L}                                      │")
print("│                                                            │")
print(f"│  Stopband: F ≥ {F_cutoff} Hz                                │")
print("│  Gain in stopband: 0 (complete attenuation)                │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n┌─ PRACTICAL FILTER DESIGN ─────────────────────────────────┐")
print("│                                                            │")
print("│  For a realizable filter with transition band:            │")
print("│                                                            │")
print(f"│  Passband edge:     Fp ≤ {F_cutoff} Hz                      │")
print(f"│  Stopband edge:     Fs ≥ {F_cutoff} Hz (e.g., 5000 Hz)      │")
print("│  Passband ripple:   δp (e.g., 0.01)                       │")
print("│  Stopband ripple:   δs (e.g., 0.001)                      │")
print(f"│  Gain:              {L}                                      │")
print("│                                                            │")
print("└────────────────────────────────────────────────────────────┘")

print("\n" + "=" * 80)
print("WHY CUTOFF = 3000 Hz?")
print("=" * 80)
print("\nThe filter must satisfy BOTH requirements simultaneously:")
print(f"  • From upsampling: cutoff ≤ {F_signal_max} Hz")
print(f"  • From downsampling: cutoff ≤ {F_new_nyquist} Hz")
print(
    f"\n  → Take the minimum: fc = min({F_signal_max}, {F_new_nyquist}) = {F_cutoff} Hz"
)
print(f"\nThis ensures:")
print("  ✓ No information loss from original signal (0-3 kHz preserved)")
print("  ✓ No aliasing after downsampling (nothing above 3 kHz)")
print("=" * 80)
