import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("PROBLEM 2d: MAGNITUDE SPECTRA AT ALL STAGES")
print("=" * 80)

# System parameters
Fsx = 8000  # Hz
Fsy = 6000  # Hz
L = 3
M = 4
F_intermediate = L * Fsx  # 24000 Hz
F_cutoff = 3000  # Hz

print(f"\nSystem: {Fsx} Hz → ↑{L} → Filter → ↓{M} → {Fsy} Hz")
print(f"Intermediate rate: {F_intermediate} Hz")
print(f"Filter cutoff: {F_cutoff} Hz")

# Frequency axis
F = np.linspace(-30000, 30000, 10000)


def triangular(F, F_max):
    spec = np.zeros_like(F)
    mask = np.abs(F) <= F_max
    spec[mask] = 1 - np.abs(F[mask]) / F_max
    return spec


def periodic_spectrum(F, period, num_periods=5):
    result = np.zeros_like(F)
    for k in range(-num_periods, num_periods + 1):
        F_shifted = F - k * period
        result += triangular(F_shifted, 4000)
    return result


X_spec = periodic_spectrum(F, Fsx, num_periods=4)
W_spec = periodic_spectrum(F, Fsx, num_periods=4)

H_spec = np.zeros_like(F)
H_spec[np.abs(F) <= F_cutoff] = L

V_spec = W_spec * H_spec

Y_spec = periodic_spectrum(F, Fsy, num_periods=5) * L

fig, axes = plt.subplots(5, 1, figsize=(16, 26))

color_original = "blue"
color_intermediate = "green"
color_filter = "red"
color_final = "purple"

ax = axes[0]
ax.plot(F / 1000, X_spec, color=color_original, linewidth=2.5)
ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
ax.axvline(x=-4, color="r", linestyle="--", linewidth=2, alpha=0.7)
ax.axvline(x=4, color="r", linestyle="--", linewidth=2, alpha=0.7)
ax.axvline(x=-8, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
ax.axvline(x=8, color="gray", linestyle=":", linewidth=1.5, alpha=0.5)
ax.axvline(x=-3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax.fill_between([-4, 4], 0, 1.3, alpha=0.08, color="blue")
ax.fill_between([-3, 3], 0, 1.3, alpha=0.12, color="orange")
ax.set_xlim([-25, 25])
ax.set_ylim([0, 1.3])
ax.set_title("STAGE 1: Input Signal x[n] at Fsx = 8 kHz")
ax.grid(True, alpha=0.3, linestyle=":")

ax = axes[1]
ax.plot(F / 1000, W_spec, color=color_intermediate, linewidth=2.5)
ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)
ax.axvline(x=-12, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)
ax.axvline(x=12, color="purple", linestyle=":", linewidth=1.5, alpha=0.7)
ax.axvline(x=-4, color="r", linestyle="--", linewidth=1.5, alpha=0.6)
ax.axvline(x=4, color="r", linestyle="--", linewidth=1.5, alpha=0.6)
ax.axvline(x=-3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.7)
ax.set_xlim([-25, 25])
ax.set_ylim([0, 1.3])
ax.set_title(
    f"STAGE 2: After Upsampling ↑{L} at {F_intermediate/1000:.0f} kHz (BEFORE filtering)"
)
ax.grid(True, alpha=0.3, linestyle=":")

ax = axes[2]
ax.plot(F / 1000, H_spec, color=color_filter, linewidth=3.5)
ax.axvline(x=-F_cutoff / 1000, color="k", linestyle="--", linewidth=2.5, alpha=0.8)
ax.axvline(x=F_cutoff / 1000, color="k", linestyle="--", linewidth=2.5, alpha=0.8)
ax.axhline(y=L, color="orange", linestyle=":", linewidth=2, alpha=0.7)
ax.set_xlim([-25, 25])
ax.set_ylim([0, 3.8])
ax.set_title(f"STAGE 3: Lowpass Filter - Cutoff = {F_cutoff} Hz, Gain = {L}")
ax.grid(True, alpha=0.3, linestyle=":")

ax = axes[3]
ax.plot(F / 1000, V_spec, color=color_intermediate, linewidth=2.5)
ax.axvline(x=-3, color="orange", linestyle="-.", linewidth=2, alpha=0.8)
ax.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.8)
ax.set_xlim([-25, 25])
ax.set_ylim([0, 3.8])
ax.set_title(
    f"STAGE 4: After Filtering at {F_intermediate/1000:.0f} kHz (BEFORE downsampling)"
)
ax.grid(True, alpha=0.3, linestyle=":")

ax = axes[4]
ax.plot(F / 1000, Y_spec, color=color_final, linewidth=2.5)
ax.axvline(x=-3, color="orange", linestyle="-.", linewidth=2, alpha=0.8)
ax.axvline(x=3, color="orange", linestyle="-.", linewidth=2, alpha=0.8)
ax.set_xlim([-25, 25])
ax.set_ylim([0, 3.8])
ax.set_title(f"STAGE 5: Final Output y[k] at Fsy = {Fsy/1000:.0f} kHz (after ↓{M})")
ax.grid(True, alpha=0.3, linestyle=":")

plt.tight_layout(pad=3.0)
plt.savefig("problem2d_all_spectra_large.png", dpi=150, bbox_inches="tight")
plt.show()
