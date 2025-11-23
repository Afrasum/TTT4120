import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

print("=" * 80)
print("PROBLEM 1c: INTERPOLATION FROM 8 kHz TO 24 kHz")
print("=" * 80)

# ============================================================================
# PART 1: FREQUENCY DOMAIN ANALYSIS (Problem 1c - The Assignment)
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: FREQUENCY DOMAIN SPECTRA (What the assignment asks for)")
print("=" * 80)

# Given parameters
Fsx = 8000  # Original sampling frequency (Hz)
Fsy = 24000  # Target sampling frequency (Hz)
I = 3  # Upsampling factor: Fsy/Fsx = 24000/8000 = 3

print(f"\nGiven:")
print(f"  - Original sampling rate: Fsx = {Fsx} Hz")
print(f"  - Target sampling rate: Fsy = {Fsy} Hz")
print(f"  - Upsampling factor: I = {I}")
print(f"  - Original analog signal bandwidth: -4 kHz to +4 kHz (triangular)")

# Create frequency axis for plotting
F = np.linspace(-30000, 30000, 10000)  # Analog frequency in Hz


def triangular_spectrum(F, F_max):
    """Create a triangular magnitude spectrum centered at 0"""
    spectrum = np.zeros_like(F)
    mask = np.abs(F) <= F_max
    spectrum[mask] = 1 - np.abs(F[mask]) / F_max
    return spectrum


def create_periodic_spectrum(F, Fs, num_periods=3):
    """
    Create periodic repetitions of triangular spectrum
    The base signal has bandwidth ±4 kHz
    """
    result = np.zeros_like(F)
    for k in range(-num_periods, num_periods + 1):
        # Shift base spectrum by k*Fs
        F_shifted = F - k * Fs
        result += triangular_spectrum(F_shifted, 4000)
    return result


# Spectrum 1: |X(F/Fsx)| - Original sampled signal
X_spectrum = create_periodic_spectrum(F, Fsx, num_periods=3)

# Spectrum 2: |W(F/Fsy)| - After upsampling (before filtering)
# From part (b): W(F/Fsy) = X(F/Fsx)
# The spectral pattern repeats every 8 kHz (same as X)
W_spectrum = create_periodic_spectrum(F, Fsx, num_periods=5)

# Spectrum 3: |H(F/Fsy)| - Ideal lowpass filter
# Cutoff at 4 kHz, gain = I = 3
H_spectrum = np.zeros_like(F)
mask = np.abs(F) <= 4000
H_spectrum[mask] = I

# Spectrum 4: |Y(F/Fsy)| - Output spectrum
# Y = W * H (multiplication in frequency domain)
Y_spectrum = W_spectrum * H_spectrum

# Create the figure with 4 subplots
fig1, axes = plt.subplots(4, 1, figsize=(14, 14))

# ---------- Plot 1: |X(F/Fsx)| ----------
axes[0].plot(F / 1000, X_spectrum, "b", linewidth=2)
axes[0].axvline(
    x=-4, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="±4 kHz (Nyquist)"
)
axes[0].axvline(x=4, color="r", linestyle="--", linewidth=1.5, alpha=0.7)
axes[0].axvline(
    x=-Fsx / 1000,
    color="g",
    linestyle=":",
    linewidth=1.5,
    alpha=0.7,
    label=f"±{Fsx/1000:.0f} kHz (period = Fsx)",
)
axes[0].axvline(x=Fsx / 1000, color="g", linestyle=":", linewidth=1.5, alpha=0.7)
axes[0].fill_between(
    [-4, 4], 0, 1.3, alpha=0.1, color="red", label="Baseband (original signal)"
)
axes[0].set_xlim([-25, 25])
axes[0].set_ylim([0, 1.3])
axes[0].set_xlabel("F [kHz]", fontsize=11)
axes[0].set_title(
    "1. Spectrum of x[n] sampled at Fsx = 8 kHz", fontsize=12, fontweight="bold"
)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=9)
axes[0].text(
    0,
    1.15,
    "Baseband: our signal",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.3),
)
axes[0].text(8, 0.6, "Copy at +8 kHz", ha="center", fontsize=9, style="italic")
axes[0].text(-8, 0.6, "Copy at -8 kHz", ha="center", fontsize=9, style="italic")

# ---------- Plot 2: |W(F/Fsy)| ----------
axes[1].plot(F / 1000, W_spectrum, "b", linewidth=2)
axes[1].axvline(
    x=-4, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="±4 kHz (baseband)"
)
axes[1].axvline(x=4, color="r", linestyle="--", linewidth=1.5, alpha=0.7)
axes[1].axvline(
    x=-Fsy / 2000,
    color="m",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label=f"±{Fsy/2000:.0f} kHz (NEW Nyquist)",
)
axes[1].axvline(x=Fsy / 2000, color="m", linestyle="-.", linewidth=2, alpha=0.7)
axes[1].fill_between(
    [-12, 12], 0, 1.3, alpha=0.05, color="magenta", label="New Nyquist range"
)
axes[1].fill_between([-4, 4], 0, 1.3, alpha=0.1, color="red")
axes[1].set_xlim([-25, 25])
axes[1].set_ylim([0, 1.3])
axes[1].set_xlabel("F [kHz]", fontsize=11)
axes[1].set_title(
    "2. Spectrum after upsampling ↑3 (BEFORE filtering) - W(F/Fsy) = X(F/Fsx)",
    fontsize=12,
    fontweight="bold",
)
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=9)
axes[1].text(
    0,
    1.15,
    "Wanted: baseband",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)
axes[1].text(
    8,
    0.6,
    "UNWANTED!\nImage at +8 kHz",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
)
axes[1].text(
    -8,
    0.6,
    "UNWANTED!\nImage at -8 kHz",
    ha="center",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="red", alpha=0.3),
)

# ---------- Plot 3: |H(F/Fsy)| ----------
axes[2].plot(F / 1000, H_spectrum, "r", linewidth=3)
axes[2].axvline(
    x=-4, color="k", linestyle="--", linewidth=1.5, alpha=0.7, label="±4 kHz (cutoff)"
)
axes[2].axvline(x=4, color="k", linestyle="--", linewidth=1.5, alpha=0.7)
axes[2].axhline(
    y=I, color="orange", linestyle=":", linewidth=2, alpha=0.7, label=f"Gain = I = {I}"
)
axes[2].fill_between([-4, 4], 0, 3.5, alpha=0.1, color="green", label="Passband")
axes[2].set_xlim([-25, 25])
axes[2].set_ylim([0, 3.5])
axes[2].set_xlabel("F [kHz]", fontsize=11)
axes[2].set_title(
    f"3. Ideal Lowpass Filter: cutoff = 4 kHz, gain = {I}",
    fontsize=12,
    fontweight="bold",
)
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=9)
axes[2].text(
    0,
    1.5,
    f"Passband gain = {I}",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
)
axes[2].text(
    10, 0.3, "Stopband: blocks images", ha="center", fontsize=9, style="italic"
)

# ---------- Plot 4: |Y(F/Fsy)| ----------
axes[3].plot(F / 1000, Y_spectrum, "g", linewidth=2.5)
axes[3].axvline(
    x=-4, color="r", linestyle="--", linewidth=1.5, alpha=0.7, label="±4 kHz"
)
axes[3].axvline(x=4, color="r", linestyle="--", linewidth=1.5, alpha=0.7)
axes[3].axvline(
    x=-Fsy / 2000,
    color="m",
    linestyle="-.",
    linewidth=2,
    alpha=0.7,
    label=f"±{Fsy/2000:.0f} kHz (Nyquist)",
)
axes[3].axvline(x=Fsy / 2000, color="m", linestyle="-.", linewidth=2, alpha=0.7)
axes[3].axhline(
    y=I, color="orange", linestyle=":", linewidth=2, alpha=0.5, label=f"Peak = {I}"
)
axes[3].fill_between([-4, 4], 0, 3.5, alpha=0.1, color="green")
axes[3].set_xlim([-25, 25])
axes[3].set_ylim([0, 3.5])
axes[3].set_xlabel("F [kHz]", fontsize=11)
axes[3].set_title(
    f"4. Final output Y(F/Fsy) = W(F/Fsy) × H(F/Fsy) at Fsy = {Fsy} Hz",
    fontsize=12,
    fontweight="bold",
)
axes[3].grid(True, alpha=0.3)
axes[3].legend(fontsize=9)
axes[3].text(
    0,
    1.8,
    f"Clean signal!\nMagnitude = {I}",
    ha="center",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)
axes[3].text(
    10, 0.3, "Images removed!", ha="center", fontsize=9, style="italic", color="green"
)

plt.tight_layout()
plt.savefig("problem1c_frequency_domain.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "-" * 80)
print("INTERPRETATION OF THE FREQUENCY DOMAIN PLOTS:")
print("-" * 80)
print("\n1. |X(F/Fsx)| - Original signal at 8 kHz:")
print("   ✓ Triangular baseband from -4 to +4 kHz (our signal)")
print("   ✓ Copies at ±8, ±16, ±24 kHz due to sampling")
print("   ✓ Nyquist frequency = 4 kHz (just enough!)")

print("\n2. |W(F/Fsy)| - After upsampling (zero insertion):")
print("   ✓ Same spectral shape as X (proved in part b!)")
print("   ✓ But now interpreted at Fsy = 24 kHz")
print("   ✗ PROBLEM: Copies at ±8, ±16 kHz are INSIDE new Nyquist range!")
print("   → These are 'spectral images' from zero insertion")

print("\n3. |H(F/Fsy)| - Interpolation filter:")
print(f"   ✓ Cutoff at 4 kHz (preserves original signal)")
print(f"   ✓ Gain = {I} in frequency domain")
print("   ✓ Removes unwanted images at ±8, ±16 kHz")

print("\n4. |Y(F/Fsy)| - Final output:")
print(f"   ✓ Clean triangular spectrum -4 to +4 kHz")
print(f"   ✓ Magnitude = {I} (in frequency domain)")
print("   ✓ All images removed - no aliasing!")
print("   ✓ Successfully interpolated to 24 kHz")

# ============================================================================
# PART 2: TIME DOMAIN ANALYSIS (Understanding what the magnitude = 3 means)
# ============================================================================

print("\n\n" + "=" * 80)
print("PART 2: TIME DOMAIN VERIFICATION (Why magnitude = 3 in freq domain?)")
print("=" * 80)

# Create a test signal
f_test = 1000  # 1 kHz test tone
duration = 0.015  # 15ms
t_original = np.arange(0, duration, 1 / Fsx)
x_test = np.sin(2 * np.pi * f_test * t_original)

print(f"\nTest signal: {f_test} Hz sinusoid")
print(f"Duration: {duration*1000} ms")
print(f"Number of samples at Fsx: {len(x_test)}")

# Step 1: Upsample by inserting zeros
w_test = np.zeros(len(x_test) * I)
w_test[::I] = x_test

print(f"After zero insertion: {len(w_test)} samples")

# Step 2: Design and apply interpolation filter
numtaps = 61
h_interp = I * signal.firwin(numtaps, cutoff=1 / I, window="hamming")
y_test = signal.lfilter(h_interp, 1, w_test)

# Remove filter transient
delay = numtaps // 2
y_test = y_test[delay:]
t_interp = np.arange(len(y_test)) / Fsy

print(f"After filtering: {len(y_test)} samples (transient removed)")

# Compute FFTs for frequency domain comparison
N_fft = 8192
X_fft = np.fft.fft(x_test, N_fft)
Y_fft = np.fft.fft(y_test[: len(x_test) * I], N_fft)
freq_x = np.fft.fftfreq(N_fft, 1 / Fsx)
freq_y = np.fft.fftfreq(N_fft, 1 / Fsy)

# Find peak magnitudes
peak_X = np.max(np.abs(X_fft[: N_fft // 2]))
peak_Y = np.max(np.abs(Y_fft[: N_fft // 2]))

print(f"\nFrequency domain peaks:")
print(f"  |X(f)| peak: {peak_X:.1f}")
print(f"  |Y(f)| peak: {peak_Y:.1f}")
print(f"  Ratio: {peak_Y/peak_X:.2f} ≈ {I} ✓")

print(f"\nTime domain amplitudes:")
print(f"  x[n] peak: {np.max(np.abs(x_test)):.3f}")
print(f"  y[m] peak: {np.max(np.abs(y_test)):.3f}")
print(f"  Ratio: {np.max(np.abs(y_test))/np.max(np.abs(x_test)):.2f} ≈ 1 ✓")

# Create time domain plots
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 12))

# TIME DOMAIN PLOTS
# Original signal
axes2[0, 0].plot(t_original * 1000, x_test, "b.-", linewidth=2, markersize=8)
axes2[0, 0].axhline(y=1, color="r", linestyle="--", alpha=0.5)
axes2[0, 0].axhline(y=-1, color="r", linestyle="--", alpha=0.5)
axes2[0, 0].set_xlabel("Time [ms]", fontsize=11)
axes2[0, 0].set_title(
    f"TIME: Original x[n] at Fsx = {Fsx} Hz", fontsize=12, fontweight="bold"
)
axes2[0, 0].grid(True, alpha=0.3)
axes2[0, 0].set_ylim([-1.5, 1.5])
axes2[0, 0].text(
    7.5,
    1.2,
    f"Peak = 1.0",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
)

# After upsampling (zeros)
t_zeros = np.arange(len(w_test)) / Fsy
axes2[1, 0].stem(
    t_zeros[:120] * 1000, w_test[:120], basefmt=" ", linefmt="b-", markerfmt="bo"
)
axes2[1, 0].axhline(y=1, color="r", linestyle="--", alpha=0.5)
axes2[1, 0].axhline(y=-1, color="r", linestyle="--", alpha=0.5)
axes2[1, 0].set_xlabel("Time [ms]", fontsize=11)
axes2[1, 0].set_title(
    f"TIME: After ↑{I} (zeros inserted)", fontsize=12, fontweight="bold"
)
axes2[1, 0].grid(True, alpha=0.3)
axes2[1, 0].set_ylim([-1.5, 1.5])
axes2[1, 0].set_xlim([0, 8])
axes2[1, 0].text(
    4,
    1.2,
    f"Mostly zeros!",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5),
)

# Interpolated signal
axes2[2, 0].plot(
    t_interp[: len(t_original) * I] * 1000,
    y_test[: len(t_original) * I],
    "g-",
    linewidth=1.5,
    label="Interpolated",
)
axes2[2, 0].plot(
    t_original * 1000, x_test, "b.", markersize=10, label="Original samples"
)
axes2[2, 0].axhline(y=1, color="r", linestyle="--", alpha=0.5)
axes2[2, 0].axhline(y=-1, color="r", linestyle="--", alpha=0.5)
axes2[2, 0].set_xlabel("Time [ms]", fontsize=11)
axes2[2, 0].set_title(
    f"TIME: Final y[m] at Fsy = {Fsy} Hz", fontsize=12, fontweight="bold"
)
axes2[2, 0].grid(True, alpha=0.3)
axes2[2, 0].set_ylim([-1.5, 1.5])
axes2[2, 0].legend(fontsize=10)
axes2[2, 0].text(
    7.5,
    1.2,
    f"Peak ≈ 1.0 (SAME!)",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)

# FREQUENCY DOMAIN PLOTS
# Original spectrum
axes2[0, 1].plot(
    freq_x[: N_fft // 2] / 1000, np.abs(X_fft[: N_fft // 2]), "b", linewidth=2
)
axes2[0, 1].axvline(
    x=f_test / 1000, color="r", linestyle="--", alpha=0.7, label=f"{f_test} Hz tone"
)
axes2[0, 1].axhline(
    y=peak_X, color="orange", linestyle=":", alpha=0.7, label=f"Peak = {peak_X:.0f}"
)
axes2[0, 1].set_xlabel("Frequency [kHz]", fontsize=11)
axes2[0, 1].set_title(f"FREQ: |X(f)| at {Fsx} Hz", fontsize=12, fontweight="bold")
axes2[0, 1].grid(True, alpha=0.3)
axes2[0, 1].set_xlim([0, 4])
axes2[0, 1].legend(fontsize=9)
axes2[0, 1].text(
    2,
    peak_X * 0.7,
    f"Peak ≈ {peak_X:.0f}",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
)

# Spectrum after zero insertion (before filter)
W_fft = np.fft.fft(w_test, N_fft)
freq_w = np.fft.fftfreq(N_fft, 1 / Fsy)
axes2[1, 1].plot(
    freq_w[: N_fft // 2] / 1000, np.abs(W_fft[: N_fft // 2]), "b", linewidth=2
)
axes2[1, 1].axvline(x=f_test / 1000, color="r", linestyle="--", alpha=0.7)
axes2[1, 1].set_xlabel("Frequency [kHz]", fontsize=11)
axes2[1, 1].set_title(
    f"FREQ: |W(f)| after ↑{I} (before filter)", fontsize=12, fontweight="bold"
)
axes2[1, 1].grid(True, alpha=0.3)
axes2[1, 1].set_xlim([0, 4])
axes2[1, 1].text(
    2,
    peak_X * 0.7,
    f"Same peak ≈ {peak_X:.0f}",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5),
)

# Interpolated spectrum
axes2[2, 1].plot(
    freq_y[: N_fft // 2] / 1000, np.abs(Y_fft[: N_fft // 2]), "g", linewidth=2
)
axes2[2, 1].axvline(
    x=f_test / 1000, color="r", linestyle="--", alpha=0.7, label=f"{f_test} Hz tone"
)
axes2[2, 1].axhline(
    y=peak_Y, color="orange", linestyle=":", alpha=0.7, label=f"Peak = {peak_Y:.0f}"
)
axes2[2, 1].set_xlabel("Frequency [kHz]", fontsize=11)
axes2[2, 1].set_title(f"FREQ: |Y(f)| at {Fsy} Hz", fontsize=12, fontweight="bold")
axes2[2, 1].grid(True, alpha=0.3)
axes2[2, 1].set_xlim([0, 4])
axes2[2, 1].legend(fontsize=9)
axes2[2, 1].text(
    2,
    peak_Y * 0.7,
    f"Peak ≈ {peak_Y:.0f} ({I}× larger!)",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
)

plt.tight_layout()
plt.savefig("problem1c_time_vs_freq.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("FINAL SUMMARY: WHY FREQUENCY MAGNITUDE = 3 BUT TIME AMPLITUDE = 1")
print("=" * 80)

print("\n┌─ TIME DOMAIN (What you actually measure/hear) ─┐")
print("│                                                  │")
print(f"│  Original x[n]:     amplitude = 1.0             │")
print(f"│  After zeros w[m]:  amplitude = 1.0 (sparse!)   │")
print(f"│  Interpolated y[m]: amplitude = 1.0             │")
print("│                                                  │")
print("│  ✓ Physical signal stays same amplitude!        │")
print("└──────────────────────────────────────────────────┘")

print("\n┌─ FREQUENCY DOMAIN (Mathematical representation) ─┐")
print("│                                                    │")
print(f"│  X at {Fsx} Hz:  {len(x_test)} samples → |X| peak ≈ {peak_X:.0f}      │")
print(f"│  Y at {Fsy} Hz: {len(x_test)*I} samples → |Y| peak ≈ {peak_Y:.0f}     │")
print("│                                                    │")
print(
    f"│  Ratio: {peak_Y:.0f}/{peak_X:.0f} = {peak_Y/peak_X:.1f} ≈ {I}                           │"
)
print("│                                                    │")
print("│  WHY? DTFT = Σ x[n]e^(-jωn)                       │")
print(f"│       More samples → larger sum → {I}× magnitude   │")
print("└────────────────────────────────────────────────────┘")

print("\n┌─ THE KEY INSIGHT ─────────────────────────────────┐")
print("│                                                    │")
print("│  The filter gain = I in FREQUENCY domain ensures: │")
print("│                                                    │")
print("│  1. Frequency representation scales correctly     │")
print("│  2. Time-domain amplitude stays at proper value   │")
print("│  3. Energy distributed across interpolated samples│")
print("│                                                    │")
print("│  It's about maintaining consistency between       │")
print("│  time and frequency at different sampling rates!  │")
print("└────────────────────────────────────────────────────┘")

print("\n" + "=" * 80)
print("Plots saved:")
print("  - problem1c_frequency_domain.png (the assignment answer)")
print("  - problem1c_time_vs_freq.png (the explanation)")
print("=" * 80)
