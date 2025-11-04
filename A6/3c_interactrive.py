# zero_padding_demo_v2.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, RadioButtons, Slider


def make_window(N, kind):
    n = np.arange(N)
    if kind == "Rektangulær":
        return np.ones(N)
    if kind == "Hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (max(N - 1, 1)))
    if kind == "Hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (max(N - 1, 1)))
    return np.ones(N)


def mag_fft(x, Nfft):
    X = np.fft.fft(x, n=Nfft)
    f = np.arange(Nfft) / Nfft
    return f, np.abs(X)


def compute_signals(N, f1, f2, win_kind):
    n = np.arange(N)
    w = make_window(N, win_kind)
    x = np.sin(2 * np.pi * f1 * n) + np.sin(2 * np.pi * f2 * n)
    xw = x * w
    return n, x, w, xw


# Init
N_init = 100
f1_init = 0.175
f2_init = 0.225
win_init = "Rektangulær"
Nfft_ref = 16384
Nfft_cap = 65536
n, x, w, xw = compute_signals(N_init, f1_init, f2_init, win_init)

fig = plt.figure(figsize=(13, 9))
ax_time = fig.add_axes([0.06, 0.61, 0.60, 0.32])
ax_win = fig.add_axes([0.70, 0.61, 0.25, 0.32])
ax_freq = fig.add_axes([0.06, 0.12, 0.89, 0.42])

axN = fig.add_axes([0.06, 0.96, 0.30, 0.02])
axf1 = fig.add_axes([0.38, 0.96, 0.30, 0.02])
axf2 = fig.add_axes([0.70, 0.96, 0.24, 0.02])
axwin = fig.add_axes([0.90, 0.12, 0.08, 0.22])
axchecks = fig.add_axes([0.70, 0.12, 0.18, 0.15])
axNcust = fig.add_axes([0.06, 0.07, 0.30, 0.03])

fig.suptitle(
    "Zero-padding = tettere prøving i frekvens (ingen ny informasjon)", y=0.995
)


def compute_reference(xw):
    fR, XR = mag_fft(xw, Nfft_ref)
    XR = XR / (np.max(XR) if np.max(XR) > 0 else 1.0)
    return fR, XR


fR, XR = compute_reference(xw)

# Tid
(line_x,) = ax_time.plot(n, x, label="x[n]")
(line_xw,) = ax_time.plot(n, xw, label="x[n]·w[n]")
ax_time.set_xlim(0, max(n[-1], 1))
ax_time.set_xlabel("n")
ax_time.set_ylabel("Amplitude")
ax_time.grid(True, linestyle=":")
ax_time.legend(loc="upper right")
ax_time.set_title("Tid: original og vinduet signal")

# Vindu (uten use_line_collection)
ax_win.stem(n, w)
ax_win.set_xlim([-1, max(n[-1], 1)])
ax_win.set_xlabel("n")
ax_win.set_ylabel("w[n]")
ax_win.grid(True, linestyle=":")
ax_win.set_title("Vindu w[n]")

# Frekvens
(ref_line,) = ax_freq.plot(
    fR[: Nfft_ref // 2 + 1], XR[: Nfft_ref // 2 + 1], label="Referanse (svært høy FFT)"
)
ax_freq.axvline(f1_init, linestyle="--")
ax_freq.axvline(f2_init, linestyle="--")
ax_freq.set_xlim(0, 0.5)
ax_freq.set_xlabel("Normalisert frekvens f")
ax_freq.set_ylabel("Normalisert |X(f)|")
ax_freq.grid(True, linestyle=":")
ax_freq.set_title("Frekvens: samme kurve, ulike NFFT-prøver (Δf = 1/NFFT)")

# Plassholdere for punkter
ax_freq.plot([], [], "o", label="N")
ax_freq.plot([], [], "o", label="2N")
ax_freq.plot([], [], "o", label="4N")
ax_freq.plot([], [], "o", label="8N")
ax_freq.plot([], [], ".", label="Custom")
txt_df = ax_freq.text(0.51, 0.90, "", transform=ax_freq.transAxes)
ax_freq.legend(loc="upper right")

# Kontroller
sN = Slider(axN, "N", 10, 4000, valinit=N_init, valstep=1)
sf1 = Slider(axf1, "f1", 0.02, 0.48, valinit=f1_init, valstep=0.001)
sf2 = Slider(axf2, "f2", 0.02, 0.48, valinit=f2_init, valstep=0.001)
rwin = RadioButtons(axwin, ("Rektangulær", "Hann", "Hamming"))
checks = CheckButtons(
    axchecks,
    ("Tegn N", "Tegn 2N", "Tegn 4N", "Tegn 8N", "Tegn Custom"),
    (True, True, False, False, False),
)
scust = Slider(axNcust, "Custom NFFT", 128, Nfft_cap, valinit=1024, valstep=128)


def draw_fft_points(xw, Nfft):
    f, X = mag_fft(xw, int(Nfft))
    X = X / (np.max(X) if np.max(X) > 0 else 1.0)
    half = int(Nfft) // 2 + 1
    return f[:half], X[:half]


def on_change(_):
    N = int(sN.val)
    f1 = float(sf1.val)
    f2 = float(sf2.val)
    win_kind = rwin.value_selected
    use_N, use_2N, use_4N, use_8N, use_cust = checks.get_status()
    Ncust = int(scust.val)

    n, x, w, xw = compute_signals(N, f1, f2, win_kind)

    # Oppdater tid
    line_x.set_data(n, x)
    line_xw.set_data(n, xw)
    ax_time.set_xlim(0, max(n[-1], 1))

    # Oppdater vindu
    ax_win.clear()
    ax_win.stem(n, w)
    ax_win.set_xlim([-1, max(n[-1], 1)])
    ax_win.set_xlabel("n")
    ax_win.set_ylabel("w[n]")
    ax_win.grid(True, linestyle=":")
    ax_win.set_title("Vindu w[n]")

    # Oppdater frekvens: tegn ALT på nytt (enkelt og robust)
    ax_freq.clear()
    fR, XR = compute_reference(xw)
    ax_freq.plot(
        fR[: Nfft_ref // 2 + 1],
        XR[: Nfft_ref // 2 + 1],
        label="Referanse (svært høy FFT)",
    )
    ax_freq.axvline(f1, linestyle="--")
    ax_freq.axvline(f2, linestyle="--")

    df_texts = []

    def add_points(Nfft, label, style):
        fpts, Xpts = draw_fft_points(xw, Nfft)
        ax_freq.plot(fpts, Xpts, style, label=label)
        df_texts.append(f"{label}: Δf=1/{Nfft:d} = {1.0/Nfft:.5f}")

    if use_N:
        add_points(N, "N", "o")
    if use_2N:
        add_points(2 * N, "2N", "o")
    if use_4N:
        add_points(4 * N, "4N", "o")
    if use_8N:
        add_points(8 * N, "8N", "o")
    if use_cust:
        add_points(Ncust, "Custom", ".")

    ax_freq.set_xlim(0, 0.5)
    ax_freq.set_xlabel("Normalisert frekvens f")
    ax_freq.set_ylabel("Normalisert |X(f)|")
    ax_freq.grid(True, linestyle=":")
    ax_freq.set_title("Frekvens: samme kurve, ulike NFFT-prøver (Δf = 1/NFFT)")
    ax_freq.legend(loc="upper right")
    txt_df.set_text("\n".join(df_texts))
    fig.canvas.draw_idle()


sN.on_changed(on_change)
sf1.on_changed(on_change)
sf2.on_changed(on_change)
rwin.on_clicked(on_change)
checks.on_clicked(on_change)
scust.on_changed(on_change)
on_change(None)
plt.show()
