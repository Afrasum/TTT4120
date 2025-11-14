# window_fft_demo.py
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


def compute_all(N, f1, f2, Nfft, win_kind):
    n = np.arange(N)
    w = make_window(N, win_kind)
    x = np.sin(2 * np.pi * f1 * n) + np.sin(2 * np.pi * f2 * n)
    xw = x * w
    fW, Wmag = mag_fft(w, Nfft)
    fXw, Xwmag = mag_fft(xw, Nfft)
    Wplot = Wmag / np.max(Wmag) if np.max(Wmag) > 0 else Wmag
    Xwplot = Xwmag / np.max(Xwmag) if np.max(Xwmag) > 0 else Xwmag
    return n, w, x, xw, fW, Wplot, fXw, Xwplot


N_init = 200
f1_init = 0.175
f2_init = 0.225
Nfft_init = 4096
win_init = "Rektangulær"
show_shift_init = True

n, w, x, xw, fW, Wplot, fXw, Xwplot = compute_all(
    N_init, f1_init, f2_init, Nfft_init, win_init
)

fig_w = plt.figure(figsize=(7, 3))
ax_w = fig_w.add_axes([0.1, 0.2, 0.85, 0.7])
ax_w.stem(n, w, use_line_collection=True)
ax_w.set_xlim([-1, N_init])
ax_w.set_xlabel("n")
ax_w.set_ylabel("w[n]")
ax_w.set_title(f"Vindu i tid: w[n]  (type: {win_init}, N={N_init})")
ax_w.grid(True, linestyle=":")

fig_xt = plt.figure(figsize=(7, 3))
ax_xt = fig_xt.add_axes([0.1, 0.2, 0.85, 0.7])
ax_xt.plot(n, x, label="x[n]")
ax_xt.plot(n, xw, label="x[n]·w[n]")
ax_xt.set_xlim([0, N_init - 1])
ax_xt.set_xlabel("n")
ax_xt.set_ylabel("Amplitude")
ax_xt.legend(loc="upper right")
ax_xt.grid(True, linestyle=":")
ax_xt.set_title("Signal og vinduet signal i tid")

fig_f = plt.figure(figsize=(7, 4))
ax_f = fig_f.add_axes([0.1, 0.2, 0.85, 0.7])
ax_f.plot(fW[: Nfft_init // 2 + 1], Wplot[: Nfft_init // 2 + 1], label="|W(f)| (norm.)")
ax_f.plot(
    fXw[: Nfft_init // 2 + 1], Xwplot[: Nfft_init // 2 + 1], label="|X_w(f)| (norm.)"
)
ax_f.axvline(f1_init, linestyle="--")
ax_f.axvline(f2_init, linestyle="--")
ax_f.set_xlim([0, 0.5])
ax_f.set_xlabel("Normalisert frekvens f")
ax_f.set_ylabel("Normalisert magnitude")
ax_f.grid(True, linestyle=":")
ax_f.legend(loc="upper right")
ax_f.set_title("Frekvensdomene: |W(f)|, |X_w(f)| og skiftede |W| (0–0.5)")

fig_ctrl = plt.figure(figsize=(7, 3.6))
axN = fig_ctrl.add_axes([0.12, 0.70, 0.75, 0.10])
axf1 = fig_ctrl.add_axes([0.12, 0.50, 0.75, 0.10])
axf2 = fig_ctrl.add_axes([0.12, 0.30, 0.75, 0.10])
axfft = fig_ctrl.add_axes([0.12, 0.10, 0.75, 0.10])
axwin = fig_ctrl.add_axes([0.88, 0.55, 0.10, 0.35])
axchk = fig_ctrl.add_axes([0.88, 0.15, 0.10, 0.25])

sN = Slider(axN, "N", 10, 2000, valinit=N_init, valstep=1)
sf1 = Slider(axf1, "f1", 0.02, 0.48, valinit=f1_init, valstep=0.001)
sf2 = Slider(axf2, "f2", 0.02, 0.48, valinit=f2_init, valstep=0.001)
sfft = Slider(axfft, "Nfft", 256, 16384, valinit=Nfft_init, valstep=256)
rwin = RadioButtons(axwin, ("Rektangulær", "Hann", "Hamming"))
chk = CheckButtons(axchk, ["Vis skiftede |W|"], [show_shift_init])
fig_ctrl.suptitle("Kontroller", y=0.98)


def on_change(_):
    N = int(sN.val)
    f1 = float(sf1.val)
    f2 = float(sf2.val)
    Nfft = int(sfft.val)
    win_kind = rwin.value_selected
    show_shift = chk.get_status()

    n, w, x, xw, fW, Wplot, fXw, Xwplot = compute_all(N, f1, f2, Nfft, win_kind)

    ax_w.clear()
    ax_w.stem(n, w, use_line_collection=True)
    ax_w.set_xlim([-1, N])
    ax_w.set_xlabel("n")
    ax_w.set_ylabel("w[n]")
    ax_w.set_title(f"Vindu i tid: w[n]  (type: {win_kind}, N={N})")
    ax_w.grid(True, linestyle=":")
    fig_w.canvas.draw_idle()

    ax_xt.clear()
    ax_xt.plot(n, x, label="x[n]")
    ax_xt.plot(n, xw, label="x[n]·w[n]")
    ax_xt.set_xlim([0, max(N - 1, 1)])
    ax_xt.set_xlabel("n")
    ax_xt.set_ylabel("Amplitude")
    ax_xt.legend(loc="upper right")
    ax_xt.grid(True, linestyle=":")
    ax_xt.set_title("Signal og vinduet signal i tid")
    fig_xt.canvas.draw_idle()

    ax_f.clear()
    ax_f.plot(fW[: Nfft // 2 + 1], Wplot[: Nfft // 2 + 1], label="|W(f)| (norm.)")
    ax_f.plot(fXw[: Nfft // 2 + 1], Xwplot[: Nfft // 2 + 1], label="|X_w(f)| (norm.)")
    ax_f.axvline(f1, linestyle="--")
    ax_f.axvline(f2, linestyle="--")
    if show_shift:
        k1 = int(np.round(f1 * Nfft)) % Nfft
        k2 = int(np.round(f2 * Nfft)) % Nfft
        Wshift1 = np.roll(Wplot, k1)
        Wshift2 = np.roll(Wplot, k2)
        ax_f.plot(
            fW[: Nfft // 2 + 1],
            Wshift1[: Nfft // 2 + 1],
            linestyle="--",
            label="|W(f-f1)| (norm.)",
        )
        ax_f.plot(
            fW[: Nfft // 2 + 1],
            Wshift2[: Nfft // 2 + 1],
            linestyle="--",
            label="|W(f-f2)| (norm.)",
        )
    ax_f.set_xlim([0, 0.5])
    ax_f.set_xlabel("Normalisert frekvens f")
    ax_f.set_ylabel("Normalisert magnitude")
    ax_f.grid(True, linestyle=":")
    ax_f.legend(loc="upper right")
    ax_f.set_title("Frekvensdomene: |W(f)|, |X_w(f)| og skiftede |W| (0–0.5)")
    fig_f.canvas.draw_idle()


sN.on_changed(on_change)
sf1.on_changed(on_change)
sf2.on_changed(on_change)
sfft.on_changed(on_change)
rwin.on_clicked(lambda _: on_change(None))
chk.on_clicked(lambda _: on_change(None))
on_change(None)
plt.show()
