import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("PROBLEM 3b: DOWNSAMPLING WITH AND WITHOUT FILTERING")
print("=" * 80)

F1 = 900
F2 = 2000
Fs = 6000
D = 2

f1 = F1 / Fs
f2 = F2 / Fs

Fs_new = Fs / D
f1_new = F1 / Fs_new
f2_new = F2 / Fs_new

fc = 0.25

fig, axes = plt.subplots(4, 1, figsize=(16, 40), constrained_layout=True)


def plot_impulse(ax, f, height, color, label=None):
    ax.plot([f, f], [0, height], color=color, linewidth=3, label=label)
    ax.plot(f, height, "o", color=color, markersize=10)


def plot_periodic_impulses(ax, frequencies, height, color, num_periods=2):
    for freq in frequencies:
        for k in range(-num_periods, num_periods + 1):
            f_shifted = freq + k
            if -0.55 <= f_shifted <= 0.55:
                ax.plot(
                    [f_shifted, f_shifted],
                    [0, height],
                    color=color,
                    linewidth=2.5,
                    alpha=0.7,
                )
                ax.plot(f_shifted, height, "o", color=color, markersize=8, alpha=0.7)


ax = axes[0]
ax.axhline(y=0, color="k", linewidth=0.8)
ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

plot_impulse(ax, f1, 0.5, "blue", label=f"±{f1}")
plot_impulse(ax, -f1, 0.5, "blue")
plot_impulse(ax, f2, 0.5, "red", label=f"±{f2:.3f}")
plot_impulse(ax, -f2, 0.5, "red")

ax.axvline(x=fc, color="orange", linestyle="--", linewidth=2.5, alpha=0.7)
ax.axvline(x=-fc, color="orange", linestyle="--", linewidth=2.5, alpha=0.7)

ax.fill_between([-fc, fc], 0, 0.7, alpha=0.1, color="green")
ax.fill_between([fc, 0.5], 0, 0.7, alpha=0.1, color="red")
ax.fill_between([-0.5, -fc], 0, 0.7, alpha=0.1, color="red")

ax.set_xlim([-0.5, 0.5])
ax.set_ylim([0, 0.7])
ax.set_title("SIGNAL x[n] at Fs = 6000 Hz")
ax.set_xlabel("Normalized frequency f")
ax.set_ylabel("|X(f)|")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right")

ax = axes[1]
ax.axhline(y=0, color="k", linewidth=0.8)
ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

plot_impulse(ax, f1, 0.5, "blue", label=f"Surviving: ±{f1}")
plot_impulse(ax, -f1, 0.5, "blue")

plot_impulse(ax, f2, 0.5, "lightgray", label=f"Removed: ±{f2:.3f}")
plot_impulse(ax, -f2, 0.5, "lightgray")

ax.axvline(x=fc, color="orange", linestyle="--", linewidth=2.5, alpha=0.7)
ax.axvline(x=-fc, color="orange", linestyle="--", linewidth=2.5, alpha=0.7)

ax.set_xlim([-0.5, 0.5])
ax.set_ylim([0, 0.7])
ax.set_title("AFTER FILTERING at Fs = 6000 Hz")
ax.set_xlabel("Normalized frequency f")
ax.set_ylabel("|W(f)|")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right")

ax = axes[2]
ax.axhline(y=0, color="k", linewidth=0.8)
ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

plot_impulse(ax, f1_new, 0.5, "blue", label=f"±{f1_new}")
plot_impulse(ax, -f1_new, 0.5, "blue")

plot_periodic_impulses(ax, [f1_new, -f1_new], 0.5, "blue", num_periods=1)

ax.axvline(x=0.5, color="purple", linestyle="-.", linewidth=2, alpha=0.6)
ax.axvline(x=-0.5, color="purple", linestyle="-.", linewidth=2, alpha=0.6)

ax.set_xlim([-0.5, 0.5])
ax.set_ylim([0, 0.7])
ax.set_title("OUTPUT WITH FILTER at Fs_new = 3000 Hz")
ax.set_xlabel("Normalized f (Fs_new)")
ax.set_ylabel("|Y(f)|")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right")

ax = axes[3]
ax.axhline(y=0, color="k", linewidth=0.8)
ax.axvline(x=0, color="k", linewidth=0.5, alpha=0.3)

plot_impulse(ax, f1_new, 0.5, "blue", label=f"±{f1_new}")
plot_impulse(ax, -f1_new, 0.5, "blue")

f2_aliased = 1 - f2_new

plot_impulse(ax, f2_aliased, 0.5, "red", label=f"ALIASED ±{f2_aliased:.3f}")
plot_impulse(ax, -f2_aliased, 0.5, "red")

ax.axvline(x=0.5, color="purple", linestyle="-.", linewidth=2, alpha=0.6)
ax.axvline(x=-0.5, color="purple", linestyle="-.", linewidth=2, alpha=0.6)

ax.set_xlim([-0.5, 0.5])
ax.set_ylim([0, 0.7])
ax.set_title("OUTPUT WITHOUT FILTER (ALIASING!)", color="red")
ax.set_xlabel("Normalized f (Fs_new)")
ax.set_ylabel("|Y(f)| without filter")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10, loc="upper right")

plt.savefig(
    "problem3b_downsampling_spectra_no_overlap.png", dpi=150, bbox_inches="tight"
)
plt.show()
