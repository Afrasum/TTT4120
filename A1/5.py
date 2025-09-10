import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Common definitions (Task 5)
# ----------------------------
x = np.array([1, 2, 3])  # input signal
h1 = np.array([1, 1, 1])  # FIR: δ[n]+δ[n-1]+δ[n-2]
h2 = 0.9 ** np.arange(11)  # IIR approx: (0.9)^n, n=0..10


def stem_seq(ax, n, y, title):
    ax.stem(n, y)
    ax.set_title(title)
    ax.set_xlabel("n")
    ax.set_ylabel("value")
    ax.grid(True)


# ----------------------------
# 5(a) y1 = x * h1
# ----------------------------
y1 = np.convolve(x, h1)
n1 = np.arange(len(y1))

# ----------------------------
# 5(b) y2_only = x * h2
# ----------------------------
y2_only = np.convolve(x, h2)
n2_only = np.arange(len(y2_only))

# ----------------------------
# 5(d) Cascade: y2 = (x*h1)*h2
# ----------------------------
h_cas = np.convolve(h1, h2)  # cascaded impulse response
y2 = np.convolve(x, h_cas)  # output after both systems
n2 = np.arange(len(y2))

# ----------------------------
# EXTRA: (x*h2)*h1
# ----------------------------
y2_alt = np.convolve(np.convolve(x, h2), h1)
n2_alt = np.arange(len(y2_alt))

# ----------------------------
# Plots
# ----------------------------
fig, axs = plt.subplots(5, 1, figsize=(10, 14), constrained_layout=True)

stem_seq(axs[0], n1, y1, "5(a): y1[n] = x * h1")
stem_seq(axs[1], n2_only, y2_only, "5(b): y2_only[n] = x * h2")
stem_seq(
    axs[2],
    np.arange(len(h_cas)),
    h_cas,
    "5(d): cascaded impulse response h_cas = h1 * h2",
)
stem_seq(axs[3], n2, y2, "5(d): y2[n] = (x*h1)*h2")
stem_seq(axs[4], n2_alt, y2_alt, "Extra: (x*h2)*h1")

plt.show()

# ----------------------------
# Checks
# ----------------------------
print("y1 (5a)      =", y1)
print("y2_only (5b) =", np.round(y2_only, 4))
print("h_cas (5d)   =", np.round(h_cas, 4))
print("y2 (x*h1*h2) =", np.round(y2, 4))
print("y2_alt (x*h2*h1) =", np.round(y2_alt, 4))
print("Associativity holds?", np.allclose(y2, y2_alt))
