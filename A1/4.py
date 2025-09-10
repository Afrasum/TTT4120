import matplotlib.pyplot as plt
import numpy as np


def x(n):
    # 1 if n=0 else 0
    return 1 * (n == 0)


def y1(n):
    return x(n) + 2 * x(n - 1) + x(n - 2)


def y2(n):
    if n < 0:
        return 0
    return -0.9 * y2(n - 1) + x(n)


# plit y1 from -2 - 5
n = np.arange(-2, 6)
plt.stem(n, y1(n))
plt.xlabel("n")
plt.ylabel("y1[n]")
plt.title("y1[n] = x[n] + 2x[n-1] + x[n-2]")
plt.grid(True)

# plot y2 from -5-10
# ---- System 2 (iterativ)
n2 = np.arange(-5, 50)
h2 = np.zeros_like(n2, dtype=float)
for i, ni in enumerate(n2):
    if ni < 0:
        h2[i] = 0.0  # kausalitet → h[n]=0 for n<0
    elif ni == 0:
        h2[i] = 1.0  # h[0] = -0.9*h[-1] + δ[0] = 1
    else:
        h2[i] = -0.9 * h2[i - 1]  # rekursjon for n≥1

plt.figure(figsize=(9, 3))
plt.stem(n2, h2)
plt.title("h2[n] via iterasjon")
plt.xlabel("n")
plt.ylabel("h2[n]")
plt.grid(True)


plt.show()
