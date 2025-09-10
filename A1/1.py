# x[n] = 5-n if n in [0,4] else 0
# y[n] = 1 if 2 <= n <= 4 else 0

import matplotlib.pyplot as plt
import numpy as np


def x(n):
    if 0 <= n <= 4:
        return 5 - n
    else:
        return 0


def y(n):
    if 2 <= n <= 4:
        return 1
    else:
        return 0


# plot x[n] and y[n] in separate windows
n = np.arange(-5, 10, 1)
x_n = [x(i) for i in n]
y_n = [y(i) for i in n]

# Plot x[n] in first window
plt.figure(1)
plt.stem(n, x_n, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal x[n]")
plt.grid()

# Plot y[n] in second window
plt.figure(2)
plt.stem(n, y_n, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal y[n]")
plt.grid()


# plot x[n-k] for k=3 and -3
k1 = 3
k2 = -3
x_n_k1 = [x(i - k1) for i in n]
x_n_k2 = [x(i - k2) for i in n]
plt.figure(3)
plt.stem(n, x_n_k1, basefmt=" ", linefmt="C1-", markerfmt="C1o", label="x[n-3]")
plt.stem(n, x_n_k2, basefmt=" ", linefmt="C2-", markerfmt="C2o", label="x[n+3]")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal x[n-k] for k=3 and -3")
plt.legend()
plt.grid()


# plot x[-n]
x_n_neg = [x(-i) for i in n]
plt.figure(4)
plt.stem(n, x_n_neg, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal x[-n]")
plt.grid()


# plot x[5-n]
x_n_5_neg = [x(5 - i) for i in n]
plt.figure(5)
plt.stem(n, x_n_5_neg, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal x[5-n]")
plt.grid()

# ploit x*y
x_y = [x(i) * y(i) for i in n]
plt.figure(6)
plt.stem(n, x_y, basefmt=" ")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.title("Signal x[n] * y[n]")
plt.grid()
plt.show()

# energy of x[n]
energy_x = sum([x(i) ** 2 for i in n])
print("Energy of x[n]:", energy_x)
