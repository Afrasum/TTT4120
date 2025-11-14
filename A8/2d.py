import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz

w = np.linspace(0, np.pi, 1024)
psd_ma = 1.25 - np.cos(w)

ars = {
    1: (np.array([0.4]), 1.05),
    2: (np.array([0.47619048, 0.19047619]), 1.0119047619),
    3: (np.array([0.49411765, 0.23529412, 0.09411765]), 1.0029411765),
}

plt.figure()
plt.plot(w / np.pi, psd_ma, label="MA(1) true PSD")
for p, (a, s2) in ars.items():
    ww, H = freqz([1.0], np.r_[1.0, a], worN=w)
    psd_ar = s2 / (np.abs(H) ** 2)
    plt.plot(ww / np.pi, psd_ar, label=f"AR[{p}] PSD")

plt.xlabel("Normalized frequency (×π rad/sample)")
plt.ylabel("PSD")
plt.legend()
plt.title("PSD: MA(1) vs AR[1..3]")
plt.show()
