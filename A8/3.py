import numpy as np
import sounddevice as sd
from scipy.io import loadmat
from scipy.linalg import toeplitz
from scipy.signal import lfilter


def lpc(x, p=10):
    x -= np.mean(x)
    r = np.correlate(x, x, "full")[len(x) - 1 : len(x) + p + 1] / len(x)
    a = np.linalg.solve(toeplitz(r[:p]) + 1e-6 * np.eye(p), r[1 : p + 1])
    return a, r[0] - np.dot(a, r[1 : p + 1])


def transform(x, y, p=10):
    a1, e1 = lpc(x, p)
    a2, e2 = lpc(y, p)
    exc = lfilter(np.r_[1, a1], [1], x) * np.sqrt(abs(e2 / e1))
    z = lfilter([1], np.r_[1, 1.5 * a2], exc)
    z /= np.max(np.abs(z)) + 1e-9
    return z


d = loadmat("vowels.mat", squeeze_me=True, struct_as_record=False)
v = d["v"]
fs = int(d["fs"])
for i, x in enumerate(v):
    print(i, len(x))
tgt = np.array(v[int(input())], dtype=np.float32).flatten()
src = sd.rec(int(2 * fs), samplerate=fs, channels=1, dtype="float32")
sd.wait()
src = src.flatten()
out = transform(src, tgt)
sd.play(src / np.max(np.abs(src)), fs)
sd.wait()
sd.play(out, fs)
sd.wait()
