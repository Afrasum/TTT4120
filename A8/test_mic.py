import sounddevice as sd

fs = 8000
print("Recording...")
rec = sd.rec(int(2 * fs), samplerate=fs, channels=1, dtype="float32")
sd.wait()
print("Playing...")
sd.play(rec, fs)
sd.wait()
