import numpy as np
import sounddevice as sd
from scipy import signal
from scipy.io.wavfile import read, write

# Load the audio file
# Note: You need to have 'piano.wav' in your working directory
fs_audio, audio_data = read("piano.wav")

# Normalize the audio to prevent clipping
audio_normalized = audio_data / np.max(np.abs(audio_data))

print(f"Audio sample rate: {fs_audio} Hz")
print(
    f"Audio length: {len(audio_data)} samples = {len(audio_data)/fs_audio:.2f} seconds"
)

# Try different parameter combinations
test_cases = [
    {"R": 2205, "alpha": 0.3, "desc": "Short delay, weak echo (0.1s)"},
    {"R": 4410, "alpha": 0.5, "desc": "Medium delay, medium echo (0.2s)"},
    {"R": 11025, "alpha": 0.7, "desc": "Long delay, strong echo (0.5s)"},
    {"R": 22050, "alpha": 0.9, "desc": "Very long delay, very strong echo (1.0s)"},
]

for i, case in enumerate(test_cases):
    R = case["R"]
    alpha = case["alpha"]

    # Create filter
    b = np.zeros(R + 1)
    a = np.ones(1)
    b[0] = 1
    b[R] = alpha

    # Apply filter
    audio_filtered = signal.lfilter(b, a, audio_normalized)

    # Normalize output
    audio_filtered = audio_filtered / np.max(np.abs(audio_filtered))

    print(f"\n{i+1}. {case['desc']}")
    print(f"   Delay in seconds: {R/fs_audio:.3f}s")
    print("   Playing original...")
    sd.play(audio_normalized, fs_audio)
    sd.wait()

    print("   Playing filtered (with echo)...")
    sd.play(audio_filtered, fs_audio)
    sd.wait()

    # Optional: save to file
    output_filename = f"piano_echo_R{R}_alpha{int(alpha*100)}.wav"
    write(output_filename, fs_audio, (audio_filtered * 32767).astype(np.int16))
    print(f"   Saved to: {output_filename}")
