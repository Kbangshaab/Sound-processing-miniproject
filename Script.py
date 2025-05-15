import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import librosa
import matplotlib.pyplot as plt

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

print("Optager...")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Optagelse færdig.")

# Konverter til 1D og gem som .wav
write("optagelse.wav", fs, myrecording)

# Brug librosa til at loade og analysere
y, sr = librosa.load("optagelse.wav")

# PITCH DETEKTION
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Find den mest tydelige pitch per frame
pitches_detected = []
for i in range(pitches.shape[1]):
    index = magnitudes[:, i].argmax()
    pitch = pitches[index, i]
    pitches_detected.append(pitch)

# Konverter 0'er til NaN så vi ikke får flad linje
pitches_detected = np.array(pitches_detected)
pitches_detected[pitches_detected == 0] = np.nan

# Plot lyd og pitch
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(y)
plt.title("Tidsdomæne (signal)")

plt.subplot(2, 1, 2)
plt.plot(pitches_detected)
plt.title("Estimeret Pitch (Hz)")

plt.tight_layout()
plt.show()

#Correlation
def autocorrelation_pitch(signal, sr, fmin=50, fmax=1000):
    # Begræns signalet til f.eks. 1 sekund (ellers bliver det for meget at arbejde med)
    signal = signal[:sr]
    corr = np.correlate(signal, signal, mode='full')
    corr = corr[len(corr)//2:]  # behold kun positiv lag
    # Find peak i det relevante område
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    peak_index = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch = sr / peak_index
    return pitch, corr

# Eksempel brug:
signal = y  # din lyd
pitch, corr = autocorrelation_pitch(signal, sr)
print(f"Estimeret pitch: {pitch:.2f} Hz")

# Visualisering
plt.figure(figsize=(12, 4))
plt.plot(corr)
plt.title("Autokorrelation")
plt.xlabel("Lag")
plt.ylabel("Amplitude")
plt.show()
