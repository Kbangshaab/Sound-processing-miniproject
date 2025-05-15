import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
import numpy as np
import matplotlib.pyplot as plt


fs = 44100
seconds = 5
cutoff_frequencies = [200, 500, 1000, 2000]
output_filename_base = "filtreret_lyd_cutoff_"

#Optag lyd
print("Optager...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Optagelse færdig.")

#Afspil original lyd
print("Afspiller original lyd...")
sd.play(recording, fs)
sd.wait()
print("Afspilning af original lyd færdig.")

#Lav low-pass filter funktioner
def butter_lowpass(cutoff, fs, order=6):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_filter(data, cutoff, fs):
    b, a = butter_lowpass(cutoff, fs)
    y = lfilter(b, a, data)
    return y

#Anvend og afspil filter med forskellige cutoff-frekvenser
filtered_signals = {}
num_plots = len(cutoff_frequencies) + 1  # Antal plots inklusive original
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 6 * num_plots), dpi=100) # Opret alle subplots på én gang

#Plot det originale signal øverst
ax = axes[0]
ax.plot(np.arange(len(recording)) / fs, recording, label="Originalt signal")
ax.set_title("Originalt lydsignal")
ax.set_xlabel("Tid (sekunder)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True)

for i, cutoff in enumerate(cutoff_frequencies):
    print(f"\nAnvender low-pass filter med cutoff: {cutoff} Hz")
    filtered = apply_filter(recording[:, 0], cutoff, fs)
    filtered_signals[cutoff] = filtered

    print(f"Afspiller filtreret lyd (cutoff={cutoff} Hz)...")
    sd.play(filtered, fs)
    sd.wait()
    print(f"Afspilning af filtreret lyd (cutoff={cutoff} Hz) færdig.")

    # Gem den filtrerede lyd
    filename = f"{output_filename_base}{cutoff}Hz.wav"
    write(filename, fs, (filtered * 32767).astype(np.int16))
    print(f"Filtreret lyd (cutoff={cutoff} Hz) gemt som {filename}")

    # Plot det filtrerede signal
    ax = axes[i + 1]
    ax.plot(np.arange(len(filtered)) / fs, filtered, label=f"Filtreret (cutoff={cutoff} Hz)")
    ax.set_title(f"Filtreret lydsignal (cutoff={cutoff} Hz)")
    ax.set_xlabel("Tid (sekunder)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True)

def plot_frequency_domain(signal, fs, title="Frekvensdomæne"):
    from scipy.fft import fft, fftfreq
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[:N//2]
    plt.figure(figsize=(10, 4))
    plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    plt.title(title)
    plt.xlabel("Frekvens (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.tight_layout()
    plt.show()

# Eksempel:
plot_frequency_domain(recording[:, 0], fs, "Originalt signal - frekvensdomæne")

plt.tight_layout(pad=3.0)
plt.show()