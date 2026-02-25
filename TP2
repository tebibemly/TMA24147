import numpy as np
import matplotlib.pyplot as plt

# 2.1 Signal sonore composé de deux fréquences
f1, f2 = 440, 880  # Hz
fs_audio = 44100   # Hz
duration = 1  # 1 seconde

t_audio = np.arange(0, duration, 1/fs_audio)
signal = np.sin(2*np.pi*f1*t_audio) + np.sin(2*np.pi*f2*t_audio)

# FFT
N = len(signal)
X = np.fft.fft(signal)
freqs = np.fft.fftfreq(N, 1/fs_audio)

plt.figure(figsize=(10,5))
plt.plot(freqs[:N//2], np.abs(X[:N//2]))
plt.title("Spectre du signal sonore pur")
plt.xlabel("Fréquence [Hz]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# 2.2 Ajout de bruit et filtrage
f_bruit = 5000   # Hz
A_bruit = 0.2    # Amplitude du bruit
signal_bruite = signal + A_bruit*np.sin(2*np.pi*f_bruit*t_audio)

# FFT du signal bruité
X_bruite = np.fft.fft(signal_bruite)

# Filtrage : couper autour de f_bruit
bandwidth = 10  # largeur de bande en Hz
idx_bruit = np.where((freqs > f_bruit - bandwidth) & (freqs < f_bruit + bandwidth))
X_bruite[idx_bruit] = 0
X_bruite[-idx_bruit] = 0  # symétrie pour partie négative

# Reconstruction du signal filtré
signal_filtre = np.fft.ifft(X_bruite)

plt.figure(figsize=(10,5))
plt.plot(t_audio[:1000], signal_bruite[:1000], label="Signal bruité")
plt.plot(t_audio[:1000], signal_filtre[:1000].real, label="Signal filtré", alpha=0.8)
plt.title("Comparaison signal bruité et filtré")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()