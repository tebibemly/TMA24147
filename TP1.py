import numpy as np
import matplotlib.pyplot as plt

# 1.1 Génération d'un signal sinusoïdal
A = 1          # Amplitude
f0 = 10        # Fréquence en Hz
phi = 0        # Phase
fs = 100       # Fréquence d'échantillonnage
T = 1          # Durée en secondes

t = np.arange(0, T, 1/fs)
x = A * np.sin(2 * np.pi * f0 * t + phi)

plt.figure(figsize=(10,5))
plt.plot(t, x)
plt.title("Signal sinusoïdal x(t)")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# 1.2 Ajout de bruit
sigma = 0.3  # Ecart-type du bruit
bruit = np.random.normal(0, sigma, size=x.shape)
y = x + bruit

plt.figure(figsize=(8,4))
plt.plot(t, x, label="Signal pur")
plt.plot(t, y, label="Signal bruité", alpha=0.7)
plt.title("Signal sinusoïdal avec bruit")
plt.xlabel("Temps [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# 1.3 Convolution d'un signal porte
rect = np.zeros(100)
rect[20:41] = 1  # indices de 20 à 40 inclus

conv_rect = np.convolve(rect, rect)

plt.figure(figsize=(10,5))
plt.plot(conv_rect)
plt.title("Convolution du signal porte avec lui-même")
plt.xlabel("Indice")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()