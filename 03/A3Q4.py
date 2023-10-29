import numpy as np
import matplotlib.pyplot as plt

# Constants
N = 12000  # Record length, 10 minutes with delta_t = 0.05 s
z = 80  # Height
U = 7.0  # Wind speed
u_star = 0.25  # Friction velocity
delta_t = 0.05  # Time step

# Generate the frequency values
frequencies = np.fft.fftfreq(N, d=delta_t)
#print("FREQUENCIES:",frequencies)
#frequencies = np.linspace(0, 10, N)#//=division by whole numbers
#print("FREQUENCIES:",frequencies)
# Kaimal spectrum function
def kaimal_spectrum(f, u_star, z, U):
    #print("yah")
    return u_star**2 * (52.5 * z / U) / ((1 + 33 * f * z / U) ** 5/3)

def ks(f, u_star, z, U):
    x = u_star**2 * (52.5 * z / U) / ((1 + 33 * f * z / U) ** 5/3)
    #print(x)
    return u_star**2 * ((52.5 * z / U) / (1 + 33 * f * z / U)**(5/3))

# Generate random complex numbers
print(len(frequencies))
X_l = np.random.normal(0, 1, size=N) + 1j * np.random.normal(0, 1, size=N)
print(len(X_l))
#print("X_l:",X_l)

# Calculate the correct Fourier amplitudes using the Kaimal spectrum
#print("kaimal")
#amplitudes = np.sqrt(2 * kaimal_spectrum(frequencies, u_star, z, U) * delta_t)
kaimal = kaimal_spectrum(frequencies, u_star, z, U)
amplitudes = []
for i in range(len(kaimal)):
    if kaimal[i] < 0:
        #print(kaimal[i])
        amplitudes.append(np.sqrt(2*(kaimal[i]+0J)*delta_t))
        #print(kaimal[i]+0J)
    else:
        amplitudes.append(np.sqrt(2*(kaimal[i])*delta_t))
#amplitudes = np.sqrt(2 * kaimal * delta_t)



#print("AMPLITUDES:",amplitudes)
"""
for i in range(len(amplitudes)):
    if (np.isnan(amplitudes[i])):
        print(i,": ", amplitudes[i])
        #x = ks(frequencies[i], u_star, z, U)
        #print(x)
"""

# Apply the amplitudes to the random complex numbers
X_l *= amplitudes

# Complex conjugate
X_conjugate = np.conjugate(X_l)
print(len(X_conjugate))


# Combine the complex conjugates to create a real-valued signal
X_bar = np.zeros(N, dtype=np.complex)
X_bar[0] = X_l[0]
X_bar[1:(N//2)-1] = X_l[1:(N // 2)-1] #+ X_conjugate[1:N // 2]
X_bar[N // 2] = X_l[(N//2)-1]
X_bar[N // 2 + 1:N] = np.flip(X_conjugate[1:N//2])

"""
# Combine the complex conjugates to create a real-valued signal
X_bar = np.zeros(N, dtype=np.complex)
X_bar[0] = X_l[0]
X_bar[1:N // 2] = X_l[1:N // 2] + X_conjugate[1:N // 2]
#print(len(X_l))
#print(N//2)
X_bar[N // 2] = X_l[(N // 2)-1]

X_bar[N // 2 + 1:N] = X_conjugate[N // 2 + 1:N]
"""


# Inverse Fourier transform and normalize
u_t = np.fft.ifft(X_bar).real / N

# Calculate the spectrum from the synthesized time series
spectrum = np.abs(np.fft.fft(u_t))**2 / (N * delta_t)

# Plot u(t)
"""
time = np.arange(0, N) * delta_t
plt.figure(figsize=(10, 6))
plt.plot(time, u_t)
#print(u_t)
plt.xlabel("Time (s)")
plt.ylabel("u(t)")
plt.title("Synthesized u(t) with Kaimal Spectrum")
plt.grid(True)
#plt.show()
"""

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, spectrum)
plt.xlim(0,10)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectrum")
plt.title("Spectrum of Synthesized u(t)")
plt.grid(True)
plt.show()
