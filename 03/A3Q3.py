import numpy as np
import matplotlib.pyplot as plt

# Function to simulate the auto-regressive process
def simulate_ar_process(c_ar, N):
    # Calculate the initial value X1
    variance = 1 / (1 - c_ar**2)
    X1 = np.random.normal(0, np.sqrt(variance))

    # Simulate the time-series
    xi = np.random.normal(0, 1, N)  # Generate N random Gaussian variables
    time_series = np.empty(N)
    time_series[0] = X1

    for n in range(1, N):
        #time_series[n] = c_ar * time_series[n - 1] + xi[n]
        time_series[n] = c_ar * time_series[n - 1] + xi[n - 1]

    return time_series

# Parameters
N = 10**4  # Length of the time series
c_ar_values = [0.1, 0.4, 0.9]  # Different c_ar values
#c_ar_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# A: Simulate and store time series for each c_ar value
time_series_data = {}

for c_ar in c_ar_values:
    time_series = simulate_ar_process(c_ar, N)
    time_series_data[c_ar] = time_series

# Print or use the time series data as needed
#for c_ar, time_series in time_series_data.items():
    #print(f"Time Series for c_ar = {c_ar}:")
    #print(time_series)

# You can also save the time series data to a file for further analysis
# Example: np.savetxt("time_series_c0.1.txt", time_series_data[0.1])

###########
# B: Calculate mean and variance for each time series
mean_values = {}
variance_values = {}

for c_ar, time_series in time_series_data.items():
    mean_value = np.mean(time_series)  # Calculate mean
    variance_value = np.var(time_series)  # Calculate variance
    
    mean_values[c_ar] = mean_value
    variance_values[c_ar] = variance_value

# Print or use the mean and variance values as needed
for c_ar, mean_value in mean_values.items():
    print(f"Mean for c_ar = {c_ar}: {mean_value:.4f}")

for c_ar, variance_value in variance_values.items():
    print(f"Variance for c_ar = {c_ar}: {variance_value:.4f}")

############
# C

# Define the function to compute the autocorrelation function
def autocorrelation(data, max_lag):
    acf = [1.0]  # Autocorrelation at lag 0 is always 1
    N = len(data)
    
    for lag in range(1, max_lag + 1):
        acf_lag = np.corrcoef(data[:-lag], data[lag:])[0, 1]
        acf.append(acf_lag)
    
    return acf

# Define the theoretical ACF function based on the expression ρ(τ) = c_ar^τ
def theoretical_acf(c_ar, lags):
    return [c_ar**tau for tau in lags]

# Plot the autocorrelation functions for each c_ar value and the theoretical ACF
max_lag = 50  # Maximum lag to consider
lags = np.arange(max_lag + 1)

for c_ar, time_series in time_series_data.items():
    acf = autocorrelation(time_series, max_lag)
    theory_acf = theoretical_acf(c_ar, lags)
    
    plt.plot(lags, acf, label=f'Simulated c_ar = {c_ar}')
    #plt.semilogy(lags, acf, label=f'Simulated c_ar = {c_ar}')
    #plt.plot(lags, theory_acf, label=f'Theoretical c_ar = {c_ar}', linestyle='--')

#plt.semilogy(years, indexValues )

#plt.ylim([-0.1,1])

#plt.xlim([0,50])




plt.xlabel('Lag (τ)')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function Comparison with Theoretical ACF')
plt.legend()
plt.grid(True)
plt.show()


########
#D

# Function to calculate the power spectral density (PSD)
def calculate_psd(time_series, sampling_rate):
    N = len(time_series)
    fft_result = np.fft.fft(time_series)
    psd = np.abs(fft_result) ** 2 / (N * sampling_rate)
    return psd[:N // 2]  # Only the positive frequencies

# Parameters
sampling_rate = 1.0  # Adjust the sampling rate as needed (e.g., 1 Hz)
frequencies = np.fft.fftfreq(len(time_series_data[0.1]), 1 / sampling_rate)[:len(time_series_data[0.1]) // 2]

# Calculate and plot the power spectral density for each c_ar value
for c_ar, time_series in time_series_data.items():
    psd = calculate_psd(time_series, sampling_rate)
    plt.loglog(frequencies, psd, label=f'c_ar = {c_ar}')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (Log scale)')
plt.title('Power Spectral Density in Log-Log Coordinates')
plt.legend()
plt.grid(True)
plt.show()