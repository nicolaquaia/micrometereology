import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.optimize import minimize
from scipy.special import gamma


def plot_CDF(array_data, color, label):
    # plots the CDF given an array of data
    sorted_data_x = np.sort(array_data)
    cumulative_data = np.cumsum(sorted_data_x) / np.sum(sorted_data_x)
    plt.plot(sorted_data_x, cumulative_data, color=color, label=label)

def plot_Gaussian(array_data, mu, sig, color, label):
    # computes the gaussian distribution given mu and sig and plots it in the range of the data to avoid zeros
    x = np.linspace(min(array_data), max(array_data), num=100)
    y = 1/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sig)**2)
    plt.plot(x, y, color=color, label=label)


def plot_histogram(array_data, color, label):
    # plots the histogram of an array of data
    x = np.linspace(min(array_data), max(array_data), num=100)
    y = np.histogram(array_data, bins=100, density=True)
    width_bar = (max(array_data) - min(array_data)) / 100 + 0.1
    plt.bar(x, y[0], width=width_bar, align='center', color=color, label=label)
    area = np.sum(y[0]*(width_bar-0.1))
    print(f"area under graph = {area:.5f}")


def plot_Weibull(array_data, k, lam, color, label):
    # computes the Weibull distribution given k and lamda and plots it in the range of the data to avoid zeros
    x = np.linspace(0.001, max(array_data), num=100)
    y = k/x * (x / lam)**k * np.exp(-(x/lam)**k)
    plt.plot(x, y, color=color, label=label)

def method1(mean,variance):
    # Define the moments (mean and variance) of your data
    #mean = 8.236
    #variance = 15.272

    # Define a function to calculate the moments of the Weibull distribution
    def weibull_moments(params):
        k, lambd = params
        mu = lambd * gamma(1 + 1/k)
        m2 = (lambd**2 * gamma(1 + 2/k)) - mu**2
        return np.array([mu, m2])

    #Define a function to calculate the difference between observed and estimated moments
    def moment_error(params):
        observed_moments = np.array([mean, variance])# + mean**2])
        estimated_moments = weibull_moments(params)
        return np.sum((observed_moments - estimated_moments)**2)

    # Initial guesses for k and lambd
    initial_guess = [1.0, 1.0]

    # Minimize the error function to estimate parameters
    result = minimize(moment_error, initial_guess, method='Nelder-Mead')

    # Extract estimated parameters
    estimated_k, estimated_lambda = result.x

    # Print the estimated parameters
    #print("Estimated k:", estimated_k)
    #print("Estimated lambda:", estimated_lambda)
    mu = estimated_lambda * gamma(1 + 1/estimated_k)
    m2 = (estimated_lambda**2 * gamma(1 + 2/estimated_k)) - mu**2
    return([estimated_k,estimated_lambda])


def divide_in_sector(data_matrix):
    # takes a list with entries [velocity, direction] and spits out a list of velocities divided in 12 sectors
    data_sorted = sorted(data_matrix, key=lambda x: x[1])
    data_sectors = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for j in range(1, len(data_sectors)):
        for i in range(len(data_sorted)):
            low = (j - 1) * 30 + 15
            high = (j) * 30 + 15
            value = data_sorted[i][1]
            if low <= value < high:
                data_sectors[j].append(data_sorted[i][0])
    for i in range(len(data_sorted)):
        value = data_sorted[i][1]
        if value < 15 or value >= 345:
            data_sectors[0].append(data_sorted[i][0])
    return data_sectors

def plot_weibull_sector(data_sectors, num, colorweibull, colorhistogram):
    sector_mean = np.mean(data_sectors[num])
    sector_var = np.var(data_sectors[num])
    est_k, est_lambda = method1(sector_mean, sector_var)

    plot_Weibull(data_sectors[num], est_k, est_lambda, colorweibull, 'theoretical')
    plot_histogram(data_sectors[num], colorhistogram, 'histrogram')
    print(f"# for sector {num:.0f}: k = {est_k:.5f} and lam = {est_lambda:.5f}")


# Hovsore data set

hov = pd.read_csv('hovsore_1.txt', delimiter=',',header=None)

hov_mean = np.mean(hov[1])    #= mu
hov_std = np.std(hov[1])      #= sigma
#print(hov_mean, hov_std)

#plot_histogram(hov[1], 'b', 'Histrogram')

#plot_Gaussian(hov[1], hov_mean, hov_std, 'r', 'Gaussian')

#plot_CDF(hov[1], 'b', 'cumulative')

#plt.title("Hovsore data distribution")
#plt.xlabel("x")
#plt.ylabel("PDF")



# sprog

sprog_data = []

# returns matrx with 2 columns, [0] is the 'cleaned' velocity and [1] is the direction
with open("sprog.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        if float(line[1]) < 90:
            if float(line[2]) < 900:
                sprog_data.append([float(line[1]), float(line[2])])
            elif float(line[3]) < 900:
                sprog_data.append([float(line[1]), float(line[3])])

sprog_mean = np.mean(sprog_data, axis=0)    # 8.224218093841214
sprog_var = np.var(sprog_data, axis=0)      # 15.241837914881087
sprog_std = np.std(sprog_data, axis=0)      # 3.9040796501712265

#print(sprog_mean[0])
#print(sprog_std[0])

sprog_velocity = [row[0] for row in sprog_data]

#plot_CDF(sprog_velocity, 'r', label='sprog')

#plot_histogram(sprog_velocity, 'r', 'Histrogram')

#est_k, est_lambda = method1(sprog_mean[0],sprog_var[0])

#plot_Weibull(sprog_velocity, est_k, est_lambda, 'b', 'theoretical')

sprog_sectors = divide_in_sector(sprog_data)

plot_weibull_sector(sprog_sectors, 11, 'k', 'b')
# b r g c m y
# for sector 0: k = 1.92232 and lam = 7.77200
# for sector 1: k = 1.88088 and lam = 7.58810
# for sector 2: k = 1.94274 and lam = 7.59869
# for sector 3: k = 2.14987 and lam = 8.35945
# for sector 4: k = 2.35385 and lam = 9.72695
# for sector 5: k = 2.19503 and lam = 8.89905
# for sector 6: k = 2.04671 and lam = 8.85117
# for sector 7: k = 2.36101 and lam = 9.80219
# for sector 8: k = 2.47373 and lam = 9.91745
# for sector 9: k = 2.65953 and lam = 10.06390
# for sector 10: k = 2.37962 and lam = 10.20500
# for sector 11: k = 1.98185 and lam = 8.45294




plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.xlabel('x')
plt.ylabel('f (x)')

#plt.title("Høvsøre data and function")
plt.title("Sector 11")

plt.legend()
plt.savefig("sec11.pdf", format='pdf')
plt.show()




