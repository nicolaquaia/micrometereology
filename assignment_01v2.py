import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.optimize import minimize
from scipy.special import gamma


def plot_CDF(array_data, color, label):
    sorted_data_x = np.sort(array_data)
    cumulative_data = np.cumsum(sorted_data_x) / np.sum(sorted_data_x)
    plt.plot(sorted_data_x, cumulative_data, color=color, label=label)

def plot_Gaussian(array_data, mu, sig, color, label):
    x = np.linspace(min(array_data), max(array_data), num=100)
    y = 1/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sig)**2)
    plt.plot(x, y, color=color, label=label)


def plot_histogram(array_data, color, label):
    x = np.linspace(min(array_data), max(array_data), num=100)
    y = np.histogram(array_data, bins=100, density=True)
    width_bar = (max(array_data) - min(array_data)) / 100 + 0.1
    plt.bar(x, y[0], width=width_bar, align='center', color=color, label=label)

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




# Hovsore data set

hov = pd.read_csv('hovsore_1.txt', delimiter=',',header=None)
hov_mean = np.mean(hov[1])    #= mu
hov_std = np.std(hov[1])      #= sigma
#print(hov_mean, hov_std)

#plot_Gaussian(hov[1], hov_mean, hov_std, 'r', 'Gaussian')
#plot_histogram(hov[1], 'b', 'Histrogram')
#plot_CDF(hov[1], 'b', 'velocity')

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

sprog_mean = np.mean(sprog_data, axis=0)
sprog_var = np.var(sprog_data, axis=0)

#print(sprog_mean[0])                        # 8.224218093841214
#print(sprog_var[0])                         # 15.241837914881087

#sprog_velocity = [row[0] for row in sprog_data]
#sprog_velocity_array = np.array(sprog_velocity)


#counter = 0
#for i in range(len(sprog_velocity)):
#    if sprog_velocity[i] < 0.4:
#        counter = counter + 1
#print(counter)
#print(counter/len(sprog_velocity)*100)


#plot_CDF(sprog_velocity, 'r', label='sprog')
#plot_histogram(sprog_velocity, 'b', 'histrogram')

plt.legend()
plt.show()

est_k,est_lambda = method1(sprog_mean[0],sprog_var[0])
print(est_k)
print(est_lambda)
