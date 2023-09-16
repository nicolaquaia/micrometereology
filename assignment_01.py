import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gamma

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
    print("Estimated k:", estimated_k)
    print("Estimated lambda:", estimated_lambda)
    mu = estimated_lambda * gamma(1 + 1/estimated_k)
    m2 = (estimated_lambda**2 * gamma(1 + 2/estimated_k)) - mu**2
    print(mu)
    print(m2)
    return([estimated_k,estimated_lambda])




# Hovsore data set

hov = pd.read_csv('hovsore_1.txt', delimiter=',',header=None)
velocity = hov[1]
hov_mean = np.mean(velocity)    #= mu
hov_std = np.std(velocity)      #= sigma
#print(hov_mean, hov_std)

# Gaussian distribution
def gaussian(x, mu, sig):
    f = 1/(sig*np.sqrt(2*np.pi)) * np.exp(-1/2*((x-mu)/sig)**2)
    return(f)
x = np.linspace(min(velocity),max(velocity),num = 100)
y_gau = gaussian(x, hov_mean, hov_std)
plt.plot(x, y_gau, color='b',label='Gaussian dist')


# histogram
y_hist = np.histogram(velocity, bins=100, density=True)
width_bar = (max(velocity) - min(velocity))/100
#print(width_bar)
plt.bar(x, y_hist[0], width=0.2, align='center', color='r', label='histogram')

# plot
plt.title("Hovsore data distribution")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()

#plt.show()




# sprog
sprog = pd.read_csv('sprog.tsv', delimiter='\t', header=None)
speed70 = sprog[1]
direct65 = sprog[2]
speed70_mean = np.mean(speed70)    #= mu
speed70_std = np.std(speed70)      #= sigma
speed70_var = np.var(speed70)
#print(speed70_mean, speed70_std)


counter = 0
for i in direct65:
    if i == 999:
        counter = counter + 1
#print(counter)


est_k,est_lambda = method1(speed70_mean, speed70_var) #the invalids haven't been filtered out
print(speed70_mean, speed70_var)
print("##########################################################")
est_k,est_lambda = method1(8.236,15.272)

#mean = 8.236
#variance = 15.272