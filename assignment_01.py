import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt


# Hovsore data set

hov = pd.read_csv('hovsore_1.txt', delimiter=',',header=None)
velocity = hov[1]
hov_mean = np.mean(velocity)    #= mu
hov_std = np.std(velocity)      #= sigma
print(hov_mean, hov_std)

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
print(width_bar)
plt.bar(x, y_hist[0], width=0.2, align='center', color='r', label='histogram')

# plot
plt.title("Hovsore data distribution")
plt.xlabel("x")
plt.ylabel("PDF")
plt.legend()

plt.show()




# sprog
sprog = pd.read_csv('sprog.tsv', delimiter='\t', header=None)
speed70 = sprog[1]
direct65 = sprog[2]
speed70_mean = np.mean(speed70)    #= mu
speed70_std = np.std(speed70)      #= sigma
#print(speed70_mean, speed70_std)


counter = 0
for i in direct65:
    if i == 999:
        counter = counter + 1
#print(counter)
