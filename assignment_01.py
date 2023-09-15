import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt


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
# columns: time | wind speed 70 | direct 67.5 | direct 70

speed70 = sprog[1]  # 23249 of 99.99
direct67 = sprog[2]
direct70 = sprog[3]

speed_list = []
direct_list = []

#for i in range(len(speed70)):
for i in range(len(speed70)):
    if speed70[i] < 90:
        speed_list.append(speed70[i])
        if direct70[i] < 900:
            direct_list.append(direct70[i])
        elif direct67[i] < 900:
            direct_list.append(direct67[i])
        else:
            direct_list.append(0)

speed_list_mean = np.mean(speed_list)       #= mu
speed_list_std = np.std(speed_list)
print(speed_list_mean)                      # 8.235566499680502
print(speed_list_std)                       # 3.907911767517807
