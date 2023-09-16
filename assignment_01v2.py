import numpy as np
import math as m
import pandas as pd
import matplotlib.pyplot as plt
import csv

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

with open("sprog.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")
    for line in tsv_file:
        if float(line[1]) < 90:
            if float(line[2]) < 900:
                sprog_data.append([float(line[1]), float(line[2])])
            elif float(line[3]) < 900:
                sprog_data.append([float(line[1]), float(line[3])])

#sprog_mean = np.mean(sprog_data, axis=0)
#sprog_var = np.var(sprog_data, axis=0)

#print(sprog_mean[0])                        # 8.224218093841214
#print(sprog_var[0])                         # 15.241837914881087

sprog_velocity = [row[0] for row in sprog_data]
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


