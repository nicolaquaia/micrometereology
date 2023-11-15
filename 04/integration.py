

import numpy as np
from scipy.integrate import quad

def Weibull(U, k, A):
    y = k/U * (U/k)**k * np.exp(-(U/A)**k)
    return y

def Power1(U, Prated, Urp):
    P = Prated * (U/Urp)**3
    return P

def integrand1(U, k, A, Prated, Urp):
    return Weibull(U, k, A) * Power1(U, Prated, Urp)

def integrand2(U, k, A, Prated, Urp):
    return Weibull(U, k, A) * Prated

# Parameters
Prated = 13*10**9
Urp = 12
Ucutoff = 25
T=365*24

# sector 0
fi = 0.0552
k = 1.92
A = 7.77


# Integrate the product of Weibull and Power1
result1, error1 = quad(integrand1, 0.000001, Urp, args=(k, A, Prated, Urp))
result2, error2 = quad(integrand2, Urp, Ucutoff, args=(k, A, Prated, Urp))

result = T*fi*(result1 + result2)

print(f'integral1 = {result1/10**9:.2f} *10^9 and integral2 = {result2/10**9:.2f} *10^9')
print(f"The result of the integral is: {result/10**9:.4f} *10^9")


# integral1 = 37.19 *10^9 and integral2 = 19.00 *10^9
# The result of the integral is: 27171.8612 *10^9
