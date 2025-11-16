import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random as rnd

a = 0
b = 1

x_i = np.random.uniform(a, b)

def function_f(y_i, y, u_y):
    global sigma
    global my
    sigma = u_y
    my = y
    f = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(y_i-my)**2/(2*sigma**2))

    #Normierungskonstante berechnen

    integral = sp.quad(f, my - 3*sigma, my + 3*sigma)

    norm = 1/integral
    f_norm = norm * f

    return f_norm

    return f


print(function_f(x_i, 1, 2))