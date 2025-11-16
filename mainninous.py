import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import random as rnd

a = 0
b = 1

x_i = np.random.uniform(a,b)
print(x_i)

def function_f(y_i, my_i, sigma_i):
    global sigma
    sigma = sigma_i
    my = my_i

def function_f_norm(y_i, y, u_y):

    f = 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(y_i-my_i)**2/(2*sigma**2))
    return f

print(function_f(3, 4, 5))