import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random as rnd

a = 0
b = 1
m = 1

x_i = 1
r = 1

def function_f(y_i, my, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(y_i-my)**2/(2*sigma**2))



def norm_function(function_f, y_i, y, u_y):
    global sigma
    global my
    sigma = u_y
    my = y

    integral, err = sp.integrate.quad(function_f, my - 3 * sigma, my + 3 * sigma)


    norm = 1 / integral
    f_norm = norm * function_f

    return f_norm


while(r >= norm_function(function_f, x_i, 6, 0.5)):
    r = np.random.uniform(0, m)
    x_i = np.random.uniform(a, b)
    print("x_i = ", x_i, "r= ", r)

