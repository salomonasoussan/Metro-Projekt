import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random as rnd

all_x = np.array([3, 6, 9, 12])
all_y = np.array([6, 12, 15, 18])
all_u_y = np.array([0.50, 0.8, 2.3, 4.4])

def function_f(y_i, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(y_i-mu)**2/(2*sigma**2))

def generate_random_y(mu, sigma):
    a = mu - 3 * sigma
    b = mu + 3 * sigma

    m = 1.0


    while(True):
        r = np.random.uniform(0, m)
        x_i = np.random.uniform(a, b)
        f_x_i = function_f(x_i, 6, 0.5)
        if r < f_x_i:
            return x_i

        print(x_i, r, f_x_i)

def koeffizienten_berechnung(all_x, all_y):
    a_0 = ((np.dot(all_x, all_y)*np.sum(all_x)) - (np.sum(all_y) * np.sum(all_x**2)))/((np.sum(all_x) * np.sum(all_x)) - len(all_y)*np.sum(all_x**2))
    a_1 = (len(all_y)*np.dot(all_x, all_y) - (np.sum(all_y) * np.sum(all_x)))/(len(all_y)*np.sum(all_x**2) - (np.sum(all_x))**2)
    coefficients = [a_0, a_1]

    return coefficients

def alle_koeffizienten(all_x, all_y, all_u_y):
    new_random_y = []
    coefficients = []
    coefficients.append(koeffizienten_berechnung(all_x, all_y))
    for i in range(1, 10):
        for j in range(len(all_y)):
            new_random_y.append(generate_random_y(all_y[j], all_u_y[j]))

        coefficients.append(koeffizienten_berechnung(all_x, new_random_y))

    return coefficients

print(alle_koeffizienten(all_x, all_y, all_u_y))
