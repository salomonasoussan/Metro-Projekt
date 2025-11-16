import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import random as rnd

all_x = np.array([3, 6, 9, 12])
all_y = np.array([6, 12, 15, 18])
all_u_y = np.array([0.5, 0.8, 2.3, 4.4])

def function_f(y_i, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma)*np.exp(-((y_i-mu)**2)/(2*sigma**2))

def generate_random_y(mu, sigma):
    a = mu - (3 * sigma)
    b = mu + (3 * sigma)

    m = 1/(np.sqrt(2*np.pi)*sigma)

    while True:
        r = np.random.uniform(0, m)
        x_i = np.random.uniform(a, b)
        f_x_i = function_f(x_i, mu, sigma)
        if r <= f_x_i:
            return x_i


def koeffizienten_berechnung(all_x, all_y):
    a_0 = ((np.dot(all_x, all_y)*np.sum(all_x)) - (np.sum(all_y) * np.sum(all_x**2)))/((np.sum(all_x) * np.sum(all_x)) - len(all_y)*np.sum(all_x**2))
    a_1 = (len(all_y)*np.dot(all_x, all_y) - (np.sum(all_y) * np.sum(all_x)))/(len(all_y)*np.sum(all_x**2) - (np.sum(all_x))**2)
    coefficients = [a_0, a_1]

    return coefficients

def alle_koeffizienten(all_x, all_y, all_u_y):
    coefficients = []
    coefficients.append(koeffizienten_berechnung(all_x, all_y))
    for i in range(1000000):
        new_random_y = []
        for j in range(len(all_y)):
            new_random_y.append(generate_random_y(all_y[j], all_u_y[j]))

        new_random_y = np.array(new_random_y)
        coefficients.append(koeffizienten_berechnung(all_x, new_random_y))

    return np.array(coefficients)

def plot_all(coefficients):
    a_0_values = coefficients[:,0]
    a_1_values = coefficients[:, 1]

    a0_mean, a0_std, a1_mean, a1_std = koeffizienten_analyse(coefficients)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.hist(a_0_values, bins="auto", density=True, alpha=0.5, label="a_0_Koeffizienten")
    plt.xlabel('Wert')
    plt.ylabel('Häufigkeit')
    plt.title('Histogramm a_0')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist(a_1_values, bins="auto", density=True, alpha=0.5, label="a_1_Koeffizienten", color='orange')
    plt.xlabel('Wert')
    plt.ylabel('Häufigkeit')
    plt.title('Histogramm a_1')
    plt.legend()

    plt.tight_layout()

    plt.figure(2)
    x_plot = np.linspace(min(all_x)-1, max(all_x)+1, 100)
    a0_upper = a0_mean + a0_std
    a0_lower = a0_mean - a0_std
    a1_upper = a1_mean + a1_std
    a1_lower = a1_mean - a1_std

    plt.plot(x_plot, a0_mean + a1_mean * x_plot, color='red', label='Mittelwertgerade', linewidth=2)
    plt.plot(x_plot, a0_upper + a1_upper * x_plot, color='green', linestyle='--', label='Oberes Unsicherheitsband')
    plt.plot(x_plot, a0_lower + a1_lower * x_plot, color='blue', linestyle='--', label='Unteres Unsicherheitsband')
    plt.errorbar(all_x, all_y, yerr=all_u_y, fmt='o', color='blue', capsize=5, label='Messwerte ± Unsicherheit')

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Monte-Carlo Simulation lineare Regression")
    plt.legend()
    plt.show()

def koeffizienten_analyse(coefficients):
    a_0_values = coefficients[:,0]
    a_1_values = coefficients[:, 1]

    a0_mean = np.mean(a_0_values)
    a0_std = np.std(a_0_values, ddof=1)

    a1_mean = np.mean(a_1_values)
    a1_std = np.std(a_1_values, ddof=1)

    return a0_mean, a0_std, a1_mean, a1_std

coefficients = alle_koeffizienten(all_x, all_y, all_u_y)
plot_all(coefficients)
a0_mean, a0_std, a1_mean, a1_std = koeffizienten_analyse(coefficients)

print("Mittelwert a0 = ", a0_mean, " +- ", a0_std, "Mittelwert a1 = ", a1_mean, " +- ", a1_std)

