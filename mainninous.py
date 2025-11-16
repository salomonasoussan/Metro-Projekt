import numpy as np
import matplotlib.pyplot as plt
import random as rnd

# Scipy wird fuer diese Implementierung nicht benoetigt.

# --- Original-Messdaten ---
# (Sie muessen hier Ihre echten Daten aus dem Projektordner eintragen)
all_x = np.array([3, 6, 9, 12])
all_y = np.array([6, 12, 15, 18])
all_u_y = np.array([0.5, 0.8, 2.3, 4.4])


# --- SCHRITT 0: Hit-and-Miss-Zufallsgenerator ---

def gauss_density_shape(y_val, mu, sigma):
    """
    Definiert die FORM der Gauß-Dichte (nicht normiert).
    Das Maximum (bei y_val = mu) ist exp(0) = 1.0.
    """
    return np.exp(-(y_val - mu) ** 2 / (2 * sigma ** 2))


def generate_random_y(mu, sigma):
    """
    Implementiert den "Hit-and-Miss" Algorithmus (PDF, Seite 5) [cite: 116-121].
    Zieht eine Zufallszahl fuer einen Messpunkt (mu, sigma).
    """

    # 1. Definiere Grenzen [a, b] (3 Standardabweichungen)
    a = mu - 3 * sigma
    b = mu + 3 * sigma

    # 'm' ist der Maximalwert der Dichte-FORM.
    # Da gauss_density_shape(mu, mu, sigma) = 1 ist:
    m = 1.0

    # Start des Algorithmus [cite: 117-121]
    while True:
        # Schritt 1: Würfele x aus [a, b[
        x_candidate = rnd.uniform(a, b)

        # Schritt 2: Berechne f(x)
        f_x = gauss_density_shape(x_candidate, mu, sigma)

        # Schritt 3: Würfele r aus [0, m[
        r_candidate = rnd.uniform(0, m)

        # Schritt 4 & 5: Prüfe "Hit" (r < f(x)) [cite: 120-121]
        if r_candidate < f_x:
            return x_candidate  # Treffer!


# --- SCHRITT 1: Geradenanpassung & Monte-Carlo-Schleife ---

def koeffizienten_berechnung(x_data, y_data):
    """
    Berechnet die Parameter a0 und a1 einer Ausgleichsgeraden
    f(x) = a0 + a1*x nach der Methode der kleinsten Quadrate
    (Formeln gemaess PDF Seite 3 [cite: 65-68]).
    """
    n = len(y_data)
    sum_x = np.sum(x_data)
    sum_y = np.sum(y_data)
    sum_xx = np.sum(x_data ** 2)
    sum_xy = np.dot(x_data, y_data)

    # Nenner (Determinante)
    D = n * sum_xx - sum_x ** 2

    # Parameter berechnen
    a_0 = (sum_y * sum_xx - sum_x * sum_xy) / D
    a_1 = (n * sum_xy - sum_x * sum_y) / D

    return [a_0, a_1]


def alle_koeffizienten(all_x, all_y, all_u_y, n_sims=1000000):
    """
    Fuehrt die vollstaendige Monte-Carlo-Simulation (Schritt 1) durch .
    """

    print(f"Starte Monte-Carlo-Simulation mit {n_sims} Durchlaeufen...")

    coefficients_list = []

    # Berechne einmal fuer die Originaldaten
    original_coeffs = koeffizienten_berechnung(all_x, all_y)
    coefficients_list.append(original_coeffs)

    # Hauptschleife der Simulation
    for i in range(n_sims):

        # Erstelle einen neuen, gewuerfelten Datensatz
        new_random_y = []

        for j in range(len(all_y)):
            # Rufe den "Schritt 0"-Generator fuer jeden Datenpunkt auf
            new_y = generate_random_y(all_y[j], all_u_y[j])
            new_random_y.append(new_y)

        # Fuehre Regression auf dem neuen Datensatz aus
        new_coeffs = koeffizienten_berechnung(all_x, np.array(new_random_y))
        coefficients_list.append(new_coeffs)

    print("Simulation abgeschlossen.")
    return np.array(coefficients_list)


# --- PROGRAMMAUSFUEHRUNG UND AUSWERTUNG ---

# SCHRITT 1 ausfuehren
# (Dies kann einige Minuten dauern!)
results_array = alle_koeffizienten(all_x, all_y, all_u_y)

# --- Datenaufbereitung ---
# Trenne die Ergebnisse in zwei separate Arrays
a0_results = results_array[:, 0]
a1_results = results_array[:, 1]

# --- SCHRITT 2: Empirische Wahrscheinlichkeitsdichten (Histogramme) ---
print("Erstelle Histogramme (Schritt 2)...")

plt.figure(figsize=(12, 6))

# Histogramm fuer a0
plt.subplot(1, 2, 1)
plt.hist(a0_results, bins=100, density=True, label='Simulation $a_0$')
plt.title('Empirische Dichte fuer Parameter $a_0$ (Achsenabschnitt)')
plt.xlabel('Wert von $a_0$')
plt.ylabel('Dichte')
plt.legend()

# Histogramm fuer a1
plt.subplot(1, 2, 2)
plt.hist(a1_results, bins=100, density=True, label='Simulation $a_1$', color='orange')
plt.title('Empirische Dichte fuer Parameter $a_1$ (Steigung)')
plt.xlabel('Wert von $a_1$')
plt.ylabel('Dichte')
plt.legend()

plt.tight_layout()
plt.show()

# --- SCHRITT 3: Unsicherheiten (Standardabweichung) ermitteln ---
print("\n--- AUSWERTUNG (Schritt 3) ---")

# Fuer Parameter a0 (Achsenabschnitt)
best_a0 = np.mean(a0_results)
u_a0 = np.std(a0_results)  # Unsicherheit ist die Standardabweichung [cite: 104, 158]

# Fuer Parameter a1 (Steigung)
best_a1 = np.mean(a1_results)
u_a1 = np.std(a1_results)  # Unsicherheit ist die Standardabweichung [cite: 104, 158]

print(f"Bester Schaetzer a0 (Achsenabschnitt): {best_a0:.4f} +/- {u_a0:.4f}")
print(f"Bester Schaetzer a1 (Steigung):         {best_a1:.4f} +/- {u_a1:.4f}")

# --- SCHRITT 4: Diskussion ---
print("\n--- AUFGABE (Schritt 4) ---")
print("Diskutieren Sie nun die Vor- und Nachteile dieses Verfahrens")
print("im Vergleich zur klassischen Methode (siehe PDF).")