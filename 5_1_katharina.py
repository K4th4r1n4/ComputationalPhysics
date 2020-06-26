"""Quantenmechanik von 1D-Potentialen 1:
Eigenwerte, Eigenfunktionen

In diesem Programm werden die Eigenfunktionen und Eigenenergien eines Teilchens
im asymmetrischen Doppelmuldenpotential V(x) = x^4 - x^2 -A*x betrachtet. Dabei
werden die Eigenenergien und zugehoerigen Eigenfunktionen mittels Ortsraumdis-
kretisierung bestimmt. Alle Eigenenergien E < 0,25 und die dazugehoerigen Eig-
enfunktionen Psi werden zusammen mit dem Potential fuer den Wert A = 0,15 graf-
isch dargestellt. Der effektive Wert fuer das dimensionslose h_quer ist dabei
auf 0,07 festgelegt.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def potential(x, A):
    """potential gibt das asymmetrische Doppelmuldenpotential
       V(x)= x**4 - x**2 - A*x zurueck.

       Parameter: x: x-Werte, an denen das Potential betrachtet wird
                  A: Parameter fuer Asymmetrie des Potentials
    """
    return x**4 - x**2 - A*x

def diskretisierung(a, b, N):
    """diskretisierung berechnet die Diskretisierung und gibt die x-Werte als
       Array zurueck.

       Parameter: a: untere Intervallgrenze
                  b: obere Intervallgrenze
                  N: Anzahl der Diskretisierungsschritte
    """
    delta_x = (b-a)/(N+1)                 # Diskretisierungsschrittweite
    return np.linspace(a + delta_x, b - delta_x, N)

def diagonalisieren(N, h_eff, x, V):
    """diagonalisieren gibt die numerisch berechneten Eigenfunktionen und
       -energien der Schroedingergleichung eines Teilchens im Potential V im
       Intervall x mit dimensionslosen Wert fuer h_quer zurueck.

       Parameter: N:     Dimension der Matrix
                  h_eff: einheitenloser Wert für h_quer
                  x:     Array der x-Werte
                  V:     betrachtetes Potential
    """
    # Diskretisierungsschrittweite:
    delta_x = x[1] - x[0]
    z = h_eff**2 / (2*delta_x**2)
    # Matrixform:
    matrix = (np.diag(V, k=0) + np.diag(np.ones(N)*2*z, k=0) + # Hauptdiagonale
              np.diag(np.ones(N-1)*-z, k=-1) +                 # Nebendiagonale
              np.diag(np.ones(N-1)*-z, k=+1))                  # Nebendiagonale
    eigenwert, eigenfunktion = eigh(matrix)
    return eigenwert, eigenfunktion

def ef_plotten(energien, eigenfunktion, x, farben, faktor, V):
    """ef_plotten plottet die normierten Eigenfunktion und zugehoerigen Ener-
       gieeigenwerte im Intervall x mit wechselnder Farbe fuer unterschiedliche
       Energien. Zur besseren Visualisierung werden die Eigenfunktionen mit
       einem Faktor skaliert. Zusaetzlich wird das betrachte Potential geplot-
       tet.

       Parameter: energien:      Array mit Energieigenwerten
                  eigenfunktion: Matrix mit Eigenfunktionen als Spalten
                  x:             betrachtetes x Intervall
                  farben:        Liste der zu plottenden Farben
                  faktor:        Faktor zur Skalierung der Eigenfunktionen
                  V:             Potential
    """
    delta_x = x[1] - x[0]                    # Diskretisierungsschrittweite
    plt.plot(x, V, c="k", linewidth=2)       # Potential plotten (schwarz)
    # Schleife zum Plotten der Eigenfunktionen und zugehoerigen Eigenenergien,
    # umgekehrte Reihenfolge damit Legendeneintraege passend zu den Eigenfunk-
    # tionen dargestellt werden (von unten nach oben):
    for i, energie in enumerate(reversed(energien)):
        # len(farben) eventuell kleiner als Anzahl der zu plotteten
        # Eigenfunktion -> Wiederholung der Farben durch modulo Operation
        plt.plot(x, energie + np.zeros(len(x)), c=farben[i % len(farben)],
                label="$\Psi_{}$".format(len(energien)-i-1))
        # Faktor 1/sqrt(delta_x) wegen Normierung,
        # Faktor faktor zur besseren Visualisierung:
        plt.plot(x, eigenfunktion[:,len(energien)-i-1]/np.sqrt(delta_x)*faktor
                 + energie, c=farben[i % len(farben)])
        plt.draw()

def main():
    """Hauptprogramm:"""
    print(__doc__)                        # Programmbeschreibung ausgeben
    A = 0.15                              # Parameter Doppelmuldenpotential
    N = 500                               # Dimension der Matrix
    h_eff = 0.07                          # dimensionsloses h_quer
    l = 1.5                               # Intervallgrenze
    x = diskretisierung(-l, l, N)         # x-Werte fuer betrachtetes Intervall
    V = potential(x, A)                   # Potential in Intervallgrenzen
    E_max = 0.25                          # maximal betrachtete Energie
    delta_x = x[1] - x[0]                 # Diskretisierungsschrittweite
    # Eigenwerte und Eigenfunktionen:
    eigenwert, eigenfunktion = diagonalisieren(N, h_eff, x, V)

    energien = eigenwert[eigenwert < E_max]  # Energiewerte kleiner E_max
    farben = ["r", "g", "b", "y"]            # Liste mit Farben fuer Plot
    faktor = 1/50                            # Faktor (bessere Visualisierung)
    plt.figure(0, figsize=(10,10))
    # Eigenfunktionen und dazugehoerige Eigeneinergien plotten:
    ef_plotten(energien, eigenfunktion, x, farben, faktor, V)
    plt.legend(loc="center left")
    plt.axis([-l, l, min(V), E_max])                     # Plotbereich,
    plt.title("Eigenfunktionen eines Teilchens im "
              "asymmetrischen Doppelmuldenpotential",    # Titel,
                fontsize=14)
    plt.xlabel("x")                                      # Labels
    plt.ylabel("V(x)")                                   # definieren
    plt.show()

if __name__ == "__main__":
    main()

"""Diskussion:
a) N = 500 gewaehlt, da fuer kleinere N die Funktionen nicht genau genug darge-
stellt werden (sind eher eckig), da zu wenig Punkte vorhanden sind. Fuer groes-
sere N nimmt die Berechnungszeit zu und fuer das gewaehlte N sehen die Eigen-
funktionen ausreichend gut/genau aus.
Das betrachtete Intervall wurde zu x =[-1.5, 1.5] gewaehlt, da hier alle Eigen-
funktionen mit E < 0.25 gut sichtbar dargestellt werden und der relevante An-
teil des Potentials enthalten ist.

b) Mit zunehmender Energie nimmt die Anzahl der Knoten nach dem Knotensatz zu.
So hat die erste Eigenfunktion (Psi_{0}) noch keinen Knoten, die zweite einen
usw. Ausserdem ist kein Energiewert entartet, wie man es auch fuer 1D Poten-
tiale erwartet. Die ersten beiden Eigenfunktionen (im Plot in der tieferen
Mulde) aehneln stark denen des harmonischen Oszillators, nur in Richtung der
Mulde verschoben. Fuer die dritte Eigenfunktion (Psi_{2}) lassen sich die bei-
den Knoten ohne Zoom schwerlich erkennen, sie sind aber vorhanden. Die Wellen-
berge der Eigenfunktionen sind aufgrund des lokalen Maximums des Potentials bei
x = 0 etwas "verzogen". Fuer groessere Eigenenergien, bei denen das asymmetri-
sche Doppelmuldenpotential dem des harmonischen Oszillators wieder stark
aehnelt und das lokale Mamximum ueberwunden ist, sehen auch die Eigenfunktionen
denen des harmonischen Oszillators wieder sehr aehnlich.
Fuer groessere h_eff verschieben sich die Eigenenergien- und funktionen nach
oben, der Abstand zwischen ihnen wird groesser, fuer kleinere h_eff genau umge-
dreht: der Abstand wird kleiner und sie verschieben sich nach unten. Der Grund
dafuer steckt in der Matrixdarstellung, mit der diese bestimmt werden. Dort
taucht im z h_eff**2 auf.

c) Im Falle A = 0 ist der Knotensatz wieder erfuellt und kein Energieeigenwert
ist entartet. Allerding liegen Psi_{0}, Psi_{1} und Psi_{2}, Psi_{3} und die
dazugehoerigen Eigenwerte jeweils paarweise sehr nah beieinander. Im Falle der
ersten beiden kommt es stellenweise fast zur kompletten Ueberschneidung der
Funktionen. Wie in b) erwaehnt kommt es aufgrund des lokalen Maximums bei x = 0
zu "verzogenen" Wellenfunktionen im Vergleich zum harmonischen Oszillator.
Fuer groessere Energieeigenwerte nimmt dieser Effekt ab und die Eigenfunktionen
sehen wieder denen des harmonischen Oszillators sehr aehnlich, da sich hier
auch der Verlauf der Potentiale staerker aehnelt.
"""
