"""Elementare numerische Methoden 1: Ableitung

Dieses Programm enthaelt drei numerische Methoden zur Berechnung der ersten
Ableitung einer Funktion im Punkt x_0 mit Diskretisierungsparamter h: 
die Methode der Vorwaerts-, der Zentral-, sowie der extrapolierten Differenz.
Fuer die Funktion arctan(x^3) wird im Punkt x_0 = 1/3 der Betrag des relativen 
Fehlers der drei Differentiationsmethoden zum analytischen Wert in Abhaengig-
keit von h doppelt logarithmisch dargestellt. Zusaetzlich wird das erwartete 
Skalierungsverhalten fuer alle drei Verfahren mitgeplottet.
"""

import numpy as np
import matplotlib.pyplot as plt

def vor_diff(funktion, x_0, h):
    """vor_diff berechnet die erste Ableitung einer Funktion mittels Vorwaerts-
       differenz.
       
       Parameter: Funktion: funktion
                  Stelle, an der die Ableitung ausgewertet wird: x_0
                  Diskretisierungsparamter: h
    """
    return (funktion(x_0+h) - funktion(x_0)) / h

def zentral_diff(funktion, x_0, h):
    """zentral_diff berechnet die erste Ableitung einer Funktion mittels
       Zentraldifferenz.
       
       Parameter: Funktion: funktion
                  Stelle, an der die Ableitung ausgewertet wird: x_0
                  Diskretisierungsparamter: h
    """
    return (funktion(x_0+h/2) - funktion(x_0-h/2)) / h

def extrapol_diff(funktion, x_0, h):
    """extrapol_diff berechnet die erste Ableitung einer Funktion mittels
       extrapolierter Differenz.
       
       Parameter: Funktion: funktion
                  Stelle, an der die Ableitung ausgewertet wird: x_0
                  Diskretisierungsparameter: h
    """
    return (8*(funktion(x_0+h/4) - funktion(x_0-h/4)) - \
           (funktion(x_0+h/2) - funktion(x_0-h/2))) / (3*h)

def arctanx3(x):
    """arctanx3 gibt die Funktion arctan(x^3) zurueck."""
    return np.arctan(x**3)

def analytisch(x_0):
    """analytisch berechnet die erste Ableitung der Funktion arctan(x^3)
       analytisch.
       
       Parameter: Stelle, an der die Ableitung ausgewertet wird: x_0
    """
    return 3*(x_0**2) / (x_0**6+1)

def fehler(numerisch, analytisch):
    """fehler berechnet den Betrag des relativen Fehlers zwischen den zwei 
       Parametern numerisch und analytisch zum Wert analytsich.
    """
    return abs((numerisch - analytisch) / analytisch)

def main():
    """Hauptprogramm:"""
    print(__doc__)                                    # Programmbeschreibung
    
    x_0 = 1/3                                         # Festlegung von x_0
    h = 10.0**np.linspace(-10, 0.0, 5000)             # Wertebereich von h
    
    plt.figure(1, figsize=(12,10))                     
    # Plot mit doppelt logarithmischer Achseneinteilung definieren
    plt.subplot(111, xscale="log", yscale="log", autoscale_on=False)
    plt.axis([10e-11, 1, 10e-16, 10])                 # Plotbereich,
    plt.title("Numerische Differentiationsmethoden")  # Titel,
    plt.xlabel("h")                                   # Labels
    plt.ylabel("Betrag relativer Fehler")             # definieren
    
    # Plotten des relativen Fehlers zum analytischen Wert und des erwarteten
    # Skalierungsverhalten der drei Ableitungsmethoden fuer die Funktion 
    # arctanx3 im Punkt x_0 in Abhaengigkeit von h:
    
    # Vorwaertsdifferenz (rot)
    plt.plot(h, fehler(vor_diff(arctanx3, x_0, h), analytisch(x_0)), c="r",
             linestyle='',marker='.',markersize=0.8,label="Vorwaertsdifferenz")
    plt.plot(h, h, "--", c="r", linewidth=0.8, 
             label="erwartetes Skalierungsverhalten Vorwaertsdifferenz")
    
    # Zentraldifferenz (gruen)
    plt.plot(h, fehler(zentral_diff(arctanx3, x_0, h), analytisch(x_0)), c="g",
             linestyle='', marker='.',markersize=0.8, label="Zentraldifferenz")
    plt.plot(h, h**2, "--", c="g", linewidth=0.8,
             label="erwartetes Skalierungsverhalten Zentraldifferenz")
    
    # extrapolierte Differenz (blau)
    plt.plot(h, fehler(extrapol_diff(arctanx3, x_0, h),analytisch(x_0)),c="b",
             linestyle='', marker='.', markersize=0.8,
             label="extrapolierte Differenz")
    plt.plot(h, h**4, "--", c="b", linewidth=0.8,
             label="erwartetes Skalierungsverhalten extrapolierte Differenz")
    
    plt.legend(loc="upper left",prop={"size":12})     # Legende anlegen
    # Endlos-Schleife, die auf Ereignisse wartet
    plt.show()
    
if __name__ == "__main__":
    main()

"""Fragen:
a) Die Ursache für das 1/h Verhalten der Fehler bei kleinen h Werten sind 
   Rundungsfehler: Zahlen werden bei numerischen Rechnungen nur mit einer 
   gewissen Genauigkeit dargestellt. Man kann sich dies am Beispiel der Vor-
   waertsdifferenz verdeutlichen, indem man die Werte f(x+h) und f(x) mit einer
   kleinen Ungenauigkeit von (1+e) multipliziert, wobei e eine kleine Zahl dar-
   stellt. Bei der Berechnung des Verhalten des Fehlers ergibt sich (wie in der
   Vorlesung gezeigt) ein Term, der mit h^1 skaliert und durch die Addition der
   kleinen Ungenauigkeit e ein weiterer Term, der proportional zu e/h ist. Fuer
   kleine Werte von h ist der 1/h Term der Relevante, wodurch es zum beobachte-
   ten Skalierungsverhalten kommt.

b) Die optimale Groessenordnung von h wurde mit Hilfe des Plots bestimmt, indem
   ungefaehr die Koordinaten der Minima der jeweiligen Kurve mit Hilfe des
   Mauszeigers abgelesen wurden:

   Methode                      h          |relstiver Fehler(h)|
   Vorwaertsdifferenz:        1e-09                1e-10
   Zentraldifferenz:          1e-06                1e-12
   extrapolierte Differenz:   1e-04                1e-14
"""
