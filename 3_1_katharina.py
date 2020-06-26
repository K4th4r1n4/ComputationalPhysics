"""Elementare numerische Methoden 2: Integration

Dieses Programm enthaelt drei numerische Methoden zur Integration einer Funk-
tion im Intervall [a,b] mit Hilfe von N Teilintervallen: die Mittelpunkt-, die
Trapez-, sowie die Simpson-Methode.
Fuer die Funktion cosh(2*x) wird der Betrag des relativen Fehlers der drei Int-
egrationsmethoden zum analytischen Wert in Abhaengigkeit des Diskretisierungs-
parameters h fuer die Integrationsgrenzen a = -pi/2 bis b = pi/3 doppelt loga-
rithmisch dargestellt. Zusaetzlich wird das erwartete Skalierungsverhalten der
drei Verfahren mitgeplottet.
"""

import numpy as np
import matplotlib.pyplot as plt

def mittelpunkt_int(funktion, a, b, N):
    """mittelpunkt_int berechnet das Integral einer Funktion mittels Mittel-
       punktsmethode.

       Parameter: funktion: zu integrierende Funktion
                  a: untere Integrationsgrenze
                  b: obere Integrationsgrenze
                  N: Anzahl der Teilintervalle
    """
    if N==1:                 # im Fall N=1: h=nan -> setze Streifenbreite h=b-a
        x = np.linspace(a, b, N, endpoint=False)
        h = b-a
    else:
        x, h = np.linspace(a, b, N, endpoint=False, retstep=True)
    return h * np.sum(funktion(x+h/2))

def trapez_int(funktion, a, b, N):
    """trapez_int berechnet das Integral einer Funktion mittels Trapezmethode.

       Parameter: funktion: zu integrierende Funktion
                  a: untere Integrationsgrenze
                  b: obere Integrationsgrenze
                  N: Anzahl der Teilintervalle
    """
    if N==1:                 # im Fall N=1: h=nan -> setze Streifenbreite h=b-a
        x = np.linspace(a, b, N, endpoint=False)
        h = b-a
    else:
        x, h = np.linspace(a, b, N, endpoint=False, retstep=True)
    return (h/2) * (np.sum(funktion(x)) + np.sum(funktion(x+h)))

def simpson_int(funktion, a, b, N):
    """simpson_int berechnet das Integral einer Funktion mittels der Simpson-
       Methode.

       Parameter: funktion: zu integrierende Funktion
                  a: untere Integrationsgrenze
                  b: obere Integrationsgrenze
                  N: Anzahl der Teilintervalle
    """
    if N==1:                 # im Fall N=1: h=nan -> setze Streifenbreite h=b-a
        x = np.linspace(a, b, N, endpoint=False)
        h = b-a
    else:
        x, h = np.linspace(a, b, N, endpoint=False, retstep=True)
    return (h/6) * (np.sum(funktion(x)) + 4*np.sum(funktion(x+h/2)) +
                    np.sum(funktion(x+h)))

def cosh2x(x):
    """cosh2x gibt die Funktion cosh(2*x) zurueck.
    """
    return np.cosh(2*x)

def analytisch_cosh2x(x):
    """analytisch_cosh2x berechnet die Stammfunktion der Funktion cosh(2*x)
       analytisch.

       Parameter: x: Stelle, an der die Stammfunktion ausgewertet wird
    """
    return np.sinh(2*x)/2

def main():
    """Hauptprogramm:"""
    print(__doc__)           # Programmbeschreibung ausgeben
    a = -np.pi/2             # untere Integrationsgrenze
    b = np.pi/3              # obere Integrationsgrenze
    Anzahl = 1000            # Anzahl der Plotpunkte

    # analytischer Wert des Integrals für cosh(2*x) mit Grenzen a und b:
    analytisch = analytisch_cosh2x(b) - analytisch_cosh2x(a)

    # Arrays der Groesse Anzahl fuer N, h und die 3 Integrationsmetoden anlegen
    N_plot = np.int32(10**np.linspace(0, 5, Anzahl))
    h_plot = np.zeros(Anzahl)
    mittelpunkt_plot = np.zeros(Anzahl)
    trapez_plot = np.zeros(Anzahl)
    simpson_plot = np.zeros(Anzahl)

    # fuer alle Werte in N_plot dazugehoerige Werte fuer h und des Integrals
    # mit Hilfe der 3 Methoden berechnen und Ergebnisse fuer Plot speichern

    for counter, i in enumerate(N_plot):
        h_plot[counter] = (b-a)/i
        mittelpunkt_plot[counter] = mittelpunkt_int(cosh2x, a, b, i)
        trapez_plot[counter] = trapez_int(cosh2x, a, b, i)
        simpson_plot[counter] = simpson_int(cosh2x, a, b, i)

    plt.figure(1, figsize=(12,10))
    # Plot mit doppelt logarithmischer Achseneinteilung definieren
    plt.subplot(111, xscale="log", yscale="log", autoscale_on=False)
    plt.axis([10e-5, 1, 10e-18, 10])                          # Plotbereich,
    plt.title("Numerische Integrationsmethoden")              # Titel,
    plt.xlabel("h")                                           # Labels
    plt.ylabel("Betrag relativer Fehler")                     # definieren

    # Plotten des relativen Fehlers zum analytischen Wert und des erwarteten
    # Skalierungsverhalten der drei Integrationsmethoden fuer die Funktion
    # cosh(2*x) mit Integrationsgrenzen a bis b in Abhaengigkeit von h:

    # Mittelpunkt-Methode (rot)
    plt.plot(h_plot, abs((mittelpunkt_plot - analytisch)/(analytisch)), c="r",
             linestyle='', marker='.', markersize=0.8,
             label="Mittelpunkt-Methode")
    # Trapez-Methdode (gruen)
    plt.plot(h_plot, abs((trapez_plot - analytisch)/(analytisch)), c="g",
             linestyle='', marker='.', markersize=0.8, label="Trapez-Methode")
    # Simpson-Methode (blau)
    plt.plot(h_plot, abs((simpson_plot - analytisch)/(analytisch)), c="b",
             linestyle='', marker='.', markersize=0.8, label="Simpson-Methode")
    # erwartetes Skalierungsverhalten
    plt.plot(h_plot, h_plot**4, "--", c="b", linewidth=0.8,   # O(h^4) (blau)
             label="h$^{4}$-Verhalten")
    plt.plot(h_plot, h_plot**2, "--", c="y", linewidth=0.8,   # O(h^2) (gelb)
             label="h$^{2}$-Verhalten")
    plt.legend(loc="best")                                    # Legende anlegen
    plt.show()


if __name__ == "__main__":
    main()

"""Diskussion:
a) cosh(2*x):
Die Mittelpunkt- und Trapez-Methoden zeigen das analytisch erwartete Skalier-
ungsverhalten von h^2.
Die Simpson-Methode zeigt fuer kleine h Werte ein h^0 Verhalten, dann (ab unge-
faehr h > 10e-3) ebenfallls das erwartete Skalierungsverhalten von h^4. Das h^0
Verhalten kommt daher, dass fuer kleine h Werte der Rundungsfehler dominiert,
waehrend der Beitrag des Diskretisierungsfehler vergleichsweise winzig wird.
Die Funktion cosh(2*x) besitzt ein konstantes Kruemmungsverhalten (zweite Ab-
leitung), daher ist der Verlauf des relativen Fehlers fuer die Integrations-
methoden wie erwartet.

b) exp(-100*x^2):
Zunaechst zeigen alle 3 Methoden ein h^0 Verhalten. Dies kommt durch das Kruem-
mungsverhalten (2.Ableitung) der Funktion, das sich für positive zu negativen
Werten von x aendert. Dadurch heben sich die Fehler des Wertes des Integrals,
die man nach oben macht, mit denen, die man nach unten macht weg. Da die
Funktion extrem schnell und auch innerhalb der Integrationsgrenzen auf an-
naehernd 0 abfaellt, spielt es dabei keine Rolle, dass das die Funktion nicht
bezueglich des Intergrationsintervalls symmetrisch ist (sondern zu x=0 sym.).
Sobald die h-Werte gross genug sind (rund h>10e-1), kommt es zu einem rasanten
Anstieg des Fehlers, da oben betrachtete Ueberlegung aufgrund der zu geringen
Teilintervallanzahl nicht mehr zutreffen.

c) Heaviside:
Alle drei Methoden zeigen Fehler ein h^1 Verhalten. Dabei ergeben sich fuer
jede Methode mehrere parallele Geraden. Dieses Verhalten kommt durch die Unste-
tigkeit der Heaviside-Funktion an der Stelle 0 zustande. Dabei ist fuer den
Fehler das Teilintervall um 0 relevant, da ansonsten die Heaviside-Funktion
konstant ist und sich damit nur Rundungsfehler ergeben duerften. Je nachdem,
wie gross bei diesem Teilintervall der Anteil von positiven zu negativen x ist,
variiert der Fehler, aber auf jedenfall ist er proportional zu h (siehe Defini-
tion der Integralberechnungen).
"""
