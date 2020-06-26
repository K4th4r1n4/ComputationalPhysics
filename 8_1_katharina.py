"""Ideales Gas:
Druckmessung

In diesem Programm wird ein ideales Gas bestehend aus N=8 Teilchen betrachtet.
Dabei wird die Wahrscheinlichkeitsverteilung des mittleren Drucks p, der auf
die rechte Seitenflaeche eines Quaders im Zeitintervall delta_t=4 ausgeuebt
wird, fuer R=10000 Realisierungen betrachtet und in einem Histogramm darge-
stellt. Zusaetzlich wird der Erwartungswert und die Standardabweichung von p
fuer die R Realisierungen ausgegeben und mit Hilfe dieser Werte eine normierte
Gauss-Verteilung zum Vergleich im Histogramm mit dargestellt.
"""

import numpy as np
import matplotlib.pyplot as plt

def reflexionen(x_0, v, delta_t):
    """reflexionen berechnet die Anzahl der Reflexionen an der rechten Wand.
       Herleitung der Formel siehe *) in Diskussion.

       Parameter: x_0:     Anfangsort des Teilchens
                  v:       Geschwindigkeit des Teilchens
                  delta_t: betrachtetes Zeitintervall
    """
    x_end = x_0 + v*delta_t
    n = np.int32((abs(x_end) + 1)/2)    # int32 rundet auf int ab
    return n

def druck(N, delta_t, v, n):
    """druck berechnet den Druck den ein Ensemble aus N Teilchen auf die rechte
       Wand eines Quaders ausuebt. Dabei muss das Array der Geschwindigkeiten
       passend zum  Array der Stoesse mit der rechten Wand sortiert sein.

       Parameter: N:       Anzahl der Teilchen
                  delta_t: betrachtetes Zeitintervall
                  v:       Geschwindigkeiten der Teilchen
                  n:       Stoesse der Teilchen mit der rechten Wand
    """
    return 2/(N*delta_t) * np.sum(abs(v)*n)

def gauss(mu, sigma, x):
    """gauss gibt die normierte Gauss-Verteilung (Dichtefunktion) zurueck.

       Parameter: mu:    Erwartungswert
                  sigma: Standardabweichung
                  x:     Traeger
    """
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(((x - mu)/sigma)**2)/2)

def main():
    print(__doc__)          # Programmbeschreibung ausgeben
    N = 8                   # Anzahl der Teilchen
    R = 10000               # Anzahl der Realisierung
    delta_t = 4             # Zeitintervall
    n = np.zeros((R, N))    # "leere" RxN-Matrix fuer Anzahl der Reflexionen
    p = np.zeros(R)         # "leeres" Array fuer Druecke mit Laenge R
    # Schleife ueber die R Realisierungen:
    for i in np.arange(R):
        # zufaellige Orte x (gleichverteilt) fuer N Teilchen zwischen 0 und 1:
        x_0 = np.random.uniform(0, 1, size=N)
        # zufaellige Geschwindigkeiten fuer die N Teilchen:
        v = np.random.randn(N)
        # Anzahl Reflex. fuer Realisierung i in i-ter Zeile von n speichern:
        n[i, :] = reflexionen(x_0, v, delta_t)
        p[i] = druck(N, delta_t, v, n[i])

    # Mittelwert des Drucks berechnen und auf Konsole ausgeben:
    p_mean = np.mean(p)
    print("Der Erwartungswert des Drucks beträgt:", p_mean)
    # Standardabweichung des Drucks berechnen und auf Konsole ausgeben:
    sigma = np.sqrt((1/(R - 1) * np.sum((p - p_mean)**2)))
    print("Die Standardabweichung des Drucks beträgt:",sigma)

    plt.figure(1, figsize=(12,10))
    plt.title("Wahrscheinlichkeitsverteilung Druck p")      # Titel,
    plt.axis([0, p.max(), 0, 1])                            # Plot-
    plt.xlabel("Druck p")                                   # bereich,
    plt.ylabel("Häufigkeit")                                # Labels definieren

    # Binanzahl mittels Scott- (fuer R>200 i.A. zu wenig Bins) und Rice-Methode
    # (i.A. zu viele Bins) berechnet und Mittelwert aus beiden genommen
    # -> liefert opt. Binanzahl=28.69, muss aber int sein -> auf 30 aufgerundet
    plt.hist(p, bins=30, normed=True, label="Wahrscheinlichkeitsverteilung")

    # Druckwerte des Histogramm fuer Gauss-Verteilung wiederverwenden;
    # p-Werte fuer Plot der Gauss-Verteilung erst sortieren und dann plotten:
    p_sort = np.sort(p)
    plt.plot(p_sort, gauss(p_mean, sigma, p_sort), ls="dashed", linewidth=2,
             label="normierte Gauss-Verteilung")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    main()

"""Diskussion:
*) Betrachte die Abhaengigkeit von n_i von |x_i| aus der Vorlesung. Fuer die
untere Grenze der Ungleichungen ergibt sich der Zusammenhang |x_i| = 2*n_i - 1.
Umstellen liefert n_i = (|x_i| + 1)/2. Da n_i ein Integer sein muss (es gibt
keine halben Zusammenstoesse oder dergleichen), ergibt sich mit Hilfe der Un-
gleichungen aus der Vorlesung, dass der sich aus n_i = (|x_i| + 1)/2 ergebende
Wert auf die naechst kleinere ganze Zahl abgerundet werden muss.

a) N | Erwartungswert | Standardabweichung
   8 |      ~ 1       |     ~ 0.51
  80 |      ~ 1       |     ~ 0.16

Die (ungefaehren) Werte fuer den Erwartungswert und die Std.abweichung sind der
Tabelle zu entnehmen. Diese varieren minimal aufgrund der erzeugten Zufallsorte
und Geschwindigkeiten. Dabei stimmt die Form der Verteilung fuer N=8 Teilchen
grob mit der eingezeichneten Gauss-Verteilung ueberein, fuer N=80 ergibt sich
eine deutlich bessere Uebereinstimmung. Dabei ergibt sich fuer N=80 eine klein-
ere Standardabweichung, der Erwartungswert von p liegt in beiden Faellen wie zu
erwarten (siehe Vorlesung) ungefaehr bei 1 (und damit auch der Peak der Ver-
teilung). Durch die geringe Standardabweichung ergibt sich eine kleinere Breite
und groessere Hoehe des Gauss-Peaks fuer groessere N.

b) Fuer N=6*10**23 Teilchen wird sich der Gauss-Peak einem Delta-Peak um den Er
wartungswert von p=1 annaehern und die Standardabweichung dementsprechend gegen
0 gehen. Dies kommt daher, dass durch eine groessere Anzahl an betrachteten
Teilchen die Standardabweichung zunehmend abnimmt (siehe a)). Dies fuehrt zu
einem schmalen Gauss-Peak mit zunehmender Hoehe (siehe Definition des Gauss).

"""
