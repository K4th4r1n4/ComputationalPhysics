"""Diffusion mit Drift und Absorption

In diesem Programm wird eine gerichtete Diffusion mit absorbierendem Rand ueber
einem Zeitintervall von [0, T_max=40] betrachtet. Dabei wird die Wahrschein-
lichkeitsdichte P(x, t_n) zu Zeiten t_n = 1, 2, 3.. fuer R=10000 Realisierungen
mit Anfangsort x_0=0 dynamisch dargestellt. Der absorbierende Rand befindet
sich bei x_abs=15. Weiterhin wird die Norm, der Erwatungswert und die Varianz
zu den Zeiten t_n in seperaten Plotbereichen dynamisch dargestellt. In allen 4
Subplots ist zusaetzlich die theoretische Vorhersage fuer den Fall ohne Absorp-
tion eingezeichnet. Fuer das Histogrammist zusaetzlich die theoretisch erwarte-
te Verteilung im Fall mit Absorption zu sehen.

Die Dynamik kann mittels Linksklick der Maus in einen der 4 Plotbereiche ge-
startet werden.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt

def gauss(x, mu, var):
    """gauss gibt die normierte Gauss-Verteilung (Dichtefunktion) zurueck.

       Parameter: x:     Traeger
                  mu:    Erwartungswert
                  var:   Varianz

    """
    # Division durch var = 0 vermeiden, Test ob mu!=x fuer alle Arrayeintraege
    # -> falls ja, gaussian auf 0 setzen, da
    # lim(var -> 0+) gauss = 0 fuer mu ungleich x:
    if var == 0 and (mu!=x).all():
        gauss = 0.0 * x
    else:
        gauss = (1/(np.sqrt(2*np.pi*var))) * np.exp(-(((x - mu)**2)/(2*var)))
    return gauss

def linksklick(event, ax1, ax2, ax3, ax4, x_0, T_max, delta_t, R, D, v_drift,
               x_abs):
    """linksklick startet nach Linksklick der Maus in einen der Plotbereiche
       axi (mit i=1,2,3,4) eine dynamische Zeitentwicklung der WSK-Dichte
       P(x, t_n), der Norm, des Erwartungswertes und der Varianz der numerisch-
       en Implemetation der Langevin-Gleichung zur Beschreibung einer gerichte-
       ten Diffusion mit absorbierenden Rand. Dabei wird die Zeitentwicklung
       fuer alle Zeitschritte delta_t ab t=0 berechnet, aber nur zu Zeiten
       t_n=1, 2, 3... bis T_max dynamisch dargestellt. Zusaetzlich wird die
       theoretische Vorhersage fuer den Fall ohne Abs. eingezeichnet.

       Parameter: ax1:      Subplot fuer WSK-Dichte P(x, t_n)
                  ax2:      Subplot fuer Norm
                  ax3:      Subplot fuer Erwartungswert
                  ax4:      Subplot fuer Varianz
                  x_0:      Anfangsorte
                  T_max:    maximal betrachtete Zeit
                  delta_t:  Zeitschrittweite
                  R:        Anzahl d. Realisierungen
                  D:        Diffusionskonstante
                  v_drift:  Driftgeschwindigkeit
                  x_abs:    Position des absorbierenden Randes
    """
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = plt.get_current_fig_manager().toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        x_t = x_0           # Startort zu t=0
        # Array mit betrachteten Zeiten:
        t = np.arange(0, T_max + delta_t, delta_t)
        for i in t:
            # nur Zeiten t_n = 1,2,3.. dynamsich darstellen:
            if i % 1.0 == 0.0:
                # neue Orte berechnen:
                x_t = (x_t + v_drift*delta_t + np.sqrt(2*D*delta_t) *
                       np.random.randn(R))
                # nur Teilchen mit Orten kleiner als absorb. Rand betrachten:
                x_plot = x_t[x_t < x_abs]
                # Wichtung R(t_n)/R:
                weight = np.ones(len(x_plot)) * (len(x_plot)/R)

                # alte Histogrammdaten in Plotbereich ax1 loeschen:
                ax1.patches = []
                # neue Daten fuer Histogramm in ax1 plotten:
                ax1.hist(x_plot, bins=30, normed=True, weights=weight,
                         color="b")
                # alte Plotdaten in Plotbereich ax1 loeschen:
                ax1.lines = []
                # rote vert. Linie des absorb. Randes an x_abs neueinzeichnen:
                ax1.axvline(x=x_abs, ls="dashed", c="r",
                            label="absorbierender Rand")
                # theoretische WSK-Dichte fuer Fall ohne Absorption berechnen:
                p_ohne_abs = gauss(x_t, x_0 + v_drift*i, 2*D*i)
                # theor. WSK-Dichte fuer Fall mit Absorption berechnen, Teilen
                # durch 0 vermeiden, falls Div. durch 0 WSK auf 0 setzen:
                try:
                    p_mit_abs = (gauss(x_t, x_0 + v_drift*i, 2*D*i) -
                                 gauss(x_t, 2*x_abs - x_0 + v_drift*i, 2*D*i) *
                                (gauss(x_abs, x_0 + v_drift*i, 2*D*i) /
                                 gauss(x_abs, 2*x_abs-x_0 + v_drift*i, 2*D*i)))
                except ZeroDivisionError:
                    p_mit_abs = np.zeros(len(x_t))

                # nur einmal Legendeneintraege f. dynam. Darstellung plotten
                # -> seperate Behandlung von Startzeit t=0:
                if i == 0.0:
                    ax1.plot(x_t, p_ohne_abs, ls="", marker="o", ms=1.2, c="k",
                             label="theoretische Erwartung ohne Absorption")
                    ax1.plot(x_t, p_mit_abs, c="y", ls="", marker="o",ms=0.8,
                             label="theoretische Erwartung mit Absorption")
                    ax2.plot(i, len(x_t)/R, c="k", marker="o", ms=2,
                             label="ohne Absorption")
                    ax2.plot(i, len(x_plot)/R, c="c", marker="o", ms=2,
                             label="mit Absorption")
                    ax3.plot(i, np.mean(x_t), c="k", marker="o", ms=2,
                             label="ohne Absorption")
                    ax3.plot(i, np.mean(x_plot), c="g", marker="o", ms=2,
                             label="mit Absorption")
                    ax4.plot(i, np.var(x_t, ddof=1), c="k", marker="o", ms=2,
                             label="ohne Absorption")
                    ax4.plot(i, np.var(x_plot, ddof=1), c="m", marker="o",ms=2,
                             label="mit Absorption")
                # Plotbefehle zu Zeiten t groesser 0 und t % 1.0 == 0.0:
                else:
                    ax1.plot(x_t, p_ohne_abs, ls="", marker="o", ms=1.2, c="k",
                             label="theoretische Erwartung ohne Absorption")
                    ax1.plot(x_t, p_mit_abs, c="y", ls="", marker="o", ms=0.8,
                             label="theoretische Erwartung mit Absorption")
                    ax2.plot(i, len(x_t)/R, c="k", marker="o", ms=2)
                    ax2.plot(i, len(x_plot)/R, c="c", marker="o", ms=2)
                    ax3.plot(i, np.mean(x_t), c="k", marker="o", ms=2)
                    ax3.plot(i, np.mean(x_plot), c="g", marker="o", ms=2)
                    ax4.plot(i, np.var(x_t, ddof=1), c="k", marker="o", ms=2)
                    ax4.plot(i, np.var(x_plot, ddof=1), c="m", marker="o",ms=2)
                # Legendeneintraege anzeigen und links oben fixieren:
                ax1.legend(loc="upper left")
                ax2.legend(loc="upper left")
                ax3.legend(loc="upper left")
                ax4.legend(loc="upper left")

                plt.gcf().canvas.flush_events()
                plt.draw()
            # fuer alle anderen Zeitschritte delta_t die x_t berechnen,
            # aber nichts plotten:
            else:
                x_t = (x_t + v_drift*delta_t + np.sqrt(2*D*delta_t) *
                       np.random.randn(R))

def main():
    print(__doc__)      # Programmbeschriebung ausgeben
    T_max = 40          # maximale Zeit
    R = 10000           # Anzahl der Realisierungen
    x_0 = np.zeros(R)   # Anfangsort fuer R Realisierungen
    v_drift = 0.1       # Driftgeschwindigkeit
    D = 1.5             # Diffusionskonstante
    x_abs = 15          # Position absorbierender Rand
    delta_t = 0.01      # Zeitschrittweite

    # figure anlegen und Ueberschrift festlegen:
    figure = plt.figure(0, figsize=(14,10))
    figure.suptitle("Diffusion mit Drift und Absorption", fontsize=18)
    # subplot ax1 fuer Wahrscheinlichkeitsdichte P(x, t_n):
    ax1 = plt.subplot(221)
    # rot gestrichelte, vertikale Linie an Stelle des absorbierenden Randes:
    ax1.axvline(x=x_abs, ls="dashed", c="r", label="absorbierender Rand")
    ax1.set_xlim([-20, 20])                                     # Plot-
    ax1.set_ylim([0.0, 0.3])                                    # bereich,
    ax1.set_title("Wahrscheinlichkeitsdichte $P(x, t_{n})$")    # Titel,
    ax1.set_xlabel("x")                                         # Labels,
    ax1.legend(loc="upper left")                                # Legende
    ax1.set_ylabel("$P(x, t_{n})$")                             # definieren
    ax1.set_autoscale_on(False)

    # subplot ax2 fuer Norm:
    ax2 = plt.subplot(222)
    ax2.set_xlim([0, T_max])                                    # Plot-
    ax2.set_ylim([0, 2])                                        # bereich,
    ax2.set_title("Norm $R(t_{n})/R$")                          # Titel,
    ax2.set_xlabel("t")                                         # Labels
    ax2.set_ylabel("$R(t_{n})/R$")                              # definieren
    ax2.set_autoscale_on(False)

    # subplot ax3 fuer Erwartungswert:
    ax3 = plt.subplot(223)
    ax3.set_xlim([0, T_max])                                    # Plot-
    ax3.set_ylim([0, 5])                                        # bereich,
    ax3.set_title("Erwartungswert $\mu$")                       # Titel,
    ax3.set_xlabel("t")                                         # Labels
    ax3.set_ylabel("$\mu$")                                     # definieren
    ax3.set_autoscale_on(False)

    # subplot ax4 fuer Varianz:
    ax4 = plt.subplot(224)
    ax4.set_xlim([0, T_max])                                    # Plot-
    ax4.set_ylim([0, 125])                                      # bereich,
    ax4.set_title("Varianz $\sigma^{2}$")                       # Titel,
    ax4.set_xlabel("t")                                         # Labels
    ax4.set_ylabel("$\sigma^{2}$")                              # definieren
    ax4.set_autoscale_on(False)

    # bei Linksklick der Maus im Plotbereich linksklick anwenden:
    klick_funktion = functools.partial(linksklick, ax1=ax1, ax2=ax2, ax3=ax3,
                                       ax4=ax4, x_0=x_0, D=D, R=R, T_max=T_max,
                                       delta_t=delta_t, v_drift=v_drift,
                                       x_abs=x_abs)
    plt.connect("button_press_event", klick_funktion)
    plt.show()

if __name__ == "__main__":
    main()

"""Diskussion:
a) Der absorbierende Rand fuehrt im Falle der Norm, des Erwartungswertes und
der Varianz dazu, dass diese im Vergleich zur theoretischen Vorhersage ohne
absorbierenden Rand mit zunehmender Zeit immer kleiner werden. Dies war zu er-
warten, da mit absorbierendem Rand die Zahl R(t_n) mit zunehmendem t_n kleiner
wird (Teilchen werden abs.) und damit die Norm immer geringer wird. Da die
Teilchen nicht beliebig weit nach rechts driften koennen, ohne abs. zu werden,
ist der Erwartungswert im Fall des abs. Randes begrenzt, wohingegen er ohne ihn
immer weiter zunimmt, da die Teilchen ungestoert weiter driften koennen.
Dies wird ungefaehr ab Zeiten groesser gleich 7 sichtbar

b) Mit veraenderten delta_t = 1 sieht man fast gar keinen Unterschied in den
untersuchten Groessen. Einzig der Erwartungswert ist beim Fall delta_t=1 gering
fuegig kleiner, als bei delta_t = 0.01. Ausserdem stimmt die WSK-Dichte gerade
zu Beginn der Dynamik fuer den Fall delta_t=1 nicht so gut mit den Theoriever-
laeufen ueberein, wie im Fall delta_t = 0.01. Dies liegt an den nicht vorhand-
enen Zwischenzeitschritte zur Berechnung der Dynamik.

c) Fuer den Fall einer Driftgescheschwindigkeit von v_drift=0.5 driften die
Teilchen viel schneller in positive x-Richtung. Daher erreichen im betrachteten
Zeitraum deutlich mehr Teilchen den abs. Rand, weshalb die Norm viel schneller
abnimmt. Auch der Unterschied zwischen den Varianzen und den Erwartungswerten
mit und ohne abs. Rand wird schneller sichtbar und ist im betrachten Zeitraum
auch groesser als im Vergleich zu v_drift=0.1. Wie im Falle von v_drift=0.1
sind dabei die Werte fuer Varianz, Norm und Erwartungswert im Falle mit Rand
kleiner als ohne.
"""
