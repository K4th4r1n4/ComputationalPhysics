"""Quantenmechanik von 1D-Potentialen 3:
Periodische Potentiale

In diesem Programm wird ein Teilchen im periodischen Potential
V(x) = A*cos(2*pi*x) betrachtet. Dabei sind A = 1 und h_eff =0.2. Links ist das
Eigenwertspektrum E_n(k) innerhalb der 1. Brillouin-Zone fuer Energien < 7 dar-
gestellt. Im rechten Plotbereich ist das periodische Potential V(x) fuer
n_per = 4 Perioden zu sehen.
Per Linksklick der Maus in den linken Plotbereich kann der Wert fuer k festge-
legt werden, fuer den im rechten Plotbereich anschliessend die betragsquadrier-
ten Eigenfunktionen farblich passend zum Eigenwertspektrum auf Hoehe der Eigen-
energien dargestellt werden.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def potential(x, A=1):
    """potential gibt das periodische Potential V(x) = A*cos(2*pi*x) mit De-
       faultwert A = 1 zurueck.

       Parameter: x: x-Werte, an denen das Potential betrachtet wird
                  A: Amplitude des Potentials
    """
    return A*np.cos(2*np.pi*x)

def diskretisierung(xmin, xmax, N, retstep=False):
    """diskretisierung berechnet die quantenmechanisch korrekte Ortsdiskreti-
       sierung.

    Parameter: xmin:    unteres Ende des Bereiches
               xmax:    oberes Ende des Bereiches
               N:       Anzahl der Diskretisierungspunkte
               retstep: entscheidet, ob Schrittweite zurueckgegeben wird

    Rueckgabe: x:       Array mit diskretisierten Ortspunkten
               delta_x: Ortsgitterabstand (nur wenn 'retstep=True')
    """
    delta_x = (xmax - xmin) / (N + 1)                      # Ortsgitterabstand
    # Ortsgitterpunkte, endpoint=False wegen Periodizitaet x=[0, 1):
    x = np.linspace(xmin+delta_x, xmax-delta_x, N, endpoint=False)

    if retstep:
        return x, delta_x
    else:
        return x

def diagonalisierung(h_eff, x, V, k):
    """diagonalisierung berechnet sortierte Eigenwerte und zugehoerige Eigen-
       funktionen.

    Parameter: h_eff: effektives hquer
               x:     Ortspunkte
               V:     Potential als Funktion einer Variable
               k:     Bloch-Phase

    Rueckgabe: ew: sortierte Eigenwerte (Array der Laenge N)
               ef: entsprechende Eigenvektoren, ef[:, i] (Groesse N*N)
    """
    delta_x = x[1] - x[0]
    v_werte = V(x)                              # Werte Potential

    N = len(x)
    z = h_eff**2 / (2.0*delta_x**2)             # Nebendiagonalelement

    # +0j wegen komplexer Eintraege der Matrix an zwei Ecken:
    h = (np.diag(v_werte + 2.0*z + 0j) +
         np.diag(-z*np.ones(N-1) + 0j, k=-1) +  # Matrix-Darstellung
         np.diag(-z*np.ones(N-1) + 0j, k=1))    # Hamilton-Operator

    h[0, -1] = -z * np.exp(-1j*k)               # kompl. Eintraege rechte obere
    h[-1, 0] = -z * np.exp(+1j*k)               # und linke untere Ecke

    ew, ef = eigh(h)                            # Diagonalisierung
    ef = ef/np.sqrt(delta_x)                    # WS-Normierung
    return ew, ef

def plot_eigenfunktionen(ax, ew, ef, x, V, anz, width=1, Emax=0.15, fak=0.01,
                         betragsquadrat=False, basislinie=True, alpha=1.0,
                         title=None):
    """plot_eigenfunktionen stellt die Eigenfunktionen dar.

    Dargestellt werden die niedrigsten Eigenfunktionen 'ef' im Potential 'V'
    auf Hoehe der Eigenwerte 'ew' in den Plotbereich 'ax'. Dabei werden genau
    die ersten 'anz' Eigenfunktionen mit zugehoerigen Eigenenergien geplottet.
    Die Eigenwerte werden hierbei als sortiert angenommen.

    Optionale Parameter:
        width: (mit Default-Wert 1) gibt die Linienstaerke beim Plot der
               Eigenfunktionen an. width kann auch ein Array von Linienstaerken
               sein mit einem spezifischen Wert fuer jede Eigenfunktion
        Emax:  (mit Default-Wert 0.15) legt die Energieobergrenze
               fuer den Plot fest
        fak:   ist ein Skalierungsfaktor fuer die graphische Darstellung
               der Eigenfunktionen mit Defaulwert 0.01
        betragsquadrat: gibt an, ob das Betragsquadrat der Eigenfunktion oder
                        die (reelle!) Eigenfunktion selbst dargestellt wird.
        basislinie: gibt an, ob auf Hoehe der jeweiligen Eigenenergie eine
                    gestrichelte graue Linie gezeichnet wird
        alpha: gibt die Transparenz beim Plot der Eigenfunktionen an (siehe
               auch Matplotlib Dokumentation von plot()). alpha kann auch ein
               Array von Transparenzwerten sein mit einem spezifischen Wert
               fuer jede Eigenfunktion
        title: Titel fuer den Plot

    """
    if title is None:
        title = "Betragsquadrierte Eigenfunktionen im periodischen Potential"

    plt.axes(ax)                                      # Ortsraumplotfenster
    plt.setp(ax, autoscale_on=False)
    plt.axis([np.min(x), np.max(x), np.min(V), Emax])
    plt.xlabel("x")
    plt.title(title)

    plt.plot(x, V, linewidth=2, color="0.7")          # Potential plotten

    if basislinie:                                    # Plot Basislinie bei Ew
        for i in np.arange(anz):
            plt.plot(x, ew[i] + np.zeros(len(x)), ls='--', color="0.7")

    try:                                              # Verhaelt sich width
        iter(width)                                   # wie ein array?
    except TypeError:                                 # Falls `width` skalar:
        width = width * np.ones(anz)                  # konst. Linienstaerke

    try:                                              # entsprechend fuer
        iter(alpha)                                   # Transparenz alpha
    except TypeError:
        alpha = alpha * np.ones(anz)

    colors = ['b', 'g', 'r', 'c', 'm', 'y']           # feste Farbreihenfolge
    if betragsquadrat:                                # Plot Betragsquadr. Efkt
        plt.ylabel(r'$V(x)\ \rm{,\ \|Efkt.\|^{2}\ bei\ EW}$')
        for i in np.arange(anz):
            plt.plot(x, ew[i] + fak*np.abs(ef[:, i])**2, linewidth=width[i],
                     color=colors[i % len(colors)], alpha=alpha[i])
    else:                                             # Plot Efkt
        plt.ylabel(r'$V(x)\ \rm{,\ Efkt.\ bei\ EW}$')
        for i in np.arange(anz):
            plt.plot(x, ew[i] + fak*ef[:, i], linewidth=width[i],
                     color=colors[i % len(colors)], alpha=alpha[i])

def linksklick(event, ax1, ax2, h_eff, potential, x, E_max, n_per, a, b, anz):
    """linksklick plottet nach Linksklick der Maus in Plotbereich ax1 die
       betragsquadrierten Eigenfunktionen und zugehoerigen -energien sowie das
       zugehoerige Potential fuer n_per Perioden in den Plotbereich ax2.
       Dabei wird mittels des Linksklick der Wert fuer k festgelegt und die al-
       ten eingezeichneten Linien in ax2 geloescht.

       Parameter: ax1:       Plotbereich, in den geklickt werden kann
                  ax2:       Plotbereich, in den eingezeichnet wird
                  h_eff:     effektives h_quer
                  potential: betrachtetes Potential
                  x:         x-Werte
                  E_max:     maximal betrachtete Energie
                  n_per:     Anzahl der betrachteten Perioden
                  a:         untere Intervallgrenze fuer x
                  b:         obere Intervallgrenze fuer x
                  anz:       Anzahl der zu plottenden Eigenfunktion/-energien
    """
    # Test, ob Klick mit linker Maustaste und im Plotbereich ax1
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = plt.get_current_fig_manager().toolbar.mode
    if event.button == 1 and event.inaxes == ax1 and mode == '':
        ax2.lines = []      # Linien in ax2 loeschen
        # Berechnung ef und zugehoerige ew fuer Plot,
        # Wert fuer k mit Linksklick festgelegt:
        ew_plot, ef_plot = diagonalisierung(h_eff, x, potential,
                                            event.xdata)

        # periodische Fortsetzung von x auf x_per (fuer n_per Perioden):
        x_per = x
        # i = 0 muss fuer Erweiterung uebersprungen werden, ansonsten x-Werte
        # doppelt, (b - a) entspricht Intervallbreite:
        for i in np.arange(n_per-1) + 1:
            x_per = np.append(x_per, x + i*(b - a))

        # periodische Fortsetzung der Eigenfunktionen, zugeh. EW und des Pot-
        # tentials fuer n_per Perioden (ef_plot ist dabei Matrix):
        ew_plot = np.tile(ew_plot, n_per)
        ef_plot = np.tile(ef_plot, (n_per, n_per))
        V_per = np.tile(potential(x), n_per)
        # plotten der betragsquadrierten ef im rechten Plotbereich ax2:
        plot_eigenfunktionen(ax2, ew_plot, ef_plot, x_per, V_per, anz,
                             Emax=E_max, fak=0.2, betragsquadrat=True)
        # Plotbereich fuer ax2:
        ax2.set_xlim([a, n_per*(b - a)])
        ax2.set_ylim([potential(x_per).min(), E_max])
        plt.draw()

def main():
    """Hauptprogramm:"""
    print(__doc__)          # Programmbeschreibung ausgeben
    A = 1                   # Amplitude des Potentials
    h_eff = 0.2             # effektives hquer
    N_k = 100               # Anzahl der k-Werte
    N_x = 150               # Anzahl der x-Werte
    a = 0                   # untere Intervallgrenze fuer x
    b = 1                   # obere  Intervallgrenze fuer x
    n_per = 4               # Periodenanzahl
    E_max = 7               # maximal betrachtete Energie

    x = diskretisierung(a, b, N_x, retstep=False)  # Ortsdiskretisierung
    k = np.linspace(-np.pi, np.pi, N_k)            # Variation der Bloch-Phase

    # periodische Fortsetzung von x auf x_per (fuer n_per Perioden):
    x_per = x
    # i = 0 muss fuer Erweiterung uebersprungen werden (sonst x-Werte doppelt):
    for i in np.arange(n_per-1) + 1:
        x_per = np.append(x_per, x + i*(b - a))

    ew = np.zeros((N_k, N_x))               # "leere" (N_k)x(N_x)-Matrix für EW
    # EW fuer alle k-Werte berechnen,
    # in i-te Zeile der Matrix ew die N_x Eigenwerte zwischenspeichern:
    for i in np.arange(N_k):
        ew[i, :] = diagonalisierung(h_eff, x, potential, k[i])[0]

    energien = ew[:, ew.min(0) < E_max]     # EW < E_max in Matrix energien
    # Anzahl der Spalten der Matrix energien entspricht Anzahl zu plottender
    # Eigenfunktionen und Energien im Eigenwertspektrum:
    anz = energien.shape[1]

    colors = ['b', 'g', 'r', 'c', 'm', 'y'] # feste Farbreihenfolge fuer Plots

    plt.figure(0, figsize=(16, 10))
    # subplot fuer Eigenwertspektrum (links):
    ew_spektrum = plt.subplot(121)
    ew_spektrum.set_xlim([-np.pi, np.pi])                          # Plot-
    ew_spektrum.set_ylim([potential(x_per, A).min(), E_max])       # bereich,
    ew_spektrum.set_title("1. Brillouin-Zone")                     # Titel,
    ew_spektrum.set_xlabel("k")                                    # Labels
    ew_spektrum.set_ylabel("E$_{n}$(k)")                           # definieren
    # Eigenwertspektrum mit unterschiedlichen Farben fuer E_n plotten:
    for i in np.arange(anz):
        ew_spektrum.plot(k, energien[:,i], color=colors[i % len(colors)])

    # subplot fuer betragsquadrierte Eigenfunktionen (rechts):
    betragsquadrat = plt.subplot(122)
    betragsquadrat.set_xlim([a, n_per*(b - a)])                    # Plot-
    betragsquadrat.set_ylim([potential(x_per, A).min(), E_max])    # bereich,
    betragsquadrat.set_title("Betragsquadrierte Eigenfunktionen "  # Titel,
                             "im periodischen Potential")          # Labels
    betragsquadrat.set_xlabel("x")                                 # definieren
    betragsquadrat.set_ylabel(r"$V(x)\ \rm{,\ \|Efkt.\|^{2}\ bei\ EW}$")
    # Potential plotten:
    betragsquadrat.plot(x_per, potential(x_per, A), linewidth=2, color="0.7")

    # bei Linksklick der Maus in linken Plotbereich linksklick anwenden:
    klick_funktion = functools.partial(linksklick, h_eff=h_eff, a=a, anz=anz,
                                       ax1=ew_spektrum, ax2=betragsquadrat,
                                       potential=potential, n_per=n_per, x=x,
                                       E_max=E_max, b=b)
    plt.connect('button_press_event', klick_funktion)
    plt.show()

if __name__ == "__main__":
    main()

"""Diskussion:
a) Eine Variation der k-Werte hat auf die erste (unterste) Eigenfunktion fast
gar keinen Einfluss. Auch fuer die zweite betragsquadrierte Eigenfunktion er-
gibt eine Variation der k-Werte keine grossen Veranderungen. Lediglich die Eig-
enenergien veraendern sich entsprechend des Eigenwertspektrums (etwas groessere
Energien fuer betragsmaessig kleine k, fuer |k| -> pi minimal kleiner werdende
Energien). Dies liegt daran, dass es sich hierbei um die im periodischen Poten-
tial gebundenen Zustaende handelt.
Am Rand der ersten Brillouin-Zone (|k|=pi) liegen die 3. und 4. sowie die 5.
und 6. Eigenfunktion eng beieinander und sind wellenfoermig. Dabei sind die Am-
plituden der 3. und 4. deutlich groesser als die der 5. und 6., da das Potenti-
al auf die unteren beiden aufgrund der niedrigeren Energien einen groesseren
Einfluss hat. Passend dazu laufen im Eigenwertspektrum die Energiewerte der 3.
und 4. sowie der 5. und 6. EF an den Raendern der 1.BZ als Folge des period.
Potentials zusammen, schneiden sich aber nicht(!), was durch hineinzoomen er-
kennbar ist.
Fuer betragsmaessig kleiner werdende k-Werte nehmen die Amplituden der 3.-6. EF
bis |k|=pi/2 ab, um anschliessend wieder bis k=0 zuzunehmen. Fuer den Fall k=0
liegen die 2. und 3. sowie die 4. und 5. (passend zum Eigenwertspektrum) nahe
beinander. Dabei beobachtet man fuer die beiden "Paare" von EF jeweils, dass
die Minima der einen EF den Maxima der anderen und umgekehrt entsprechen. Dabei
sind die Maxima/Minima der Aufenthaltswahrscheinlichkeiten immer jeweils an den
Stellen, an denen sich Atome befinden, bzw. genau zwischen zwei Atomen.
Weiterhin beobachtet man, dass die betragsquadrierten EF nie genau auf dem je-
weiligen EW liegen. Es gibt also keine Bereiche, in denen das Teilchen nie an-
zutreffen ist (lediglich mit sehr geringer WSK).


b) Fuer den Grenzfall eines sehr schwachen Potentials A << 1 ergeben sich fuer
die obersten Eigenfunktionen unabhaengig vom betrachteten k-Wert nahezu waage-
rechte Linien. Nur die Hoehe dieser veraendert sich entsprechend des im linken
Plotbereich dargestellten Eigenwertspektrums. Das Potential ist hierbei so
schwach, dass es kaum Einfluss auf die Eigenfunktionen hat. Man kann also von
nahezu freien Teilchen ausgehen. Nur fuer k=0 ist fuer die 2. und 3. EF wieder
ein wellenfoermiger Verlauf erkennbar, wobei die Minima der 2. genau auf Hoehe
der Maxima der 3. liegen und umgekehrt. Durch Hineinzoomen laesst sich auch
fuer die 4. und 5. betragsquadrierte EF ein Wellencharakter der Aufenthaltswsk
erkennen, allerdings mit sehr kleiner Amplitude.
Fuer die 1. und 2. EF ergeben sich wellenfoermige betragsquadrierte EF fuer k-
Werte nahe dem Rand der 1.BZ.

"""
