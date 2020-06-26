"""Quantenmechanik von 1D-Potentialen 2:
Zeitentwicklung

In diesem Programm wird die Zeitentwicklung eines Gauss'schen Wellenpakets im
asymmetrischen Doppelmuldenpotential V(x) = x^4 + x^2 - A*x betrachtet. A ist
hierbei auf A = 0.15 festgelegt.
Mit einem Linksklick der Maus kann der Startpunkt fuer den mittleren Ort x_0
festgelegt werden. Anschliessend wird die Zeitentwicklung des Betragsquadrates
des Wellenpakets in Hoehe des Energieerwartungswertes geplottet.
Fuer jeden Startpunkt wird die Norm der Differenz zwischen dem Anfangswellen-
paket phi und dem aus den Entwicklungskoeffizienten c_n rekonstruierten Wellen-
paket phi', also sqrt(|phi - phi'|^2), ausgegeben.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
import quantenmechanik as qm

def wellenpaket(x, x_0, del_x, h_eff, p_0):
    """wellenpaket gibt das Gauss'sche Wellenpaket

       phi(x, t=0) = 1/(2*pi*(del_x)^2)^(1/4) * exp(-(x-x_0)^2/(4*(del_x)^2) *
                     exp(i/h_eff * p_0 * x)
       zurueck.

       Parameter: x:     Ortspunkte
                  x_0:   mittlerer Ort
                  del_x: Breite des Gauss'schen Wellenpakets
                  h_eff: einheitenloser Wert fuer h_quer
                  p_0:   mittlerer Impuls
    """
    return (1/(2*np.pi * (del_x)**2)**(1/4) *
            np.exp(- (x - x_0)**2 / (4 * del_x**2)) *
            np.exp(1j/h_eff * p_0 * x))

def potential(x, A=0.15):
    """potential gibt das asymmetrische Doppelmuldenpotential
       V(x)= x**4 - x**2 - A*x
       mit Defaultwert A = 0.15 zurueck.

       Parameter: x: x-Werte, an denen das Potential betrachtet wird
                  A: Parameter fuer Asymmetrie des Potentials
    """
    return x**4 - x**2 - A*x

def linksklick(event, ew, ef, x, h_eff, del_x, p_0, faktor=0.01):
    """linksklick plottet nach Linksklick der Maus die Zeitentwicklung eines
       Gauss'schen Wellenpaketes, wobei der mittlere Ort mittels des Linksklick
       festgelegt wird. Die Norm der Differenz des urspruenglichen Wellenpakets
       und des aus den Entwicklungskoeffizienten c_n rekonstruierten Paketes
       wird bei jeder geplotteten Zeitentwicklung mit ausgegeben. Der Skalier-
       ungsfaktor zur besseren Visualisierug ist standardmaessig auf 0.01 fest-
       gelegt.

       Parameter: ew:     sortierte Eigenwerte (Array der Laenge N)
                  ef:     entsprechende Eigenvektoren, ef[:, i]
                          (Matrix der Dimension N*N)
                  x:      Ortspunkte
                  h_eff:  einheitenloser Wert fuer h_quer (fuer Wellenpaket)
                  del_x:  Breite des Gauss'schen Wellenpaketes
                  p_0:    mittlerer Impuls (fuer Wellenpaket)
                  faktor: Skalierungsfaktor fuer graphische Darstellung
    """
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = plt.get_current_fig_manager().toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        # Startwellenpaket (x_0 mit Linksklick festgelegt):
        phi_0 = wellenpaket(x, event.xdata, del_x, h_eff, p_0)
        delta_x = x[1] - x[0]                # Ortsgitterabstand
        # Entwicklungskoeffizienten c_n berechnen (Konjugation wegen Skalarpro-
        # dukt, aber in ef eigentlich nur reelle Eintraege),
        # Eigenfunktionen sind spaltenweise in ef angeordnet:
        # -> Transponieren noetig, wegen Matrixmultiplikation (Zeilen * Vektor)
        c = delta_x * np.dot(np.conjugate(np.transpose(ef)), phi_0)
        # mittels Koeffizienten rekonstruiertes phi' := phi_test:
        phi_test = np.dot(ef, c)
        # Norm der Differenz phi - phi' berechnen und auf Konsole ausgeben:
        print("Norm der Differenz =",np.sum(np.sqrt(abs(phi_test - phi_0)**2)))
        # Energieerwartungswert <phi|H|phi> berechnen:
        energie = np.dot(abs(c)**2, ew)
        ax = plt.plot(x, abs(phi_0)**2)      # |phi_0|^2 plotten
        zeiten = np.linspace(0, 5, 100)      # Array mit betrachteten Zeiten
        for t in zeiten:
            # Konstruktion von phi(t):
            phi_t = np.dot(np.conjugate(ef), c*np.exp((-1j*ew*t)/h_eff))
            # Zeitentwicklung des Betragsquadrates des Wellenpakets auf Hoehe
            # des Energieerwartungswertes plotten, dafuer mittels plt.setp
            # Daten immer neu setzen (dynamische Darstellung):
            plt.setp(ax[0], ydata=faktor*abs(phi_t)**2 + energie)
            plt.gcf().canvas.flush_events()
            plt.draw()
        # Benutzerfuehrung:
        print("Mittels Linksklick neue Zeitentwicklung starten.")

def main():
    """Hauptprogramm:"""
    print(__doc__)                   # Programmbeschreibung ausgeben
    p_0 = 0.0                        # mittlerer Impuls
    h_eff = 0.07                     # dimensionsloses h_quer
    del_x = 0.1                      # delta_x im Gausspaket
    l = 1.5                          # Intervallgrenze
    N = 150                          # Anzahl der Diskretisierungspunkte
    # diskretisierte Ortspunkte x  mit Hilfe von quantenmechanik.py berechnen:
    x = qm.diskretisierung(-l, l, N, retstep=False)
    # Eigenwerte und Eigenfunktion fuer asym. Doppelmuldenpotential mit Hilfe
    # von quantenmechanik.py berechnen:
    ew, ef = qm.diagonalisierung(h_eff, x, potential)
    plt.figure(0, figsize=(12,10))   # figure,
    ax = plt.subplot(111)            # subplot ax festlegen
    # mittels quantenmechnanik.py Betragsquadrate der Eigenfunktionen fuer asym
    # Doppelmuldenpotential bis Eigenenergien E_max = 0.15 plotten:
    qm.plot_eigenfunktionen(ax, ew, ef, x, potential, betragsquadrat=True,
                            title="Zeitentwicklung im asymmetrischen "
                                  "Doppelmuldenpotential")
    # bei Linksklick der Maus im Plotbereich linksklick anwenden:
    klick_funktion = functools.partial(linksklick, ew=ew, ef=ef, x=x,
                                       h_eff=h_eff, del_x=del_x, p_0=p_0)
    plt.connect('button_press_event', klick_funktion)
    plt.show()

if __name__ == "__main__":
    main()

"""Diskussion:
a) Beim Startpunkt eines Pakets im Minimum (also einer der beiden Mulden) be-
wegt sich das Wellenpaket leicht nach links und rechts. Dabei wird es an den
Potentialwaenden reflektiert. Die Breite des Paketes bleibt hier relativ konst-
ant, die Form des Paktes andert sich nur leicht (verglichen mit anderen Start-
punkten). Wenn man den Startpunkt in der linken Mulde (also des betragsmaessig
kleineren Minimums) waehlt, so wird ausserdem ein minimaler Anteil des ur-
spruenglichen Wellenpaktes durch den Potentialwall in die rechte Mulde trans-
mittiert. Man beobachtet also den Tunneleffekt.
Beim Startpunkt im Maximum ist ein gleichmaessiges zerfliessen des Wellenpaket-
es nach rechts und links zu erkennen. An den Potentialgrenzen werden beide Pa-
kete reflektiert. Anschliessend treffen diese reflektierten Pakete wieder auf-
einander und ueberlagern sich.

b) Fuer p_0 = 0.3 sind die Energieerwartungswerte fuer gleiche Startpunkte im
Vergleich zu p_0 = 0.0 groesser.
Bei Startpunkt in einem Minimum ist zunaechst die beobachtete Bewegung aehnlich
zu denen bei p_0 = 0.0. Allerdings kommt es dadurch, dass der mittlere Impuls
nicht mehr 0 ist, nach wenigen Reflektionen an den Potentialwaenden zu Fluktua-
tioen der Amplitude des Wellenpakets. Auch ist das "Zerfliessen" des Wellenpa-
kets etwas staerker, als bei verschwindenden mittleren Impuls p_0. Es ist ein
Eindringen in den "verbotenen Bereich" ueber das Potential hinaus (in x-Richt-
ung) zu sehen. Hier tunnelt im Vergleich zum Fall mit p_0 0 0.0 ein groesserer
Anteil des urspruenglichen Gauss'schen Wellenpaketes.
Beim Startpunkt im Maximum zerfliesst das Wellenpaket fuer p_0 = 0.3, im Gegen-
satz zum Fall p_0 = 0.0, nicht mehr gleichmaessig in positive und negative x-
Richtung. Es ist aufgrund des positiven mittleren Impulses ein "Zerlaufen" des
Paketes in positive x-Richtung zu beobachten (fuer negatives p_0 wuerde man
eine anfaengliche Bewegung des Paketes in negative x-Richtung beobachten),
waehrend nur ein minimaler Anteil in negative x-Richtung zu erkennen ist. An
den Grenzen des Potentals wird das Wellenpaket wieder reflektiert, allerdings
ist ein Eindringen in den "verbotenen Bereich" ueber die Potentialgrenzen hin-
aus zu erkennen.

c) Fuer den Fall A = 0.0 (symmetrisches Potential) und p_0 = 0.0 (verschwinden-
der mittlerer Impuls) laesst sich fuer grosse Zeiten beim Startpunkt in einem
Minimum ein Tunneln in das jeweils andere Minimum beobachten. Dabei tunnelt das
Wellenpaket fuer t_max = 20000 genau 7 mal von einer Mulde in die andere. Zum
Schluss befindet sich das Paket also in der jeweils anderen Mulde. Man muss
fuer diese Beobachtung grosse Zeiten betrachten, da die Wahrscheinlichkeit zum
Tunneln nur sehr gering ist.
"""
