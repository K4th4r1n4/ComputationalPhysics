"""Das Programm "Standardabbildung" berechnet 1000 Iterationen der Standard-
   abbildung auf dem Torus und zeichnet diese bei Linksklick mit Startpunkt im
   geklickten Punkt in den Plotbereich ein. """

import functools
import numpy as np
import matplotlib.pyplot as plt

def torus_iteration(theta=0.0, p=0.0,K=0.0, n=1000):
    """torus_iteration berechnet n Iterationen der Standardabbildung auf dem
    Torus mit Parameter K und gibt die beiden Arrays plottheta und plotp
    zurueck, die jeweils die n berechneten Punkte fuer theta und p enthalten"""
    plottheta = []                                      # leere Arrays
    plotp = []                                          # erzeugen
    # n Werte theta und p iterativ berechnen
    for i in range(n):
        theta += (p % (2*np.pi))
        p += ((K*np.sin(theta) + np.pi) % (2*np.pi) - np.pi)
        plottheta.append((theta) % (2*np.pi))           # Werte an Arrays
        plotp.append((p + np.pi) % (2*np.pi) - np.pi)   # uebergeben
    return plottheta, plotp 
    
def linksklick(event, K):
    """linksklick plottet nach Linksklick die Standardabbildung auf dem Torus
     mit Startpunkt im ausgewaehlten Punkt."""
    # Test, ob Klick mit linker Maustaste und im Plotbereich erfolgt
    # und ob die Zoomfunktion des Plotfensters deaktiviert ist
    mode = plt.get_current_fig_manager().toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        x, y = torus_iteration(event.xdata, event.ydata,K=K, n=1000)
        plt.plot(x, y, linestyle='', marker='.', markersize=1)  
        plt.draw()                                      # Plotbefehl


def main():
    """Hauptprogramm: """
    # Festlegung des Paramters K der Standardabbildung
    K = 2.6
    
    plt.figure(0)                                       # Fenster und quadrat.
    plt.subplot(111)                                    # Plot erzeugen,
    plt.axis([0, 2*np.pi, -np.pi, np.pi])               # Plotbereich
    plt.title("Phasenraum Standardabbildung auf Torus") # Titel
    plt.xlabel(r"$\theta$")                             # Labels
    plt.ylabel("p")                                     # definieren
    x_pkt, y_pkt = torus_iteration()          
    plt.plot(x_pkt, y_pkt)
    
    # Bedienungsinformation fuer Benutzer des Programms
    print("""Mit Linksklick bitte den Startpunkt fuer die graphische"""
    """ Darstellung der Standardabbildung auf dem Torus"""
    """ Torus festlegen.""")
                            
    # Bei einem Linksklick wird die Funktion linksklick aufgerufen,
    # an diese wird der Parameter K (hier K=2.6) uebergeben, der vorher am
    # Anfang des Hauptprogramms festgelegt wurde
    klick_funktion = functools.partial(linksklick, K=K)
    plt.connect('button_press_event', klick_funktion)

    # Endlos-Schleife, die auf Ereignisse wartet
    plt.show()


if __name__ == "__main__":
    main()

# Beobachtungen:
# -fuer kleine Parameter K (bis K=1) ist die Dynamik noch ueberwiegnd regulaer
# -fuer Werte ab K=1 bilden sich inselartige Bereiche, beim Hereinzoomen stellt
#  man fest, dass sich das Muster dem im ungezoomten Zustand aehnelt: insel-
#  artige Bereiche mit weiteren darinliegenden kleineren regulaeren Inseln
# -fuer die Werte groesser K=6 ist die Dynamik fast vollkommen chaotisch



