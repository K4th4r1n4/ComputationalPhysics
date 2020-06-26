"""Differentialgleichungen

In diesem Programm wird die Dynamik eines Teilchens im angetriebenen Doppelmul-
denpotential V(x,t) = x^4 - x^2 + x*[A * B*sin(omega*t)] mit den Parametern
A = 0.2, B = 0.1 und omega = 1 betrachtet. Dieses wird durch die dimensionslose
Hamiltonfunktion H(x,p,t) = p^2/2 + V(x,t) beschrieben.
Es werden zwei Plotbereiche dargestellt, die Phasenraeumen entsprechen: links
wird die Trajektorie (x(t),p(t)) abgebildet und rechts die stroboskopische Dar-
stellung der Trajektorie, bei der nur die Zeiten t_i = (2*pi)/omega * i nach
einer vollen Periode betrachtet werden (wobei i eine natuerliche Zahl ist).
Durch einen Linksklick der Maus in eines der beiden Phasenraumdiagramme wird
der Startpunkt (x(0), p(0)) fuer eine Trajektorie des Teilchens fuer 200 Perio-
den festgelegt und diese in beide eingezeichnet.
Zusaetzlich sind in beiden Plotbereichen fuer den Fall B = 0 fuer die 8 Ener-
giewerte [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5] Konturlinien einge-
zeichnet.
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def ableitung(y, t, A, B, omega):
    """ableitung gibt die rechte Seite der DGL der Dynamik eines Teilchens im
       angetriebenen Doppelmuldenpotential als Array [dot_x, dot_p] zurueck.
       DGL: dot_x = p
            dot_p = (-4)*x^3 + 2*x - A - B*sin(omega*t)

       Parameter: y:     y = [x, p] (x und p Wertepaare)
                  t:     Zeiten
                  A:     Parameter der Hamiltonfunktion
                  B:     Antrieb
                  omega: Kreisfrequenz der Hamiltionfunktion

    """
    return np.array([y[1], (-4)*y[0]**3 + 2*y[0] - A - B*np.sin(omega*t)])

def linksklick(event, A, B, omega, perioden, N):
    """linksklick plottet nach Linksklick die Trajektorie eines Teilchens im
       angetriebenen Doppelmuldenpotential im Phasenraum (x(t), p(t)) sowie in
       stroboskopischer Darstellung des Phasenraums mit Startpunkt im geklickt-
       en Punkt.
       Parameter: A:        Parameter der Hamiltonfunktion
                  B:        Antrieb Hamiltonfunktion
                  omega:    Kreisfrequenz der Hamiltonfunktion
                  perioden: Anzahl der betrachteten Perioden (entspricht Anzahl
                            der geplotteten Punkte der stroboskopischen Dar-
                            stellung -1)
                  N:        Anzahl der Punkte, die pro Periode fuer die Trajek-
                            torie betrachtet werden
    """
    # Test, ob Klick mit linker Maustaste und im Koordinatensystem
    # erfolgt sowie ob Zoomfunktion des Plotfensters deaktiviert ist:
    mode = plt.get_current_fig_manager().toolbar.mode
    if event.button == 1 and event.inaxes and mode == '':
        # Definition der betrachteten Zeiten:
        zeiten = np.linspace(0.0, (2*np.pi/omega)*perioden, perioden*N+1)

        # Integration der DGL, Anfangsbedingung wird mit Mausklick festgelegt:
        y_t = odeint(ableitung, [event.xdata, event.ydata], zeiten,
                     args=(A, B, omega))
        x_t = y_t[:, 0]                 # Auslesen der Spalte mit x
        p_t = y_t[:, 1]                 # Auslesen der Spalte mit p
        trajekt = plt.subplot(121)      # links Plot fuer Trajektorie erzeugen
        trajekt.plot(x_t, p_t, marker="", ls="-", linewidth=1)

        strob = plt.subplot(122)        # rechts stroboskopischen Plot erzeugen
        counter = np.arange(perioden+1) # Array fuer stroboskopischen Plot
        # Periode (2*pi) unterteilt in N Teilpunkte -> fuer stroboskopischen
        # Plot nur jeder (counter * N)-te Eintrag aus zeiten relevant
        # (entspricht Vielfachen von 2*pi):
        strob.plot(x_t[counter*N], p_t[counter*N], marker=".", markersize=2.5,
                   ls="")
        plt.draw()

def main():
    """Hauptprogramm:"""
    print(__doc__)           # Programmbeschreibung ausgeben
    A = 0.2                  # Parameter der Hamiltonfunktion
    B = 0.1                  # Antrieb
    omega = 1                # Kreisfrequenz

    perioden = 200           # Anzahl der Perioden
    N = 100                  # Anzahl der Punkte pro Periode fuer Trajektorie

    # x und p Werte fuer die Konturlinien erzeugen:
    x = np.linspace(-1.5, 1.5, 100)          # 1D x-Werte
    p = np.linspace(-2.0, 2.0, 100)          # 1D p-Werte
    x2D, p2D = np.meshgrid(x, p)             # 2D-Arrays fuer x und p erzeugen
    H = p2D**2/2 + x2D**4 - x2D**2 + A*x2D   # Hamiltonfunktion
    # Energiewerte fuer Konturlinien:
    energien = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    # figure anlegen und Ueberschrift festlegen:
    figure = plt.figure(0, figsize=(14,10))
    figure.suptitle("Teilchendynamik im angetriebenen Doppelmuldenpotential",
                     fontsize=20)
    # Plot Trajektorie
    trajekt = plt.subplot(121)
    trajekt.set_xlim([-1.5, 1.5])                     # Plot-
    trajekt.set_ylim([-2.0, 2.0])                     # bereich,
    trajekt.set_title("Trajektorie im Phasenraum")    # Titel,
    trajekt.set_xlabel("x(t)")                        # Labels
    trajekt.set_ylabel("p(t)")                        # definieren
    # schwarze Konturlinien fuer die 8 Energiewerte einzeichnen:
    trajekt.contour(x2D, p2D, H, levels=energien, ls="", linewidths=1,
                    colors="black")

    # Plot stroboskopische Darstelllung der Trajektorie
    strob = plt.subplot(122)
    strob.set_xlim([-1.5, 1.5])                                    # Plot-
    strob.set_ylim([-2.0, 2.0])                                    # bereich,
    strob.set_title("Trajektorie in stroboskopischer Darstellung") # Titel,
    strob.set_xlabel("x(t)")                                       # Labels
    strob.set_ylabel("p(t)")                                       # definieren
    # schwarze Konturlinien fuer die 8 Energiewerte einzeichnen:
    strob.contour(x2D, p2D, H, levels=energien, ls="", linewidths=1,
                  colors="black")
    # bei linkem Mausklick linksklick anwenden:
    klick_funktion = functools.partial(linksklick, A=A, B=B, omega=omega, N=N,
                                       perioden=perioden)
    plt.connect('button_press_event', klick_funktion)
    plt.show()


if __name__ == "__main__":
    main()

"""Diskussion:
a) B = 0 entspricht dem Fall ohne Antrieb, die Hamiltonfunktion ist also zeit-
unabhaengig. Damit ist die Energie des Teilchens konstant. Die Bewegung im Pha-
senraum erfolgt aequivalent zu den eingezeichneten Konturlinien. Im Ortsraum
entspricht das bei niedrigen Energien je nach Startpunkt einer periodischen Be-
wegung in einer der beiden Mulden. Ist die Energie gross genug, um das lokale
Maximum des Potentials bei ungefaehr x = 0,102 zu ueberwinden, werden beide
Mulden passiert, bis der Umkehrpunkt erreicht wird. Die Bewegung ist in jedem
Fall periodisch, da Energieerhaltung gilt.

b) Fuer B = 0.1 ist die Hamiltonfunktion zeitabhaengig, da ein schwacher perio-
discher aeusserer Antrieb erfolgt. Die Energie des Teilchens ist damit nicht
mehr zeitlich konstant. Die Bewegung erfolgt nun nicht mehr aequivalent zu den
eingezeichneten Konturlinien (da diese konstanten Energien entsprechen),sondern
springt mitunter zwischen konstanten Energiekonturlinien hin und her. Das
liegt daran, dass je nach Zeitpunkt der Sinus positiv oder negativ ist, somit
die Energie mal kleiner und mal groesser wird. Dies ist beispielsweise fuer
Startpunkte in der Naehe der "zwei Mitten" der eingezeichneten Konturlinien der
Fall.
Weiterhin gibt es Startpunkte, bei denen sich in der stroboskopischen Darstel-
lung Halbmonde ergeben.
Bei Startpunkten in der Naehe von x = 0,093 und p = -0,587 ergeben sich zwi-
schen der dritten und vierten eingezeichneten Konturlinie (von innen gezaehlt)
in der stroboskopischen Darstellung eine Insel beim Startpunkt. Hier wird der
Sprung zwischen den eingezeichneten Energieniveaukonturlinien im Phasenraum zu
einem schmalen Band im Phasenraum.
Es gibt auch Startpunkte, nach denen die Bewegung in stroboskopischer Betracht-
ung chaotisch verlaueft (zB. beim Startpunkt x = -0,818, p = -1,038). Im Pha-
senraum ist die Bewegung hier sehr zwischen den unterschiedichen Energieniveaus
ausgedehnt.
Zu noch groesseren Energien hin werden wieder Bewegungen beobachtet, die grob
nach den eingezeichneten Konturlinien verlaufen, aber zwischen ihnen aufgrund
des zeitabhaengigen Potentials hin und herspringen. Die Bewegungen wuerden also
aehnlich zu denen ohne aeusseren Antrieb erfolgen, allerdings wuerde das Teil-
chen aufgrund der geringen auesseren Energiezu- bzw. abfuhr zeitlich immer
zwischen konstanten Energie hin und her gezogen werden.

c)
x = -0,776
p = -0,095
Periode (= Anzahl der Punkte im stroboskopischen Plot -1) = 65*2*pi
Fuer den Startpunkt x = -0.776, p = -0.095 sind in der stroboskopischen Dar-
stellung 60 verschiedene Punkte sichtbar. Da aber im Hauptprogramm die Anzahl
der betrachteten Perioden zu 200 festgelegt ist, bedeutet das, dass ab dem 60.
Punkt der Ausgangspunkt wieder erreicht ist, also eine volle Periode vorliegt.
Der Zeitpunkt t = 0 wird mitgeplottet, daher ist die Periode (66-1)*2*pi.
"""
