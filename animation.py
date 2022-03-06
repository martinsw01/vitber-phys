'''Denne koden animerer bevegelsen til båtet, gitt at dere sender inn arrays som inneholder tidsverdier t,
skipets helningsvinkel theta, x- og y-koordinatet til skipets massesenter. Disse arraysene brukes i funksjonen
definert nederst "animate_deck_movement". Den grønne sirkelen viser posisjonen til skipets massesenter
Man kan sende inn optional argumenter, disse står beskrevet i funksjonen. For eksempel kan man sende inn et array
som inneholder lastens posisjon relativt metasenteret. Lasten vil da animeres som en rød sirkel.
'''
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from variables import R, H


def init_anim():
    """ Initialises the animation.
    """

    global ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory
    boat, = plt.plot([], [],
                     color="k", linewidth=1)
    deck, = plt.plot([], [], color="k", linewidth=1)
    sea_surface, = plt.plot([-R * 10, R * 10], [0, 0], color='blue', linewidth=2)  # The surface
    last, = plt.plot([], [], color="r", marker="o", markersize=10)
    CM, = plt.plot([], [], color="g", marker="o", markersize=10)
    venstre_gjerde, = plt.plot([], [], color="k", marker="|", markersize=25)
    høyre_gjerde, = plt.plot([], [], color="k", marker="|", markersize=25)
    ax.set_xlim([-R * 1.3, R * 1.3])
    ax.set_ylim([-R * 1.1, R * 1.1])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect("equal")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textbox_theory = ax.text(0.775, 0.95, '', transform=ax.transAxes, fontsize=12,
                             verticalalignment='top', bbox=props)

    return ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory


def animate(M, theta, t, x_C, y_C, s_L, gjerde=False):
    global ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory
    ax.set_xlim([-R * 1.1 + np.amin(x_C), R * 1.1 + np.amax(x_C)])
    ax.set_ylim([-R * 1.1, R * 1.1])
    angle_values = np.linspace(0, np.pi, 100)
    metasenter_x = x_C[M] - H * np.sin(theta[M])
    metasenter_y = y_C[M] + H * np.cos(theta[M])
    xs = R * np.cos(angle_values + np.pi + theta[M]) + metasenter_x
    ys = R * np.sin(angle_values + np.pi + theta[M]) + metasenter_y
    boat.set_data(xs, ys)
    deck.set_data([xs[0], xs[-1]], [ys[0], ys[-1]])
    if s_L[M] != -42:
        last.set_data(metasenter_x + s_L[M] * np.cos(theta[M]),
                      metasenter_y + s_L[M] * np.sin(theta[M]))
    CM.set_data(x_C[M], y_C[M])
    if gjerde:
        venstre_gjerde.set_data([metasenter_x - R * np.cos(theta[M])], [metasenter_y - R * np.sin(theta[M])])
        høyre_gjerde.set_data([metasenter_x + R * np.cos(theta[M])], [metasenter_y + R * np.sin(theta[M])])
    theta_string = r'$\theta = %.2f$' % (theta[M] * 180 / np.pi) + r"$\degree$"
    time_string = '$t =  %.2f$' % (t[M])
    textbox_theory.set_text(
        theta_string + '\n' + time_string)

    M += 1
    return ax, boat, deck, last, CM, venstre_gjerde, høyre_gjerde, textbox_theory


def animate_deck_movement(t, theta, x_C, y_C, s_L=None, gjerde=False, stepsize=0.01, vis_akse_verdier=False):
    """

    :param t: Array som inneholder tidsverdiene man har beregnet \vec{w} for systemet
    :param theta: Array som inneholder utslagsvinkelen til skipet
    :param x_C: Array som inneholder massesenterets x-koordinat
    :param y_C: Array som inneholder massesenterets y-koordinat
    :param s_L: Optional array som inneholder lastens posisjon relativt massesenteret
    :param gjerde: Optional Boolean som forteller om vi skal tegne inn gjerder på skipet
    :param stepsize: Hvor lang tid som skal gå mellom hver frame
    :param vis_akse_verdier: Hvis akse-verdier vises går animasjonen litt mer hakkete, men man kan se tallverdier
    :return: Animasjon som viser dynamikken til skipet
    """
    global fig, ax
    fig, ax = plt.subplots()
    dt = t[1] - t[0]
    skips = max(int(stepsize / dt), 1)
    theta_anim = theta[::skips]
    t_anim = t[::skips]
    x_C_anim = x_C[::skips]
    y_C_anim = y_C[::skips]
    if s_L is None:
        s_L_anim = - 42 * np.ones_like(theta_anim, dtype=int)
    else:
        s_L_anim = s_L[::skips]
    h_anim = animation.FuncAnimation(fig, animate, init_func=init_anim, frames=len(t_anim) - 1, interval=1,
                                     blit=not vis_akse_verdier,
                                     fargs=(theta_anim, t_anim, x_C_anim, y_C_anim, s_L_anim, gjerde))
    plt.show()
