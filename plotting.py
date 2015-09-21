import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import casadi as ca

__author__ = 'belousov'


def plot_arrows(name, ax, x, y, phi):
    ax.quiver(x, y, ca.cos(phi), ca.sin(phi),
              units='xy', angles='xy', scale=2, width=0.1,
              headwidth=4, headlength=3, headaxislength=3,
              color='r', lw='0.1')
    return [Patch(color='red', label=name)]


def plot_trajectory(ax, x, y, phi):
    ax.scatter(x, y, label='coordinate')
    plot_arrows('gaze', ax, x, y, phi)
    ax.set_title("Trajectory")
    ax.grid(True)


def plot_controls(ax, v, w, t):
    ax.step(t, ca.veccat([0, v]), label='v')
    ax.step(t, ca.veccat([0, w]), label='w')
    ax.set_title("Controls")
    ax.legend(loc=2)
    ax.grid(True)


def plot_policy(ax, x, y, phi, v, w, t):
    plot_trajectory(ax[0], x, y, phi)
    plot_controls(ax[1], v, w, t)
    plt.show()