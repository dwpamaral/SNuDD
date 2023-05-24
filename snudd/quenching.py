"""
The quenching factor objects used in 'detector'.
For Xenon this is motivated by Sorensen and Dahl (1101.6080)
For Germanium this is motivated by Scholz et al (1608.03588)
For Argon this is motivated by Mei et al (0712.2470)
"""
from typing import Callable

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import snudd.config as config
#from snudd.targets import nucleus_ge, nucleus_xe
from snudd.targets import nucleus_xe


class Quenching:
    """Provides integrated functionality to quenching factor, including a derivative and a keV_nr to keV_ee converter.
    """

    def __init__(self, quenching: Callable):
        self.quenching = quenching

    @property
    def _differentiated_quenching(self):
        """Return the quenching derivative."""
        E_nrs = np.logspace(-2, 3, 10000) / 1e6  # NR energies to use with spline
        quenchings = self.quenching(E_nrs)
        interpolated_quenching = InterpolatedUnivariateSpline(E_nrs * 1e6, quenchings)  # Quenching/keV_nr, so 1e6
        return interpolated_quenching.derivative()

    def derivative(self, E_nr):
        """Call the derivative at NR energy in GeV"""
        return self._differentiated_quenching(E_nr * 1e6)  # 1e6 to get into keV from GeV

    def convert_nr2ee(self, E_nr) -> float:
        """Converts energy from NR-equivalent to electron-equivalent energy."""
        return self.quenching(E_nr) * E_nr

    def __call__(self, E_nr):
        return self.quenching(E_nr)


def quenching_func_xe(E_nr, k=0.1735):
    """Return quenching factor used for xenon with E_nr in GeV_nr"""
    return lindhard_factor(E_nr * 1e6, nucleus_xe, k)


def lindhard_k(nucleus):
    """Return classic Lindhard k"""
    return 0.133 * nucleus.Z ** (2 / 3) * nucleus.A ** (-1 / 2)


def lindhard_factor(E_nr, nucleus, k=None):
    """Return the classical Lindhard theory implementation of the quenching factor."""
    if k is None:
        k = lindhard_k(nucleus)  # Classic lindhard if k not supplied
    eps = 11.5 * nucleus.Z ** (-7 / 3) * E_nr
    g = 3 * eps ** 0.15 + 0.7 * eps ** 0.6 + eps
    L = k * g / (1 + k * g)
    return L


quenching_xe = Quenching(quenching_func_xe)
quenching_electron = Quenching(lambda E_R: 1. + 0 * E_R)  # Electron energies are ideally unquenched (0*E_R to get arr)
