"""
The quenching factor objects used in 'detector'.
For Xenon this is motivated by Sorensen and Dahl (1101.6080)
For Germanium this is motivated by Scholz et al (1608.03588)
For Argon this is motivated by Mei et al (0712.2470)
"""
from typing import Callable

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

import nuddnsi.config as config
from nuddnsi.targets import nucleus_ge, nucleus_xe

INTERP_DATA_AR = np.loadtxt(config.get_data("exps/darkside/quenching_argon_mei.txt"),
                            unpack=True)  # Quenching data for Argon

quenching_ar_interp = InterpolatedUnivariateSpline(*INTERP_DATA_AR, k=1)


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


def quenching_func_ge(E_nr, k=config.ge_quenching_consts["k"]):
    """Return quenching factor used for germanium with E_nr in GeV_nr. Implemented after 1608.03588."""
    return (lindhard_factor(E_nr * 1e6, nucleus_ge, k)
            * f_ac(E_nr * 1e6, config.ge_quenching_consts["xi"])
            * eta(config.ge_quenching_consts["x"], config.ge_quenching_consts["delta"],
                  config.ge_quenching_consts["tau"])
            * config.ge_quenching_consts["gamma"])


def quenching_func_ar(E_nr):
    """Return quenching factor used for argon with E_nr in GeV_nr. Simple interpolation after DarkSide 0712.2470"""
    return quenching_ar_interp(E_nr * 1e6)


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


def f_ac(E_nr, xi):
    """Return form factor correction."""
    return 1 - np.exp(-E_nr / xi)


def eta(x, delta, tau):
    """Return charge correction efficiency profile."""
    exp = np.exp((x - (delta + 0.5 * tau)) / (0.17 * tau))
    return 1 - 1 / (exp + 1)


quenching_xe = Quenching(quenching_func_xe)
quenching_ge = Quenching(quenching_func_ge)
quenching_ar = Quenching(quenching_func_ar)
quenching_electron = Quenching(lambda E_R: 1. + 0 * E_R)  # Electron energies are ideally unquenched (0*E_R to get arr)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from targets import Nucleus

    nucleus_ge = Nucleus(32, 73)
    nucleus_xe = Nucleus(54, 132)
    nucleus_si = Nucleus(14, 28)

    plt.figure()

    E_Rs_ar = np.linspace(1.5, 298, 1000)
    E_Rs_lindhard = np.linspace(0.2, 10, 1000)

    q_ar = quenching_ar(E_Rs_ar)
    q_ge_full = quenching_ge(E_Rs_lindhard)
    q_xe = quenching_xe(E_Rs_lindhard)

    plt.figure()
    plt.plot(E_Rs_ar, q_ar, label='Ar', c='y')
    plt.legend(frameon=False, loc=2)
    plt.xlabel(r'$E_{nr}\,\mathrm{(keV)}$')
    plt.ylabel(r'$Q\,(E_{nr})$')
    plt.xlim(E_Rs_ar[0], E_Rs_ar[-1])
    plt.show()

    plt.figure()
    plt.plot(E_Rs_lindhard, q_ge_full, label='Ge Full', c='g', ls='--')
    plt.semilogx(E_Rs_lindhard, q_xe, label='Xe', c='b')
    plt.legend(frameon=False)
    plt.xlabel(r'$E_{nr}\,\mathrm{(keV)}$')
    plt.ylabel(r'$Q\,(E_{nr})$')
    plt.xlim(E_Rs_lindhard[0], E_Rs_lindhard[-1])
    plt.show()
