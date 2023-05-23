"""Profiles of electron and electron/neutron inside the Sun. Differentiated quantities relevant for calculating
 adiabaticity parameter also included."""

import numpy as np
from snudd import config

N_A = 6.02214076e23  # Avogadro's constant
M2PERGEV = config.fm_conv * 1e15  # Conversion from m to /GeV
R_SUN = 696340e3  # Radius of Sun in m


# Constants used in electron number denisty
e1 = 2.36
e2 = 4.52
e3 = 0.33
e4 = 0.075
e5 = 1.1

# Constants used in neutron number density
n1 = 1.72
n2 = 4.80


def log10_electron_density(x):
    """Return log_10 of electron number density over Avogadro's number inside of the Sun as a function of distance
    fraction r / R_SUN. Expression taken from https://arxiv.org/pdf/astro-ph/0511337.pdf.
    """

    return e1 - e2 * x - e3 * np.exp(-(x / e4) ** e5)


def log10_neutron_electron_density(x):
    """Return log_10 of neutron number density over electron number density inside of the Sun as a function of distance
    fraction r / R_SUN. Expression derived from neutron density taken from https://arxiv.org/pdf/astro-ph/0511337.pdf.
    """

    return (n1 - e1) - (n2 - e2) * x + e3 * np.exp(-(x / e4) ** e5)


def electron_density(x):
    """Return electron number density in GeV^3."""

    return N_A * 10 ** (log10_electron_density(x)) / (M2PERGEV * 1e-2) ** 3  # 1e-2 m -> cm conv


def neutron_electron_fraction(x):
    """Return fraction of neutron and electron densities. This is our Y(x)."""

    return 10 ** log10_neutron_electron_density(x)


def electron_density_derivative_scaling(x):
    """Return scaling relationship for derivative of electron number density profile."""

    return np.log(10) * (-e2 + (e3 * e5 / e4) * (x / e4) ** (e5 - 1) * np.exp(-(x / e4) ** e5))


def neutron_electron_fraction_derivative_scaling(x):
    """Return scaling relationship for derivative of neutron/electron number density profile."""

    return np.log(10) * (- (n2 - e2) - e3 * e5 / e4 * (x/e4)**(e5 - 1) * np.exp(-(x/e4)**e5))


def electron_density_derivative(x):
    """Return the derivative of the electron number density profile."""

    return electron_density(x) * electron_density_derivative_scaling(x) / (R_SUN * M2PERGEV)


def neutron_electron_fraction_derivative(x):
    """Return the derivative of the electron number density profile."""

    return neutron_electron_fraction(x) * neutron_electron_fraction_derivative_scaling(x) / (R_SUN * M2PERGEV)
