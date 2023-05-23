"""Contains cross sections to be used in targets for any model you like."""
from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import numpy as np

import nuddnsi.config as config


if typing.TYPE_CHECKING:
    from nudd.targets import Nucleus, Electron


class Model(ABC):
    """Provides model interface."""

    @abstractmethod
    def nucleus_cross_section_flavour(self, nucleus: Nucleus, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        pass

    @abstractmethod
    def electron_cross_section_flavour(self, electron: Electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        pass


class SM(Model):
    """The standard model neutrino scattering behaviour."""

    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour-breakdown cross section for nucleus. Energy in GeV. model_params = g_x, m_A"""
        Z, m_N, Q_nu_N = nucleus.Z, nucleus.mass, nucleus.Q_nu_N
        cs_SM = Q_nu_N ** 2 / 4

        cs_e = _nuclear_prefactor(nucleus, E_R, E_nu) * cs_SM
        cs_mu = cs_e
        cs_tau = cs_e

        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function

    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        y = E_R / E_nu

        cs_e = (2 * electron.mass * config.G_F ** 2 / np.pi *
                ((1 + config.g_L) ** 2 + config.g_R ** 2 * (1 - y) ** 2 - (1 + config.g_L) * config.g_R *
                 (config.m_e * y / E_nu)))
        cs_mu = 2 * electron.mass / np.pi * (config.G_F ** 2 * (
                config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R * config.m_e * y / E_nu))
        cs_tau = cs_mu

        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function


class LMuTau(Model):
    """Provides cross sections for the L_mu - L_tau model relevant for DD experiments."""

    def __init__(self, g_x, m_A):
        self.g_x = g_x
        self.m_A = m_A  # Mass in GeV

    @property
    def epsilon(self):
        """Return the g_x dependent kinetic mixing factor."""
        return self.g_x * self._epsilon_prime

    @property
    def _epsilon_prime(self):
        """Return the coupling-independent kinetic mixing factor."""
        return config.e / (6 * np.pi ** 2) * np.log(config.m_mu / config.m_tau)

    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour-breakdown cross section for nucleus. Energy in GeV. model_params = g_x, m_A"""
        Z, m_N, Q_nu_N = nucleus.Z, nucleus.mass, nucleus.Q_nu_N
        q = nucleus.momentum_transfer(E_R)

        cs_SM = Q_nu_N ** 2 / 4
        cs_int = self.epsilon * config.e * self.g_x * Q_nu_N * Z / (np.sqrt(2) * config.G_F * (q ** 2 + self.m_A ** 2))
        cs_sq = (self.epsilon * config.e * self.g_x * Z / (np.sqrt(2) * config.G_F * (q ** 2 + self.m_A ** 2))) ** 2

        cs_e = _nuclear_prefactor(nucleus, E_R, E_nu) * cs_SM
        cs_mu = _nuclear_prefactor(nucleus, E_R, E_nu) * (cs_SM + cs_int + cs_sq)
        cs_tau = _nuclear_prefactor(nucleus, E_R, E_nu) * (cs_SM - cs_int + cs_sq)

        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function

    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        q = electron.momentum_transfer(E_R)
        y = E_R / E_nu

        cs_e = (2 * config.m_e * config.G_F ** 2 / np.pi
                * ((1 + config.g_L) ** 2 + config.g_R ** 2 * (1 - y) ** 2 - (1 + config.g_L) * config.g_R *
                   (config.m_e * y / E_nu)))

        cs_mu = (2 * config.m_e / np.pi *
                 (config.G_F ** 2 * (config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R *
                                     config.m_e * y / E_nu)
                  + config.G_F / np.sqrt(2) * config.e * self.epsilon * self.g_x / (q ** 2 + self.m_A ** 2)
                  * ((config.g_L + config.g_R) * (1 - config.m_e * y / (2 * E_nu)) - config.g_R * y * (2 - y))
                  + 1 / 4 * (config.e * self.epsilon * self.g_x) ** 2 / (q ** 2 + self.m_A ** 2) ** 2
                  * (1 - y * (1 - (E_R - config.m_e) / (2 * E_nu)))))

        cs_tau = (2 * config.m_e / np.pi *
                  (config.G_F ** 2 * (config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R *
                                      config.m_e * y / E_nu)
                   - config.G_F / np.sqrt(2) * config.e * self.epsilon * self.g_x / (q ** 2 + self.m_A ** 2)
                   * ((config.g_L + config.g_R) * (1 - config.m_e * y / (2 * E_nu)) - config.g_R * y * (2 - y))
                   + 1 / 4 * (config.e * self.epsilon * self.g_x) ** 2 / (q ** 2 + self.m_A ** 2) ** 2
                   * (1 - y * (1 - (E_R - config.m_e) / (2 * E_nu)))))
        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function

    def nucleus_cross_section_reduced(self, nucleus, E_R, E_nu):
        """Return term-breakdown cross section for nucleus without g_x dependence. Energy in GeV."""
        Z, m_N, Q_nu_nucleus = nucleus.Z, nucleus.mass, nucleus.Q_nu_N
        q = nucleus.momentum_transfer(E_R)

        c_sm = _nuclear_prefactor(nucleus, E_R, E_nu) * Q_nu_nucleus ** 2 / 4
        c_int = _nuclear_prefactor(nucleus, E_R, E_nu) * config.eps_mu_tau_prime * config.e * Q_nu_nucleus * Z / (
                np.sqrt(2) * config.G_F * (q ** 2 + self.m_A ** 2))  # minus this for tau
        c_bsm = _nuclear_prefactor(nucleus, E_R, E_nu) * (config.eps_mu_tau_prime * config.e * Z / (
                np.sqrt(2) * config.G_F * (q ** 2 + self.m_A ** 2))) ** 2  # minus of mu due to L_mu_tau charge

        return np.array([c_sm, c_int, c_bsm])  # np array in order to work with vectorized function

    def electron_cross_section_reduced(self, electron, E_R, E_nu):
        """Return term-breakdown cross section for electron without g_x dependence. Energy in GeV."""
        q = electron.momentum_transfer(E_R)
        y = E_R / E_nu

        c_sm = 2 * config.m_e / np.pi * config.G_F ** 2 * (
                config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R * config.m_e * y / E_nu)

        c_int = (2 * config.m_e / np.pi * config.G_F / np.sqrt(2) * config.e * config.eps_mu_tau_prime
                 / (q ** 2 + self.m_A ** 2)
                 * ((config.g_L + config.g_R) * (1 - config.m_e * y / (2 * E_nu)) - config.g_R * y * (2 - y)))

        c_bsm = (2 * config.m_e / np.pi / 4 * (config.e * config.eps_mu_tau_prime / (q ** 2 + self.m_A ** 2)) ** 2
                 * (1 - y * (1 - (E_R - config.m_e) / (2 * E_nu))))

        return np.array([c_sm, c_int, c_bsm])  # np array in order to work with vectorized function


class LMu(Model):
    """Provides cross sections for the L_mu - L_tau model relevant for DD experiments."""

    def __init__(self, g_x, m_A, mass_scale):
        self.g_x = g_x
        self.m_A = m_A  # Mass in GeV
        self.mass_scale = mass_scale

    @property
    def epsilon(self):
        """Return the g_x dependent kinetic mixing factor."""
        return self.g_x * self._epsilon_prime

    @property
    def _epsilon_prime(self):
        """Return the coupling-independent kinetic mixing factor."""
        return config.e / (6 * np.pi ** 2) * np.log(config.m_mu / self.mass_scale)

    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour-breakdown cross section for nucleus. Energy in GeV. model_params = g_x, m_A"""
        Z, m_N, Q_nu_N = nucleus.Z, nucleus.mass, nucleus.Q_nu_N
        q = nucleus.momentum_transfer(E_R)

        cs_SM = Q_nu_N ** 2 / 4
        cs_int = self.epsilon * config.e * self.g_x * Q_nu_N * Z / (np.sqrt(2) * config.G_F * (q ** 2 + self.m_A ** 2))
        cs_sq = (self.epsilon * config.e * self.g_x * Z / (np.sqrt(2) * config.G_F * (q ** 2 + self.m_A ** 2))) ** 2

        cs_e = _nuclear_prefactor(nucleus, E_R, E_nu) * cs_SM
        cs_mu = _nuclear_prefactor(nucleus, E_R, E_nu) * (cs_SM + cs_int + cs_sq)
        cs_tau = cs_e

        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function

    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return cross section for target by flavour. Energies in GeV."""
        q = electron.momentum_transfer(E_R)
        y = E_R / E_nu

        cs_e = (2 * config.m_e * config.G_F ** 2 / np.pi
                * ((1 + config.g_L) ** 2 + config.g_R ** 2 * (1 - y) ** 2 - (1 + config.g_L) * config.g_R
                   * (config.m_e * y / E_nu)))

        cs_mu = (2 * config.m_e / np.pi
                 * (config.G_F ** 2 * (config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R
                                       * config.m_e * y / E_nu)
                    + config.G_F / np.sqrt(2) * config.e * self.epsilon * self.g_x / (q ** 2 + self.m_A ** 2)
                    * ((config.g_L + config.g_R) * (1 - config.m_e * y / (2 * E_nu)) - config.g_R * y * (2 - y))
                    + 1 / 4 * (config.e * self.epsilon * self.g_x) ** 2 / (q ** 2 + self.m_A ** 2) ** 2
                    * (1 - y * (1 - (E_R - config.m_e) / (2 * E_nu)))))

        cs_tau = 2 * config.m_e / np.pi * (config.G_F ** 2 * (config.g_L ** 2 + config.g_R ** 2 * (1 - y) ** 2 - config.g_L * config.g_R * config.m_e * y / E_nu))
        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function

class Scalar(Model):
    """A scalar particle."""

    def __init__(self, y, m_phi):
        self.y = y
        self.m_phi = m_phi


    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour-breakdown cross section for nucleus in the scalar case."""

        Z, m_N, Q_nu_N = nucleus.Z, nucleus.mass, nucleus.Q_nu_N
        cs_SM = Q_nu_N ** 2 / 4

        cs_e_sm = _nuclear_prefactor(nucleus, E_R, E_nu) * cs_SM
        cs_mu_sm = cs_e_sm
        cs_tau_sm = cs_e_sm


        q = nucleus.momentum_transfer(E_R)
        Q_prime = 13.8 * nucleus.A - 0.02 * nucleus.Z

        cs_scalar = nucleus.form_factor(E_R)**2 * self.y**4 * Q_prime**2 * nucleus.mass**2 * E_R / \
            (4 * np.pi * E_nu**2 * (q**2 + self.m_phi**2)**2)

        cs_e = cs_e_sm + cs_scalar
        cs_mu = cs_mu_sm + cs_scalar
        cs_tau = cs_tau_sm + cs_scalar


        return np.array([cs_e, cs_mu, cs_tau])  # np array in order to work with vectorized function



    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        pass


class GeneralNSI(Model):
    """A general NSI model, which takes a matrix of NSI couplings and angles."""

    def __init__(self, eps_matrix, eta, phi):
        self.eps_matrix = eps_matrix
        self.eta = eta
        self.phi = phi

    @property
    def xi_p(self):
        """Return proton-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) * np.cos(self.eta) * np.cos(self.phi)

    @property
    def xi_n(self):
        """Return the neutron-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) * np.sin(self.eta)

    @property
    def xi_u(self):
        """Return the up-quark-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) / 3 * (2 * np.cos(self.eta) * np.cos(self.phi) - np.sin(self.eta))

    @property
    def xi_d(self):
        """Return the down-quark-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) / 3 * (2 * np.sin(self.eta) - np.cos(self.eta) * np.cos(self.phi))

    @property
    def xi_e(self):
        """Return the electron-rotated charged part of the NSI factorisation."""
        return np.sqrt(5) * np.cos(self.eta) * np.sin(self.phi)

    def G_nucleus_coupling_matrix(self, nucleus):
        """Return the G coupling matrix."""

        return (self.xi_p * nucleus.Z + self.xi_n * nucleus.N) * self.eps_matrix

    def nucleus_cross_section_flavour(self, nucleus, E_R, E_nu):
        """Return flavour cross section matrix. Eneregy in GeV"""

        Z, N, m_N, Q_nu_N = nucleus.Z, nucleus.N, nucleus.mass, nucleus.Q_nu_N

        G_matrix = self.G_nucleus_coupling_matrix(nucleus)

        cs_sm = Q_nu_N ** 2 / 4 * np.diag((1,1,1))
        cs_int = - Q_nu_N * G_matrix.real
        cs_bsm = np.matmul(G_matrix, G_matrix.conjugate())

        return np.multiply.outer(_nuclear_prefactor(nucleus, E_R, E_nu),
                                 (cs_sm + cs_int + cs_bsm))


    def electron_cross_section_flavour(self, electron, E_R, E_nu):
        """Return flavour cross section matrix. Energies in GeV."""

        eps_matrix, eta, phi =  self.eps_matrix, self.eta, self.phi  # This is the change!

        g_L = config.g_L
        g_R = config.g_R
        xi_e = self.xi_e
        xi_p = self.xi_p
        xi_n = self.xi_n 

        GL_matrix = (np.array([[1,0,0],[0,0,0],[0,0,0]])   
                    + g_L * np.diag([1,1,1]) 
                    + 0.5*eps_matrix*xi_e)

        GR_matrix = (g_R*np.diag([1,1,1]) 
                     + 0.5 * eps_matrix*xi_e)

        prefactor  = 2 * config.G_F ** 2 * config.m_e / np.pi

        Lterm_shape_enhancement = (E_R / E_nu).shape  # To get Lterm to be correct shape for sum

        Lterm = np.multiply.outer(np.ones(Lterm_shape_enhancement), np.matmul(GL_matrix, GL_matrix.conjugate()))
        Rterm = np.multiply.outer((1 - E_R/E_nu)**2, np.matmul(GR_matrix, GR_matrix.conjugate()))
        LRterm = np.multiply.outer(((config.m_e * E_R)/(2 * E_nu**2)), (np.matmul(GL_matrix, GR_matrix.conjugate())
                                    + np.matmul(GR_matrix, GL_matrix.conjugate())))

        return prefactor * (Lterm + Rterm - LRterm)

        
def _nuclear_prefactor(nucleus, E_R, E_nu):
    """Return commonly used nuclear model prefactor."""
    F_helm = nucleus.form_factor(E_R)
    return config.G_F ** 2 / np.pi * nucleus.mass * (
            1 - nucleus.mass * E_R / (2 * E_nu ** 2)) * F_helm ** 2




