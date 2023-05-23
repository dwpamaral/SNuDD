"""Provides the DD recoil differential rate spectrum."""
from __future__ import annotations
from tkinter import E

import typing

import numpy as np
from scipy.interpolate import interp1d
from nuddnsi import config
from nuddnsi.nsi.nsi_probabilities import ProbabilityCalculator, DensityMatrixCalculator, interp_probabilities_sm, interp_density_sm

if typing.TYPE_CHECKING:
    from nuddnsi.targets import Target


class Spectrum:
    """Target (nucleus or electron) spectrum."""

    def __init__(self, target):
        """Spectrum defined by target, model parameters, and probability type for neutrino probabilities."""
        self.target = target
        self.model = target.model
        self.osc_params = target.osc_params
        self.prob_calc = ProbabilityCalculator(self.model, self.osc_params, adiabatic_check=False)
        self.nu_probabilities = interp_probabilities_sm


    def nu_minimum_energy(self, E_R):
        """Return neutrino minimum energy given a recoil in GeV."""
        m = self.target.mass
        E_nu_min = 1. / 2. * (E_R + np.sqrt(E_R ** 2 + 2 * self.target.mass * E_R))

        return E_nu_min

    def _rate_nu(self, E_R, nu):
        """Return differential rate for a neutrino source. Overridden for each breakdown by subclasses."""

        nu_ind = config.NU_SOURCE_INDS[nu]  # TODO Monochromatic if clause. Check speed too...
        p_fn = self.nu_probabilities[nu]
        E_nu_min = self.nu_minimum_energy(E_R)  # Minimum neutrino energy
        # TODO THIS IS ALL VERY UGLY. REFACTOR. START WITH FUNCTION FOR MONO CASE
        if nu in config.NU_SOURCE_KEYS_MONO:
            E_nu_mono = config.E_nus[nu][0] / 1000
            E_nus_mins = (E_nu_min < E_nu_mono)
            ps = p_fn(E_nu_mono)
            dsigmas = self.target.cross_section_flavour(E_R, E_nu_mono)
            v_flux = np.array([[config.nu_flux[nu]]])
            integrated = (v_flux * ps).T * dsigmas

            return self.target.number_targets_mass(E_R) * integrated * config.rate_conv * E_nus_mins

        nu_flux_fn = config.nu_flux_interp[nu]
        np.putmask(E_nu_min, E_nu_min < nu_flux_fn.x.min() / 1000, nu_flux_fn.x.min() / 1000)
        np.putmask(E_nu_min, E_nu_min > nu_flux_fn.x.max() / 1000, (1 - 1e-6) * nu_flux_fn.x.max() / 1000)
        E_nus = np.geomspace(E_nu_min, nu_flux_fn.x.max() / 1000, 500)  # The relevant neutrino energies (in GeV)

        nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3  # Convert to per GeV
        probs = p_fn(E_nus)

        N_targets = self.target.number_targets_mass(E_R)

        E_R = np.array([E_R])

        integrands = self._integrand(E_R, E_nus, nu_fluxes, probs)
        rates = N_targets * np.trapz(integrands, E_nus.T) * config.rate_conv
        return np.where(rates < 0, 0, rates)

    def _integrand(self, E_R, E_nu, v_fluxes_need, probs_need):
        """Override _integrand method to give flavour specific integrands."""
        with np.errstate(all="ignore"):
            dsigma_ee, dsigma_em, dsigma_et = np.nan_to_num(
                self.target.cross_section_flavour(E_R, E_nu), posinf=0., neginf=0.)

        Ps_ee_need, Ps_em_need, Ps_et_need = probs_need

        integrand_ee = v_fluxes_need * Ps_ee_need.T * dsigma_ee.T
        integrand_em = v_fluxes_need * Ps_em_need.T * dsigma_em.T
        integrand_et = v_fluxes_need * Ps_et_need.T * dsigma_et.T

        return np.array([integrand_ee, integrand_em, integrand_et])

    def _spectrum_nu(self, E_Rs, nu):
        """Same as above but for specific source."""
        spectrum = self._rate_nu(E_Rs, nu)
        return np.squeeze(spectrum)

    def _total_spectrum(self, spectrum, total: bool):
        """Sum of zeroth axis of spectrum."""
        if total:
            return spectrum.sum(axis=0)
        return spectrum

    def spectrum(self, E_Rs, total: bool = True, nu: str = None):
        """
        Return neutrino spectrum for ER in GeV, coupling g_x, A mass m_A, and neutrino type nu (if string).
        If nu an integer, returns sum over all neutrino spectra. If g_x = 0, we retrieve the SM spectrum (tested).
        """
        if nu is not None:
            spectrum_nu = self._spectrum_nu(E_Rs, nu)
            return self._total_spectrum(spectrum_nu, total)
        else:
            spectrum = np.array([self._spectrum_nu(E_Rs, key) for key in config.NU_SOURCE_KEYS])
            source_summed_spectrum = spectrum.sum(axis=0)
            return self._total_spectrum(source_summed_spectrum, total)

    def prepare_probabilities(self):
        """Return dictionary of interpolated probabilities for all nu sources.
        Interpolation done between neutrinos energies of E_nu_min and
        E_nu_max (MeV)
        """

        interp_probabilities = self.prob_calc.interpolate_probabilities()
        self.nu_probabilities = interp_probabilities


class SpectrumTrace(Spectrum):
    """Target (nucleus or electron) spectrum."""

    def __init__(self, target):
        """Spectrum defined by target, model parameters, and probability type for neutrino probabilities."""
        super().__init__(target)
        self.density_calc = DensityMatrixCalculator(self.model, self.osc_params, adiabatic_check=False)
        self.nu_density_elements = interp_density_sm


    def _rate_nu(self, E_R, nu):
        """Return differential rate for a neutrino source. Overridden for each breakdown by subclasses."""

        nu_ind = config.NU_SOURCE_INDS[nu]  # TODO Monochromatic if clause. Check speed too...
        density_elements = self.nu_density_elements[nu]
        E_nu_min = self.nu_minimum_energy(E_R)  # Minimum neutrino energy
        # TODO THIS IS ALL VERY UGLY. REFACTOR. START WITH FUNCTION FOR MONO CASE
        if nu in config.NU_SOURCE_KEYS_MONO:
            E_nu_mono = config.E_nus[nu][0] / 1000
            E_nus_mins = (E_nu_min < E_nu_mono)
            density_mat = self.density_calc.matrix_from_elements(density_elements(E_nu_mono))
            dsigma_mat = self.target.cross_section_flavour(E_R, E_nu_mono)
            matrix_mult = np.matmul(density_mat, dsigma_mat)
            v_flux = np.array([[config.nu_flux[nu]]])
            integrated = v_flux * matrix_mult.trace(axis1=-2, axis2=-1)

            return self.target.number_targets_mass(E_R) * integrated * config.rate_conv * E_nus_mins

        nu_flux_fn = config.nu_flux_interp[nu]
        np.putmask(E_nu_min, E_nu_min < nu_flux_fn.x.min() / 1000, nu_flux_fn.x.min() / 1000)
        np.putmask(E_nu_min, E_nu_min > nu_flux_fn.x.max() / 1000, (1 - 1e-6) * nu_flux_fn.x.max() / 1000)
        E_nus = np.geomspace(E_nu_min, nu_flux_fn.x.max() / 1000, 500)  # The relevant neutrino energies (in GeV)

        nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3  # Convert to per GeV
        density_mat = self.density_calc.matrix_from_elements(density_elements(E_nus))

        N_targets = self.target.number_targets_mass(E_R)

        E_R = np.array([E_R])
        dsigma_mat = self.target.cross_section_flavour(E_R, E_nus)
        dsigma_mat = dsigma_mat.swapaxes(0,1)
        density_mat = np.rollaxis(density_mat, 3)
        matrix_mult = np.matmul(density_mat, dsigma_mat)
        matrix_mult = matrix_mult.swapaxes(0,1)

        integrands = nu_fluxes * matrix_mult.trace(axis1=-2, axis2=-1).T
        rates = N_targets * np.trapz(integrands, E_nus.T) * config.rate_conv

        return np.where(rates < 0, 0, rates)

    def prepare_density(self):
        """Return dictionary of interpolated probabilities for all nu sources.
        Interpolation done between neutrinos energies of E_nu_min and
        E_nu_max (MeV)
        """

        interp_density_elements = self.density_calc.interpolate_density_elements()
        self.nu_density_elements = interp_density_elements

    def spectrum(self, E_Rs, total=True, nu: str = None):
        """
        Return neutrino spectrum for ER in GeV, coupling g_x, A mass m_A, and neutrino type nu (if string).
        If nu an integer, returns sum over all neutrino spectra. If g_x = 0, we retrieve the SM spectrum (tested).
        """
        if nu is not None:
            spectrum_nu = self._spectrum_nu(E_Rs, nu)
            return spectrum_nu
        else:
            spectrum = np.array([self._spectrum_nu(E_Rs, key) for key in config.NU_SOURCE_KEYS])
            source_summed_spectrum = spectrum.sum(axis=0)
            return source_summed_spectrum

# class SpectrumIso(Spectrum):
#     """Target (nucleus or electron) spectrum."""

#     def __init__(self, nuclei, iso_fractions):
#         """Spectrum defined by target, model parameters, and probability type for neutrino probabilities."""
#         super().__init__(nuclei[0])
#         self.nuclei = nuclei
#         self.iso_fractions = iso_fractions

#     def sort_isotopes(self):
#         """Sort isotopes in ascending order."""
#         iso_masses = np.array([nucleus.mass for nucleus in self.nuclei])
#         sorted_indices = np.argsort(iso_masses)
#         self.nuclei = self.nuclei[sorted_indices]
#         self.iso_fractions = self.iso_fractions[sorted_indices]
#

#     def _rate_nu(self, E_R, nu):
#         """Return differential rate for a neutrino source. Overridden for each breakdown by subclasses."""
#         nu_ind = config.NU_SOURCE_INDS[nu]  # TODO Monochromatic if clause. Check speed too...

#         E_nu_min_0 = self.nu_minimum_energy(E_R)  # Minimum nu energy of lightest nucleus

#         for nucleus

#         cross_section_matrix = self.nu

#         p_fn = self.nu_probabilities[nu]
#         E_nu_min = self.nu_minimum_energy(E_R)  # Minimum neutrino energy

#         if nu in config.NU_SOURCE_KEYS_MONO:
#             E_nu_mono = config.E_nus[nu][0] / 1000
#             E_nus_mins = (E_nu_min < E_nu_mono)
#             ps = p_fn(E_nu_mono)
#             dsigmas = self.target.cross_section_flavour(E_R, E_nu_mono)
#             v_flux = np.array([[config.nu_flux[nu]]])
#             integrated = (v_flux * ps).T * dsigmas

#             return self.target.number_targets_mass(E_R) * integrated * config.rate_conv * E_nus_mins

#         nu_flux_fn = config.nu_flux_interp[nu]
#         np.putmask(E_nu_min, E_nu_min < nu_flux_fn.x.min() / 1000, nu_flux_fn.x.min() / 1000)
#         np.putmask(E_nu_min, E_nu_min > nu_flux_fn.x.max() / 1000, (1 - 1e-6) * nu_flux_fn.x.max() / 1000)
#         E_nus = np.geomspace(E_nu_min, nu_flux_fn.x.max() / 1000, 500)  # The relevant neutrino energies (in GeV)

#         nu_fluxes = nu_flux_fn(E_nus * 1000).T * 1e3  # Convert to per GeV
#         probs = p_fn(E_nus)

#         N_targets = self.target.number_targets_mass(E_R)

#         E_R = np.array([E_R])

#         integrands = self._integrand(E_R, E_nus, nu_fluxes, probs)
#         rates = N_targets * np.trapz(integrands, E_nus.T) * config.rate_conv
#         return np.where(rates < 0, 0, rates)

#     def _integrand(self, E_R, E_nu, v_fluxes_need, probs_need):
#         """Override _integrand method to give flavour specific integrands."""
#         with np.errstate(all="ignore"):
#             dsigma_ee, dsigma_em, dsigma_et = np.nan_to_num(
#                 self.target.cross_section_flavour(E_R, E_nu), posinf=0., neginf=0.)

#         Ps_ee_need, Ps_em_need, Ps_et_need = probs_need

#         integrand_ee = v_fluxes_need * Ps_ee_need.T * dsigma_ee.T
#         integrand_em = v_fluxes_need * Ps_em_need.T * dsigma_em.T
#         integrand_et = v_fluxes_need * Ps_et_need.T * dsigma_et.T

#         return np.array([integrand_ee, integrand_em, integrand_et])

#     def _spectrum_nu(self, E_Rs, nu):
#         """Same as above but for specific source."""
#         spectrum = self._rate_nu(E_Rs, nu)
#         return np.squeeze(spectrum)

#     def _total_spectrum(self, spectrum, total: bool):
#         """Sum of zeroth axis of spectrum."""
#         if total:
#             return spectrum.sum(axis=0)
#         return spectrum

#     def spectrum(self, E_Rs, total: bool = True, nu: str = None):
#         """
#         Return neutrino spectrum for ER in GeV, coupling g_x, A mass m_A, and neutrino type nu (if string).
#         If nu an integer, returns sum over all neutrino spectra. If g_x = 0, we retrieve the SM spectrum (tested).
#         """
#         if nu is not None:
#             spectrum_nu = self._spectrum_nu(E_Rs, nu)
#             return self._total_spectrum(spectrum_nu, total)
#         else:
#             spectrum = np.array([self._spectrum_nu(E_Rs, key) for key in config.NU_SOURCE_KEYS])
#             source_summed_spectrum = spectrum.sum(axis=0)
#             return self._total_spectrum(source_summed_spectrum, total)
