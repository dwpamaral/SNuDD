"""Supplies an ad-hoc RRPA scaling ready for use in a bound electron."""
import numpy as np
from scipy import interpolate

from snudd import config

E_R_RRPA, SPEC_RRPA = np.loadtxt(config.get_data("exps/lz/rrpa_rate.txt"), unpack=True)
E_R_PPBE7, SPEC_PPBE7 = np.loadtxt(config.get_data("exps/lz/pp.txt"), unpack=True)

spec_step_interp = interpolate.interp1d(E_R_PPBE7, SPEC_PPBE7)
rrpa_step_ratio = SPEC_RRPA / spec_step_interp(E_R_RRPA)
rrpa_step_ratio_interp = interpolate.interp1d(E_R_RRPA / 1e6, rrpa_step_ratio)


def rrpa_scaling(E_R):
    """Scale factor to apply for RRPA approx in energy window 0.25 - 30 keV"""
    condlist = [(E_R >= E_R_RRPA.min() / 1e6) & (E_R <= E_R_RRPA.max() / 1e6), ]  # 1e6 for GeV
    funclist = [lambda E: rrpa_step_ratio_interp(E), 1.]
    return np.piecewise(E_R, condlist, funclist)
