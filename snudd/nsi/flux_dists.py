"""Module to interpolate flux distributions taken from 1302.2791."""

import numpy as np
from scipy.interpolate import interp1d
from snudd import config


DATA_PATH = config.get_data('flux_distributions/')

def create_dist_keys():
    "Return neutrino source keys with Be sources replaced with a common Be source."

    nu_sources = []
    for nu in config.NU_SOURCE_KEYS:
        if nu == '7Be_3':
            nu = '7Be'
        elif nu == '7Be_8':
            continue
        nu_sources.append(nu)

    return nu_sources


def combine_beryllium(nu_source):
    """Return combined beryllium source if needed."""
    if nu_source == '7Be_3' or nu_source == '7Be_8':
        return '7Be'

    return nu_source


def read_dist_data(nu_source):
    """Return the x- and distribution-values from data for a source."""
    nu_source = combine_beryllium(nu_source)
    file_name = DATA_PATH + f'/{nu_source}_dist.txt'
    xs_data, fs_data = np.loadtxt(file_name, unpack=True)

    return xs_data, fs_data


def interpolate_dist(nu_source):
    """Return interpolated distribution function for neutrino source."""

    xs_data, fs_data = read_dist_data(nu_source)
    return interp1d(xs_data, fs_data, kind='linear', bounds_error=False, fill_value=0.)


def create_dist_dict() -> dict:
    """Return the interpolated distributions for each neutrino source."""

    keys = config.NU_SOURCE_KEYS
    return {key: interpolate_dist(key) for key in keys}


dist_dict = create_dist_dict()
