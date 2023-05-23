# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# +
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from nudd import config
from nudd.models import NeutrinoDipole
from nudd.targets import nucleus_xe_dd
from nudd.coherent import config as coh_config
from nudd.coherent.dd_detector import lz
from scipy.special import factorial

# -

def Llik(n_obs, n_th):
    '''
    This function returns the logarithm of the likelihood
    assuming a poisson distribution for each bin
    
    Parameters
    ----------
    n_obs: Array (float). Observed number of events in each bin
    n_th: Array (float). Expected number of events in each bin
    
    Return
    ---------
    Llik: float. Logarithm of the likelihood
    '''
    return np.sum( -n_th - np.log(factorial(n_obs)) + n_obs * np.log(n_th) )


def Llik_ratio(n_obs, n_th):
    '''
    This function returns the logarithm of the likelihood
    assuming a poisson distribution for each bin
    
    Parameters
    ----------
    n_obs: Array (float). Observed number of events in each bin
    n_th: Array (float). Expected number of events in each bin
    
    Return
    ---------
    Llik: float. Logarithm of the likelihood
    '''
    return 2 * np.sum(n_th - n_obs + n_obs * np.log(n_obs / n_th) )


def chi2(n_obs, n_th):
    '''
    This function returns the chi2
    
    Parameters
    ----------
    n_obs: Array (float). Observed number of events in each bin
    n_th: Array (float). Expected number of events in each bin
    
    Return
    ---------
    Llik: float. Logarith of the likelihood
    '''
    return np.sum( (n_obs - n_th)**2 )


# # Full Spectra example

E_Rs = np.logspace(-2, 2, 1000) / 1e6
spectrum_sm = lz.spectrum(E_Rs, total=True)

# +
d = 2.1e-7 # * coh_config.BOHR_MAG
m4 = 2e-3
ndp = NeutrinoDipole(d, m4)

nucleus_xe_dd_ndp = deepcopy(nucleus_xe_dd)
nucleus_xe_dd_ndp.update_model(ndp)

lz_bsm = deepcopy(lz)
lz_bsm.nucleus = nucleus_xe_dd_ndp

spectrum_ndp = lz_bsm.spectrum(E_Rs, total=False)[1]
spectrum_bsm = spectrum_ndp + spectrum_sm

# +
plt.loglog(E_Rs * 1e6, spectrum_sm, c = 'black', label = 'SM')
plt.loglog(E_Rs * 1e6, spectrum_bsm, c = 'blue', label = 'BSM')
plt.loglog(E_Rs * 1e6, lz.eff_nr(E_Rs), linestyle=':',label = 'Eff')

plt.xlim(1e0, 1e1)
plt.ylim(1e-1, 1e3)
plt.legend(frameon=False)
plt.xlabel('$E_{R}$ [keV]')
plt.ylabel(r'$\frac{dE_{R}}{E_{R}}$')
# -

# ## Convolved Spectra

spectrum_sm = lz.convolved_spectrum(E_Rs, total=True)
spectrum_ndp = lz_bsm.convolved_spectrum(E_Rs, total=False, nu=1)
spectrum_bsm = spectrum_ndp + spectrum_sm

# +
plt.loglog(E_Rs * 1e6, spectrum_sm, c = 'black', label = 'SM')
plt.loglog(E_Rs * 1e6, spectrum_bsm, c = 'blue', label = 'BSM')
plt.loglog(E_Rs * 1e6, lz.eff_nr(E_Rs), linestyle=':',label = 'Eff')

plt.xlim(1e0, 1e1)
plt.ylim(1e-1, 1e3)
plt.legend(frameon=False)
plt.xlabel('$E_{R}$ [keV]')
plt.ylabel(r'$\frac{dE_{R}}{E_{R}}$')
# -

print(np.trapz(spectrum_sm, E_Rs * 1e6) * 15.34)
print(np.trapz(spectrum_bsm, E_Rs * 1e6) * 15.34)

# # Binned spectra example

BIN_WIDTH = 10 / 1e6 # Detector bin width in GeV
E_Rs = np.logspace(-2, 2, 1000) / 1e6 # Energies where to compute spectra in GeV

bin_edges, n_sm = lz.bin_spectrum(E_Rs, total=True, bin_width=BIN_WIDTH)

bin_edges

# +
ndp = NeutrinoDipole(d, m4)

nucleus_xe_dd_ndp = deepcopy(nucleus_xe_dd)
nucleus_xe_dd_ndp.update_model(ndp)

lz_bsm = deepcopy(lz)
lz_bsm.nucleus = nucleus_xe_dd_ndp

_, n_ndp = lz_bsm.bin_spectrum(E_Rs, total=False, bin_width=BIN_WIDTH, nu=1)
n_bsm = n_ndp + n_sm
# -


n_sm

n_bsm

# +
plt.bar(bin_edges[:-1:] * 1e6, 
        n_sm, 
        BIN_WIDTH * 1e6, 
        align='edge',
        fc='None',
        ec='k',
        label = 'SM',
        zorder=0.2)

plt.bar(bin_edges[:-1:] * 1e6, 
        n_bsm, 
        BIN_WIDTH * 1e6, 
        align='edge',
        fc='None',
        ec='b',
        label = 'BSM',
        zorder=0.1)
plt.legend(frameon=False)
plt.xlabel('$E_{R}$ [keV]')
plt.ylabel('#Events')


# -

# # Grid analysis

def grid_routine(detector_sm, d, m4, E_Rs, BIN_WIDTH,
                 dmin = 1e-7, dmax = 1e-6, logm4min = -3, logm4max = -1.5, 
                 grid_dim=20,stat_mth = Llik):
    """Return an array including the chi squared over a grid of
    masses and couplings, both of which are also returned. Grid dimension is
    given by grid_dim.
    """

    m4s = np.logspace(logm4min, logm4max, grid_dim)  # Mediator masses in GeV
    dip_couplings = np.geomspace(dmin, dmax, grid_dim)  # Couplings
    stats = np.zeros((grid_dim, grid_dim))  # Array to contain sigs
    
    # Let's compute the binned spectra for the SM
    bin_edges, n_sm = lz.bin_spectrum(E_Rs, total=True, bin_width=BIN_WIDTH)
    #-----------------------------------------------------------------------------------------------}
    
    # Let's compute the binned spectra for BP
    ndp = NeutrinoDipole(d, m4)

    nucleus_xe_dd_ndp = deepcopy(nucleus_xe_dd)
    nucleus_xe_dd_ndp.update_model(ndp)

    lz_bmp = deepcopy(lz)
    lz_bmp.nucleus = nucleus_xe_dd_ndp

    _, n_ndp = lz_bmp.bin_spectrum(E_Rs, total=False, bin_width=BIN_WIDTH, nu=1)
    n_bmp = n_ndp + n_sm
    #-----------------------------------------------------------------------------------------------}

    
    # This is the grid scan
    print("Beginning grid scan...")
    for im4, mass4 in tqdm(enumerate(m4s)):
        if np.remainder(im4+1,1)==0:
            print(f'At mass {im4 + 1}/{len(m4s)}')
        for idip, dip in enumerate(dip_couplings):
            ndp = NeutrinoDipole(dip, mass4)

            nucleus_xe_dd_ndp = deepcopy(nucleus_xe_dd)
            nucleus_xe_dd_ndp.update_model(ndp)

            lz_bsm = deepcopy(lz)
            lz_bsm.nucleus = nucleus_xe_dd_ndp

            _, n_ndp = lz_bsm.bin_spectrum(E_Rs, total=False, bin_width=BIN_WIDTH, nu=1)
            n_bsm = n_ndp + n_sm
    
            stats[idip, im4] = stat_mth(n_obs = n_bmp, n_th = n_bsm)
    
    print('Grid scan completed!')
    return stats, m4s, dip_couplings

stats, m4s, dip_couplings = grid_routine(lz, d, m4, E_Rs, BIN_WIDTH,
                                         grid_dim = 10,
                                         stat_mth = Llik)

# +
plt.pcolor(m4s, dip_couplings, stats)
plt.plot(m4, d, marker='*', color='r')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('$m_4$ $(GeV)$')
plt.ylabel('$d$ $(GeV^{-1})$')
plt.colorbar(label='$stats$')
# -


