from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
import matplotlib.cm as cm

'''Calculates neutrino normalisations based on total neutrino fluxes'''

data_8B = np.loadtxt("B8_neutrino_flux.txt")#8B neutrino data (Bahcall 1995)
data_7Be_3 = np.loadtxt("7Be_3843_neutrino.txt")
data_7Be_8 = np.loadtxt("7Be_8613_neutrino.txt")
data_15O = np.loadtxt("15O_neutrino.txt")
data_17F = np.loadtxt("17F_neutrino.txt")
data_hep = np.loadtxt("hep_neutrino.txt")
data_13N = np.loadtxt("N13_neutrino.txt")
data_pp = np.loadtxt("pp_neutrino.txt")
data_atmos_mu = np.loadtxt("Atmos_mu_neutrino_flux.txt")#Atmos v_mu flux at Super-K (Battistoni 2005)
data_atmos_e = np.loadtxt("Atmos_e_neutrino_flux.txt")#Atmos v_e flux at Super-K (Battistoni 2005)
data_pep = np.loadtxt("pep_neutrino.txt")

total_flux = {} # Total neutrino fluxes
integral = {}  # Integral of fluxes over energy range to be completed using trapezium rule
normalisation = {}  # Normalisations

E_v = {}  # Neutrino energy ranges in MeV
v_flux = {}  # Neutrino fluxes

E_v['8B'] = data_8B[:,0] #v energy range MeV
v_flux['8B'] = data_8B[:,1]
E_v['7Be_3'] = data_7Be_3[:,0]/1000+0.3843 #v energy range MeV  # Why the additional operations?
v_flux['7Be_3'] = data_7Be_3[:,1]
E_v['7Be_8'] = data_7Be_8[:,0]/1000+0.8613 #v energy range MeV
v_flux['7Be_8'] = data_7Be_8[:,1]
E_v['15O'] = data_15O[:,0] #v energy range MeV
v_flux['15O'] = data_15O[:,1]
E_v['17F'] = data_17F[:,0] #v energy range MeV
v_flux['17F'] = data_17F[:,1]
E_v['13N'] = data_13N[:,0] #v energy range MeV
v_flux['13N'] = data_13N[:,1]
E_v['hep'] = data_hep[:,0] #v energy range MeV
v_flux['hep'] = data_hep[:,1]
E_v['pp'] = data_pp[:,0] #v energy range MeV
v_flux['pp'] = data_pp[:,1]
E_v['atmos_mu'] = data_atmos_mu[:, 0]*1000 #v energy range MeV
E_v['atmos_mu_bar'] = data_atmos_mu[:, 0]*1000
E_v['atmos_e'] = data_atmos_mu[:, 0]*1000
E_v['atmos_e_bar'] = data_atmos_mu[:, 0]*1000
v_flux['atmos_mu'] = data_atmos_mu[:, 1]/(100**2*1000)#flux MeV
v_flux['atmos_mu_bar'] = data_atmos_mu[:, 3]/(100**2*1000)
v_flux['atmos_e'] = data_atmos_e[:, 1]/(100**2*1000)
v_flux['atmos_e_bar'] = data_atmos_e[:, 3]/(100**2*1000)
E_v['pep'] = data_pep[:,0] # np.array([1.438, 1.440, 1.442, 1.444, 1.446])
v_flux['pep'] = data_pep[:,1] # np.array([0.0, 0.5,1.0,0.5, 0.0])



#del E_v['7Be_3']
#del E_v['7Be_8']
#del v_flux['7Be_3']
#del v_flux['7Be_8']



total_flux['8B'] = 5.46e6
total_flux['pp'] = 5.98e10
total_flux['hep'] = 7.98e3
#total_flux['7Be'] = 4.93e9
total_flux['7Be_3'] = 4.93e9*0.1
total_flux['7Be_8'] = 4.93e9*0.9
total_flux['13N'] = 2.78e8
total_flux['15O'] = 2.05e8
total_flux['17F'] = 5.29e6
total_flux['pep'] = 1.448e8


for key in total_flux:
    total_flux[key] = total_flux[key]#*1692

for key in E_v:
    normalisation[key] = 1.0


#integral['7Be_3'] = integ.trapz(v_flux['7Be_3'], E_v['7Be_3']) + integ.trapz(v_flux['7Be_8'], E_v['7Be_8'])
#normalisation['7Be_3'] = normalisation['7Be_8'] = total_flux['7Be']/integral['7Be_3']
#print '7Be_3'
#print normalisation['7Be_3']

for key in total_flux:
    integral[key] = integ.trapz(v_flux[key], E_v[key])
    normalisation[key] = total_flux[key]/integral[key]  # So that integral over energies does indeed give the total flux!
    print(key)
    print(normalisation[key])

for key in v_flux:
    v_flux[key] = v_flux[key]*normalisation[key]  # This checks out

#Be = np.zeros(len(E_v['7Be']))
#for i in range(len(E_v['7Be'])):
#    Be[i] = integ.trapz(v_flux['7Be'][i:], E_v['7Be'][i:])

E_v['7Be'] = np.append(E_v['7Be_3'],E_v['7Be_8'], axis=0)
v_flux['7Be'] = np.append(v_flux['7Be_3'],v_flux['7Be_8'], axis=0)

del E_v['7Be_3']  # We are deleting these entries
del E_v['7Be_8']
del v_flux['7Be_3']
del v_flux['7Be_8']

#plt.figure()
#plt.plot(E_v['7Be'], v_flux['7Be'])
#plt.plot(E_v['7Be'], Be)

#plt.loglog()
#plt.show()
cmap = cm.gist_rainbow
nstep = 7.0
plt.figure()
i = 0.5
for key in E_v:
    print(normalisation[key])
    i+=0.5
    if int(i)-i == 0:
        plt.plot(E_v[key], v_flux[key], color = cmap(int(i)/nstep),label = key)
    else:
        plt.plot(E_v[key], v_flux[key], color = cmap(int(i)/nstep),linestyle = '--',label = key)
plt.legend(frameon = False, fontsize = 12)
plt.loglog()
plt.xlabel(r"$E_\nu [\mathrm{MeV}]$")
plt.ylabel(r"$d\phi /dE_\nu [\mathrm{cm^{-1} s^{-1} MeV^{-1}}]$")
plt.xlim(1e-2, 1e3)
plt.ylim(1e-5, 1e13)
plt.show()
