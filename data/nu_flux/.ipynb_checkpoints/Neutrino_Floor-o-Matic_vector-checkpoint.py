from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as integ
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
from math import erf

data_8B = np.loadtxt("B8_neutrino.txt")#8B neutrino data (Bahcall 1995)
data_7Be_3 = np.loadtxt("7Be_3843_neutrino.txt")
data_7Be_8 = np.loadtxt("7Be_8613_neutrino.txt")
data_15O = np.loadtxt("15O_neutrino.txt")
data_17F = np.loadtxt("17F_neutrino.txt")
data_hep = np.loadtxt("hep_neutrino.txt")
data_13N = np.loadtxt("N13_neutrino.txt")
data_pp = np.loadtxt("pp_neutrino.txt")
data_atmos_mu = np.loadtxt("Atmos_mu_neutrino_flux.txt")#Atmos v_mu flux at Super-K (Battistoni 2005)
data_atmos_e = np.loadtxt("Atmos_e_neutrino_flux.txt")#Atmos v_e flux at Super-K (Battistoni 2005)

normalisation = {'pep': 36200000000.0,'pp':59800327526.4, 'hep': 7980.00904375, '8B': 5460131.86218, '17F': 5290001.1186, '13N': 278001239.903, '7Be_3': 493056454477.0, '7Be_8': 4.43751697596e+12, '15O': 204999990.076}
suppression = 1#e16 #In some versions we only want to consider certain flux sources
E_v = {}
v_flux = {}
E_v['8B'] = data_8B[:,0] #v energy range MeV
v_flux['8B'] = data_8B[:,1]
E_v['7Be_3'] = data_7Be_3[:,0]/1000+0.3843 #v energy range MeV
v_flux['7Be_3'] = data_7Be_3[:,1]/suppression
E_v['7Be_8'] = data_7Be_8[:,0]/1000+0.8613 #v energy range MeV
v_flux['7Be_8'] = data_7Be_8[:,1]/suppression
E_v['15O'] = data_15O[:,0] #v energy range MeV
v_flux['15O'] = data_15O[:,1]
E_v['17F'] = data_17F[:,0] #v energy range MeV
v_flux['17F'] = data_17F[:,1]
E_v['13N'] = data_13N[:,0] #v energy range MeV
v_flux['13N'] = data_13N[:,1]
E_v['hep'] = data_hep[:,0] #v energy range MeV
v_flux['hep'] = data_hep[:,1]/1e15
E_v['pp'] = data_pp[:,0] #v energy range MeV
v_flux['pp'] = data_pp[:,1]
E_v['atmos_mu'] = data_atmos_mu[:, 0]*1000 #v energy range MeV
E_v['atmos_mu_bar'] = data_atmos_mu[:, 0]*1000
E_v['atmos_e'] = data_atmos_mu[:, 0]*1000
E_v['atmos_e_bar'] = data_atmos_mu[:, 0]*1000
v_flux['atmos_mu'] = data_atmos_mu[:, 1]/(100**2*1000)#/suppression#flux MeV
v_flux['atmos_mu_bar'] = data_atmos_mu[:, 3]/(100**2*1000)#/suppression
v_flux['atmos_e'] = data_atmos_e[:, 1]/(100**2*1000)#/suppression
v_flux['atmos_e_bar'] = data_atmos_e[:, 3]/(100**2*1000)#/suppression
E_v['pep'] = np.array([1.438, 1.440, 1.442, 1.444, 1.446])
v_flux['pep'] = np.array([0.0, 0.5,1.0,0.5, 0.0])/suppression

for key in normalisation:
    v_flux[key] = v_flux[key]*normalisation[key] #Apply correct normalisations to spectra

for key in E_v:
    E_v[key] = E_v[key]/1000#Convert to GeV
    v_flux[key] = v_flux[key]*2.563016e-52*1000#Convert to GeV-1


log_E_r = np.linspace(-4.2, 3, 2500)*2.303#RECOIL ENERGIES
E_r = np.exp(log_E_r)#Energy range KeV
E_R = E_r/1000000

c = 299800.0
v_vals = 270.0/c #DM avg velocity
rho = 0.4*(0.197*10**(-13))**3 #Local DM density GeV/cm**2 -> GeV**4
v_circ = 220.0/c #156.0*2.0**0.5/c #km/s
v_lag = 230.0/c #km/s
v_esc = 544.0/c

t = 365.0*24.0*3600.0/(6.58*10**(-25)) #365.0*24.0*3600.0/(6.58*10**(-25)) #s
M_T = 1000.0/(1.8*10**(-27)) #GeV

epsilon = M_T*t
efficiency_Xe = np.ones(len(E_r))
#for i in range(int(len(E_r)*(2.5+.7)/4.27)):
#    efficiency_Xe[i] = 0.5+0.5*E_r[i]/5

efficiency = np.ones(len(E_r))
#for i in range(int(len(E_r)*(2.5)/4.27)):
#    efficiency[i] = 0.5+0.5*E_r[i]


m_p = 0.938 #proton mass GeV
m_e = 0.511e-3
m_W = 80.385#W mass GeV
G_f = 1.166e-5#Fermi Coupling Constant/GeV-2
S_w2 = 0.238 #sin^2(theta_W) weak mixing

g_v = 2.0*S_w2-0.5#muon/tau neutrinos
g_a = -0.5
g_ve = 2.0*S_w2+0.5#electron neutrinos
g_ae = 0.5

def F_SI_integrand(r, Q, R_a, a):
    rho_A = 1.0/(1.0+np.exp((r-R_a)/a))
    return 4.0*np.pi*r*np.sin(Q*r)*rho_A/Q

def E_min_n_calc(Z,N):

    A = Z+N
    m_N = A*m_p
    R_a = (1.23*A**(1/3)-0.6)*10**(-15)/(1.97*10**(-16))#fm->GeV
    #R_a_SD = (0.92*A**(1/3)+2.68-0.78*((A**(1/3)-3.8)**2+0.2)**0.5)*10**(-15)/(1.97*10**(-16))#fm->GeV
    a = 0.5*10.0**(-15)/(1.97*10**(-16)) #fm->GeV
    
    
    q = (2.0*m_N*E_R)**0.5

    F_SI = np.zeros(len(q))
    for i in range(len(q)):
        F_SI[i], F_err = integ.quad(F_SI_integrand, 0, 100, args=(q[i], R_a, a))
        #if F_SI[i]<0:
        #    F_SI[i] = 0
    F_SI = F_SI/F_SI[0]

    E_min_n = (m_N*E_R/2)**0.5

#    print 'Establishing E_min_arg_n'
    E_min_arg_n = {}
    for key in E_v:
        E_min_arg_n[key] =  np.zeros(len(E_r))
        for i in range(len(E_r)):
            E_min_arg_n[key][i] = (np.abs(E_v[key]-E_min_n[i])).argmin()
            if E_v[key][int(E_min_arg_n[key][i])]-E_min_n[i]<=0:
                E_min_arg_n[key][i]+=1
    return E_min_n, E_min_arg_n, F_SI

def E_min_e_calc(E_r, Z,N):
    E_R = E_r/1000000
    A = Z+N
    m_N = A*m_p
    
    E_min = 0.5*(E_R+(E_R*(E_R+2*m_e))**0.5)

##    print 'Establishing E_min_arg_e'
    E_min_arg = {}
    for key in E_v:
        E_min_arg[key] =  np.zeros(len(E_r))
        for i in range(len(E_r)):
            E_min_arg[key][i] = (np.abs(E_v[key]-E_min[i])).argmin()
            if E_v[key][E_min_arg[key][i]]-E_min[i]<=0:
                if E_min_arg[key][i] != len(E_v[key]-1):
                    E_min_arg[key][i]+=1
    return E_min, E_min_arg


def v_calc_scalar(E_r, Y, m_Z_prime, Z, N, E_min_n, E_min_arg_n, F_SI):
    '''Calculate count rate for scalar-mediated neutrino-nucleus coherent scattering'''
    E_R = E_r/1000000
    Q_w = N-(1-4*S_w2)*Z
    #g_prime = g_BL
    #m_Z_prime = 2.5e-2
    A = Z+N
    m_N = A*m_p
    Q_prime = (14*A+1.1*Z)*Y**2
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min_n[i]<=E_v[key].max():
                
                d_sigma_SM = G_f**2/(4*np.pi)*Q_w**2*m_N*(1.0-m_N*E_R[i]/(2*E_v[key]**2))*F_SI[i]**2 #DIFFERENTIAL CROSS SECTION
                d_sigma_BL = m_N**2*Q_prime**2*E_R[i]/(4*np.pi*(2*m_N*E_R[i]+m_Z_prime**2)**2*E_v[key]**2)*F_SI[i]**2
                d_sigma = d_sigma_SM+d_sigma_BL
                
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg_n[key][i]):], E_v[key][int(E_min_arg_n[key][i]):])
        
    dNv_dEr_dict = {}
    #Nv_dict = {}
    for key in E_v:
        if Z == 54.0 or Z == 18.0:
            dNv_dEr_dict[key] = epsilon*efficiency_Xe*integral[key]/(m_N)
        else:
            dNv_dEr_dict[key] = epsilon*efficiency*integral[key]/(m_N)
    
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N


def v_calc_vector_BL(E_r, m_Z_prime, g_BL, Z, N, E_min_n, E_min_arg_n, F_SI):
    '''Calculate count rate for vector-mediated neutrino-nucleus coherent scattering'''
    E_R = E_r/1000000
    Q_w = N-(1-4*S_w2)*Z
    A = Z+N
    m_N = A*m_p
    #Q_prime = A*g_BL**2
    Q_prime = -A*g_BL**2
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min_n[i]<=E_v[key].max():
                
                d_sigma_SM = G_f**2/(4*np.pi)*Q_w**2*m_N*(1.0-m_N*E_R[i]/(2*E_v[key]**2))*F_SI[i]**2 #DIFFERENTIAL CROSS SECTION
                d_sigma_BL = F_SI[i]**2*(-G_f*m_N*Q_w*Q_prime*(2*E_v[key]**2-m_N*E_R[i])/(2**1.5*np.pi*E_v[key]**2*(2*E_R[i]*m_N+m_Z_prime**2))+Q_prime**2*m_N*(2*E_v[key]**2-E_R[i]*m_N)/(4*np.pi*E_v[key]**2*(2*E_R[i]*m_N+m_Z_prime**2)**2))
                #d_sigma = F_SI[i]**2*m_N/(4*np.pi)*(1-E_R[i]/E_v[key]-m_N*E_R[i]/(2*E_v[key]**2))*(G_f**2*Q_w**2+2*g_BL**4*A**2/(2*m_N*E_R[i]+m_Z_prime**2)**2
                #            +4*g_BL**2*A*G_f*Q_w/(2**1.5*m_N*E_R[i]+m_Z_prime**2))
                d_sigma = d_sigma_SM+d_sigma_BL
                
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg_n[key][i]):], E_v[key][int(E_min_arg_n[key][i]):])
        
    dNv_dEr_dict = {}
    #Nv_dict = {}
    for key in E_v:
        if Z == 54.0 or Z == 18.0:
            dNv_dEr_dict[key] = epsilon*efficiency_Xe*integral[key]/(m_N)
        else:
            dNv_dEr_dict[key] = epsilon*efficiency*integral[key]/(m_N)
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N

def v_calc_vector_BLT(E_r, g_BL, s_BL, m_Z_prime, Z, N, E_min_n, E_min_arg_n, F_SI):
    '''Calculate count rate for vector-mediated neutrino-nucleus coherent scattering'''
    E_R = E_r/1000000
    Q_w = N-(1-4*S_w2)*Z
    A = Z+N
    m_N = A*m_p
    #Q_prime = A*g_BL**2
    Q_prime = -A*g_BL*s_BL*(8*0.5**0.5*m_W**2*G_f*(1-S_w2))**0.5/2 #BL(3) Model
    #Q_prime = -3*A*g_BL*s_BL #Miniboone model
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min_n[i]<=E_v[key].max():
                
                d_sigma_SM = G_f**2/(4*np.pi)*Q_w**2*m_N*(1.0-m_N*E_R[i]/(2*E_v[key]**2))*F_SI[i]**2 #DIFFERENTIAL CROSS SECTION
                d_sigma_BL = F_SI[i]**2*(-G_f*m_N*Q_w*Q_prime*(2*E_v[key]**2-m_N*E_R[i])/(2**1.5*np.pi*E_v[key]**2*(2*E_R[i]*m_N+m_Z_prime**2))+Q_prime**2*m_N*(2*E_v[key]**2-E_R[i]*m_N)/(4*np.pi*E_v[key]**2*(2*E_R[i]*m_N+m_Z_prime**2)**2))
                #d_sigma = F_SI[i]**2*m_N/(4*np.pi)*(1-E_R[i]/E_v[key]-m_N*E_R[i]/(2*E_v[key]**2))*(G_f**2*Q_w**2+2*g_BL**4*A**2/(2*m_N*E_R[i]+m_Z_prime**2)**2
                #            +4*g_BL**2*A*G_f*Q_w/(2**1.5*m_N*E_R[i]+m_Z_prime**2))
                d_sigma = d_sigma_SM+d_sigma_BL
                if key == "atmos_mu" or key == "atmos_mu_bar":
                	d_sigma = d_sigma_SM+d_sigma_BL*0.3#Assume 40% v_mu survival rate
                elif key == "atmos_e" or key == "atmos_e":
                	d_sigma = d_sigma_SM+d_sigma_BL*0.3# 40% v_e survival rate
                else:
                	d_sigma = d_sigma_SM+d_sigma_BL*0.225#55% survival rate
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg_n[key][i]):], E_v[key][int(E_min_arg_n[key][i]):])
        
    dNv_dEr_dict = {}
    #Nv_dict = {}
    for key in E_v:
        if Z == 54.0 or Z == 18.0:
            dNv_dEr_dict[key] = epsilon*efficiency_Xe*integral[key]/(m_N)
        else:
            dNv_dEr_dict[key] = epsilon*efficiency*integral[key]/(m_N)
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N

def v_calc_vector_seq(E_r, m_Z_prime, s_x, Z, N, E_min_n, E_min_arg_n, F_SI):
    '''Calculate count rate for vector-mediated neutrino-nucleus coherent scattering'''
    E_R = E_r/1000000
    Q_w = N-(1-4*S_w2)*Z
    A = Z+N
    m_N = A*m_p
    g_prime = s_x*(8*0.5**0.5*m_W**2*G_f*(1-S_w2))**0.5
    Qv_prime = -0.5
    #Q_prime = -3*A*g_BL**2
    aZ_prime = (N-Z*(1-4*S_w2))/4
    #aZ_prime = 1.5*(3*Z+N)
    
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min_n[i]<=E_v[key].max():
                d_sigma_SM = G_f**2/(4*np.pi)*Q_w**2*m_N*(1.0-m_N*E_R[i]/(2*E_v[key]**2))*F_SI[i]**2 #DIFFERENTIAL CROSS SECTION
                d_sigma_BL = m_N/(4*np.pi)*(1-E_R[i]/E_v[key]-m_N*E_R[i]/(2*E_v[key]**2))*(2*g_prime**4*Qv_prime**2*aZ_prime**2*F_SI[i]**2/(2*m_N*E_R[i]+m_Z_prime**2)**2
                                -4*g_prime**2*Qv_prime*aZ_prime*G_f*Q_w*F_SI[i]**2/(2**1.5*m_N*E_R[i]+m_Z_prime**2))
                #d_sigma_BL = m_N/(4*np.pi)*(1-E_R[i]/E_v[key]-m_N*E_R[i]/(2*E_v[key]**2))*(2*g_prime**4*Qv_prime**2*aZ_prime**2*F_SI[i]**2/(2*m_N*E_R[i]+m_Z_prime**2)**2
                #                -4*g_prime**2*Qv_prime*aZ_prime*G_f*Q_w*F_SI[i]**2/(2**1.5*m_N*E_R[i]+m_Z_prime**2))
                d_sigma = d_sigma_SM+d_sigma_BL
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg_n[key][i]):], E_v[key][int(E_min_arg_n[key][i]):])
        
    dNv_dEr_dict = {}
    #Nv_dict = {}
    for key in E_v:
        if Z == 54.0 or Z == 18.0:
            dNv_dEr_dict[key] = epsilon*efficiency_Xe*integral[key]/(m_N)
        else:
            dNv_dEr_dict[key] = epsilon*efficiency*integral[key]/(m_N)
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N


def v_calc_vector_SO(E_r, g_BL, m_Z_prime, Z, N, E_min_n, E_min_arg_n, F_SI):
    '''Calculate count rate for vector-mediated neutrino-nucleus coherent scattering'''
    E_R = E_r/1000000
    Q_w = N-(1-4*S_w2)*Z
    A = Z+N
    m_N = A*m_p
    g_prime = g_BL
    Qv_prime = -3
    #Q_prime = -3*A*g_BL**2
    aZ_prime = 4*N+2*Z
    
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min_n[i]<=E_v[key].max():
                d_sigma_SM = G_f**2/(4*np.pi)*Q_w**2*m_N*(1.0-m_N*E_R[i]/(2*E_v[key]**2))*F_SI[i]**2 #DIFFERENTIAL CROSS SECTION
                d_sigma_BL = m_N/(4*np.pi)*(1-E_R[i]/E_v[key]-m_N*E_R[i]/(2*E_v[key]**2))*(2*g_prime**4*Qv_prime**2*aZ_prime**2*F_SI[i]**2/(2*m_N*E_R[i]+m_Z_prime**2)**2
                                -4*g_prime**2*Qv_prime*aZ_prime*G_f*Q_w*F_SI[i]**2/(2**1.5*m_N*E_R[i]+m_Z_prime**2))
                #d_sigma = F_SI[i]**2*m_N/(4*np.pi)*(1-E_R[i]/E_v[key]-m_N*E_R[i]/(2*E_v[key]**2))*(G_f**2*Q_w**2+2*g_BL**4*A**2/(2*m_N*E_R[i]+m_Z_prime**2)**2
                #            +4*g_BL**2*A*G_f*Q_w/(2**1.5*m_N*E_R[i]+m_Z_prime**2))
                d_sigma = d_sigma_SM+d_sigma_BL
                
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg_n[key][i]):], E_v[key][int(E_min_arg_n[key][i]):])
        
    dNv_dEr_dict = {}
    #Nv_dict = {}
    for key in E_v:
        if Z == 54.0 or Z == 18.0:
            dNv_dEr_dict[key] = epsilon*efficiency_Xe*integral[key]/(m_N)
        else:
            dNv_dEr_dict[key] = epsilon*efficiency*integral[key]/(m_N)
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N

def v_calc_e_scalar(E_r, g_BL, m_Z_prime, Z, N, E_min, E_min_arg):
    '''Calculate count rate for vector-mediated neutrino-electron coherent scattering'''
    E_R = E_r/1000000
    A = Z+N
    m_N = A*m_p
    g_prime = g_BL
    
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min[i]<=E_v[key].max():
                d_sigma_SM_e = (G_f**2*m_e/(2.0*np.pi))*((g_ve+g_ae)**2+(g_ve-g_ae)**2*(1.0-E_R[i]/E_v[key])**2+(g_ae**2-g_ve**2)*m_e*E_R[i]/E_v[key]**2) #DIFFERENTIAL CROSS SECTION: electron neutrinos
                d_sigma_SM_mt = (G_f**2*m_e/(2.0*np.pi))*((g_v+g_a)**2+(g_v-g_a)**2*(1.0-E_R[i]/E_v[key])**2+(g_a**2-g_v**2)*m_e*E_R[i]/E_v[key]**2)#muon/tau neutrinos
   
                d_sigma_BL_e = g_prime**4*E_R[i]*m_e**2/(4*np.pi*E_v[key]**2*(2*E_R[i]*m_e+m_Z_prime**2)**2)
                d_sigma_BL_mt = d_sigma_BL_e#g_prime**4*E_R[i]*m_e**2/(4*np.pi*E_v[key]**2*(2*E_R[i]*m_e+m_Z_prime**2)**2)
                #print d_sigma_BL_e
                #print d_sigma_BL_mt
                if key == "atmos_mu" or key == "atmos_mu_bar":
                	d_sigma = d_sigma_BL_mt+d_sigma_SM_mt
                elif key == "atmos_e" or key == "atmos_e":
                	d_sigma = d_sigma_BL_e+d_sigma_SM_e
                else:
                	d_sigma = (d_sigma_BL_e+d_sigma_SM_e)*0.55+(d_sigma_BL_mt+d_sigma_SM_mt)*0.45#Account for solar neutrino oscillations
                
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg[key][i]):], E_v[key][int(E_min_arg[key][i]):])
    
    dNv_dEr_dict = {}
    
    for key in E_v:
        dNv_dEr_dict[key] = epsilon*efficiency*integral[key]*Z/(m_N)    
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N



def v_calc_e_vector(E_r, g_BL, m_Z_prime, Z, N, E_min, E_min_arg):
    '''Calculate count rate for vector-mediated neutrino-electron coherent scattering'''
    E_R = E_r/1000000
    A = Z+N
    m_N = A*m_p
    g_prime = g_BL
    
    integral = {}
    for key in E_v:
        integral[key] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            if E_min[i]<=E_v[key].max():
                d_sigma_SM_e = (G_f**2*m_e/(2.0*np.pi))*((g_ve+g_ae)**2+(g_ve-g_ae)**2*(1.0-E_R[i]/E_v[key])**2+(g_ae**2-g_ve**2)*m_e*E_R[i]/E_v[key]**2) #DIFFERENTIAL CROSS SECTION: electron neutrinos
                d_sigma_SM_mt = (G_f**2*m_e/(2.0*np.pi))*((g_v+g_a)**2+(g_v-g_a)**2*(1.0-E_R[i]/E_v[key])**2+(g_a**2-g_v**2)*m_e*E_R[i]/E_v[key]**2)#muon/tau neutrinos
   
                d_sigma_BL_e = 2**0.5*G_f*m_e*g_ve*g_prime**2/(np.pi*(2*E_R[i]*m_e+m_Z_prime**2))+m_e*g_prime**4/(2.0*np.pi*((2.0*E_R[i]*m_e+m_Z_prime**2)**2))
                d_sigma_BL_mt = 2**0.5*G_f*m_e*g_v*g_prime**2/(np.pi*(2*E_R[i]*m_e+m_Z_prime**2))+m_e*g_prime**4/(2.0*np.pi*((2.0*E_R[i]*m_e+m_Z_prime**2)**2))
                #print d_sigma_BL_e
                #print d_sigma_BL_mt
                if key == "atmos_mu" or key == "atmos_mu_bar":
                	d_sigma = d_sigma_BL_mt+d_sigma_SM_mt
                elif key == "atmos_e" or key == "atmos_e":
                	d_sigma = d_sigma_BL_e+d_sigma_SM_e
                else:
                	d_sigma = (d_sigma_BL_e+d_sigma_SM_e)*0.55+(d_sigma_BL_mt+d_sigma_SM_mt)*0.45#Account for solar neutrino oscillations
                
                integrand = d_sigma*v_flux[key]
                integral[key][i] = integ.trapz(integrand[int(E_min_arg[key][i]):], E_v[key][int(E_min_arg[key][i]):])
    
    dNv_dEr_dict = {}
    
    for key in E_v:
        dNv_dEr_dict[key] = epsilon*efficiency*integral[key]*Z/(m_N)    
    
    dNv_dEr = np.zeros(len(E_r))
    for key in dNv_dEr_dict:
        dNv_dEr = dNv_dEr + dNv_dEr_dict[key]#sum contributions from each flux source
    
    N = np.zeros(len(E_r))
    for i in range(len(E_r)):
        N[i] = integ.trapz((dNv_dEr[i:])/1000000, E_r[i:])
    return N


'''DARK MATTER'''

def f_v(v, v_circ, v_lag):
    sigma = v_circ/2**0.5
    const = (2.0/np.pi)**0.5*(v_lag*sigma)**(-1)
    ex = -(v**2+v_lag**2)/(2*sigma**2)
    return const*np.exp(ex)*np.sinh(v*v_lag/(sigma**2))

log_m = np.linspace(-1, 3, 200)*2.303
m_array = np.exp(log_m)

def dm_calc(m_array, Z, N):
    A = N+Z
    m_N = A*m_p
    m_red_N = m_array*m_N/(m_array+m_N)
    mu_p = m_array*m_p/(m_array+m_p)
    
    vmin_dict = {}
    inv_mean_dict = {}
    for j in range(len(m_array)):
        #print 'm_array'
        #print j
        vmin_dict[j] = (m_N*(E_R)/(2*m_red_N[j]**2))**0.5
        inv_mean_dict[j] = np.zeros(len(E_r))
        for i in range(len(E_r)):
            sigma = v_circ/2**0.5
            N_esc = erf(v_esc/(2**0.5*sigma)) - (2/np.pi)**0.5*v_esc/sigma * np.exp(-v_esc**2/(2*sigma**2))
            if vmin_dict[j][i]<=v_esc-v_lag:
                inv_mean_dict[j][i] = ((1/(2*N_esc*v_lag))*(erf((vmin_dict[j][i]+v_lag)/v_circ) - erf((vmin_dict[j][i]-v_lag)/v_circ)) - (2/(v_circ*N_esc*np.pi**0.5) * np.exp(-(v_esc/v_circ)**2)))
            elif vmin_dict[j][i]<=v_esc+v_lag:
                inv_mean_dict[j][i] = ((1/(2*N_esc*v_lag))*(erf((v_esc)/v_circ) - erf((vmin_dict[j][i]-v_lag)/v_circ)) - ((2/(N_esc*v_circ*np.pi**0.5))*((v_esc+v_lag-vmin_dict[j][i])/(2*v_lag))*np.exp(-(v_esc/v_circ)**2)))
            #if vmin_dict[j][i]<=v_esc:
            #    inv_mean_dict[j][i], v_err = integ.quad(f_v, vmin_dict[j][i], v_esc, args=(v_circ, v_lag))
    return inv_mean_dict

def v_floor_calc(Nv, inv_mean_dict, Z, N, F_SI, exp_jump = 50):
    A = N+Z
    m_N = A*m_p
    m_red_N = m_array*m_N/(m_array+m_N)
    mu_p = m_array*m_p/(m_array+m_p)
    floor_lines = {}
    exposure = 1/Nv#Exposure to give 1 v count (in ton years)
    for i in range(int(len(exposure)/(exp_jump))):
        floor_lines[i] = np.zeros(len(m_array))
        for j in range(len(m_array)):
            dNx_dEr = epsilon*exposure[i*exp_jump]*rho*(F_SI**2)*inv_mean_dict[j]/(2*m_array[j]*m_red_N[j]**2)
            Nx = integ.trapz((dNx_dEr[i*exp_jump:]), E_R[i*exp_jump:])
            floor_lines[i][j] = 2.3*3.8938e-28/(Nx*A**2*(m_red_N[j]/mu_p[j])**2)
    
    v_floor = np.zeros(len(m_array))
    for i in range(len(v_floor)):
        v_floor[i] = min(data[i] for data in floor_lines.values() if data[i]>0)
    return v_floor

def Yield(E_r, V, Z = 32):
    #Z = 32.0
    epsilon = 11.5*E_r*Z**(-7/3)
    k = 0.157
    g = 3*epsilon**0.15 + 0.7*epsilon**0.6 + epsilon
    Y = k*g/(1+k*g)
    return E_r*(1+Y*V/3)/(1+V/3)


'''GENERATE NEUTRINO FLOOR'''

#E_ee = {}

#E_ee['70'] = Yield(E_r, 70)
#E_ee['150'] = Yield(E_r, 150)
'''CONSTRAINTS'''
BL = {}#(mass, coupling)

'''#OG
BL[0] = (0.02950549551825165, 0.000043287612810830706)
BL[1] = (0.054197665787207694, 0.00007742636826811293)
BL[2] = (0.08942445678545866, 0.00009006280202112832)
BL[3] = (0.19294649476442313, 0.0001519911082952938)
BL[4] = (0.6452213624158634, 0.00040370172585965663)
BL[5] = (1.5919951765531937, 0.0004328761281083075)
BL[6] = (3.6896935009615266, 0.0004695881961786378)
BL[7] = (9.778928306368183, 0.0005336699231206335)
'''

#Bauer
BL[0] = (0.020113265529181412, 0.000028411121153323796)
BL[1] = (0.03291644578600374, 0.000045391730941267094)
BL[2] = (0.05145680133874588, 0.00007155684151903584)
BL[3] = (0.06620572143707974, 0.00007964578382764386)
BL[4] = (0.10650332930564013, 0.00009867020863649179)
BL[5] = (0.20579406028699215, 0.00015347813444464704)
BL[6] = (0.2334313619803343, 0.0001619206824459435)
BL[7] = (0.2836196510846913, 0.0002005975301100495)
BL[8] = (0.7552825531919002, 0.0004725182693587074)
BL[9] = (1.0468886648272397, 0.00047888646014676665)
BL[10] = (65.45161965659264, 0.002879402156530748)


'''#Ilten
BL[0] = (0.012469942812837045, 0.000017547378627125316)
BL[1] = (0.026624950054062978, 0.000036118243849564166)
BL[2] = (0.05257889391460313, 0.0000729186231397832)
BL[3] = (0.08305884185346683, 0.00008516567530433997)
BL[4] = (0.12001244397475588, 0.00010346599162710776)
BL[5] = (0.18132214378524492, 0.00013591584079594685)
BL[6] = (0.26494421489757436, 0.00018568671398078637)
BL[7] = (0.34626994199121486, 0.00023011021626592298)
BL[8] = (0.790505834753686, 0.0005021849464873106)
BL[9] = (1.0387998157707405, 0.00045502443501054234)
BL[10] = (10.574613221707812, 0.006599118111169704)
'''

BLT = {}#(mass, coupling(3), mixing angle)
BLT[0] = (0.2630764466275547, 0.005361956770568816, 0.00004749979434661412)#tanb = 10
BLT[1] = (1.053273077968177, 0.021052722727657028, 0.00018649907912142961)
BLT[2] = (1.8818723614330761, 0.003858923467029907, 0.00003418492146175941)
BLT[3] = (2.6060551766045177, 0.02376511113811073, 0.00021052722727657092)
BLT[4] = (5.596942170754293, 0.018015200408020238, 0.0001595906776404731)
BLT[5] = (7.861253539546596, 0.008124084577862966, 0.00007196856730011529)
BLT[6] = (9.584233305564016, 0.019987196386572485, 0.00017705993512268998)
BLT[7] = (75.70043248078939, 0.23357214690901176, 0.00206913808111479)
BLT[8] = (28.91116376721265, 0.08557183144939998, 0.0007580524367559262)
BLT[9] = (0.11575095695223918, 0.002335721469090121,0.0000206913808111479)
BLT[10] = (0.1472423658261952, 0.0030283286655749457,0.000026826957952797274)

'''
BLT={} #Miniboone model
BLT[0] = (0.0020524913732975787, 0.177, 0.000007306483356846009)
BLT[0] = (0.005688565509349191, 0.177, 0.00001572220557885352)
BLT[0] = (0.015998036055074147,0.177,  0.000037470764292633866)
BLT[0] = (0.05204933004069108, 0.177, 0.00012133711394622009)
BLT[0] = (0.1126341373494067,0.177,  0.0002818897206905312)
BLT[0] = (1.5935290261385806, 0.177, 0.00028918298859687046)
BLT[0] = (8.150705518681809,0.177,  0.01144556099145304)
BLT[0] = (44.11423433488659,0.177,  0.01033387191932061)
'''

Seq = {}
Seq[0] = (0.7895627848632042, 0.0008547968093841695)
Seq[1] = (2.7504435375968144, 0.002970198678477084)
Seq[2] = (0.26981449203897523, 0.00029286445646252375)
Seq[3] = (7.684290622978283, 0.00824592378177437)


inv_mean = {}
Nv_SM = {}
v_floor_SM = {}
Nv = {}
v_floor = {}
New_floor = {}
BL_floor = {}

E_min_n_Xe, E_min_arg_n_Xe, F_SI_Xe = E_min_n_calc(54.0,78.0)
E_min_n_Ge, E_min_arg_n_Ge, F_SI_Ge = E_min_n_calc(32.0,40.0)
E_min_n_He, E_min_arg_n_He, F_SI_He = E_min_n_calc(2.0,2.0)

inv_mean['Ge'] = dm_calc(m_array, 32.0, 40.0)
Nv_SM['Ge'] = v_calc_vector_BL(E_r, 0, 0.0, 32.0, 40.0, E_min_n_Ge, E_min_arg_n_Ge, F_SI_Ge)
v_floor_SM['Ge'] = v_floor_calc(Nv_SM['Ge'], inv_mean['Ge'], 32.0, 40.0, F_SI_Ge, exp_jump = 100)

Nv['Ge'] = {}
v_floor['Ge'] = {}

Nv['Ge']['BL'] = {}
v_floor['Ge']['BL'] = {}

for key in BL:
    Nv['Ge']['BL'][key] = v_calc_vector_BL(E_r, BL[key][0], BL[key][1], 32.0, 40.0, E_min_n_Ge, E_min_arg_n_Ge, F_SI_Ge)
    v_floor['Ge']['BL'][key] = v_floor_calc(Nv['Ge']['BL'][key], inv_mean['Ge'], 32.0, 40.0, F_SI_Ge, exp_jump = 100)

BL_floor['Ge'] = {}
BL_floor['Ge']['BL'] = np.zeros(len(m_array))

Nv['Ge']['BL(3)'] = {}
v_floor['Ge']['BL(3)'] = {}

for key in BLT:
    Nv['Ge']['BL(3)'][key] = v_calc_vector_BLT(E_r, BLT[key][1], BLT[key][2], BLT[key][0], 32.0, 40.0, E_min_n_Ge, E_min_arg_n_Ge, F_SI_Ge)
    v_floor['Ge']['BL(3)'][key] = v_floor_calc(Nv['Ge']['BL(3)'][key], inv_mean['Ge'], 32.0, 40.0, F_SI_Ge, exp_jump = 100)

BL_floor['Ge']['BL(3)'] = np.zeros(len(m_array))

#Nv['Ge']['Seq'] = {}
#v_floor['Ge']['Seq'] = {}

#for key in Seq:
#    Nv['Ge']['Seq'][key] = v_calc_vector_seq(E_r, Seq[key][0], Seq[key][1], 32.0, 40.0, E_min_n_Ge, E_min_arg_n_Ge, F_SI_Ge)
#    v_floor['Ge']['Seq'][key] = v_floor_calc(Nv['Ge']['Seq'][key], inv_mean['Ge'], 32.0, 40.0, F_SI_Ge, exp_jump = 100)

#BL_floor['Ge']['Seq'] = np.zeros(len(m_array))

#Nv['Ge']['SO10'] = {}
#v_floor['Ge']['SO10'] = {}
#
#for key in BL:
#    Nv['Ge']['SO10'][key] = v_calc_vector_SO(E_r, BL[key][0], BL[key][1], 32.0, 40.0, E_min_n_Ge, E_min_arg_n_Ge, F_SI_Ge)
#    v_floor['Ge']['SO10'][key] = v_floor_calc(Nv['Ge']['SO10'][key], inv_mean['Ge'], 32.0, 40.0, F_SI_Ge, exp_jump = 100)
# 
#BL_floor['Ge']['SO10'] = np.zeros(len(m_array))

New_floor['Ge'] = np.zeros(len(m_array))




inv_mean['He'] = dm_calc(m_array, 2.0, 2.0)
Nv_SM['He'] = v_calc_vector_BL(E_r, 0, 0.0, 2.0, 2.0, E_min_n_He, E_min_arg_n_He, F_SI_He)
v_floor_SM['He'] = v_floor_calc(Nv_SM['He'], inv_mean['He'], 2.0, 2.0, F_SI_He, exp_jump = 100)

Nv['He'] = {}
v_floor['He'] = {}

Nv['He']['BL'] = {}
v_floor['He']['BL'] = {}

for key in BL:
    Nv['He']['BL'][key] = v_calc_vector_BL(E_r, BL[key][0], BL[key][1], 2.0, 2.0, E_min_n_He, E_min_arg_n_He, F_SI_He)
    v_floor['He']['BL'][key] = v_floor_calc(Nv['He']['BL'][key], inv_mean['He'], 2.0, 2.0, F_SI_He, exp_jump = 100)

BL_floor['He'] = {}
BL_floor['He']['BL'] = np.zeros(len(m_array))

Nv['He']['BL(3)'] = {}
v_floor['He']['BL(3)'] = {}

for key in BLT:
    Nv['He']['BL(3)'][key] = v_calc_vector_BLT(E_r, BLT[key][1], BLT[key][2], BLT[key][0], 2.0, 2.0, E_min_n_He, E_min_arg_n_He, F_SI_He)
    v_floor['He']['BL(3)'][key] = v_floor_calc(Nv['He']['BL(3)'][key], inv_mean['He'], 2.0, 2.0, F_SI_He, exp_jump = 100)

BL_floor['He']['BL(3)'] = np.zeros(len(m_array))

#Nv['He']['Seq'] = {}
#v_floor['He']['Seq'] = {}

#for key in Seq:
#    Nv['He']['Seq'][key] = v_calc_vector_seq(E_r, Seq[key][0], Seq[key][1], 2.0, 2.0, E_min_n_He, E_min_arg_n_He, F_SI_He)
#    v_floor['He']['Seq'][key] = v_floor_calc(Nv['He']['Seq'][key], inv_mean['He'], 2.0, 2.0, F_SI_He, exp_jump = 100)

#BL_floor['He']['Seq'] = np.zeros(len(m_array))

#Nv['He']['SO10'] = {}
#v_floor['He']['SO10'] = {}
#
#for key in BL:
#    Nv['He']['SO10'][key] = v_calc_vector_SO(E_r, BL[key][0], BL[key][1], 2.0, 2.0, E_min_n_He, E_min_arg_n_He, F_SI_He)
#    v_floor['He']['SO10'][key] = v_floor_calc(Nv['He']['SO10'][key], inv_mean['He'], 2.0, 2.0, F_SI_He, exp_jump = 100)
#
#BL_floor['He']['SO10'] = np.zeros(len(m_array))

New_floor['He'] = np.zeros(len(m_array))




inv_mean['Xe'] = dm_calc(m_array, 54.0, 78.0)
Nv_SM['Xe'] = v_calc_vector_BL(E_r, 0, 0.0, 54.0, 78.0, E_min_n_Xe, E_min_arg_n_Xe, F_SI_Xe)
v_floor_SM['Xe'] = v_floor_calc(Nv_SM['Xe'], inv_mean['Xe'], 54, 78, F_SI_Xe, exp_jump = 100)

Nv['Xe'] = {}
v_floor['Xe'] = {}

Nv['Xe']['BL'] = {}
v_floor['Xe']['BL'] = {}

for key in BL:
    Nv['Xe']['BL'][key] = v_calc_vector_BL(E_r, BL[key][0], BL[key][1], 54.0, 78.0, E_min_n_Xe, E_min_arg_n_Xe, F_SI_Xe)
    v_floor['Xe']['BL'][key] = v_floor_calc(Nv['Xe']['BL'][key], inv_mean['Xe'], 54, 78, F_SI_Xe, exp_jump = 100)

BL_floor['Xe'] = {}
BL_floor['Xe']['BL'] = np.zeros(len(m_array))

Nv['Xe']['BL(3)'] = {}
v_floor['Xe']['BL(3)'] = {}

for key in BLT:
    Nv['Xe']['BL(3)'][key] = v_calc_vector_BLT(E_r, BLT[key][1], BLT[key][2], BLT[key][0], 54.0, 78.0, E_min_n_Xe, E_min_arg_n_Xe, F_SI_Xe)
    v_floor['Xe']['BL(3)'][key] = v_floor_calc(Nv['Xe']['BL(3)'][key], inv_mean['Xe'], 54, 78, F_SI_Xe, exp_jump = 100)

BL_floor['Xe']['BL(3)'] = np.zeros(len(m_array))


#Nv['Xe']['Seq'] = {}
#v_floor['Xe']['Seq'] = {}

#for key in Seq:
#    Nv['Xe']['Seq'][key] = v_calc_vector_seq(E_r, Seq[key][0], Seq[key][1], 54.0, 78.0, E_min_n_Xe, E_min_arg_n_Xe, F_SI_Xe)
#    v_floor['Xe']['Seq'][key] = v_floor_calc(Nv['Xe']['Seq'][key], inv_mean['Xe'], 54, 78, F_SI_Xe, exp_jump = 100)

#BL_floor['Xe']['Seq'] = np.zeros(len(m_array))

#Nv['Xe']['SO10'] = {}
#v_floor['Xe']['SO10'] = {}
#
#for key in BL:
#    Nv['Xe']['SO10'][key] = v_calc_vector_SO(E_r, BL[key][0], BL[key][1], 54.0, 78.0, E_min_n_Xe, E_min_arg_n_Xe, F_SI_Xe)
#    v_floor['Xe']['SO10'][key] = v_floor_calc(Nv['Xe']['SO10'][key], inv_mean['Xe'], 54, 78, F_SI_Xe, exp_jump = 100)
#
#BL_floor['Xe']['SO10'] = np.zeros(len(m_array))


New_floor['Xe'] = np.zeros(len(m_array))

#for i in range(len(New_floor['Xe'])):
#    New_floor['Xe'][i] = max(data[i] for data in BL_floor['Xe'].values())

for key in BL_floor:
    for i in range(len(BL_floor[key]['BL'])):
        BL_floor[key]['BL'][i] = max(data[i] for data in v_floor[key]['BL'].values())

for key in BL_floor:
    #BL_floor[key]['Miniboone'] = BL_floor[key]['BL(3)']
    #del BL_floor[key]['BL(3)']
    for i in range(len(BL_floor[key]['BL(3)'])):
        BL_floor[key]['BL(3)'][i] = max(data[i] for data in v_floor[key]['BL(3)'].values())

#for key in BL_floor:
#    for i in range(len(BL_floor[key]['Seq'])):
#        BL_floor[key]['Seq'][i] = max(data[i] for data in v_floor[key]['Seq'].values())

#for key in BL_floor:
#    for i in range(len(BL_floor[key]['SO10'])):
#        BL_floor[key]['SO10'][i] = max(data[i] for data in v_floor[key]['SO10'].values())

for key in New_floor:
    for i in range(len(New_floor[key])):
        New_floor[key][i] = max(data[i] for data in BL_floor[key].values())


'''CURRENT DIRECT DETECTION BOUNDS'''

DD_bounds_x = {}
DD_bounds_y = {}

DD_bounds_ratio = {}

DD_bounds_x['Xenon1T'] = np.loadtxt("Xenon1T.dat", usecols = (0,))#Xenon1T curves
DD_bounds_y['Xenon1T'] = np.loadtxt("Xenon1T.dat", usecols = (1,))

DD_bounds_x['Xenon1TSensitivity'] = np.loadtxt("Xenon1TSensitivity.dat", usecols = (0,))
DD_bounds_y['Xenon1TSensitivity'] = np.loadtxt("Xenon1TSensitivity.dat", usecols = (1,))

DD_bounds_x['SuperCDMSHV'] = np.loadtxt("SuperCDMSHV.dat", usecols = (0,))
DD_bounds_y['SuperCDMSHV'] = np.loadtxt("SuperCDMSHV.dat", usecols = (1,))

DD_bounds_x['SuperCDMSHVSensitivity'] = np.loadtxt("SuperCDMSHVSensitivity.dat", usecols = (0,))
DD_bounds_y['SuperCDMSHVSensitivity'] = np.loadtxt("SuperCDMSHVSensitivity.dat", usecols = (1,))

DD_bounds_x['SuperCDMSiZipSensitivity'] = np.loadtxt("SuperCDMSiZipSensitivity.dat", usecols = (0,))
DD_bounds_y['SuperCDMSiZipSensitivity'] = np.loadtxt("SuperCDMSiZipSensitivity.dat", usecols = (1,))

DD_bounds_x['LUX'] = np.loadtxt("LUX.dat", usecols = (0,))
DD_bounds_y['LUX'] = np.loadtxt("LUX.dat", usecols = (1,))

DD_bounds_x['LZSensitivity'] = np.loadtxt("LZSensitivity.dat", usecols = (0,))
DD_bounds_y['LZSensitivity'] = np.loadtxt("LZSensitivity.dat", usecols = (1,))

DD_bounds_x['NEWSG'] = np.loadtxt("NEWSG.dat", usecols = (0,))
DD_bounds_y['NEWSG'] = np.loadtxt("NEWSG.dat", usecols = (1,))

DD_bounds_x['NEWSG_He'] = np.loadtxt("NEWSG_Helium.txt", usecols = (0,))
DD_bounds_y['NEWSG_He'] = np.loadtxt("NEWSG_Helium.txt", usecols = (1,))

DD_bounds_x['CDMS'] = np.loadtxt("CDMS.dat", usecols = (0,))
DD_bounds_y['CDMS'] = np.loadtxt("CDMS.dat", usecols = (1,))


DD_bounds_ratio['Xenon1T'] = np.zeros(len(DD_bounds_y['Xenon1T']))
for i in range(len(DD_bounds_y['Xenon1T'])):
    index = (np.abs(m_array-DD_bounds_x['Xenon1T'][i])).argmin()
    DD_bounds_ratio['Xenon1T'][i] = (DD_bounds_y['Xenon1T'][i])/v_floor_SM['Xe'][index]
    
DD_bounds_ratio['LUX'] = np.zeros(len(DD_bounds_y['LUX']))
for i in range(len(DD_bounds_y['LUX'])):
    index = (np.abs(m_array-DD_bounds_x['LUX'][i])).argmin()
    DD_bounds_ratio['LUX'][i] = (DD_bounds_y['LUX'][i])/v_floor_SM['Xe'][index]
    
DD_bounds_ratio['LZSensitivity'] = np.zeros(len(DD_bounds_y['LZSensitivity']))
for i in range(len(DD_bounds_y['LZSensitivity'])):
    index = (np.abs(m_array-DD_bounds_x['LZSensitivity'][i])).argmin()
    DD_bounds_ratio['LZSensitivity'][i] = (DD_bounds_y['LZSensitivity'][i])/v_floor_SM['Xe'][index]

DD_bounds_ratio['SuperCDMSHVSensitivity'] = np.zeros(len(DD_bounds_y['SuperCDMSHVSensitivity']))
for i in range(len(DD_bounds_y['SuperCDMSHVSensitivity'])):
    index = (np.abs(m_array-DD_bounds_x['SuperCDMSHVSensitivity'][i])).argmin()
    DD_bounds_ratio['SuperCDMSHVSensitivity'][i] = (DD_bounds_y['SuperCDMSHVSensitivity'][i])/v_floor_SM['Ge'][index]
    
DD_bounds_ratio['SuperCDMSiZipSensitivity'] = np.zeros(len(DD_bounds_y['SuperCDMSiZipSensitivity']))
for i in range(len(DD_bounds_y['SuperCDMSiZipSensitivity'])):
    index = (np.abs(m_array-DD_bounds_x['SuperCDMSiZipSensitivity'][i])).argmin()
    DD_bounds_ratio['SuperCDMSiZipSensitivity'][i] = (DD_bounds_y['SuperCDMSiZipSensitivity'][i])/v_floor_SM['Ge'][index]


DD_bounds_ratio['NEWSG_He'] = np.zeros(len(DD_bounds_y['NEWSG_He']))
for i in range(len(DD_bounds_y['NEWSG_He'])):
    index = (np.abs(m_array-DD_bounds_x['NEWSG_He'][i])).argmin()
    DD_bounds_ratio['NEWSG_He'][i] = (DD_bounds_y['NEWSG_He'][i])/v_floor_SM['He'][index]



'''PLOTS'''
cmap = cm.gist_rainbow
nstep = 4.0

yt = 0.625
yb = 0.175
xea = 0.255
xgap = 0.038

yedge = (1.0-(yt+yb))/2
xedge = (1.0-(xea*3+xgap*2))/2

size = 14

fig = plt.figure(figsize = (15,6))


Het = fig.add_axes((xedge, yedge+yb, xea, yt))
plt.plot(m_array, v_floor_SM['He'], linewidth = 2, linestyle = '-', color = 'silver')
plt.plot(m_array, (BL_floor['He']['BL']), linewidth = 2, linestyle = '--', color = 'Black')
plt.plot(m_array, (BL_floor['He']['BL(3)']), linewidth = 2, linestyle = '-.', color = 'Black')
plt.plot(DD_bounds_x['NEWSG_He'], DD_bounds_y['NEWSG_He'], linestyle = '--',color = 'red', linewidth = 1.2, label = 'NEWS-G')
plt.loglog()
plt.text(0.18,1e-41,"He",fontsize = 16)
plt.xlim(0.1, 1000)
plt.xticks([])
plt.ylim(1e-49, 1e-40)
#plt.xlabel('WIMP mass [GeV]')
plt.ylabel(r'$\sigma_{\chi n}$ $\mathrm{[cm^2]}$',fontsize = size)
plt.legend(frameon=False, fontsize = 12, loc = "upper right")

Get = fig.add_axes((xedge+xea+xgap, yedge+yb, xea, yt))
plt.plot(m_array, v_floor_SM['Ge'], linewidth = 2, linestyle = '-', color = 'silver')
plt.plot(m_array, (BL_floor['Ge']['BL']), linewidth = 2, linestyle = '--', color = 'Black')
plt.plot(m_array, (BL_floor['Ge']['BL(3)']), linewidth = 2, linestyle = '-.', color = 'Black')
#plt.plot(DD_bounds_x['CDMS'], DD_bounds_y['CDMS'], linestyle = ':', label = 'CDMS')
#plt.plot(DD_bounds_x['SuperCDMSHV'], DD_bounds_y['SuperCDMSHV'], linestyle = ':', label = 'SuperCDMS HV')
plt.plot(DD_bounds_x['SuperCDMSHVSensitivity'], DD_bounds_y['SuperCDMSHVSensitivity'], linestyle = '--',color = 'red', linewidth = 1.2, label = 'SuperCDMS HV')
plt.plot(DD_bounds_x['SuperCDMSiZipSensitivity'], DD_bounds_y['SuperCDMSiZipSensitivity'], linestyle = '--', color = 'blue', linewidth = 1.2, label = 'SuperCDMS iZip')
plt.loglog()
plt.text(0.18,1e-41,"Ge",fontsize = 16)
plt.xlim(0.1, 1000)
plt.xticks([])
plt.ylim(1e-49, 1e-40)
#plt.xlabel('WIMP mass [GeV]')
#plt.ylabel(r'$\sigma_{\chi N}$ $\mathrm{[cm^2]}$')
plt.legend(frameon=False, fontsize = 12, loc = 'upper right')

Xet = fig.add_axes((xedge+2*xea+2*xgap, yedge+yb, xea, yt))
plt.plot(m_array, v_floor_SM['Xe'], linewidth = 2, linestyle = '-', color = 'silver')
plt.plot(m_array, (BL_floor['Xe']['BL']), linewidth = 2, linestyle = '--', color = 'Black')
plt.plot(m_array, (BL_floor['Xe']['BL(3)']), linewidth = 2, linestyle = '-.', color = 'Black')
#plt.plot(DD_bounds_x['LUX'], DD_bounds_y['LUX'], linestyle = '-.', linewidth = 1.2, label = 'LUX')
plt.plot(DD_bounds_x['Xenon1T'], DD_bounds_y['Xenon1T'], linestyle = '-', color = 'blue', linewidth = 1.2, label = 'Xenon1T')
plt.plot(DD_bounds_x['LZSensitivity'], DD_bounds_y['LZSensitivity'], linestyle = '--', color = 'red', linewidth = 1.2, label = 'LZ')
plt.loglog()
plt.text(0.18,1e-41,"Xe",fontsize = 16)
plt.xlim(0.1, 1000)
plt.xticks([])
plt.ylim(1e-49, 1e-40)
#plt.xlabel('WIMP mass [GeV]')
#plt.ylabel(r'$\sigma_{\chi N}$ $\mathrm{[cm^2]}$')
plt.legend(frameon=False, fontsize = 12, loc = 'upper right')



Heb = fig.add_axes((xedge, yedge, xea, yb))
plt.plot(m_array, (BL_floor['He']['BL'])/v_floor_SM['He'], linewidth = 2, linestyle = '--', color = 'Black')
plt.plot(m_array, (BL_floor['He']['BL(3)'])/v_floor_SM['He'], linewidth = 2, linestyle = '-.', color = 'Black')
plt.fill_between(DD_bounds_x['NEWSG_He'], DD_bounds_ratio['NEWSG_He'], linestyle = '--', color = 'lightgrey', edgecolor = 'red', linewidth = 1.2, label = 'NEWS-G')
Heb.tick_params(axis='y',which='both',left=True)
plt.loglog()
plt.xlim(0.1, 1000)
plt.ylim(0.9, 2.5)
Heb.yaxis.set_major_locator(MultipleLocator(1))
Heb.yaxis.set_minor_locator(MultipleLocator(0.2))
Heb.set_yticklabels(['0','1','2'], minor=False)
Heb.set_yticklabels([], minor=True) 
#plt.yticks([1,2],['1', '2'])
plt.xlabel('$m_\chi \mathrm{[GeV]}$',fontsize = size)
plt.ylabel(r"$\sigma_{\chi n}/\sigma^{SM}_{\chi n}$",fontsize = size)
#plt.legend(frameon=False, fontsize = 12, loc = "lower right")

Geb = fig.add_axes((xedge+xea+xgap, yedge, xea, yb))
plt.plot(m_array, (BL_floor['Ge']['BL'])/v_floor_SM['Ge'], linewidth = 2, linestyle = '--', color = 'Black')
plt.plot(m_array, (BL_floor['Ge']['BL(3)'])/v_floor_SM['Ge'], linewidth = 2, linestyle = '-.', color = 'Black')
#plt.plot(DD_bounds_x['SuperCDMSHVSensitivity'], DD_bounds_ratio['SuperCDMSHVSensitivity'], linestyle = '--', color = 'red', linewidth = 1.2, label = 'SuperCDMS HV')
#plt.plot(DD_bounds_x['SuperCDMSiZipSensitivity'], DD_bounds_ratio['SuperCDMSiZipSensitivity'], linestyle = '--', color = 'blue', linewidth = 1.2, label = 'SuperCDMS iZip')
plt.fill_between(DD_bounds_x['SuperCDMSHVSensitivity'], DD_bounds_ratio['SuperCDMSHVSensitivity'], linestyle = '--', color = 'lightgrey', edgecolor = 'red', linewidth = 0, label = 'SuperCDMS HV')
plt.fill_between(DD_bounds_x['SuperCDMSiZipSensitivity'], DD_bounds_ratio['SuperCDMSiZipSensitivity'], linestyle = '--', color = 'lightgrey', edgecolor = 'blue', linewidth = 0, label = 'SuperCDMS iZip')
plt.plot(DD_bounds_x['SuperCDMSHVSensitivity'], DD_bounds_ratio['SuperCDMSHVSensitivity'], linestyle = '--', color = 'red', linewidth = 1.2, label = 'SuperCDMS HV')
plt.plot(DD_bounds_x['SuperCDMSiZipSensitivity'], DD_bounds_ratio['SuperCDMSiZipSensitivity'], linestyle = '--', color = 'blue', linewidth = 1.2, label = 'SuperCDMS iZip')
Geb.tick_params(axis='y',which='both',left=True)
plt.loglog()
plt.xlim(0.1, 1000)
plt.ylim(0.9, 2.5)
Geb.yaxis.set_major_locator(MultipleLocator(1))
Geb.yaxis.set_minor_locator(MultipleLocator(0.2))
Geb.set_yticklabels(['0','1','2'], minor=False)
Geb.set_yticklabels([], minor=True)
#plt.yticks([1,2],['1', '2'])
plt.xlabel('$m_\chi \mathrm{[GeV]}$',fontsize = size)
#plt.ylabel(r'$\sigma_{\chi N}$ $\mathrm{[cm^2]}$')
#plt.legend(frameon=False, fontsize = 12, loc = 'upper right')

Xeb = fig.add_axes((xedge+2*xea+2*xgap, yedge, xea, yb))
plt.plot(m_array, (BL_floor['Xe']['BL'])/v_floor_SM['Xe'], linewidth = 2, linestyle = '--', color = 'Black')
plt.plot(m_array, (BL_floor['Xe']['BL(3)'])/v_floor_SM['Xe'], linewidth = 2, linestyle = '-.', color = 'Black')
#plt.plot(DD_bounds_x['LUX'], DD_bounds_ratio['LUX'], linestyle = '-', linewidth = 1.2, label = 'LUX')
plt.fill_between(DD_bounds_x['Xenon1T'], DD_bounds_ratio['Xenon1T'], linestyle = '-', color = 'lightgrey', edgecolor = 'blue', linewidth = 0, label = 'Xenon1T')
plt.fill_between(DD_bounds_x['LZSensitivity'], DD_bounds_ratio['LZSensitivity'], linestyle = '--', color = 'lightgrey', edgecolor = 'red', linewidth = 0, label = 'LZ')
plt.plot(DD_bounds_x['Xenon1T'], DD_bounds_ratio['Xenon1T'], linestyle = '-',color = 'blue', linewidth = 1.2, label = 'Xenon1T')
plt.plot(DD_bounds_x['LZSensitivity'], DD_bounds_ratio['LZSensitivity'], linestyle = '--',color = 'red', linewidth = 1.2, label = 'LZ')
Xeb.tick_params(axis='y',which='both',left=True)
plt.loglog()
plt.xlim(0.1, 1000)
plt.ylim(0.9, 2.5)
Xeb.yaxis.set_major_locator(MultipleLocator(1))
Xeb.yaxis.set_minor_locator(MultipleLocator(0.2))
Xeb.set_yticklabels(['0','1','2'], minor=False)
Xeb.set_yticklabels([], minor=True)
#plt.yticks([1,2],['1', '2'])
plt.xlabel('$m_\chi \mathrm{[GeV]}$',fontsize = size)





plt.show()

