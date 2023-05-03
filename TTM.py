
# Physical Parameters for Two-Temperature Models
# Michael S. Murillo
# 2 May 2023

# plan:
# add <Z>
# move files to GitHub
# start adding to slides: JT, GMS, "visco-elastic" gen of TTM
# perhaps also start with plasma formulary? 
# check with GMS and JT for specific heat factors


import numpy as np

# chemical potential
def chemical_potential(density, temperature):
    '''
    Returns a wide ranging chemical potential valid from
    zero temperature to the classical limit.
    
    This function uses the fit from Ichimaru Volume I.
    
    Inputs: electron density [1/cc] and temperature [eV]
    Output: chemical potential [eV]
    '''
    
    Theta = temperature/Fermi_energy(density)
    
    A = 0.25954
    B = 0.072
    b = 0.858
    
    term_1 = -1.5*np.log(Theta)
    term_2 = np.log(4/(3*np.sqrt(np.pi)))
    term_3 = (A*Theta**(-(1+b)) + B*Theta**(-(1+b)/2))/(1 + A*Theta**(-b))
    
    return temperature*(term_1 + term_2 + term_3)


# Fermi energy
def Fermi_energy(density):
    '''
    Returns the Fermi energy.
    
    Inputs: electron density
    Output: Fermi energy [eV]
    '''
    hbar = 6.58211957e-16 # eV-s
    c = 2.99792458e10 # cm/s
    me_c2 = 0.511e6 # eV
    
    E_F = (hbar*c)**2*(3*np.pi**2*density)**(2/3)/(2*me_c2)
    
    return E_F


# low-temperature chemical potential
def chemical_potential_JT(density, temperature):
    '''
    Returns the low-temperature chemical potential, as given by
    Eqn. (6) in Jiang and Tsai [J. Heat Transfer 127, 1167(2005)]. 
    
    Inputs: electron density [1/cc], electron temperature [eV]
    Output: chemical potential [eV]
    '''
    
    print(temperature)
    
    return temperature