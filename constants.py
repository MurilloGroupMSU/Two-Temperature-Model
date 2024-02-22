# Constants.py
# Defines physical constants
from numpy import pi 

π = pi

m_e = 9.1093837e-31  # Ion mass (proton mass) [kg]
m_p = 1.67e-27  # Ion mass (proton mass) [kg]

aB = 5.29177210903e-11 # Bohr radius in m
k_B = 1.380649e-23 # SI J/K
hbar = 6.62607015e-34/(2*π) # J/s
α  = 0.0072973525693 # 1/c in AU
c  = 2.99792458e8 #m/s
σ_SB = 2*π**5 * k_B**4/( 15* (2*π*hbar)**3 * c**2 ) # Stefan-Boltzman 
ee = 1.602176634e-19 # C
ε_0 = 8.8541878128e-12 # F/m

eV_to_AU = 0.03674932539796232 # PDG
eV_to_K = 11604.5250061598
K_to_eV = 1/eV_to_K
erg_to_J = 1e-7  #erg to Joules 
J_to_eV = 1/ee  # Joule to eV
eV_to_J = 1/J_to_eV

J_to_erg = 1e7
AU_to_eV = 1/eV_to_AU
AU_to_J = 4.359744e-18
AU_to_erg = AU_to_J*J_to_erg
AU_to_Pa = AU_to_J / aB**3 
AU_to_bar = AU_to_J / aB**3 /1e5 
AU_to_invcc = 1/aB**3/1e6
AU_to_g  = 9.1093837e-28 
AU_to_s  = 2.4188843265e-17
AU_to_kg = 9.1093837e-31
AU_to_K = 1/(8.61732814974493e-5*eV_to_AU) #Similarly, 1 Kelvin = 3.167e-6... in natural units 
AU_to_m = aB
AU_to_cm = aB*1e2
AU_to_Angstrom = AU_to_cm*1e8
AU_to_Coulombs = 1.602176634e-19 #C
AU_to_Amps = AU_to_Coulombs/AU_to_s
AU_to_Volts = 27.211386245988
AU_to_Ohms  = AU_to_Volts/AU_to_Amps
AU_to_Siemens = 1/AU_to_Ohms

eV_to_AU   = 1/AU_to_eV
J_to_AU   = 1/AU_to_J
erg_to_AU   = 1/AU_to_erg
Pa_to_AU   = 1/AU_to_Pa
bar_to_AU   = 1/AU_to_bar
invcc_to_AU   = 1/AU_to_invcc
g_to_AU   = 1/AU_to_g
s_to_AU   = 1/AU_to_s
kg_to_AU   = 1/AU_to_kg
K_to_AU   = 1/AU_to_K
cm_to_AU = 1/AU_to_cm
m_to_AU = 1/AU_to_m
Angstrom_to_AU = 1/AU_to_Angstrom
Amps_to_AU = 1/AU_to_Amps
Coulombs_to_AU = 1/AU_to_Coulombs
Volts_to_AU = 1/AU_to_Volts
Ohms_to_AU = 1/AU_to_Ohms
Siemens_to_AU = 1/AU_to_Siemens
