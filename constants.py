# Constants.py
# Defines physical constants

from numpy import pi 

π = pi

m_p = 1.67e-27  # Ion mass (proton mass) [kg]
m_e = 9.1093837e-31  # Ion mass (proton mass) [kg]
ee = 1.6e-19  # Elementary charge [C]
ε_0 = 8.85e-12  # Vacuum permittivity [F/m]
k_B = 1.38e-23  # Boltzmann constant [J/K]
hbar=  6.62607015e-34 # Planck [J/s]

# Conversion units
# Multiply variable by this to go from first unit to second
K_to_eV = k_B / ee 
J_to_eV = 1/ee