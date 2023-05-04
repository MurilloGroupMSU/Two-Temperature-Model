
# Physical Parameters for Two-Temperature Models
# Michael S. Murillo
# 2 May 2023

# plan:
# add <Z>
# start adding to slides: JT, GMS, "visco-elastic" gen of TTM
# perhaps also start with plasma formulary? 
# check with GMS and JT for specific heat factors


# Works in SI units (sorry!)


import numpy as np
from constants import *

class Physical_Parameters(): 
    """
    Base class of physical parameters. Incorporates model-free physics
    """
    def __init__(self):
        pass

    # Static Methods
    @staticmethod
    def Fermi_energy( n): 
        '''
        Returns the Fermi energy.
        
        Args: 
            n: number density [1/m^3]
        Returns: 
            Fermi energy [J]
        '''
        E_F = hbar**2/(2*m_e) * (3*π**2 * n)**(2/3)

        return E_F

    @staticmethod
    def r_WignerSeitz(n):
        """
        Returns the Wigner-Seitz radius 
        Args: 
            n: number density [1/m^3]
        """
        rs = (3/(4*π*n))**(1/3)
        return rs

    @staticmethod
    def electron_thermal_velocity(Te):
        """
        Returns the electron thermal velocity
        Args: 
            T: Temperature [K]
        Returns:
            vthe: [m/s]
        """
        vth_e = np.sqrt(k_B*Te/m_e)
        return vth_e

    @staticmethod
    def ion_thermal_velocity(Ti, m_i):
        """
        Returns the ion thermal velocity
        Args: 
            T: Temperature [K]
        Returns:
            vthi: [m/s]
        """
        vth_i = np.sqrt(k_B*Ti/m_i)
        return vth_i
    
    @staticmethod
    def electron_Debye_length(n_e, Te):
        """
        Returns the electron Debye length
        Args: 
            n: number density [1/m^3]
            T: Temperature [K]
        """
        λDe = np.sqrt(ε_0*k_B*Te/(ee**2*n_e) )
        return λDe


    # Class Methods
    @classmethod
    def Theta(cls, n, T):
        '''
        Returns the Degneracy parameter θ
        
        Args: 
            n: number density [1/m^3]
        Returns: 
            θ
        '''
        θ = k_B *T / cls.Fermi_energy(n)
        return θ

    @classmethod
    def Gamma(cls, n, T, Z=1):
        '''
        Returns the Coulomb coupling strength parameter Γ
        
        Args: 
            n: number density [1/m^3]
            T: Temperature [K]

        Returns: 
            Γ
        '''
        Γ = (Z*ee)**2/(4*π*ε_0*cls.r_WignerSeitz(n))/(k_B*T)
        return Γ
    
    @classmethod
    def electron_deBroglie_wavelength(cls, n_e, Te):
        """
        Returns the electron de Broglie Wavelength
        Args: 
            n: number density [1/m^3]
            T: Temperature [K]
        """
        vthe = cls.electron_thermal_velocity(Te)
        λDe = hbar/ (2*m_e*vthe)
        return λDe




class Plasma_Formulary_Physics(Physical_Parameters):
    """
    Defines physical quantities using the plasma formulary
    """
    def __init__(self):
        # super().__init__()
        pass


class JT_GMS_Physics(Physical_Parameters):
    """
    Models based on two papers:

    Dense plasma temperature equilibration in the binary collision approximation
    D. O. Gericke, M. S. Murillo, and M. Schlanges
    Phys. Rev. E 65, 036418 – Published 7 March 2002

    and

    Improved Two-Temperature Model and Its Application in Ultrashort Laser Heating of Metal Films 
    Lan Jiang, Hai-Lung Tsai
    J. Heat Transfer. Oct 2005, 127(10): 1167-1173 (7 pages)
    """

    def __init__(self):
        """
        """


    # low-temperature chemical potential
    @classmethod
    def chemical_potential_JT(cls,density, temperature):
        ''' 
        Returns the low-temperature chemical potential, as given by
        Eqn. (6) in Jiang and Tsai [J. Heat Transfer 127, 1167(2005)]. 

        Inputs: electron density [1/cc], electron temperature [eV]
        Output: chemical potential [eV]
        '''

        print(temperature)

        return temperature

    # chemical potential
    @classmethod
    def chemical_potential(cls,n, T):
        '''
        Returns a wide ranging chemical potential valid from
        zero temperature to the classical limit.

        This function uses the fit from Ichimaru Volume I.

        Args: 
            n: electron density [1/m^3],
            T: temperature [K]
        Returns: 
            μ: chemical potential [J]
        '''

        Theta = cls.Theta(n, T)#temperature/self.Fermi_energy(density)

        A = 0.25954
        B = 0.072
        b = 0.858

        term_1 = -1.5*np.log(Theta)
        term_2 = np.log(4/(3*np.sqrt(np.pi)))
        term_3 = (A*Theta**(-(1+b)) + B*Theta**(-(1+b)/2))/(1 + A*Theta**(-b))
        μ = k_B*T*(term_1 + term_2 + term_3)
        
        return μ
        
    @staticmethod
    def electron_heat_capacity(n_e):
        """
        Returns the electron heat capacity
        Ideal gas
        Args: 
            n_e: electron number density [1/m^3]
        """
        Ce = 3/2 * k_B * n_e # Electron ideal gas heat capacity
        return Ce

    @staticmethod
    def ion_heat_capacity(n_i):
        """
        Returns the ion heat capacity
        Ideal gas
        Args: 
            n_i: electron number density [1/m^3]
        """
        Ci = 3/2 * k_B * n_i # Electron ideal gas heat capacity
        return Ci

    @classmethod
    def ei_coupling_factor(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns G, the electron-ion coupling factor

        CHECK: See ei_relaxation_time note for possible amiguity

        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            G: electron-ion coupling 
        """
        Ce = cls.electron_heat_capacity(n_e)
        τei_e = cls.ei_relaxation_time( n_e, n_i, m_i, Z_i, Te, Ti) 
        
        G = Ce/τei_e
        return G

    @classmethod
    def electron_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Based on J-T paper, Drude theory?
        Args:

        Returns:
            ke: [k_B /(ms)]
        """
        vthe = cls.electron_thermal_velocity(Te)
        τei  = cls.ei_relaxation_time(n_e, n_i, m_i, Z_i, Te, Ti)
        Ce   = cls.electron_heat_capacity(n_e)
        ke = 1/3 * vthe**2 *τei * Ce 
        return ke

    @classmethod
    def ion_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Not Implemented yet
        Args:

        Returns:
            ki: [k_B /(ms)]
        """
        ki = 1e-40
        return ki


    
    @classmethod   
    def ei_relaxation_time(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns Spitzer formula for electron-ion thermalization timescale


        CHECK: Amiguous in paper if electron or ion T relaxation time! 
        from "Electron-ion equilibration in a strongly coupled plasma"
            by A. Ng, P. Celliers, * G. Xu, and A. Forsman
        I think this is \tau_{ei}^e, meaning electron relaxation
        Thus, G = C_e/\tau_{ei} = C_i / ( \tau_{ei} / Z )

        Assumes No. 4 log Λ  in GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            τei relaxation time [s]

        """
        λD = cls.electron_Debye_length(n_e, Te)
        ai = cls.r_WignerSeitz(n_i)
        bmax = np.sqrt(λD**2 + ai**2)
        
        vthe, vthi = cls.electron_thermal_velocity(Te), cls.ion_thermal_velocity(Ti, m_i)
        λdBe = cls.electron_deBroglie_wavelength(n_e, Te)
        r_closest_approach = Z_i* ee**2 / (m_e *vthe)
        bref = np.sqrt(λdBe**2 + r_closest_approach**2)
        
        λ = 0.5*np.log(1+bmax**2/bref**2) # effectively logΛ

        vth_e = k_B*Te/m_e

        unit_conversion = (4*π*ε_0)**2
        τei = unit_conversion* (3 * m_i * m_e) / (4 * np.sqrt(2*π)*n_i*Z_i**2*ee**4*λ ) * ( vthe**2  + vthi**2 )**(3/2)

        return τei

    @classmethod
    def electron_diffusivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns electron thermal diffusivity D_e

        Assumes No. 4 log Λ  in GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            De diffusivity [m^2/s]
        """
        ke = cls.electron_thermal_conductivity(n_e, n_i, m_i, Z_i, Te, Ti)
        Ce = cls.electron_heat_capacity(n_e)
        De = ke/Ce
        return De

    @classmethod
    def ion_diffusivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns ion thermal diffusivity D_i

        Assumes No. 4 log Λ  in GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            Di diffusivity [m^2/s]
        """
        ki = cls.ion_thermal_conductivity(n_e, n_i, m_i, Z_i, Te, Ti)
        Ci = cls.ion_heat_capacity(n_i)
        Di = ki/Ci
        return Di