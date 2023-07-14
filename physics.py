
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
    def photon_energy_density(Tγ):
        """
        Returns the integrated-over-frequency energy density of photons at Temperature T
        Args:
            Tγ: Temperature of photons [K] 
        """
        return 4*σ_SB/c * Tγ**4
        
    @staticmethod
    def electron_thermal_velocity(Te):
        """
        Returns the electron thermal velocity, defined as the root mean square of total velocity
        Args: 
            T: Temperature [K]
        Returns:
            vthe: [m/s]
        """
        vth_e = np.sqrt(3*k_B*Te/m_e)
        return vth_e

    @staticmethod
    def ion_thermal_velocity(Ti, m_i):
        """
        Returns the ion thermal velocity, defined as the root mean square of total velocity
        Args: 
            T: Temperature [K]
        Returns:
            vthi: [m/s]
        """
        vth_i = np.sqrt(3*k_B*Ti/m_i)
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

    @staticmethod
    def ion_Debye_length(n_i, Ti, Zi):
        """
        Returns the ion Debye length
        Args: 
            ni: number density [1/m^3]
            Ti: Temperature [K]
            Zi: Ionization number
        """
        λDe = np.sqrt(ε_0*k_B*Ti/((Zi * ee)**2*n_i) )
        return λDe

    @staticmethod
    def total_Debye_length(n_e, n_i, Ti, Te, Zi):
        """
        Returns the electron Debye length
        Args: 
            n: number density [1/m^3]
            T: Temperature [K]
        """
        λD = 1/np.sqrt(  ee**2*n_e/(ε_0*k_B*Te) + (Zi * ee)**2*n_i/(ε_0*k_B*Ti)   )
        return λD

    
    @staticmethod
    def electron_plasma_frequency(n_e):
        return np.sqrt( n_e*ee**2 / (m_e*ε_0) )

    # Class Methods

    @classmethod
    def Thomas_Fermi_wavelength(cls, n_e, Te):
        """
        Thomas Fermi wavelength
        """
        λ_classical = cls.electron_Debye_length(n_e, Te)
        λ_TF = λ_classical / np.sqrt(1 + 4/9 * cls.Theta(n_e, Te))
        return λ_TF


    @classmethod
    def Fermi_velocity(cls, n, m):
        E_F = cls.Fermi_energy(n)
        v_F = np.sqrt(2*E_F/m)
        return v_F

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
    def kappa(cls, n_e, n_i, Te, Ti, Zi):
        '''
        Returns the Debye screening parameter κ
        
        Args: 
            n_e: electron number density [1/m^3]
            n_i: ion number density [1/m^3]
            Te: Electron Temperature [K]
            Ti: Ion Temperature [K]
            Zi: Ion charge

        Returns: 
            κ
        '''
        κ = cls.r_WignerSeitz(n_i) / cls.Thomas_Fermi_wavelength(n_e, Te)
        return κ
    
    @classmethod
    def electron_deBroglie_wavelength(cls, n_e, Te):
        """
        Returns the electron thermal de Broglie Wavelength
        Args: 
            n: number density [1/m^3]
            T: Temperature [K]
        """
        # λdB = np.sqrt(2*π)*hbar/ np.sqrt(m_e*k_B*Te) # Actual deBroglie wavelength
        λdB = hbar /2 /np.sqrt( k_B*Te*m_e ) # from GMS
        return λdB

    @classmethod
    def electron_classical_quantum_wavelength(cls, n_e, Te):
        """
        Modified de Broglie Wavelength
        Idea is σx σp >= hbar/2 from Heisenberg
        σx σv me >= hbar/2
        σv = sqrt( 3/5 vF**2  + 3 k_B Te/me  )
        σx = hbar/2 / sqrt( 3/5 me**2 vF**2  + 3 k_B Te) 
        
        σx_classical = hbar/( 2 sqrt( 3 me k_B Te) )
    
        Thus, to get λdB, we need
        λ = 2*sqrt(6*π)

        Args: 
            n: number density [1/m^3]
            T: Temperature [K]
        """
        # λdB = cls.electron_deBroglie_wavelength(n_e, Te)
        v_F  = cls.Fermi_velocity(n_e, m_e)
        # λ = np.sqrt(6*π) * hbar / np.sqrt( 3*k_B*Te/m_e + 0.6*v_F**2)/m_e
        λ = np.sqrt(3) * hbar / 2 /np.sqrt( 3*k_B*Te/m_e + 0.6*v_F**2)/m_e
        
        # λ = hbar /2 /np.sqrt( k_B*Te*m_e ) # Classical
        return λ

    @staticmethod
    def average_temperature(m1, T1, m2, T2):
        T_avg = (m1 * T2 + m2*T1)/(m1 + m2)
        return T_avg
    
    @staticmethod
    def Thomas_Fermi_Zbar(Z, num_density, T):
        """
        Finite Temperature Thomas Fermi Charge State using 
        R.M. More, "Pressure Ionization, Resonances, and the
        Continuity of Bound and Free States", Adv. in Atomic 
        Mol. Phys., Vol. 21, p. 332 (Table IV).
        
        Args:
            Z: atomic number
            num_density: number density [m^-3]
            T: temperature [K]
        Returns:
            Zbar: Average ionization
        """

        alpha = 14.3139
        beta = 0.6624
        a1 = 0.003323
        a2 = 0.9718
        a3 = 9.26148e-5
        a4 = 3.10165
        b0 = -1.7630
        b1 = 1.43175
        b2 = 0.31546
        c1 = -0.366667
        c2 = 0.983333
        

        convert = num_density*mc_to_cc*1.6726e-24
        R = convert/Z
        T0_in_eV = T*K_to_eV/Z**(4./3.)
        Tf_in_eV = T0_in_eV*K_to_eV/(1 + T0_in_eV)

        A = a1*T0_in_eV**a2 + a3*T0_in_eV**a4
        B = -np.exp(b0 + b1*Tf_in_eV + b2*Tf_in_eV**7)
        C = c1*Tf_in_eV + c2
        Q1 = A*R**B
        Q = (R**C + Q1**C)**(1/C)
        x = alpha*Q**beta
        Zbar = Z*x/(1 + x + np.sqrt(1 + 2.*x))
        
        return Zbar



class Plasma_Formulary_Physics(Physical_Parameters):
    """
    Defines physical quantities using the plasma formulary
    """
    def __init__(self):
        # super().__init__()
        pass


class JT_GMS(Physical_Parameters):
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

    @classmethod
    def electron_heat_capacity(cls, n_e, Te):
        """
        Returns the electron heat capacity
        Free electron model modifcation to classical result
        Args: 
            n_e: electron number density [1/m^3]
        """
        Ce_ideal = 3/2 * k_B * n_e # Electron ideal gas heat capacity
        
        return Ce_ideal

    @staticmethod
    def ion_heat_capacity(n_i, Ti):
        """
        Returns the ion heat capacity
        Ideal gas
        Args: 
            n_i: electron number density [1/m^3]
        Returns:
            Ci: ion specific heat [J / K / m^3]
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
            G: electron-ion coupling [J / K / m^3 / s]
        """
        Ce = cls.electron_heat_capacity(n_e, Te)
        τei, τie = cls.ei_relaxation_times( n_e, n_i, m_i, Z_i, Te, Ti) 

        G = Ce/τei
        return G


    @classmethod
    def electron_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Based on J-T paper, Drude theory?
        Args:

        Returns:
            ke: [k_B /(ms)]
        """
        vthe = cls.electron_thermal_velocity(Te) # root mean square so (3/2 kB Te = 1/2 me vthe^2  for classical)
        τee  = cls.ee_relaxation_time(n_e, n_i, m_i, Z_i, Te, Ti)
        # τei, τie = cls.ei_relaxation_times( n_e, n_i, m_i, Z_i, Te, Ti) 
        Ce   = cls.electron_heat_capacity(n_e, Te)
        ke = 1/3 * vthe**2 *τee * Ce  #if τee >> τei, otherwise some reciprocal addition 
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
    def coulomb_logarithm(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns lnΛ, largely from GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            lnΛ Coulomb log 
        """
        λDb = cls.electron_Debye_length(n_e, Te)
        ae = cls.r_WignerSeitz(n_e)
        bmax = np.sqrt(λDb**2 + ae**2)

        vthe= cls.electron_thermal_velocity(Te)
        λ_spread = cls.electron_classical_quantum_wavelength(n_e, Te)
        r_closest_approach = ee**2 /(4*π*ε_0) / (3/4*k_B*Te) # Assumes 90 degree deflection. Head-on is half this
        bmin = np.sqrt(λ_spread**2 + r_closest_approach**2)
        lnΛ = 0.5*np.log(1 + bmax**2/bmin**2) # effectively logΛ
        lnΛ = np.where(lnΛ==0, 1e-16, lnΛ)

        return lnΛ



    @classmethod   
    def ee_relaxation_time(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        My summary of dimensional analysis τee with model-dependent numerical prefactor
        Multiple possible factors presented, Lee_More agrees best with SMT model

        Assumes No. 4 log Λ  in GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            τee relaxation time [s]

        """
        unit_conversion = (4*π*ε_0)**2
        τee_dimensional = unit_conversion * (k_B*Te)**(3/2)*np.sqrt(m_e) / (n_e* ee**4 ) # Only certain part of τee, based on dimensional analysis

        #Construct best guess for lnΛ
        
        # Possible numerical factors
        c_GMS = 3/(2*np.sqrt(π))          # 10.1103/PhysRevE.65.036418
        c_Lee_More = 3/(4*np.sqrt(2*π))   #Without spurious Zstar?? https://doi.org/10.1063/1.864744
        c_Beckers  = 6*np.sqrt(3)/(16*np.sqrt(π)) # https://doi.org/10.1088/0963-0252/25/3/035010

        lnΛ = cls.coulomb_logarithm(n_e, n_i, m_i, Z_i, Te, Ti)
        τee_prefactor = c_Lee_More / lnΛ

        τee = τee_prefactor * τee_dimensional 
        
        return τee

    @classmethod   
    def ei_relaxation_times(cls, n_e, n_i, m_i, Z_i, Te, Ti):
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
        r_closest_approach = Z_i* ee**2 /(4*π*ε_0) / (m_e *vthe**2)

        bref = np.sqrt(λdBe**2 + r_closest_approach**2)
        
        λ = 0.5*np.log(1+bmax**2/bref**2) # effectively logΛ
        λ = np.where(λ==0, 1e-16, λ)
        
        unit_conversion = (4*π*ε_0)**2

        τei = unit_conversion* ( m_i * m_e) / (4 * np.sqrt(6*π)*n_i*Z_i**2*ee**4*λ ) * ( vthe**2  + vthi**2  )**(3/2)
        τie = unit_conversion* ( m_i * m_e) / (4 * np.sqrt(6*π)*n_e*Z_i**2*ee**4*λ ) * ( vthe**2  + vthi**2  )**(3/2)
        
        return τei, τie

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
        Ce = cls.electron_heat_capacity(n_e, Te)
        De = ke/Ce
        # print(ke, Ce)
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
        Ci = cls.ion_heat_capacity(n_i, Ti)
        Di = ki/Ci
        return Di
    

class Fraley(Physical_Parameters):
    """
    Models based on paper:

    'Thermonuclear burn characteristics of compressed deuterium‐tritium microspheres'
    G. S. Fraley; E. J. Linnebur; R. J. Mason; ... et. al
    """

    def __init__(self):
        """
        """

    @classmethod
    def electron_heat_capacity(cls, n_e, Te):
        """
        Returns the electron heat capacity
        Free electron model modifcation to classical result
        Args: 
            n_e: electron number density [1/m^3]
        """
        Ce_ideal = 3/2 * k_B * n_e # Electron ideal gas heat capacity
        
        return Ce_ideal

    @staticmethod
    def ion_heat_capacity(n_i, Ti):
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
        Ce = cls.electron_heat_capacity(n_e, Te)
        

        G = Ce * veq
        return G

    @classmethod
    def electron_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Based on J-T paper, Drude theory?
        Args:

        Returns:
            ke: [k_B /(ms)]
        """
        vthe = cls.electron_thermal_velocity(Te) # root mean square so (3/2 kB Te = 1/2 me vthe^2  for classical)
        τee  = cls.ee_relaxation_time(n_e, n_i, m_i, Z_i, Te, Ti)
        # τei, τie = cls.ei_relaxation_times( n_e, n_i, m_i, Z_i, Te, Ti) 
        Ce   = cls.electron_heat_capacity(n_e, Te)
        ke = 1/3 * vthe**2 *τee * Ce  #if τee >> τei, otherwise some reciprocal addition 
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
    def coulomb_logarithm(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns lnΛ, largely from GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            lnΛ Coulomb log 
        """
        λDb = cls.electron_Debye_length(n_e, Te)
        ae = cls.r_WignerSeitz(n_e)
        bmax = np.sqrt(λDb**2 + ae**2)

        vthe= cls.electron_thermal_velocity(Te)
        λ_spread = cls.electron_classical_quantum_wavelength(n_e, Te)
        r_closest_approach = ee**2 /(4*π*ε_0) / (3/4*k_B*Te) # Assumes 90 degree deflection. Head-on is half this
        bmin = np.sqrt(λ_spread**2 + r_closest_approach**2)
        lnΛ = 0.5*np.log(1 + bmax**2/bmin**2) # effectively logΛ
        lnΛ = np.where(lnΛ==0, 1e-16, lnΛ)

        return lnΛ


    @classmethod   
    def ee_relaxation_time(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        My summary of dimensional analysis τee with model-dependent numerical prefactor
        Multiple possible factors presented, Lee_More agrees best with SMT model

        Assumes No. 4 log Λ  in GMS paper
        Args:
            n_e: e number density [1/m^3]
            n_i: ion number density [1/m^3]
            m_i: ion mass [kg]
            Z_i: Ion ionization 
            Te: Electron Tempearture [K]
            Ti: Ion Tempearture [K]
        Returns:
            τee relaxation time [s]

        """
        unit_conversion = (4*π*ε_0)**2
        τee_dimensional = unit_conversion * (k_B*Te)**(3/2)*np.sqrt(m_e) / (n_e* ee**4 ) # Only certain part of τee, based on dimensional analysis

        #Construct best guess for lnΛ
        
        # Possible numerical factors
        c_GMS = 3/(2*np.sqrt(π))          # 10.1103/PhysRevE.65.036418
        c_Lee_More = 3/(4*np.sqrt(2*π))   #Without spurious Zstar?? https://doi.org/10.1063/1.864744
        c_Beckers  = 6*np.sqrt(3)/(16*np.sqrt(π)) # https://doi.org/10.1088/0963-0252/25/3/035010

        lnΛ = cls.coulomb_logarithm(n_e, n_i, m_i, Z_i, Te, Ti)
        τee_prefactor = c_Lee_More / lnΛ

        τee = τee_prefactor * τee_dimensional 
        
        return τee

    @classmethod
    def lnΛei(cls,n_e, n_i, m_i, Z_i, Te, Ti):
        A = m_i/m_p
        second_param = np.log(3/2 * ((4*π*ε_0)/ee**2) )**(3/2)*(A/Z*k_B**3*Ti**3/π )
        
        return np.max((1, second_param ))

    @classmethod   
    def ei_relaxation_times(cls, n_e, n_i, m_i, Z_i, Te, Ti):
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
        unit_conversion = (4*π*ε_0)**2
        A = m_i/m_p
        lnΛei
        veq = 8*np.sqrt(2*π)*np.sqrt(m_e)*ee**4/unit_conversion/3 * (Z/A)**2 * lnΛei

        λD = cls.electron_Debye_length(n_e, Te)
        ai = cls.r_WignerSeitz(n_i)
        bmax = np.sqrt(λD**2 + ai**2)
        
        vthe, vthi = cls.electron_thermal_velocity(Te), cls.ion_thermal_velocity(Ti, m_i)
        λdBe = cls.electron_deBroglie_wavelength(n_e, Te)
        r_closest_approach = Z_i* ee**2 /(4*π*ε_0) / (m_e *vthe**2)

        bref = np.sqrt(λdBe**2 + r_closest_approach**2)
        
        λ = 0.5*np.log(1+bmax**2/bref**2) # effectively logΛ
        λ = np.where(λ==0, 1e-16, λ)
        

        τei = unit_conversion* ( m_i * m_e) / (4 * np.sqrt(6*π)*n_i*Z_i**2*ee**4*λ ) * ( vthe**2  + vthi**2  )**(3/2)
        τie = unit_conversion* ( m_i * m_e) / (4 * np.sqrt(6*π)*n_e*Z_i**2*ee**4*λ ) * ( vthe**2  + vthi**2  )**(3/2)
        
        return τei, τie

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
        Ce = cls.electron_heat_capacity(n_e, Te)
        De = ke/Ce
        # print(ke, Ce)
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
        Ci = cls.ion_heat_capacity(n_i, Ti)
        Di = ki/Ci
        return Di
    

class SMT(Physical_Parameters):

    def __init__(self):
        
        pass


    @classmethod
    def effective_screening_length(cls, ne, Te, ni, Ti, Zi):
        """
        Formula 4-5 in e-SMT
        """

        lambda_i = cls.ion_Debye_length(ni, Ti, Zi)
        lambda_e = cls.electron_Debye_length(ne, Te)
        ai = cls.r_WignerSeitz(ni)

        lambda_eff = (1.0/(lambda_i**2 + ai**2) +  1.0/lambda_e**2)**(-1/2)

        return lambda_eff
    
    @classmethod
    def gij_plasma_parameter(cls, ne, Te, ni, Ti, mi, Zi, Tij, Zij):
        """
        Args:
            Tij: av temp of two species i, j
            Zij: Zi*Zj
        """
        
        lambda_eff = cls.effective_screening_length(ne, Te, ni, Ti, Zi)

        gij = Zij * ee**2/( 4 *  π * ε_0)  / (k_B * Tij * lambda_eff)

        return gij
    
    @staticmethod
    def collision_integral(n, m, g):

        if n == 1 and m == 1:
            a1, a2, a3, a4, a5 = 1.4660, -1.7836, 1.4313, -0.55833, 0.061162
            b0, b1, b2, b3, b4 = 0.081033, -0.091336, 0.051760, -0.50026, 0.17044
        elif n == 1 and m == 2:
            a1, a2, a3, a4, a5 = 0.52094, 0.25153, -1.1337, 1.2155, -0.43784
            b0, b1, b2, b3, b4 = 0.20572, -0.16536, 0.061572, -0.12770, 0.066993
        elif n == 1 and m == 3:
            a1, a2, a3, a4, a5 = 0.30346, 0.23739, -0.62167, 0.56110, -0.18046
            b0, b1, b2, b3, b4 = 0.68375, -0.38459, 0.10711, 0.10649, 0.028760
        elif n == 2 and m == 2:
            a1, a2, a3, a4, a5 = 0.85401, -0.22898, -0.60059, 0.80591, -0.30555
            b0, b1, b2, b3, b4 = 0.43475, -0.21147, 0.11116, 0.19665, 0.15195
        
        if not isinstance(g, np.ndarray):
            if g < 1:
                series_a = a1 * g + a2 * g**2 + a3 * g**3 + a4 * g**4 + a5 * g**5
                Knm = - n/4 * np.math.factorial(m - 1)*np.log(series_a)
            else:
                num = b0 + b1*np.log(g) + b2 * np.log(g)**2
                denom = 1 + b3*g + b4 * g**2
                Knm = num/denom
        else:
            Knm = np.zeros_like(g)
            for i, gi in enumerate(g):
                if gi < 1:
                    series_a = a1 * gi + a2 * gi**2 + a3 * gi**3 + a4 * gi**4 + a5 * gi**5
                    Knm[i] = - n/4 * np.math.factorial(m - 1)*np.log(series_a)
                else:
                    num = b0 + b1*np.log(gi) + b2 * np.log(gi)**2
                    denom = 1.0 + b3*gi + b4 * gi**2
                    Knm[i] = num/denom
            
        return Knm

    @staticmethod
    def electron_heat_capacity(n_e, Te):
        """
        Returns the electron heat capacity
        Ideal gas
        Args: 
            n_e: electron number density [1/m^3]
        """
        Ce = 3/2 * k_B * n_e # Electron ideal gas heat capacity
        return Ce

    @staticmethod
    def ion_heat_capacity(n_i, Ti):
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
        Ce = cls.electron_heat_capacity(n_e, Te)
        Ci = cls.ion_heat_capacity(n_i, Ti)
        τei, τie = cls.ei_relaxation_times( n_e, n_i, m_i, Z_i, Te, Ti) 
        
        Gei = Ce/τei
        Gie = Ci/τie
        # if Gei!=Gie:
        #     print("Warning: Gei!= Gie")
        G=Gei 
        return G

    @classmethod
    def electron_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        kee (not full k) from e-SMT paper
        Args:

        Returns:
            ke: [k_B /(ms)]
        """
        num = 75 * k_B * (k_B * Te)**(5/2)
        Tij, Zij = Te, 1
        gee = cls.gij_plasma_parameter(n_e, Te, n_i, Ti, m_i, Z_i, Tij, Zij) #Awkward last arguments
        K22 = cls.collision_integral(2, 2, gee)
        
        charge_factor = ( ee**2/( 4 *  π * ε_0) )**2
        denom = 64 * np.sqrt(π * m_e) *charge_factor*K22 

        ke = num/denom
        return ke

    @classmethod
    def ion_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Same as electron thermal conductivity (with appropriate masses, charges), from either SMT paper
        Args:

        Returns:
            ki: [k_B /(ms)]
        """
        num = 75 * k_B * (k_B * Ti)**(5/2)
        Tij, Zij = Ti, Z_i**2
        gii = cls.gij_plasma_parameter(n_e, Te, n_i, Ti, m_i, Z_i, Tij, Zij)
        K22 = cls.collision_integral(2,2,gii)

        charge_factor = (  Z_i**2 * ee**2/( 4 *  π * ε_0) )**2
        denom = 64 * np.sqrt(π * m_i) *charge_factor*K22 

        ki = num/denom

        return 1e-40

    @classmethod
    def total_thermal_conductivity(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Full thermal conductivity from approximate Eq. 18 of e-SMT
        Args:

        Returns:
            ke: [k_B /(ms)]
        """
        num = 75 * k_B * (k_B * Te)**(5/2)
        Tee, Zee = Te, 1
        Tei, Zei = cls.average_temperature(m_e, Te, m_i, Ti), Z_i
        gee = cls.gij_plasma_parameter(n_e, Te, n_i, Ti, m_i, Z_i, Tee, Zee) #Awkward last arguments
        gei = cls.gij_plasma_parameter(n_e, Te, n_i, Ti, m_i, Z_i, Tei, Zei) #Awkward last arguments
        
        K11_ei = cls.collision_integral(1, 1, gei)
        K12_ei = cls.collision_integral(1, 2, gei)
        K13_ei = cls.collision_integral(1, 3, gei)
        K22_ee = cls.collision_integral(2, 2, gee)

        Λ = Z_i*(25*K11_ei - 20*K12_ei + 4*K13_ei) + np.sqrt(8)*K22_ee
        
        charge_factor = ( ee**2/( 4 *  π * ε_0) )**2
        denom = 16 * np.sqrt(2* π * m_e) *charge_factor* Λ 

        κ = num/denom
        return κ

    @classmethod   
    def ei_relaxation_times(cls, n_e, n_i, m_i, Z_i, Te, Ti):
        """
        Returns SMT formula 21 in Stanton Murillo Phys Plasmas
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
        charge_factor = (  - Z_i * ee**2/( 4 *  π * ε_0) )**2
        mass_factor = np.sqrt(m_e *  m_i)/( m_i + m_e)**(3/2)
        T_avg = cls.average_temperature(m_e, Te, m_i, Ti)
        temp_factor = 1.0/(k_B * T_avg)**(3/2)
        Phi = charge_factor * mass_factor * temp_factor

        Tij = cls.average_temperature(m_e, Te, m_i, Ti)
        Zij = Z_i
        gei = cls.gij_plasma_parameter(n_e, Te, n_i, Ti, m_i, Z_i, Tij, Zij)

        K11 = cls.collision_integral(1,1,gei)

        nu = 128.0/3.0 * np.sqrt(np.pi)/2**(3/2) * Phi * K11
        nu_ei =  n_i * nu
        nu_ie =  n_e * nu

        return 1/nu_ei, 1/nu_ie
    

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
        Ce = cls.electron_heat_capacity(n_e, Te)
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
        Ci = cls.ion_heat_capacity(n_i, Ti)
        Di = ki/Ci
        return Di
    
