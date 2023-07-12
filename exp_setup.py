# Two Temperature Model 
# Experimental Setup
# Zach Johnson, Michael S. Murillo, Luciano Silvestri

# SI units everywhere


import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit


from physics import JT_GMS as jt_mod
from physics import SMT as smt_mod
from constants import *


from scipy.optimize import root

class Cylindrical_Grid():
    """
    Simple class containing numerical grid parameters for finite volume computation
    """
    def __init__(self, r_max, N=101 ):
        self.r = np.linspace(0, r_max, N) #[m]
        self.r_max = r_max
        self.dr = self.r[1] - self.r[0]
        self.cell_centers = self.r[:-1] + 0.5 * self.dr
        self.cell_volumes = 2*π*(self.r[1:]**2 - self.r[:-1]**2)
        self.N = N

        self.Vol = self.integrate_f(np.ones(N-1))
    
    def integrate_f(self, f, endpoint=-1):
        return np.sum(  (self.cell_volumes*f)[:endpoint] )

class Experiment():
    """
    Contains all of the UCLA related experimental details.
    Initializes all physical parameters according to initial conditions of experiment.
    """

    def __init__(self, grid, n0, Z, A, Te_experiment_initial, Ti_experiment_initial, intensity_full_width_at_half_maximum, temperature_width,
                gas_name='Argon', model = "SMT", electron_temperature_model='lorentz', ion_temperature_model = 'electron', ion_temperature_file = None):
        """
        Args:
            n0: Density of gas [1/m^3] 
            Z : Ionization number of gas ## Switch to atomic number, define Zstar????
            A : Weight of gas
            Te_initial: Bulk measurement of electron temperature
            Ti_initial: Bulk measurement of ion temperature
            laser_width: Full width at half maximum of approximately 
                         gaussian laser (Temperature, power???)
            gas_name: ...

        """
        self.grid = grid
        self.Z = Z # Atomic Number of ion
        self.A = A
        self.m_i = A * m_p
        self.Te_exp_init = Te_experiment_initial
        self.Ti_exp_init = Ti_experiment_initial
        self.intensity_full_width_at_half_maximum = intensity_full_width_at_half_maximum
        # self.temperature_full_width_at_half_maximum = temperature_full_width_at_half_maximum
        self.T_measurement_radius = temperature_width
        self.ion_temperature_model  = ion_temperature_model
        self.electron_temperature_model  = electron_temperature_model
        self.ion_temperature_file = ion_temperature_file 
        self.n0 = n0 # Density- may change this variable later

        if model == "SMT":
            self.params = smt_mod
        else:
            self.params = jt_mod
        
        self.make_n_i_profile()
        self.make_T_profiles()
        self.make_n_e_profile()
        self.make_physical_timescales()

    def get_bulk_T(self, T_profile, n_profile, max_r = None):
        if max_r==None:
            max_r = self.T_measurement_radius   #0.5*self.full_width_at_half_maximum
        
        width_index = np.argmin(np.abs(self.grid.r - max_r))

        return self.grid.integrate_f(n_profile[:-1]*T_profile[:-1], endpoint = width_index)/self.grid.integrate_f(n_profile[:-1], endpoint=width_index)

    def make_gaussian_Ti_profile(self):
        """
        NEEDS WORK
        Zbar, ne <-> Te needs to be self-consistent

        Makes an initial temperature profile after laser heating for the electrons and ion.
        Currently assumes measured Temperature is based on bulk average over laser width region
        Args:
            None
        Returns:
            None
        """
        self.Ti_peak = self.Te_peak*self.Ti_exp_init/self.Te_exp_init
        self.Ti = self.T_distribution(self.Ti_peak)
        print("Using gaussian model for Ti: Ti_max = {0:.3e} K".format(self.Ti_peak))

    def make_Te_profile(self):
        """
        Makes an initial temperature profile after laser heating for the electrons and ion.
        Currently assumes measured Temperature is based on bulk average over laser width region
        Args:
            None
        Returns:
            None
        """
        # Rescale so bulk Temperature is the initial one.
        def ΔT_to_min( T_max ):
            Te_profile = self.T_distribution(T_max)
            ne_profile = self.get_ionization(self.Z, self.n_i, Te_profile)
            bulk_T = self.get_bulk_T(Te_profile, ne_profile)
            ΔT = bulk_T - self.Te_exp_init
            return ΔT

        sol = root(ΔT_to_min, self.Te_exp_init )
        self.Te_peak = float(sol.x)
        
        print("Initial peak T_electron converged: ", sol.x, sol.success, sol.message)
        print("Te_max = {0:.3e} K".format(self.Te_peak))
        self.Te = self.T_distribution(self.Te_peak)
        

    def make_lorentz_Te_profile(self):
        r = self.grid.r
        
        self.T_room = 300
        
        Γ = self.intensity_full_width_at_half_maximum/2
        self.T_distribution = lambda T_max : T_max* (Γ**2 / ( r**2 + Γ**2 ))**(1/4) # 1/4 since really Lorentzian is a fit to Intensity

        self.make_Te_profile()


    def make_gaussian_Te_profile(self):

        r = self.grid.r

        σ = np.sqrt( (self.intensity_full_width_at_half_maximum/2)**2/np.log(2)/2)
        
        self.T_room = 300
        self.T_distribution = lambda T_max: T_max*np.exp(-r**2/(2*σ**2)) + self.T_room #Gaussian Laser
        # self.Ti_distribution = lambda T_max: T_max*np.exp(-r**2/(2*σ**2)) + self.T_room
        
        self.make_Te_profile()

        
    def make_MD_Ti_profile(self):

        def gaussian_Ti(r, T_peak, T_room, σ ):
            return T_room + T_peak*np.exp(-r**2/(2*σ**2))
            
        dih_file = self.ion_temperature_file #"/home/zach/plasma/TTM/data/Xe5bar_DIH_profile_data.txt"
        data = read_csv(dih_file, delim_whitespace=True, header=0 )

        T_peak_fit, T_room_fit, σ_fit = curve_fit(gaussian_Ti, data['r[m]']*1e6, data['Tion[K]']*1e-3 )[0]
        print("Ti fit params: ", T_peak_fit, T_room_fit, σ_fit)
        self.Ti = gaussian_Ti(self.grid.r*1e6, T_peak_fit, T_room_fit, σ_fit)*1e3
        

    def make_T_profiles(self):
        """
        NEEDS WORK
        Zbar, ne <-> Te needs to be self-consistent

        Makes an initial temperature profile after laser heating for the electrons and ion.
        Currently assumes measured Temperature is based on bulk average over laser width region
        Args:
            None
        Returns:
            None
        """
        if self.electron_temperature_model == 'gaussian':
            self.make_gaussian_Te_profile()
        elif self.electron_temperature_model == 'lorentz':
            self.make_lorentz_Te_profile()
            
        if self.ion_temperature_model == 'electron':
            self.make_gaussian_Ti_profile()
        elif self.ion_temperature_model == 'MD':
            self.make_MD_Ti_profile()


    def get_ionization(self, Z, n_i, Te):
        """
        Gets the ionization profile of the ion using TF AA fit.
        Args:
            None
        Returns:
            None
        """
        Zbar = self.params.Thomas_Fermi_Zbar(Z, n_i, Te)
        return Zbar#*(np.exp(-5000**2/Te**2)+1e-5)

    def set_ionization(self):
        """
        Sets the ionization profile of the ion using TF AA fit.
        Args:
            None
        Returns:
            None
        """
        self.Zbar = self.get_ionization(self.Z, self.n_i, self.Te)

    def make_n_i_profile(self):
        """
        Makes an initial temperature profile after laser heating for the ions
        Args:
            None
        Returns:
            None
        """
        self.n_i = self.n0 * np.ones(self.grid.N)
    
    def make_n_e_profile(self):
        """
        Makes an initial temperature profile after laser heating for the electrons 
        Args:
            None
        Returns:
            None
        """
        self.set_ionization()
        self.n_e = self.n_i * self.Zbar #profile, heterogeneous n_e
    
    def make_physical_timescales(self):
        """
        Find physical timescales of experiment
        Args: 
            None
        Returns:
            None
        """
        n_e, n_i, m_i, Zbar, Te, Ti = (self.n_e[0], self.n_i[0], self.m_i, 
                                      self.Zbar[0], self.Te[0], self.Ti[0])

        self.τei_Equilibration, self.τie_Equilibration = self.params.ei_relaxation_times(n_e, n_i, m_i, Zbar, Te, Ti) 
        self.τDiff_e_rmax = self.grid.r_max**2 / self.params.electron_diffusivity(n_e, n_i, m_i, Zbar, Te, Ti)
        self.τDiff_i_rmax = self.grid.r_max**2 / self.params.ion_diffusivity(n_e, n_i, m_i, Zbar, Te, Ti)
        self.τDiff_e_dr = self.τDiff_e_rmax * (self.grid.dr / self.grid.r_max)**2
        self.τDiff_i_dr = self.τDiff_i_rmax * (self.grid.dr / self.grid.r_max)**2
    
    def print_exp_info(self):
        """
        CURRENTLY INCORRECT Γ?
        """
        print("Γ_ee = {0:.2f}".format(params.Gamma(self.n_e, self.Te, Z=1)[0] ))
        print("Γ_ei = {0:.2f}".format(params.Gamma(self.n_e, self.Te, Z=1)[0] ))
        print("Γ_ii = {0:.2f}".format(params.Gamma(self.n_i, self.Ti, Z=1)[0] ))

