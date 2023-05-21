# Two Temperature Model 
# Experimental Setup
# Michael S. Murillo, Zach Johnson

# SI units everywhere


import numpy as np
from physics import JT_GMS_Physics as params
from constants import *

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

    def __init__(self, grid, n0, Z_i, A, Te_initial, Ti_initial,laser_width, gas_name='Argon'):
        """
        Args:
            n0: Density of gas [1/m^3] 
            Z : Ionization number of gas ## Switch to atomic number, define Zstar????
            A : Weight of gas
            laser_width: approximate width of laser
            gas_name: ...

        """
        self.grid = grid
        self.Z_i = Z_i
        self.A = A
        self.m_i = A * m_p
        self.Te_init = Te_initial
        self.Ti_init = Ti_initial
        self.laser_width = laser_width
        self.n0 = n0 # Density- may change this variable later

        self.make_n_profiles()
        self.make_T_profiles()
        self.make_physical_timescales()

    def make_T_profiles(self):
        """
        Makes an initial temperature profile after laser heating for the electrons and ion.
        Currently assumes measured Temperature is based on bulk average over laser width region
        Args:
            None
        Returns:
            None
        """
        r = self.grid.r
        σ = self.laser_width
        self.Te = self.Te_init*np.exp(-r**2/(2*σ**2)) #Gaussian Laser
        self.Ti = self.Ti_init*np.exp(-r**2/(2*σ**2))
        
        # Rescale so bulk Temperature is the initial one.
        width_index = np.argmin(np.abs(self.grid.r - self.laser_width))
        av_Te = self.grid.integrate_f(self.n_e[:-1]*self.Te[:-1], endpoint=width_index)/self.grid.integrate_f(self.n_e[:-1], endpoint=width_index)
        av_Ti = self.grid.integrate_f(self.n_i[:-1]*self.Ti[:-1], endpoint=width_index)/self.grid.integrate_f(self.n_i[:-1], endpoint=width_index)
        
        self.Te *= self.Te_init/av_Te
        self.Ti *= self.Ti_init/av_Ti

    def make_n_profiles(self):
        """
        Makes an initial temperature profile after laser heating for the electrons and ion
        Args:
            None
        Returns:
            None
        """
        self.n_i = self.n0 * np.ones(self.grid.N)
        self.n_e = self.n_i * self.Z_i
    
    def make_physical_timescales(self):
        """
        Find physical timescales of experiment
        Args: 
            None
        Returns:
            None
        """
        n_e, n_i, m_i, Z_i, Te, Ti = (self.n_e[0], self.n_i[0], self.m_i, 
                                      self.Z_i, self.Te_init, self.Ti_init)

        self.τei_Equilibration = params.ei_relaxation_time(n_e, n_i, m_i, Z_i, Te, Ti) 
        self.τDiff_e_rmax = self.grid.r_max**2 / params.electron_diffusivity(n_e, n_i, m_i, Z_i, Te, Ti)
        self.τDiff_i_rmax = self.grid.r_max**2 / params.ion_diffusivity(n_e, n_i, m_i, Z_i, Te, Ti)
        self.τDiff_e_dr = self.τDiff_e_rmax * (self.grid.dr / self.grid.r_max)**2
        self.τDiff_i_dr = self.τDiff_i_rmax * (self.grid.dr / self.grid.r_max)**2
    
    def print_exp_info(self):
        """
        CURRENTLY INCORRECT Γ?
        """
        print("Γ_ee = {0:.2f}".format(params.Gamma(self.n_e, self.Te, Z=1)[0] ))
        print("Γ_ei = {0:.2f}".format(params.Gamma(self.n_e, self.Te, Z=1)[0] ))
        print("Γ_ii = {0:.2f}".format(params.Gamma(self.n_i, self.Ti, Z=1)[0] ))

