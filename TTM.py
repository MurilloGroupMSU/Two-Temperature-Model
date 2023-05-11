# Two Temperature Model Solver
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
        Let's do that.
        """



class TwoTemperatureModel():
    """
    Implements a two temperature model of a plasma as a cylinder
    """

    def __init__(self, Experiment):
        self.experiment = Experiment
        self.grid = Experiment.grid

        self.Ti = Experiment.Ti # .copy() or leave as is?
        self.Te = Experiment.Te
        
        self.n_i = Experiment.n_i # .copy() or leave as is?
        self.n_e = Experiment.n_e

    #Gradient Function
    def grad_T(self, T): #Gradient at boundaries. Neumann at 0, cylinder axis. 
        grad_T   = (T - np.roll(T,1))/self.grid.dr
        grad_T[0]=0 
        return grad_T

    def ke(self, n_e, n_i, Te, Ti ):
        m_i, Z_i = self.experiment.m_i, self.experiment.Z_i 
        return params.electron_thermal_conductivity( n_e, n_i, m_i, Z_i, Te, Ti)

    def ki(self, n_e, n_i, Te, Ti ):
        return 0

    def G(self, n_e, n_i, Te, Ti ):
        m_i, Z_i = self.experiment.m_i, self.experiment.Z_i 
        return params.ei_coupling_factor(n_e, n_i, m_i, Z_i, Te, Ti)

    def get_tmax(self):
        τ_κ = self.experiment.τei_Equilibration
        return 5*τ_κ
    
    def get_dt(self):
        shortest_timescale_on_dr = np.min([self.experiment.τei_Equilibration, self.experiment.τDiff_e_dr,
                                           self.experiment.τDiff_e_dr]) 
        
        return 1e-1*shortest_timescale_on_dr

    def make_times(self, dt=None, tmax=None):
        if dt == None:
            self.dt = self.get_dt()
        else:
            self.dt = dt
        if tmax == None:
            self.tmax = self.get_tmax()   # Maximum simulation time [s]
        else:
            self.tmax = tmax
        self.t_list = np.arange(0, self.tmax, self.dt) 

        # print("\nTotal time: {0:.1e} ns,  dt = {1:.1e} ps".format(1e9*self.tmax, 1e12*self.dt))
    
    def print_timescales(self):
        print("\nSimulation time: {0:.1e} ns,  dt = {1:.1e} ps, steps = {2}".format(1e9*self.tmax, 1e12*self.dt, len(self.t_list)))
        print("  Diffusion time (r_max): e:{0:.1e} ns, i:{1:.1e} ns ".format(1e9*self.experiment.τDiff_e_rmax, 1e9*self.experiment.τDiff_i_rmax))
        print("  Diffusion time (dr): e:{0:.1e} ns, i:{1:.1e} ns ".format(1e9*self.experiment.τDiff_e_dr,1e9*self.experiment.τDiff_i_dr))

        print("  Thermalization Time: {0:.1} ps".format(self.experiment.τei_Equilibration*1e12))

    def solve_TTM(self, dt=None, tmax=None):
        """
        Solves TTM model using finite-volume method. 
        Args:
            None
        Returns:
            None
        """
        self.make_times(dt=dt, tmax=tmax)

        self.Te_list, self.Ti_list = [self.Te.copy()], [self.Ti.copy()]
        
        for t in self.t_list[:-1]:
            # Calculate new temperatures using explicit Euler method, finite volume, and relaxation
            
            ke = self.ke(self.n_e, self.n_i, self.Te,self.Ti)
            ki = self.ki(self.n_e, self.n_i, self.Te,self.Ti)
            G  = self.G (self.n_e, self.n_i, self.Te,self.Ti)[:-1]
            Ce = params.electron_heat_capacity(self.n_e, self.Te)[:-1]
            Ci = params.ion_heat_capacity(self.n_i)[:-1]

            Te_flux = ke * 2*π*self.grid.r * self.grad_T(self.Te) #Cylindrical flux first order
            Ti_flux = ki * 2*π*self.grid.r * self.grad_T(self.Ti)
            
            
            # Note- γ is a CONSTANT!!! Broken to make it a function of r right now!
            net_flux_e = (Te_flux[1:] - Te_flux[:-1]) 
            Te_new = self.Te[:-1] + self.dt/Ce * (
                    net_flux_e/ self.grid.cell_volumes - G *(self.Te - self.Ti)[:-1]
                    )

            net_flux_i = (Ti_flux[1:] - Ti_flux[:-1]) 
            Ti_new = self.Ti[:-1] + self.dt/Ci * (
                    net_flux_i/ self.grid.cell_volumes + G *(self.Te - self.Ti)[:-1]
                    )

            # Update temperatures
            self.Te[:-1] = Te_new
            self.Ti[:-1] = Ti_new
            # Make list of temperature profiles 
            self.Te_list.append(self.Te.copy()); self.Ti_list.append(self.Ti.copy())
            
            

                
    # def plot_
    # fig, ax = plt.subplots(figsize=(14,10),facecolor='w')

# #Plot temperature profiles at intermediate times
            # if plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
            #     ax.plot(cell_centers*1e6, Te[:-1]*1e-3, '--', color=colors[plot_idx], label=f"$T_e$: t={t:.1e} [s]")
            #     ax.plot(cell_centers*1e6, Ti[:-1]*1e-3, '-' , color=colors[plot_idx], label=f"$T_i$: t={t:.1e} [s]")
            #     plot_idx += 1
