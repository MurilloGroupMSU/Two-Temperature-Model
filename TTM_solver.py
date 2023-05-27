# Two Temperature Model 
# Solver class
# Zach Johnson, Michael S. Murillo, Luciano Silvestri

# SI units everywhere


import numpy as np
from physics import JT_GMS as jt_mod
from physics import SMT as smt_mod
from constants import *


class TwoTemperatureModel():
    """
    Implements a two temperature model of a plasma as a cylinder
    """

    def __init__(self, Experiment, model = "SMT"):
        self.experiment = Experiment
        self.grid = Experiment.grid

        self.Ti = Experiment.Ti # .copy() or leave as is?
        self.Te = Experiment.Te
        
        self.n_i = Experiment.n_i # .copy() or leave as is?
        self.n_e = Experiment.n_e

        self.Zbar = Experiment.Zbar

        if model == "SMT":
            self.params = smt_mod
        else:
            self.params = jt_mod

    #Gradient Function
    def grad_T(self, T): #Gradient at boundaries. Neumann at 0, cylinder axis. 
        grad_T   = (T - np.roll(T,1))/self.grid.dr
        grad_T[0]=0 
        return grad_T

    def ke(self, n_e, n_i, Zbar, Te, Ti ):
        m_i = self.experiment.m_i
        return self.params.electron_thermal_conductivity( n_e, n_i, m_i, Zbar, Te, Ti)

    def ki(self, n_e, n_i, Zbar, Te, Ti ):
        m_i = self.experiment.m_i
        return self.params.ion_thermal_conductivity( n_e, n_i, m_i, Zbar, Te, Ti)

    def G(self, n_e, n_i, Zbar, Te, Ti ):
        m_i = self.experiment.m_i
        return self.params.ei_coupling_factor(n_e, n_i, m_i, Zbar, Te, Ti)

    def get_tmax(self):
        τ_κ = self.experiment.τei_Equilibration
        return 5*τ_κ
    
    def get_dt(self):
        shortest_timescale_on_dr = np.min([self.experiment.τei_Equilibration, self.experiment.τei_Equilibration, self.experiment.τDiff_e_dr,
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

        print("  Thermalization Times: {0:.1} ps  {0:.1} ps".format(self.experiment.τei_Equilibration*1e12, self.experiment.τie_Equilibration*1e12))

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
            
            ke = self.ke(self.n_e, self.n_i, self.Zbar, self.Te,self.Ti)
            ki = self.ki(self.n_e, self.n_i, self.Zbar, self.Te,self.Ti)
            G  = self.G (self.n_e, self.n_i, self.Zbar, self.Te,self.Ti) 
            Ce = self.params.electron_heat_capacity(self.n_e, self.Te)[:-1]
            Ci = self.params.ion_heat_capacity(self.n_i, self.Ti)[:-1]


            Te_flux = ke * 2*π*self.grid.r * self.grad_T(self.Te) #Cylindrical flux first order
            Ti_flux = ki * 2*π*self.grid.r * self.grad_T(self.Ti)
            
            # Note- γ is a CONSTANT!!! Broken to make it a function of r right now!
            net_flux_e = (Te_flux[1:] - Te_flux[:-1]) 
            Te_new = self.Te[:-1] + self.dt/Ce * (
                    net_flux_e/ self.grid.cell_volumes - (G*(self.Te - self.Ti))[:-1]
                    )

            net_flux_i = (Ti_flux[1:] - Ti_flux[:-1]) 
            Ti_new = self.Ti[:-1] + self.dt/Ci * (
                    net_flux_i/ self.grid.cell_volumes + (G *(self.Te - self.Ti))[:-1]
                    )

            # Update temperatures
            self.Te[:-1] = Te_new
            self.Ti[:-1] = Ti_new
            self.experiment.make_n_e_profile()
            # Make list of temperature profiles 
            self.Te_list.append(self.Te.copy()); self.Ti_list.append(self.Ti.copy())
            
            

                
    # def plot_
    # fig, ax = plt.subplots(figsize=(14,10),facecolor='w')

# #Plot temperature profiles at intermediate times
            # if plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
            #     ax.plot(cell_centers*1e6, Te[:-1]*1e-3, '--', color=colors[plot_idx], label=f"$T_e$: t={t:.1e} [s]")
            #     ax.plot(cell_centers*1e6, Ti[:-1]*1e-3, '-' , color=colors[plot_idx], label=f"$T_i$: t={t:.1e} [s]")
            #     plot_idx += 1
