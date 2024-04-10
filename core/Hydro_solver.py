# Two Temperature Model 
# Solver class
# Zach Johnson, Michael S. Murillo, Luciano Silvestri

# SI units everywhere


import numpy as np
from physics import JT_GMS as jt_mod
from physics import SMT as smt_mod
from constants import *

from scipy.optimize import root
from scipy.special import lambertw as W0


class HydroModel():
    """
    Implements a two temperature model of a plasma as a cylinder
    """

    def __init__(self, Experiment, model = "SMT", electron_model = 'equilibrium'):
        self.experiment = Experiment
        self.grid = Experiment.grid

        self.Ti = Experiment.Ti # .copy() or leave as is?
        self.Te = Experiment.Te
        
        self.n_i = Experiment.n_i # .copy() or leave as is?
        self.n_e = Experiment.n_e

        self.Zbar = Experiment.Zbar
        self.m_i = self.experiment.m_i
        self.electron_model = electron_model

        self.gamma = 5/3
        if model == "SMT":
            self.params = smt_mod
        else:
            self.params = jt_mod

    #Gradient Function
    def grad(self, f): #Gradient at boundaries. Neumann at 0, cylinder axis. 
        grad_f   = (f - np.roll(f,1))/self.grid.dr
        grad_f[0]=0 
        return grad_f

    def ke(self, n_e, n_i, Zbar, Te, Ti ):
        return self.params.electron_thermal_conductivity( n_e, n_i, self.m_i, Zbar, Te, Ti)

    def ki(self, n_e, n_i, Zbar, Te, Ti ):
        return self.params.ion_thermal_conductivity( n_e, n_i, self.m_i, Zbar, Te, Ti)

    def G(self, n_e, n_i, Zbar, Te, Ti ):
        return self.params.ei_coupling_factor(n_e, n_i, self.m_i, Zbar, Te, Ti)

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

        print("  Thermalization Times: {0:.2e} ps  {0:.2e} ps".format(self.experiment.τei_Equilibration*1e12, self.experiment.τie_Equilibration*1e12))

    def get_P(self):
        """
        Assuming Ideal pressure right now
        """
        return self.get_Pe() + self.get_Pi()

    def get_Pe(self):
        """
        Assuming Ideal pressure right now
        """

        P_e_ideal = k_B*self.n_e * self.Te
        Γee = jt_mod.Gamma(self.n_e, self.Te, Z = 1)
        λ_tot = jt_mod.total_Debye_length(self.n_e, self.n_i, self.Ti, self.Te, self.Zbar)
        ae = jt_mod.r_WignerSeitz(self.n_e)
        
        P_e_ex    = -2/3*π*self.n_e**2 * (ee**2/(4*π*ε_0))**2/(self.Te*k_B)*λ_tot* np.exp(-W0(Γee*ae/λ_tot) )
        return P_e_ideal.real + P_e_ex.real
        
    def get_Pi(self):
        """
        Assuming Ideal pressure right now
        """
        
        P_i_ideal = k_B*self.n_i * self.Ti
        Γii = jt_mod.Gamma(self.n_i, self.Ti, Z=self.Zbar)
        λ_tot = jt_mod.total_Debye_length(self.n_e, self.n_i, self.Ti, self.Te, self.Zbar)
        ai = jt_mod.r_WignerSeitz(self.n_i)
        P_i_ex    = -2/3*π*self.n_i**2 * (self.Zbar**2*ee**2/(4*π*ε_0))**2/(self.Ti*k_B)*λ_tot* np.exp(-W0(Γii*ai/λ_tot) )
        return P_i_ideal.real + P_i_ex.real


    def get_Ek_i(self):
        """
        Thermal kinetic energy density
        """
        return 3/2 * k_B*self.Ti * self.n_i 

    def get_Ek_e(self):
        """
        Thermal kinetic energy density
        """
        return 3/2 * k_B * self.Te * self.n_e 

    def get_Ti_from_Ei(self, Eki, ni):
        return 2/3 * Eki/ni/k_B  

    def get_Te(self, Ek_e, n_i, Zbar_previous, Zbar=None, χ0_factor= 1):
        """
        χ0_factor allows setting recombination heating to zero by χ0_factor=0. χ0_factor=1 does nothing

        Finds the temperature of the electrons at a given point using the amount of kinetic energy there, plus what is released from recombination
        Zbar_previous is defined as Zbar(t_{i-1},x_i)
        Erec(t_i, x_i) = (Zbar(t_i, x_i)- Zbar(t_{i-1},x_i - v Δt)) * n_i(x_i,t_i) * χ0(x_i, t_i)
        Zbar_back_a_step = Zbar(t_{i-1},x_i- v(x_i)  Δt) ~ Zbar(t_{i-1},x_i)  - v Δt d/dx Zbar(t_{i-1},x_i) is what we actually need
        """
        Zbar_back_a_step = Zbar_previous - self.v*self.dt*np.gradient(Zbar_previous, self.grid.r)
        Erec_density = lambda Zbar, Te: (Zbar_back_a_step - Zbar)*n_i*self.experiment.χ0_func(n_i, Te)*χ0_factor # positive means energy is released into electrons, since ionization decreased
        Te_func = lambda n_e, Te: 2/3 * (Ek_e  + Erec_density(n_e/n_i, Te))/(n_e*k_B)
        if np.all(Zbar==None):
            Zbar_func = lambda Te: self.experiment.Zbar_func(n_i, Te)
            f_to_min_Te = lambda Te: np.abs(Te - Te_func(n_i*Zbar_func(Te), Te) )
            sol = root(f_to_min_Te, self.Te , options={'maxfev':2000})
            Te = sol.x
            if sol.success == False:
                print("WARNING, Zbar minimizer not converge.")
                print(sol)
                bad_ind = np.argmax(sol.fun)
                print(f"Worst point is at index = {bad_ind}, r={self.grid.r[bad_ind]}, fun = {sol.fun[bad_ind]}, Te = {self.Te[bad_ind]} ")


            success = sol.success
            Zbar = Zbar_func(Te)
        else:
            Te = Te_func(Zbar*n_i)
            success = True
        # print(f"Erec_density/Ek_e = {Erec_density(Zbar, Te)/Ek_e}")
        # print(f"Erec_density = {Erec_density(Zbar)}")
        # print(f"From get_Te: Zbar={Zbar}, ΔZbar={Zbar_back_a_step-Zbar}, Ek_e/Erec_density ={Ek_e/Erec_density(Zbar)}")        
        return Te, Zbar, Erec_density(Zbar, Te), success

    def get_FWHM(self):
        FWHM_index = np.argmin(np.abs(self.Te - 0.5*self.Te[0]))
        FWHM = 2*self.grid.r[FWHM_index]
        return FWHM


    def solve_hydro(self, dt=None, tmax=None, χ0_factor= 1 ):
        """
        Solves TTM model using finite-volume method. 
        Args:
            None
        Returns:
            None
        """
        self.make_times(dt=dt, tmax=tmax)

        self.t_saved_list = []
        self.Te_list, self.Ti_list = [self.Te.copy()], [self.Ti.copy()]
        self.Ek_e_list = [self.get_Ek_e()   ]
        self.P_list = [np.zeros_like(self.n_e)]
        self.n_e_list = [self.n_e.copy()]
        self.n_i_list = [self.n_i.copy()]
        self.FWHM_list = [self.get_FWHM()]
        self.ΔEe_recombination_list = []
        self.ΔEe_ion_equil_list = []
        self.ΔEe_work_list = []

        
        self.v = np.zeros_like(self.n_e) # initialize velocity at zero
        self.v_list = [self.v.copy()]

        for i, t in enumerate(self.t_list[:-1]):
            # print(f"Step {i}, Zbar={self.Zbar}")
            # Calculate new temperatures using explicit Euler method, finite volume, and relaxation
            
            G  = self.G(self.n_e, self.n_i, self.Zbar, self.Te,self.Ti) 
            Ce = self.params.electron_heat_capacity(self.n_e, self.Te)[:-1]
            Ci = self.params.ion_heat_capacity(self.n_i, self.Ti)[:-1]
            ke = self.params.electron_thermal_conductivity(self.n_e, self.n_i, self.m_i, self.Zbar, self.Te, self.Ti)
            ki = self.params.ion_thermal_conductivity(self.n_e, self.n_i, self.m_i, self.Zbar, self.Te, self.Ti)

            self.P = self.get_P()
            self.Ek_i = self.get_Ek_i()
            self.Ek_e = self.get_Ek_e()
            self.ρ =  self.n_i * self.m_i

            P_grad = self.grad(self.P) # Calculate gradient of pressure
            # Energy equation
            # ion
            flux_Ek_i = 2*π*self.grid.r*(  - 0*ki*self.grad(self.Ti) +  self.v * (self.Ek_i + self.get_Pi())  )
            net_flux_Ek_i = flux_Ek_i[1:]  - flux_Ek_i[:-1]
            self.Ek_i[:-1] = self.Ek_i[:-1] + self.dt * ( 
                    - net_flux_Ek_i/self.grid.cell_volumes
                    + (G*(self.Te - self.Ti) )[:-1]
                    )

            # electron
            flux_Ek_e = 2*π*self.grid.r*( - 0*ke*self.grad(self.Te)  + self.v*(  self.Ek_e + self.get_Pe() ))
            net_flux_Ek_e = flux_Ek_e[1:]  - flux_Ek_e[:-1]
            ΔEk_e_from_flux = self.dt * -net_flux_Ek_e/self.grid.cell_volumes
            ΔEk_e_from_Gei  = - self.dt*(G*(self.Te - self.Ti))[:-1]
            self.Ek_e[:-1] = self.Ek_e[:-1] + ΔEk_e_from_flux + ΔEk_e_from_Gei
                    

            # Update densities with continuity equation and quasineutrality
            flux_ni = 2*π*self.grid.r * self.n_i * self.v
            net_flux_ni = flux_ni[1:] - flux_ni[:-1]
            self.n_i[:-1] = self.n_i[:-1] - self.dt * net_flux_ni/self.grid.cell_volumes
            self.Ti = self.get_Ti_from_Ei(self.Ek_i, self.n_i)
            # flux_ne = 2*π*self.grid.r * self.n_e * self.v
            # net_flux_ne = flux_ne[1:] - flux_ne[:-1]
            # self.n_e[:-1] = self.n_e[:-1] - self.dt * net_flux_ne/self.grid.cell_volumes

            if self.electron_model == 'equilibrium':
                self.Te, self.Zbar, self.Erecombination, success = self.get_Te(self.Ek_e, self.n_i, self.Zbar, χ0_factor=χ0_factor) # Finds temperature including from recombination heating
                self.n_e = self.n_i*self.Zbar # enforce quasineutrality
                self.Ek_e = self.get_Ek_e() # Effectively adds recombination heating to Ek_e
            elif self.electron_model == 'fixed_Zbar':
                flux_ne = 2*π*self.grid.r * self.n_e * self.v
                net_flux_ne = flux_ne[1:] - flux_ne[:-1]
                self.n_e[:-1] = self.n_e[:-1] - self.dt * net_flux_ne/self.grid.cell_volumes
                self.Zbar = self.n_e/self.n_i
                self.Te, _ ,_,_ = self.get_Te(self.Ek_e, self.n_i, Zbar = self.Zbar) 

            # Update velocities
            self.v[:-1] = self.v[:-1] + self.dt * (
                 -self.grad(self.P)/self.ρ  - self.v*self.grad(self.v)
                 )[:-1] 
                
            self.v[0]=0

            
            if np.any(np.isnan(self.Ti))==True:
                print("Error: Returning nan's. Decrease dt?` ")
            # Make list of temperature profiles 
            if i%1==0:
                self.t_saved_list.append(t)
                self.Te_list.append(self.Te.copy()); self.Ti_list.append(self.Ti.copy())
                self.Ek_e_list.append(self.Ek_e.copy())
                self.v_list.append(self.v.copy())
                self.P_list.append(self.P.copy())
                self.n_e_list.append(self.n_e.copy())
                self.n_i_list.append(self.n_i.copy())
                self.FWHM_list.append(self.get_FWHM())
                self.Zbar_list = np.array(self.n_e_list)/np.array(self.n_i_list)
                self.ΔEe_recombination_list.append(self.grid.integrate_f(self.Erecombination[:-1]))
                self.ΔEe_ion_equil_list.append(self.grid.integrate_f(ΔEk_e_from_Gei))
                self.ΔEe_work_list.append(self.grid.integrate_f(ΔEk_e_from_flux))

            if success is False:
                print("Fail Zbar, breaking.")
                break