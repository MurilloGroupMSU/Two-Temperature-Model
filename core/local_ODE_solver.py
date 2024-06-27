# Two Temperature Model 
# Solver class
# Zach Johnson, Michael S. Murillo, Luciano Silvestri

# SI units everywhere

import numpy as np
from .physics import JT_GMS as jt_mod
from .physics import SMT as smt_mod
from .constants import *

from scipy.optimize import root
from scipy.special import lambertw as W0


class LocalModel():
    """
    Implements a local model of a plasma, following the evolution of ions and electrons over time assuming spatial homogeneity.
    

    """

    def __init__(self, Z, m_i, Ti_init, Te_init, ni_init, Zbar_func = None, χ_func = None, transport_model = "SMT", G_rescale = 1):
        self.Ti  = Ti_init
        self.Te  = Te_init 
        self.n_i = ni_init
        self.m_i = m_i
        self.Z   = Z
        self.G_rescale = G_rescale # Multiplies G for analysis reasons, default 1 does nothing

        if transport_model == "SMT":
            self.params = smt_mod
        else:
            self.params = jt_mod

        # If None, use TF and no recombination energy
        if Zbar_func is None:
            self.Zbar_func = lambda Te: self.params.Thomas_Fermi_Zbar(self.Z, self.n_i, Te)
        else:
            self.Zbar_func = Zbar_func

        if χ_func is None:
            self.χ_func = lambda Te : 0
        else:
            self.χ_func = χ_func

        self.Zbar = self.Zbar_func(self.Te)
        self.n_e = self.Zbar*self.n_i

    def G(self, n_e, Zbar, Te, Ti ):
        return self.params.ei_coupling_factor(n_e, Zbar * self.n_i, self.m_i, 1, Te, Ti)

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

    def get_Ti_from_Ei(self, Eki):
        return 2/3 * Eki/self.n_i/k_B  

    def get_Te(self, Ek_e, Zbar_previous, Zbar=None):
        """
        Finds the temperature of the electrons at a given point using the amount of kinetic energy there, plus what is released from recombination
        Zbar_previous is defined as Zbar(t_{i-1},x_i)
        Erec(t_i, x_i) = (Zbar(t_i, x_i)- Zbar(t_{i-1},x_i - v Δt)) * n_i(x_i,t_i) * χ0(x_i, t_i)
        Zbar_back_a_step = Zbar(t_{i-1},x_i- v(x_i)  Δt) ~ Zbar(t_{i-1},x_i)  - v Δt d/dx Zbar(t_{i-1},x_i) is what we actually need
        """
        Erec_density = lambda Zbar, Te: (Zbar_previous*self.χ_func(self.Te) - Zbar*self.χ_func(Te))*self.n_i # positive means energy is released into electrons, since ionization decreased
        Te_func = lambda n_e, Te: 2/3 * (Ek_e  + Erec_density(n_e/self.n_i, Te))/(n_e*k_B)
        # if np.all(Zbar==None):
        
        f_to_min_Te = lambda Te: np.abs(Te - Te_func(self.n_i*self.Zbar_func(Te), Te) )
        sol = root(f_to_min_Te, self.Te , options={'maxfev':2000})
        Te = sol.x
        if sol.success == False:
            print("WARNING, Zbar minimizer not converge.")
            print(sol)
            bad_ind = np.argmax(sol.fun)
            print(f"Worst point is at index = {bad_ind}, r={self.grid.r[bad_ind]}, fun = {sol.fun[bad_ind]}, Te = {self.Te[bad_ind]} ")

        success = sol.success
        Zbar = self.Zbar_func(Te)
        # else:
        #     Te = Te_func(Zbar*n_i)
        #     success = True
        # print(f"Erec_density/Ek_e = {Erec_density(Zbar, Te)/Ek_e}")
        # print(f"Erec_density = {Erec_density(Zbar)}")
        # print(f"From get_Te: Zbar={Zbar}, ΔZbar={Zbar_back_a_step-Zbar}, Ek_e/Erec_density ={Ek_e/Erec_density(Zbar)}")        
        return Te[0], Zbar[0], Erec_density(Zbar, Te)[0], success

    def solve_ode(self, dt=1e-12, tmax=10e-9, χ0_factor= 1 ):
        """
        Solves TTM model using finite-volume method. 
        Args:
            None
        Returns:
            None
        """
        self.dt = dt
        self.tmax = tmax
        self.t_list = np.arange(0, self.tmax, self.dt) 

        self.t_saved_list = [0]
        self.Te_list, self.Ti_list = [self.Te], [self.Ti]
        self.Ek_e_list = [ self.get_Ek_e() ]
        self.n_e_list = [self.n_e]
        self.G_list = [self.G_rescale * self.G(self.n_e, self.Zbar, self.Te,self.Ti) ]
        self.ΔEe_recombination_list = []
        self.ΔEe_ion_equil_list = []

        for i, t in enumerate(self.t_list[:-1]):
            # print(f"Step {i}, Zbar={self.Zbar}")
            # Calculate new temperatures using explicit Euler method, finite volume, and relaxation
            
            G  = self.G_rescale * self.G(self.n_e, self.Zbar, self.Te,self.Ti) 
            # Ce = self.params.electron_heat_capacity(self.n_e, self.Te)
            # Ci = self.params.ion_heat_capacity(self.n_i, self.Ti)
            # ke = self.params.electron_thermal_conductivity(self.n_e, self.n_i, self.m_i, self.Zbar, self.Te, self.Ti)
            # ki = self.params.ion_thermal_conductivity(self.n_e, self.n_i, self.m_i, self.Zbar, self.Te, self.Ti)

            self.Ek_i = self.get_Ek_i()
            self.Ek_e = self.get_Ek_e()
            self.ρ =  self.n_i * self.m_i

            # Energy equation
            # ion
            self.ΔEk_e_from_Gei = -self.dt * G*(self.Te - self.Ti)
            self.Ek_i += - self.ΔEk_e_from_Gei 
            # electron
            self.Ek_e += + self.ΔEk_e_from_Gei
                    
            # Update densities with EOS
            self.Ti = self.get_Ti_from_Ei(self.Ek_i)
            
            self.Te, self.Zbar, self.Erecombination, success = self.get_Te(self.Ek_e, self.Zbar ) # Finds temperature including from recombination heating
            self.n_e = self.n_i*self.Zbar # enforce quasineutrality
            self.Ek_e = self.get_Ek_e() # Effectively adds recombination heating to Ek_e
        
            if np.any(np.isnan(self.Ti))==True:
                print("Error: Returning nan's. Decrease dt?` ")
            # Make list of temperature profiles 
            if i%1==0:
                self.t_saved_list.append(t)
                self.Te_list.append(self.Te); self.Ti_list.append(self.Ti)
                self.Ek_e_list.append(self.Ek_e)
                self.n_e_list.append(self.n_e)
                self.Zbar_list = np.array(self.n_e_list)/self.n_i
                self.ΔEe_recombination_list.append(self.Erecombination)
                self.ΔEe_ion_equil_list.append(self.ΔEk_e_from_Gei)
                self.G_list.append(G)

            if success is False:
                print("Fail Zbar, breaking.")
                break


        self.t_saved_list = np.array(self.t_saved_list)
        self.Te_list = np.array(self.Te_list)
        self.Ti_list = np.array(self.Ti_list)
        self.Ek_e_list = np.array(self.Ek_e_list)
        self.n_e_list = np.array(self.n_e_list)
        self.Zbar_list = np.array(self.Zbar_list)
        self.G_list  = np.array(self.G_list)
        self.ΔEe_recombination_list = np.array(self.ΔEe_recombination_list)
        self.ΔEe_ion_equil_list = np.array(self.ΔEe_ion_equil_list)
