# Two Temperature Model 
# Experimental Setup
# Zach Johnson, Michael S. Murillo, Luciano Silvestri

# SI units everywhere


import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


from physics import JT_GMS as jt_mod
from physics import SMT as smt_mod
from constants import *


import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from scipy.optimize import root, minimize

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

    def __init__(self, grid, n0, Z, A, Te_experiment_initial, Te_full_width_at_half_maximum,
                Ti_experimental_initial = None, gas_name='Argon', model = "SMT", electron_temperature_model='lorentz', ion_temperature_model = 'electron', ion_temperature_file = None,
                Te_experiment_is_peak=False, super_gaussian_power=1):
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
        self.Te_full_width_at_half_maximum = Te_full_width_at_half_maximum
        self.Te_experiment_is_peak = Te_experiment_is_peak

        if ion_temperature_model == 'gaussian':
            self.Ti_exp_init = Ti_experiment_initial
        self.ion_temperature_model = ion_temperature_model
        self.electron_temperature_model  = electron_temperature_model
        self.ion_temperature_file = ion_temperature_file 
        self.n0 = n0 # Density- may change this variable later
        self.super_gaussian_power = super_gaussian_power

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
        if self.Te_experiment_is_peak==True:
            self.Te_peak = self.Te_exp_init            
        else:
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

        # print("Te_max = {0:.3e} K".format(self.Te_peak))
        self.Te = self.T_distribution(self.Te_peak)
        

    def make_lorentz_Te_profile(self):
        r = self.grid.r
        
        self.T_room = 300
        
        Γ = self.Te_full_width_at_half_maximum/2
        self.T_distribution = lambda T_max : T_max* (Γ**2 / ( r**2 + Γ**2 ))**(1/4) # 1/4 since really Lorentzian is a fit to Intensity

        self.make_Te_profile()


    def make_gaussian_Te_profile(self): #generic super gaussian, with power=1 corresponding to regular gaussian

        r = self.grid.r
        
        self.T_room = 300
    
        self.T_distribution = lambda T_max: T_max*np.exp(-np.log(2)* ( 4*r**2/self.Te_full_width_at_half_maximum**2)**self.super_gaussian_power )+ self.T_room #Gaussian Laser
        
        self.make_Te_profile()

        
    def make_MD_Ti_profile(self):
        "Needs improvement! Better exptrapolationg to high Z? Exp?"
  
        dih_file = self.ion_temperature_file #"/home/zach/plasma/TTM/data/Xe5bar_DIH_profile_data.txt"
        data = read_csv(dih_file, delim_whitespace=True, header=0 )
        
        def T_DIH_fit(Zbar, T_room, T_0, pow):
            return T_room + T_0 *Zbar**pow

        fit_vals = curve_fit(T_DIH_fit, data['Zbar'], data['Tion[K]'])[0]

        Ti_func = interp1d(data['Zbar'], data['Tion[K]'], bounds_error=False, fill_value='extrapolate')
        self.Ti = T_DIH_fit(self.Zbar, *fit_vals) #np.array(Ti_func(self.Zbar))
        return self.Ti
        

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
            
        self.set_ionization()

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
        return Zbar*(np.exp(-2000**2/Te**2)+1e-5)

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

class Measurements():
    
    def __init__(self, Z, A, r_array, n_e_array, n_i_array, Te_array, Ti_array, R_max = 100e-6, Nx = 100, Nz=500):
        self.Z, self.A = Z, A
        self.r_array  = r_array
        self.n_e_array = n_e_array
        self.n_i_array = n_i_array
        self.Te_array  = Te_array
        self.Ti_array  = Ti_array
        self.Zbar_array= self.n_e_array/self.n_i_array
        
        self.m_i = A*m_p
        
        self.x, self.z = np.linspace(-R_max,R_max, num=Nx), np.linspace(-R_max,R_max, num=Nz)
        self.dx, self.dz = self.x[1]-self.x[0], self.z[1]-self.z[0]
        self.X, self.Z = np.meshgrid(self.x,self.z, indexing='ij')
        self.ρ_grid = np.sqrt(self.X**2 + self.Z**2)
        
        self.make_parameter_grids()
        self.make_spectral_parameters()
        self.fit_Te_with_spectral_Intensity()

    def convert_r_array_to_grid(self, array, fill_value='extrapolate'):
        array_func = interp1d(self.r_array, array, bounds_error=False, fill_value=fill_value)
        grid = array_func(self.ρ_grid)
        return grid

    def make_parameter_grids(self):
        self.n_e_grid = self.convert_r_array_to_grid(self.n_e_array)
        self.n_i_grid = self.convert_r_array_to_grid(self.n_i_array)
        self.Te_grid  = self.convert_r_array_to_grid(self.Te_array)
        self.Ti_grid  = self.convert_r_array_to_grid(self.Ti_array)
        self.Zbar_grid= self.convert_r_array_to_grid(self.Zbar_array)
        
    def make_εeff_grid(self):
        self.ε_grid = np.zeros_like(self.κeff_grid)
        for x_i in range(self.κeff_grid.shape[0]):
            for z_i in range(self.κeff_grid.shape[1]):
                self.ε_grid[x_i, z_i] = np.exp(-np.sum(self.dz*self.κeff_grid[x_i,:z_i]))
        return self.ε_grid

    def make_ε_grid(self):
        self.ε_grid = np.zeros((len(self.x), len(self.z), len(self.ωs)))
        for x_i in range(len(self.x)):
            for z_i in range(len(self.z)):
                self.ε_grid[x_i, z_i, :] = np.exp(-np.sum(self.κ_grid[x_i,:z_i,:]*self.dz,axis=0))
        return self.ε_grid

    def make_eff_parameters(self):
        self.κeff_grid = np.nan_to_num(jt_mod.effective_photon_absorption_coefficient(self.m_i, self.n_i_grid, 
                                                                                       self.n_e_grid, self.Ti_grid, 
                                                                                       self.Te_grid, self.Zbar_grid), nan=1e8)
        self.εeff_grid = self.make_εeff_grid() 
        self.Ieff_grid = self.Te_grid**4*self.εeff_grid
        self.Ieff_unnormalized = np.array([np.sum(self.dz*self.Te_grid[x_i]**4*self.εeff_grid[x_i])/1e16 for x_i in range(len(self.x))])

    def make_spectral_parameters(self):
        self.λs = np.geomspace(0.2*2.897e-3/np.max(self.Te_grid), 2*2.897e-3/np.min(self.Te_grid),num=100 )
        self.ωs = 2*π*c/self.λs
        self.dλ = self.λs[1:]-self.λs[:-1]

        self.κ_grid = jt_mod.photon_absorption_coefficient(self.ωs[np.newaxis,np.newaxis,:], self.m_i,
                                                            self.n_i_grid[:,:,np.newaxis], 
                                                            self.n_e_grid[:,:,np.newaxis], 
                                                            self.Ti_grid[:,:,np.newaxis], 
                                                            self.Te_grid[:,:,np.newaxis], 
                                                            self.Zbar_grid[:,:,np.newaxis])
        self.κ_grid = np.nan_to_num(self.κ_grid, nan=1e10)

        self.Bλ_grid = jt_mod.photon_wavelength_density(self.λs[np.newaxis,np.newaxis,:],self.Te_grid[:,:,np.newaxis])

        self.ε_grid = self.make_ε_grid() 
        self.Iλ_grid = self.Bλ_grid * self.ε_grid

        self.Iλ_unnormalized_of_r = np.array([np.sum(self.Iλ_grid[x_i], axis=0) for x_i in range(len(self.x))])
        self.Iλ_unnormalized = np.sum(self.Iλ_unnormalized_of_r*self.dx, axis=0)
        self.I_unnormalized_of_r = np.sum(self.Iλ_unnormalized_of_r[:,1:]*self.dλ[np.newaxis,:], axis=1)
        
        def gaussian(x, FWHM , P):
            return np.exp(-np.log(2)* ( 4*x**2/FWHM**2)**P )


        FWHM_fit = curve_fit(gaussian, self.x, self.I_unnormalized_of_r/np.max(self.I_unnormalized_of_r), p0=(50e-6,1))
        self.I_of_r_fit = gaussian(self.x, *FWHM_fit[0])
        self.FWHM = FWHM_fit[0][0]
        

    def plot_parameter(self, parameter_grid, label=''):
        plt.figure(figsize=(10, 7))
        contour = plt.contourf(self.X*1e6, self.Z*1e6, parameter_grid, levels=100, cmap='viridis')

        cbar = plt.colorbar(contour)
        cbar.set_label(label, size=20)
        cbar.ax.tick_params(labelsize=20)

        plt.xlabel('x [μm]', fontsize=20)
        plt.ylabel('z [μm]', fontsize=20)

        plt.tick_params(labelsize=20)

        plt.show()
        
    def plot_emissivity_and_intensity(self, cmap = 'viridis'):
        self.make_eff_parameters()
        # Gridspec is now 2x2 with sharp width ratios
        gs = gridspec.GridSpec(2,2,height_ratios=[4,1],width_ratios=[20,1])
        fig = plt.figure(figsize=(8,6))

        # Contour plot axis
        cax = fig.add_subplot(gs[0])

        # Create the contour plot
        contour = cax.contourf(self.X*1e6, self.Z*1e6, self.εeff_grid, levels=100, cmap=cmap)
        cax.set_ylabel('z (μm)', fontsize=20)
        cax.tick_params(labelsize=20)

        # Intensity plot axis
        lax = fig.add_subplot(gs[2], sharex=cax)

        # Create the Intensity plot
        # lax.plot(self.X[:,0]*1e6, Intensity_grid/np.max(Intensity_grid), 'b.')
        lax.plot(self.x*1e6, self.I_of_r_fit, 'b-')
        lax.set_xlabel('x (μm)', fontsize=15)
        lax.set_ylabel('Intensity', fontsize=15)
        lax.tick_params(labelsize=20)
        lax.set_ylim(0,1.1)

        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        
        textstr = "FWHM = {0:.1f} μm".format(self.FWHM*1e6)
        lax.text(0.03, 0.93, textstr, transform=lax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)

        # Make a subplot for the colour bar
        bax = fig.add_subplot(gs[1])

        # Use general colour bar with specific axis given.
        cbar = plt.colorbar(contour, cax=bax)
        cbar.set_label('Emissivity Estimate', size=20)
        cbar.ax.tick_params(labelsize=20)

        plt.tight_layout()
        plt.show()
        
    def fit_Te_with_spectral_Intensity(self):
        self.test_Boltzmann = lambda Te: jt_mod.photon_wavelength_density(self.λs, Te)
        f_to_min = lambda Te: np.linalg.norm( self.Iλ_unnormalized/np.max(self.Iλ_unnormalized) - self.test_Boltzmann(Te)/np.max(self.test_Boltzmann(Te)) )
        λ_peak = self.λs[np.argmax(self.Iλ_unnormalized/np.max(self.Iλ_unnormalized))]
        T_peak = 2.897e-3/λ_peak
        sol = minimize(f_to_min, T_peak)
        self.Te_fit = float(sol.x)
        self.spectral_intensity_fit = self.test_Boltzmann(self.Te_fit)
        
    def plot_spectral_Intensity(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.λs*1e9, self.Iλ_unnormalized/np.max(self.Iλ_unnormalized),'.',  label="Integrated over profile")
        plt.plot(self.λs*1e9, self.spectral_intensity_fit/np.max(self.spectral_intensity_fit), label="Te={0:.2f} kK".format(self.Te_fit*1e-3))

        plt.xlabel('λ [nm]', fontsize=20)
        plt.ylabel('Normalized Intensity', fontsize=20)
        plt.xscale('log')
        plt.legend(fontsize=12)
        plt.tick_params(labelsize=20)
        plt.ylim(0,1.1)

        plt.show()