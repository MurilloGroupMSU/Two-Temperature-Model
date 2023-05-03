# Two Temperature Model Solver
# Michael S. Murillo, Zach Johnson

# SI units everywhere



import numpy as np



class Experiment():

    def __init__(self, n0, Z, A, Te_initial, Ti_initial,laser_width, gas_name='Argon'):
        """
        Args:
            n0: Density of gas [1/m^3] 
            Z : Atomic number of gas
            A : Weight of gas
            laser_width: approximate width of laser
            gas_name: ...

        """
        self.laser_width = laser_width
        self.Z = Z
        self.A = A



class TwoTemperatureModel():
    """
    Implements a two temperature model of a plasma as a cylinder
    """

    def __init__(self, Experiment):
        self.experiment = Experiment


    #Gradient Function
    def grad_T(self, T): #Gradient at boundaries. Neumann at 0, cylinder axis. 
        grad_T   = (T - np.roll(T,1))/self.dr
        grad_T[0]=0 
        return grad_T

    def solve_TTM(self):

        self.t_list, self.Te_list, self.Ti_list = [0], [Te.copy()], [Ti.copy()]
        
        while t < t_max:
            # Calculate new temperatures using explicit Euler method, finite volume, and relaxation
            Te_flux = ke(Te) * 2*π*r * grad_T(Te) #Cylindrical flux first order
            Ti_flux = ki(Ti) * 2*π*r * grad_T(Ti)
            
            # Note- γ is a CONSTANT!!! Broken to make it a function of r right now!
            Te_new = Te[:-1] + dt * (
                (Te_flux[1:] - Te_flux[:-1]) / cell_volumes
                - (γ(Te_init,Ti_init) * (Te - Ti))[:-1]
            )/Ce
            
            Ti_new = Ti[:-1] + dt * (
                (Ti_flux[1:] - Ti_flux[:-1]) / cell_volumes
                + (γ(Te_init,Ti_init) *(Te - Ti))[:-1]
            )/Ci

            # Update temperatures
            Te[:-1] = Te_new
            Ti[:-1] = Ti_new
            t += dt
            # Make list of temperature profiles 
            t_list.append(t); Te_list.append(Te.copy()); Ti_list.append(Ti.copy())
            
            

                
    # def plot_
    # fig, ax = plt.subplots(figsize=(14,10),facecolor='w')

# #Plot temperature profiles at intermediate times
            # if plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
            #     ax.plot(cell_centers*1e6, Te[:-1]*1e-3, '--', color=colors[plot_idx], label=f"$T_e$: t={t:.1e} [s]")
            #     ax.plot(cell_centers*1e6, Ti[:-1]*1e-3, '-' , color=colors[plot_idx], label=f"$T_i$: t={t:.1e} [s]")
            #     plot_idx += 1
