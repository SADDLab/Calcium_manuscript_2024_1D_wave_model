"""
In this code we use the model with a sink to vary thw parameters Vsink,max and KM within speciied ranges together. 
We compute the wave velocity, distance travelled by the wave and the R2 value of a linear model fit to the activation 
time of calcium channels. 
"""

import numpy as np
import scipy.stats as st
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numba
from smt.sampling_methods import LHS
import os
import time

@numba.njit
def release_calcium(t, t_i, sigma, tau, dSites):
    if t >= t_i and t <= t_i + tau:
        return sigma
    else:
        return 0
    
@numba.njit
def source_calcium(t, src_strength, src_rise_time):
    if t <= src_rise_time:
        return src_strength
    else:
        return 0
    
@numba.njit
def cicr_ros_diff(Ca, dx, dt, time_steps, theta):
    ### Defining model parameters
    src_center_strength = theta[0]
    src_center_actTime = theta[1]
    sigma = theta[2] 
    tau = theta[3] 
    d_sites = theta[4]
    Ca_threshold = theta[5]
    km = theta[6]
    nsink = theta[7]
    vmax = theta[8]
    
    D = 1200
    
    Ca_basal = 0.1
    Ca = Ca + Ca_basal
    
    ### Generating points in the grid where point osurces of CNGC are
    L = 700
    points2 = np.arange(350+d_sites, 600, d_sites)
    points1List = []
    temp = 350
    for i in range(len(points2)):
        temp = temp - d_sites
        points1List.append(temp)
    points1Numpy = np.array(points1List)
    points1 = points1Numpy[::-1]
    x_sites = np.concatenate((points1, points2))
    fired = np.zeros(len(x_sites), dtype=numba.boolean) ## np.zeros(len(x_sites), dtype=bool)
    fired_init_times = np.zeros(len(x_sites)) 
    
    ### 
    idx_center = int(350 / dx)
    
    ### Solving reaction diffusion equations
    for t in range(1, time_steps): 
        ### Solving teh diffusion
        Ca[1:-1, t] = Ca[1:-1, t-1] + D * dt / dx**2 * (Ca[2:, t-1] - 2 * Ca[1:-1, t-1] + Ca[:-2, t-1])
        Ca[0, t] = Ca_basal
        Ca[-1, t] = Ca_basal
        
        ### Sink
        sink_term = (vmax * ((Ca[:, t-1] - Ca_basal)**nsink)) / (km**nsink + (Ca[:, t-1]- Ca_basal)**nsink)
        Ca[:, t] -= dt * sink_term 
        
        ### Defining source at teh center 
        Ca[idx_center,t]+= dt*source_calcium(t*dt, src_center_strength, src_center_actTime)
        
        ### Modeling CICR
        for i, x_i in enumerate(x_sites):
            idx = int(x_i / dx)
            if not fired[i] and Ca[idx, t-1] >= Ca_threshold:
                fired[i] = True  
                fired_init_times[i] = t * dt    
            if fired[i]:
                Ca[idx, t] += dt* release_calcium(t*dt, fired_init_times[i], sigma, tau, d_sites)
                
    return Ca, x_sites, fired_init_times, fired, points2

def plotting_functions(Ca, x_sites, fired_init_times, ctr):
    
    Ca_size_in_gb = Ca.nbytes / (1024 ** 3) 
    memory_threshold_gb = 1.0
    
    if np.all(np.isfinite(Ca)):
        ### Creating directory to save files
        output_folder = 'output_data_analysis'
        os.makedirs(output_folder, exist_ok=True)
        
        ### Making eth figure
        fig, axs = plt.subplots(1, 1, figsize=(2.8, 3), dpi=200)
        Ca_sliced = Ca[:, ::100]
        
        half_index = Ca_sliced.shape[0] // 2
        Ca_sliced_half = Ca_sliced[half_index:, :]
        
        im1 = axs.imshow(Ca_sliced_half, extent=[0, 4, 0, 350], aspect='auto', origin='lower', interpolation='nearest',
                         vmin = 0.1, vmax = 0.45
                        )
        fig.colorbar(im1, ax=axs, label=r'$[Ca^{2+}]$ ($\mu M$)')
        
        time_points1 = np.linspace(0, 3.85, 100)
        y_positions1 = (1.35-0.27)* 60 * time_points1 ##+ 350
        axs.plot(time_points1, y_positions1, 'whitesmoke', linewidth=1.5, linestyle = '-')
        time_points2 = np.linspace(0, 2.57, 100)
        y_positions2 = (1.35+0.27)* 60 * time_points2 ##+ 350
        axs.plot(time_points2, y_positions2, 'whitesmoke', linewidth=1.5, linestyle = '-')
        
        axs.hlines(y=50, xmin=0, xmax=4, color='whitesmoke', linewidth=1.5, linestyle='-')
        axs.hlines(y=200, xmin=0, xmax=4, color='whitesmoke', linewidth=1.5, linestyle='-')
        
        #fired_init_times_base = np.load('fired_init_times_base.npy') #= fired_init_times.copy()
        #fired_init_times_base = fired_init_times.copy()
        #np.save('fired_init_times_base.npy', fired_init_times_base)
        
        #axs.scatter(fired_init_times_base, x_sites-350, c='tab:red', s=10)
        axs.scatter(fired_init_times, x_sites-350, c='tab:pink', s=10)
        
        axs.set_ylim(0, 350)
        axs.set_xlim(0, 4)
        axs.set_xlabel('time (min)')
        axs.set_ylabel('position (µm)')        
        filename = os.path.join(output_folder, 'Figure_' + str(ctr) + '.svg')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close(fig)
        
    else:
        print('Exceeds memory use.')

        
def main_model_1(theta, ctr):
    
    dx = 0.2 
    T = 4  
    dt = 0.00001  
    time_steps = int(T / dt)
    L = 700
    x = np.arange(0, L, dx)
    n = len(x)
    Ca = np.zeros((n, time_steps))
    
    ### Solving the CICR model
    Ca, x_sites, fired_init_times, fired, points2 = cicr_ros_diff(Ca, dx, dt, time_steps, theta) 
    #tau = theta[3]
    #plot_firing_and_activity_times(x_sites, fired_init_times, tau, ctr)
    #plotting_functions(Ca, x_sites, fired_init_times, ctr)
    indices = np.where(np.isin(x_sites, points2))[0]
    filtered_fired = fired[indices]
    filtered_fired_init_times = fired_init_times[indices]
    
    ### Plotting functions
    if np.max(np.max(Ca)) < 100:
        print('I am here')
        fired_indices = np.where(filtered_fired)[0]
        if len(fired_indices) > 1:
            first_fired_index = fired_indices[np.argmin(filtered_fired_init_times[fired_indices])]
            last_fired_index = fired_indices[np.argmax(filtered_fired_init_times[fired_indices])]
            middle_fired_index = fired_indices[len(fired_indices) // 2]
            
            total_distance = points2[last_fired_index] - points2[first_fired_index]
            total_time = filtered_fired_init_times[last_fired_index] - filtered_fired_init_times[first_fired_index]
            average_velocity = total_distance / total_time if total_time != 0 else 10000
            
            yVar = points2[fired_indices]
            xVar = filtered_fired_init_times[fired_indices]
            slope, intercept, r_value, p_value, std_err = stats.linregress(xVar, yVar)
            y_pred = intercept + slope * xVar
            ss_res = np.sum((yVar - y_pred) ** 2) 
            ss_tot = np.sum((yVar - np.mean(yVar)) ** 2)  
            r_squared = 1 - (ss_res / ss_tot)
            print('R-squared: '+str(r_squared))
            
            """
            ### Uncomment this to enable plotting
            try:
                plotting_functions(Ca, x_sites, fired_init_times, ctr)
            except Exception as e:
                print('Not succesful in saving this one')"""
        else:
            average_velocity = 0
            total_distance = 0
            
    else:
        average_velocity = 0
        total_distance = 0
    
    return average_velocity, total_distance, r_squared


if __name__ == '__main__':
    var1_range = np.linspace(0, 0.3, 20) ## Index 8, vmax
    var2_range = np.linspace(0.05, 0.2, 10) ## Index 6, km
    theta0 = np.array([2.00270355e+02, 1.95119197e-01, 1.50008776e+00, 4.18697145e-01,5.69583333e-01, 2.05740000e-01, 0.15, 2, 0.25])
    theta0[2] = 1.4*theta0[2]

    r_squared_array = np.zeros((len(var1_range), len(var2_range)))
    total_distance_array = np.zeros((len(var1_range), len(var2_range)))
    average_velocity_array = np.zeros((len(var1_range), len(var2_range)))
    
    k = 0
    
    for i, var1 in enumerate(var1_range):
        for j, var2 in enumerate(var2_range):
            k = k + 1

            theta = theta0.copy()
            theta[6] = var2  
            theta[8] = var1
            
            avg_velocity, total_distance, r_squared = main_model_1(theta, ctr=0)
            
            r_squared_array[i, j] = r_squared
            total_distance_array[i, j] = total_distance
            average_velocity_array[i, j] = avg_velocity
            
            np.save('r_squared_array.npy', r_squared_array)
            np.save('total_distance_array.npy', total_distance_array)
            np.save('average_velocity_arrayy.npy', average_velocity_array)

            print(str(k) + ' jobs complete!')