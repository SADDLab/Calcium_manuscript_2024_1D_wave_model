# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:12:47 2024
@author: nilay
"""

"""
The code performs a Latin Hypercube Sampling (LHS) of model parameters. 
For each parameter set, we run the model and save the wave statistics
inlcuding distance travelled by wave and wave velocity. 
"""


from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import numpy as np
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
    src_center_strength = 10**theta[0]
    src_center_actTime = 10**theta[1]
    sigma = 10**theta[2] 
    tau = 10**theta[3] 
    d_sites = theta[4]
    Ca_threshold = theta[5] ##0.2, between (0.2-0.4)
    D = 1200
    Ca_basal = 0.1 ### 100 nM = 100e-9 M = 0.1 uM
    Ca = Ca + Ca_basal
    
    ### Generating points in the grid where point sources of CNGC are
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
    #Ca[idx_center,0] = 1
    
    ### Solving reaction diffusion equations
    for t in range(1, time_steps): 
        ### Solving teh diffusion
        Ca[1:-1, t] = Ca[1:-1, t-1] + D * dt / dx**2 * (Ca[2:, t-1] - 2 * Ca[1:-1, t-1] + Ca[:-2, t-1])
        Ca[0, t] = Ca_basal
        Ca[-1, t] = Ca_basal
        
        ### Defining source at the center 
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
        fig, axs = plt.subplots(1, 1, figsize=(3, 3.5), dpi=200)
        Ca_sliced = Ca[:, ::100]
        
        half_index = Ca_sliced.shape[0] // 2
        Ca_sliced_half = Ca_sliced[half_index:, :]
        
        im1 = axs.imshow(Ca_sliced_half, extent=[0, 4, 250, 500], aspect='auto', origin='lower', interpolation='nearest')
        fig.colorbar(im1, ax=axs, label=r'$[Ca^{2+}]$')
        
        time_points1 = np.linspace(0, 2.81, 100)
        y_positions1 = 53.4 * time_points1 + 250
        axs.plot(time_points1, y_positions1, 'r-', linewidth=2)
        time_points2 = np.linspace(0, 1.43, 100)
        y_positions2 = 105.6 * time_points2 + 250
        axs.plot(time_points2, y_positions2, 'r-', linewidth=2)
        
        axs.scatter(fired_init_times, x_sites, c='tab:pink', s=10)
        
        axs.set_ylim(250, 500)
        axs.set_xlabel('Time (min)')
        axs.set_ylabel('Position (µm)')        
        filename = os.path.join(output_folder, 'Figure_' + str(ctr) + '.png')
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
    #plotting_functions(Ca, x_sites, fired_init_times, ctr)
    indices = np.where(np.isin(x_sites, points2))[0]
    filtered_fired = fired[indices]
    filtered_fired_init_times = fired_init_times[indices]
    
    ### Plotting functions
    if np.max(np.max(Ca)) < 100:
        #print('I am here')
        fired_indices = np.where(filtered_fired)[0]
        if len(fired_indices) > 1:
            first_fired_index = fired_indices[np.argmin(filtered_fired_init_times[fired_indices])]
            last_fired_index = fired_indices[np.argmax(filtered_fired_init_times[fired_indices])]
            
            total_distance = points2[last_fired_index] - points2[first_fired_index]
            total_time = filtered_fired_init_times[last_fired_index] - filtered_fired_init_times[first_fired_index]
            
            average_velocity = total_distance / total_time if total_time != 0 else 10000
            
            ### Uncomment this to enable plotting
            """
            try:
                plotting_functions(Ca, x_sites, fired_init_times, ctr)
            except Exception as e:
                print('Not succesful in saving this one')
            """
        else:
            average_velocity = 0
            
    else:
        average_velocity = 0
    
    return average_velocity ####Ca, x_sites, fired_init_times

if __name__ == '__main__':
    param_ranges = np.array([[-3,3.5], [-3,0] ,[-3,3.5],    
                            [-3,0],[0.5,3], [0.2,0.4]])

    num_samples = 15000
    sampling = LHS(xlimits=param_ranges)
    theta_master = sampling(num_samples)
    wave_velocity_master = np.zeros((num_samples,))

    for i in range(num_samples):
        result = main_model_1(theta_master[i,:], i)
        wave_velocity_master[i] = result
        if i % 10 == 0:
            print(str(i+1) + ' jobs complete!')
            np.save('wave_velocity_master.npy', wave_velocity_master)
            np.save('theta_master.npy', theta_master)
            #print('Time taken is: ' + str(end_time - start_time))
