"""
In this codee we add sink to the model. A sink was added to all the parameter sets that generated wave velocitoes within 
biological rane of 1um flg2 treatment. We compute the wave velocity, distance travelled by the wave and the R2 value of a
linear model fit to the activation time of calcium channels.
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

            Ca_first_fired = Ca[int(points2[first_fired_index] / dx), :]
            Ca_middle_fired = Ca[int(points2[middle_fired_index] / dx), :]
            Ca_last_fired = Ca[int(points2[last_fired_index] / dx), :]
            
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
    
    return average_velocity, total_distance, r_squared, Ca_first_fired, Ca_middle_fired, Ca_last_fired


if __name__ == '__main__':

    theta_master = np.load('output-data-15000/theta_master.npy')
    wave_velocity_master = np.load('output-data-15000/wave_velocity_master.npy')

    vel_exp = [53.4, 61.8, 62.4, 63.6, 74.4, 76.2, 84,
            84.6, 86.4, 90.6, 92.4, 97.8, 102, 105.6]

    vel_min = min(vel_exp)
    vel_max = max(vel_exp)

    mask = (wave_velocity_master >= vel_min) & (wave_velocity_master <= vel_max)
    wave_velocity_filtered = wave_velocity_master[mask]

    theta_filtered = theta_master[mask]
    theta_filtered[:, :4] = 10 ** theta_filtered[:, :4]

    wave_velocity_subset = wave_velocity_filtered[theta_filtered[:, 3] <= 0.5]
    theta_filtered_subset = theta_filtered[theta_filtered[:, 3] <= 0.5]

    indices_cicr_pos = [12, 33, 36, 48, 89, 91, 99,
                        103, 174, 187, 198, 217, 226, 243,
                        317, 324, 331, 332]

    theta_filtered_subset = theta_filtered_subset[indices_cicr_pos,:]
    wave_velocity_subset = wave_velocity_subset[indices_cicr_pos]

    theta01 = np.array([2.00270355e+02, 1.95119197e-01, 1.50008776e+00, 4.18697145e-01, 5.69583333e-01, 2.05740000e-01, 0.1, 2, 0])
    theta02 = np.array([2.00270355e+02, 1.95119197e-01, 1.50008776e+00, 4.18697145e-01, 5.69583333e-01, 2.05740000e-01, 0.1, 2, 0.2])

    wave_velocities_1 = np.zeros(len(theta_filtered_subset))
    wave_velocities_2 = np.zeros(len(theta_filtered_subset))

    wave_distances_1 = np.zeros(len(theta_filtered_subset))
    wave_distances_2 = np.zeros(len(theta_filtered_subset))

    wave_rsquared_1 = np.zeros(len(theta_filtered_subset))
    wave_rsquared_2 = np.zeros(len(theta_filtered_subset))

    ca_first_master = []
    ca_middle_master = []
    ca_last_master = []


    for i, theta in enumerate(theta_filtered_subset):
        
        if i > 9:
            theta_1 = theta01.copy()
            theta_2 = theta02.copy()
            theta_1[:6] = theta
            theta_2[:6] = theta
            
            wave_vel_1, dis1, rsq1, dummy1, dummy2, dummy3  = main_model_1(theta_1, ctr=0) 
            wave_vel_2, dis2, rsq2, caf, cam, cal = main_model_1(theta_2, ctr=0)
    
            wave_velocities_1[i] = wave_vel_1
            wave_velocities_2[i] = wave_vel_2
            
            wave_distances_1[i] = dis1
            wave_distances_2[i] = dis2
            
            wave_rsquared_1[i] = rsq1
            wave_rsquared_2[i] = rsq2
            
            ca_first_master.append(caf)
            ca_middle_master.append(cam)
            ca_last_master.append(cal)
            
            print(i)
    
            np.save('wave_velocities_1.npy', wave_velocities_1)
            np.save('wave_distances_1.npy', wave_distances_1)
            np.save('wave_rsquared_1.npy', wave_rsquared_1)
    
            np.save('wave_velocities_2.npy', wave_velocities_2)
            np.save('wave_distances_2.npy', wave_distances_2)
            np.save('wave_rsquared_2.npy', wave_rsquared_2)
    
            np.save('ca_first_master.npy', ca_first_master)
            np.save('ca_middle_master.npy', ca_middle_master)
            np.save('ca_last_master.npy', ca_last_master)
