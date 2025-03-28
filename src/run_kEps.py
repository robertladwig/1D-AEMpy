import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from numba import jit

#os.chdir("/home/robert/Projects/1D-AEMpy/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/1D-AEMpy/src")
#os.chdir("D:/bensd/Documents/Python_Workspace/1D-AEMpy/src")
os.chdir("C:/Users/au740615/Documents/Projects/1d_aempy/1D-AEMpy/src")
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, run_wq_model_time, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens, run_kE_model #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 25 # maximum lake depth
nx = 25 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
area, depth, volume = get_hypsography(hypsofile = '../input/bathymetry.csv',
                            dx = dx, nx = nx)
                           
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = '../input/Mendota_2002.csv',
                    secchifile = None, 
                    windfactor = 1.0)
                     
## time step discretization                      
daily_timestep = (86400/dt) 
total_runtime =  21 * 86400 # (365 *6) * hydrodynamic_timestep/dt  #365 *1 # 14 * 365  (365 *1.7) 
startTime =   (140+365*6) * 86400 #150 * 24 * 3600  (120 + 365*5)
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime/3600] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime)/3600]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='min')

nTotalSteps = int(total_runtime)

## here we define our initial profile
u_ini = initial_profile(initfile = '../input/observedTemp.txt', nx = nx, dx = dx,
                     depth = depth,
                     startDate = startingDate)



Start = datetime.datetime.now()

pgdl_mode = 'on'
    
res = run_kE_model(  
    u = deepcopy(u_ini),
    k = deepcopy(u_ini) * 0.0 + 1E-5,
    e = deepcopy(u_ini) * 0.0 + 1E-7,
    w = deepcopy(u_ini) * 0.0 + 1E-3,
    v = deepcopy(u_ini) * 0.0 + 1E-3,
    startTime = startTime, 
    endTime = endTime, 
    area = area,
    volume = volume,
    depth = depth,
    zmax = zmax,
    nx = nx,
    dt = dt,
    dx = dx,
    daily_meteo = meteo_all[0],
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,
    coupled = 'off',
    diffusion_method = 'hendersonSellers',#'pacanowskiPhilander',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme ='implicit',
    km = 1.4 * 10**(-7), # 4 * 10**(-6), 
    k0 = 1 * 10**(-2), #1e-2
    weight_kz = 0.5,
    kd_light = 0.6, 
    denThresh = 1e-2,
    albedo = 0.1,
    eps = 0.97,
    emissivity = 0.97,
    sigma = 5.67e-8,
    sw_factor = 1.0,
    wind_factor = 1.0,
    at_factor = 1.0,
    turb_factor = 1.0,
    p2 = 1,
    B = 0.61,
    g = 9.81,
    Cd = 0.0013, # momentum coeff (wind)
    meltP = 1,
    dt_iceon_avg = 0.8,
    Hgeo = 0.1, # geothermal heat 
    KEice = 0,
    Ice_min = 0.1,
    pgdl_mode = pgdl_mode,
    rho_snow = 250,
    p_max = 1/86400,
    mean_depth = sum(volume)/max(area)
    )

temp =  res['temp_final']
v_veloc = res['v_final']
w_veloc = res['w_final']
TKE = res['k_final']
eps = res['eps_final']
diff_temp = res['kz']
diff_momentum = res['kzz']
diff_TKE = res['kk']
diff_eps = res['ke']
H_net = res["H_net"]

plt.plot(H_net[np.isfinite(H_net)])
plt.show()

fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("", fontsize=15)    
ax.collections[0].colorbar.set_label("Water temperature  ($^\circ$C)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//2
time_label = times[::nelement]
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
time_label = times[np.array(ax.get_xticks()).astype(int)]
ax.set_xticklabels(time_label, rotation=15)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.rcParams.update({'font.size': 30})
plt.show()

fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(TKE, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("", fontsize=15)    
ax.collections[0].colorbar.set_label("TKE  (m2/s)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//2
time_label = times[::nelement]
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
time_label = times[np.array(ax.get_xticks()).astype(int)]
ax.set_xticklabels(time_label, rotation=15)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.rcParams.update({'font.size': 30})
plt.show()

fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(eps, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("", fontsize=15)    
ax.collections[0].colorbar.set_label("Epsilon  (m2/s2)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//2
time_label = times[::nelement]
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
time_label = times[np.array(ax.get_xticks()).astype(int)]
ax.set_xticklabels(time_label, rotation=15)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.rcParams.update({'font.size': 30})
plt.show()

fig, ax = plt.subplots(figsize=(20,15))
sns.heatmap(v_veloc, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("", fontsize=15)    
ax.collections[0].colorbar.set_label("Velocity  (m/s)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//2
time_label = times[::nelement]
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
time_label = times[np.array(ax.get_xticks()).astype(int)]
ax.set_xticklabels(time_label, rotation=15)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.rcParams.update({'font.size': 30})
plt.show()

