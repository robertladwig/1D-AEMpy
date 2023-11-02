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

os.chdir("/home/robert/Projects/WQ_MCL/src")
#os.chdir("C:/Users/ladwi/Documents/Projects/R/WQ_MCL/src")
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


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
hydrodynamic_timestep = 24 * dt
total_runtime =  (365) * hydrodynamic_timestep/dt  #365 *1 # 14 * 365
startTime =   (140 + 365*12) * hydrodynamic_timestep/dt #150 * 24 * 3600
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime)

## here we define our initial profile
u_ini = initial_profile(initfile = '../input/observedTemp.txt', nx = nx, dx = dx,
                     depth = depth,
                     startDate = startingDate)

wq_ini = wq_initial_profile(initfile = '../input/mendota_driver_data_v2.csv', nx = nx, dx = dx,
                     depth = depth, 
                     volume = volume,
                     startDate = startingDate)

tp_boundary = provide_phosphorus(tpfile =  '../input/Mendota_observations_tp.csv', 
                                 startingDate = startingDate,
                                 startTime = startTime)

tp_boundary = tp_boundary.dropna(subset=['tp'])

Start = datetime.datetime.now()

    
res = run_wq_model(  
    u = deepcopy(u_ini),
    o2 = deepcopy(wq_ini[0]),
    docr = deepcopy(wq_ini[1]),
    docl = 1.0 * volume,
    pocr = 0.5 * volume,
    pocl = 0.5 * volume,
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
    secview = meteo_all[1],
    phosphorus_data = tp_boundary,
    ice = False,
    Hi = 0,
    Hs = 0,
    Hsi = 0,
    iceT = 6,
    supercooled = 0,
    diffusion_method = 'pacanowskiPhilander',#'pacanowskiPhilander',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme ='implicit',
    km = 1.4 * 10**(-7), # 4 * 10**(-6), 
    k0 = 1 * 10**(-2),
    weight_kz = 0.5,
    kd_light = 0.6, 
    denThresh = 1e-2,
    albedo = 0.01,
    eps = 0.97,
    emissivity = 0.97,
    sigma = 5.67e-8,
    sw_factor = 1.0,
    wind_factor = 1.2,
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
    pgdl_mode = 'on',
    rho_snow = 250,
    p_max = 1/86400,
    IP = 3e-5/86400 ,#0.1, 3e-5
    delta= 1.1, #1.08,
    conversion_constant = 1e-4,#0.1
    sed_sink = -0.01 / 86400,
    k_half = 0.5,
    resp_docr = 0.0001/86400, # 0.001 0.0001
    resp_docl = 0.05/86400, # 0.01 0.05
    resp_poc = 0.0001/86400, # 0.1 0.001
    settling_rate = 0.3/86400,
    sediment_rate = 0.3/86400,
    piston_velocity = 1.0/86400,
    light_water = 0.125,
    light_doc = 0.02,
    light_poc = 0.7,
    mean_depth = sum(volume)/max(area),
    W_str = None)

temp=  res['temp']
o2=  res['o2']
docr=  res['docr']
docl =  res['docl']
pocr=  res['pocr']
pocl=  res['pocl']
diff =  res['diff']
avgtemp = res['average'].values
temp_initial =  res['temp_initial']
temp_heat=  res['temp_heat']
temp_diff=  res['temp_diff']
temp_mix =  res['temp_mix']
temp_conv =  res['temp_conv']
temp_ice=  res['temp_ice']
meteo=  res['meteo_input']
buoyancy = res['buoyancy']
icethickness= res['icethickness']
snowthickness= res['snowthickness']
snowicethickness= res['snowicethickness']
npp = res['npp']
docr_respiration = res['docr_respiration']
docl_respiration = res['docl_respiration']
poc_respiration = res['poc_respiration']
kd = res['kd_light']
thermo_dep = res['thermo_dep']
energy_ratio = res['energy_ratio']


End = datetime.datetime.now()
print(End - Start)

    
plt.plot(times, energy_ratio[0,:])

# heatmap of temps  
N_pts = 6



fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 30)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Water Temperature  ($^\circ$C)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()




fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(o2)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 20)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Dissolved Oxygen  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(docl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 7)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-labile  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(docr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 7)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-refractory  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(pocr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 15)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-refractory  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(pocl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 15)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-labile  (g/m3)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(npp)/volume) * 86400, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = .3)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("NPP  (g/m3/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(docr_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 2e-3)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOCr respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(docl_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 8e-2)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOCl respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(poc_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 3e-1)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC respiration  (/d)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
#time_label = time_label[::nelement]
ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
ax.set_xticklabels(time_label, rotation=0)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()

# plt.plot(npp[1,1:400]/volume[1] * 86400)
# plt.plot(o2[1,:]/volume[1])
# plt.plot(o2[1,1:(24*14)]/volume[1])
# plt.plot(o2[1,:]/volume[1])
# plt.plot(docl[1,:]/volume[1])
# plt.plot(docr[1,1:(24*10)]/volume[1])
# plt.plot(pocl[0,:]/volume[0])
# plt.plot(pocr[0,:]/volume[0])
# plt.plot(npp[0,:]/volume[0]*86400)
# plt.plot(docl_respiration[0,:]/volume[0]*86400)
# plt.plot(o2[(nx-1),:]/volume[(nx-1)])

plt.plot(o2[1,1:(24*28)]/volume[1]/4, color = 'blue', label = 'O2')
gpp = npp[1,:] -1/86400 *(docl[1,:] * docl_respiration[1,:]+ docr[1,:] * docr_respiration[1,:] + pocl[1,:] * poc_respiration[1,:] + pocr[1,:] * poc_respiration[1,:])
plt.plot(npp[1,1:(24*28)]/volume[1] * 86400, color = 'yellow', label = 'NPP') 
plt.plot(1/86400*(docl[1,1:(24*28)] * docl_respiration[1,1:(24*28)]+ docr[1,1:(24*28)] * docr_respiration[1,1:(24*28)] + pocl[1,1:(24*28)] * poc_respiration[1,1:(24*28)] + pocr[1,1:(24*28)] * poc_respiration[1,1:(24*28)])/volume[1] * 86400, color = 'red', label = 'R') 
plt.plot(gpp[1:(24*28)]/volume[1] * 86400, color = 'green', label = 'GPP')
plt.legend(loc='best')
plt.show() 

plt.plot(times, kd[0,:])
plt.ylabel("kd (/m)")
plt.show()

do_sat = o2[0,:] * 0.0
for r in range(0, len(temp[0,:])):
    do_sat[r] = do_sat_calc(temp[0,r], 982.2, altitude = 258) 

plt.plot(times, o2[0,:]/volume[0], color = 'blue')
plt.plot(times, do_sat, color = 'red')
plt.show()

plt.plot(times, thermo_dep[0,:]*dx,color= 'blue')
plt.plot(times, temp[0,:] - temp[(nx-1),:], color = 'red')
plt.show()

# TODO
# air water exchange
# sediment loss POC
# diffusive transport
# r and npp
# phosphorus bc
# ice npp
# wind mixingS
