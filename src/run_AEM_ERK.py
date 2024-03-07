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
from processBased_lakeModel_functions import get_hypsography, provide_meteorology, initial_profile, run_wq_model, wq_initial_profile, provide_phosphorus, do_sat_calc, calc_dens #, heating_module, diffusion_module, mixing_module, convection_module, ice_module


## lake configurations
zmax = 21 # maximum lake depth
nx = 21 * 2 # number of layers we will have
dt = 3600 # 24 hours times 60 min/hour times 60 seconds/min
dx = zmax/nx # spatial step

## area and depth values of our lake 
area, depth, volume = get_hypsography(hypsofile = '../../lakes/erken/bathymetry.csv',
                            dx = dx, nx = nx)
                           
## atmospheric boundary conditions
meteo_all = provide_meteorology(meteofile = '../../lakes/erken/meteodriverdata.csv',
                    secchifile = None, 
                    windfactor = 1.0)
                     
## time step discretization                      
hydrodynamic_timestep = 24 * dt
total_runtime =  (365 * 5) * hydrodynamic_timestep/dt  #365 *1 # 14 * 365  (365 *1.7) 
startTime =  1  #150 * 24 * 3600  (120 + 365*5)
endTime =  (startTime + total_runtime) # * hydrodynamic_timestep/dt) - 1

startingDate = meteo_all[0]['date'][startTime] #* hydrodynamic_timestep/dt]
endingDate = meteo_all[0]['date'][(endTime-1)]#meteo_all[0]['date'][(startTime + total_runtime)]# * hydrodynamic_timestep/dt -1]

times = pd.date_range(startingDate, endingDate, freq='H')

nTotalSteps = int(total_runtime)

## here we define our initial profile
dt_da = pd.read_csv('../../lakes/erken/observed.csv', index_col=0)
dt_da=dt_da.rename(columns = {'DateTime':'time'})
dt_da['time'] = pd.to_datetime(dt_da['time']) # pd.to_datetime(dt['time'], format='%Y-%m-%d %H')
dt_red = dt_da[dt_da['time'] >= startingDate]
dt_red = dt_red[dt_red['time'] <= endingDate]
u_ini = dt_red.iloc[0,1:].to_numpy()
u_ini = u_ini[0:(len(u_ini)-1)]

u_ini =pd.to_numeric(u_ini,errors='coerce')

wq_ini = wq_initial_profile(initfile = '../input/mendota_driver_data_v2.csv',
                            nx = nx, dx = dx,
                     depth = depth, 
                     volume = volume,
                     startDate = startingDate)

tp_boundary = provide_phosphorus(tpfile =  '../input/Mendota_observations_tp.csv', 
                                 startingDate = startingDate,
                                 startTime = startTime)

tp_boundary = tp_boundary.dropna(subset=['tp'])

#u_ini = np.ones(nx)*10

Start = datetime.datetime.now()

pgdl_mode = 'on'
    
res = run_wq_model(  
    u = deepcopy(u_ini),
    o2 = deepcopy(wq_ini[0]),
    docr = deepcopy(wq_ini[1]),
    docl = 1.0 * volume,
    pocr = 0.5 * volume,
    pocl = 0.5 * volume,
    alg = 10/1000 * volume,
    nutr = 10/1000 * volume,
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
    coupled = 'off',
    diffusion_method = 'hendersonSellers',#'pacanowskiPhilander',# 'hendersonSellers', 'munkAnderson' 'hondzoStefan'
    scheme ='implicit',
    km = 1.4 * 10**(-7), # 4 * 10**(-6), 
    k0 = 1 * 10**(-2), #1e-2
    weight_kz = 0.5,
    kd_light = 0.63, 
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
    IP = 3e-2/86400 ,#0.1, 3e-6
    theta_npp = 1.0, #1.08,
    theta_r = 1.08, #1.08,
    conversion_constant = 1e-4,#0.1
    sed_sink =0.005 / 86400, #0.01
    k_half = 3.1, #0.5,
    resp_docr = 0.008/86400, # 0.08 0.001 0.0001
    resp_docl = 0.008/86400, # 0.01 0.05
    resp_pocr = 0.004/86400, # 0.04 0.1 0.001 0.0001
    resp_pocl = 0.004/86400,
    grazing_rate = 0.9/86400, #1e-1/86400, # 3e-3/86400
    pocr_settling_rate = 1e-3/86400,
    pocl_settling_rate = 1e-3/86400,
    algae_settling_rate = 1e-5/86400,
    sediment_rate = 10/86400,
    piston_velocity = 1.0/86400,
    light_water = 0.125,
    light_doc = 0.02,
    light_poc = 0.7,
    mean_depth = sum(volume)/max(area),
    W_str = None,
    tp_inflow = 0,#np.mean(tp_boundary['tp'])/1000 * volume[0] * 1/1e6,
    alg_inflow = 0.1 * volume[0] * 5/1e6,
    pocr_inflow = 0.1 * volume[0] * 1/1e7,
    pocl_inflow = 0.1 * volume[0] * 1/1e7,
    f_sod = 0.01 / 86400,
    d_thick = 0.001,
    growth_rate = 0.5/86400, # 1.0e-3
    grazing_ratio = 0.1,
    alpha_gpp = 0.03/86400,
    beta_gpp = 0.00017/86400,
    o2_to_chla = 2.15/3600)

temp=  res['temp']
o2=  res['o2']
docr=  res['docr']
docl =  res['docl']
pocr=  res['pocr']
pocl=  res['pocl']
alg=  res['alg']
nutr=  res['nutr']
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
algae_growth = res['algae_growth']
algae_grazing = res['algae_grazing']
docr_respiration = res['docr_respiration']
docl_respiration = res['docl_respiration']
pocr_respiration = res['pocr_respiration']
pocl_respiration = res['pocl_respiration']
kd = res['kd_light']
thermo_dep = res['thermo_dep']
energy_ratio = res['energy_ratio']
differror = res['differror']
alpha = res['alpha']


End = datetime.datetime.now()
print(End - Start)

    
plt.plot(times, energy_ratio[0,:])
plt.show()

plt.plot(times, differror[0,:])
plt.show()

plt.plot(times, alpha[0,:])
plt.show()

plt.plot(differror[0,:], alpha[0,:])
plt.show()

# heatmap of temps  
N_pts = 6



fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Water Temperature  ($^\circ$C)")
xticks_ix = np.array(ax.get_xticks()).astype(int)
time_label = times[xticks_ix]
nelement = len(times)//N_pts
time_label = times[::nelement]
#time_label = time_label[::nelement]
#ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
time_label = times[np.array(ax.get_xticks()).astype(int)]
ax.set_xticklabels(time_label, rotation=90)
yticks_ix = np.array(ax.get_yticks()).astype(int)
depth_label = yticks_ix / 2
ax.set_yticklabels(depth_label, rotation=0)
plt.show()


fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(diff, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = diff.min(), vmax = diff.max())
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Diffusivity  (m2/s)")
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


# plt.plot(o2[0,:]/volume[0])
# ax = plt.gca()
# ax.set_ylim(0,20)
# plt.show()

fig, ax = plt.subplots(figsize=(15,5))
sns.heatmap(np.transpose(np.transpose(o2)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
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
sns.heatmap(np.transpose(np.transpose(alg)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Algae (g/m3)")
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
sns.heatmap(np.transpose(np.transpose(nutr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Nutrients (g/m3)")
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
sns.heatmap(np.transpose(np.transpose(docl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
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
sns.heatmap(np.transpose(np.transpose(docr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
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
sns.heatmap(np.transpose(np.transpose(pocr)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
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
sns.heatmap(np.transpose(np.transpose(pocl)/volume), cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
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
sns.heatmap(np.transpose(np.transpose(algae_growth)) , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Algae growth  (/d)")
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
sns.heatmap(np.transpose(np.transpose(algae_grazing)) , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("Algae grazing  (/d)")
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
sns.heatmap(npp , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("NPP  (/d)")
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
sns.heatmap(docr_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-refractory respiration  (/d)")
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
sns.heatmap(docl_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("DOC-labile respiration  (/d)")
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
sns.heatmap(pocr_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-refractory respiration  (/d)")
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
sns.heatmap(pocl_respiration , cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
           colors=['black', 'gray'],
           linestyles = 'dotted')
ax.set_ylabel("Depth (m)", fontsize=15)
ax.set_xlabel("Time", fontsize=15)    
ax.collections[0].colorbar.set_label("POC-labile respiration  (/d)")
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

#plt.plot(o2[1,1:(24*28)]/volume[1]/4, color = 'blue', label = 'O2')
nep = 1/86400 *alg[1,:] *npp[1,:] -1/86400 *(docl[1,:] * docl_respiration[1,:]+ docr[1,:] * docr_respiration[1,:] + pocl[1,:] * pocl_respiration[1,:] + pocr[1,:] * pocr_respiration[1,:])
plt.plot(alg[1,1:(24*90)] *npp[1,1:(24*90)]* 1/86400 * 1/ volume[1], color = 'green', label = 'GPP') 
plt.plot(1/86400*(docl[1,1:(24*90)] * docl_respiration[1,1:(24*90)]+ docr[1,1:(24*90)] * docr_respiration[1,1:(24*90)] + pocl[1,1:(24*90)] * pocl_respiration[1,1:(24*90)] + pocr[1,1:(24*90)] * pocr_respiration[1,1:(24*90)])/volume[1] , color = 'red', label = 'R') 
plt.plot(nep[1:(24*90)]/volume[1] , color = 'yellow', label = 'NEP')
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

if pgdl_mode == 'on':
    # save model output
    
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    df_volumes =  np.ones(len(times)) + sum(volume)
    df_osgood =  np.ones(len(times)) + (sum(volume)/max(area)) / sqrt(max(area/1e6))
    df_maxdepth =  np.ones(len(times)) + max(depth)
    df_meandepth =  np.ones(len(times)) +  (sum(volume)/max(area))
    df1.insert(1, "Volume_m2", df_volumes, True)
    df1.insert(2, "Osgood", df_osgood, True)
    df1.insert(3, "MaxDepth_m", df_maxdepth, True)
    df1.insert(4, "MeanDepth_m", df_meandepth, True)
    df1.to_csv('../../lakes/erken/output/py_lakecharacteristics.csv', index=None)
    
    # initial temp.
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(temp_initial)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_temp_initial00.csv', index=None)
    
    # heat temp.
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(temp_heat)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_temp_heat01.csv', index=None)
    
    # diffusion temp.
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(temp_diff)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_temp_diff03.csv', index=None)
    
    # mixing temp.
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(temp_mix)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_temp_mix05.csv', index=None)
    
    # convection temp.
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(temp_conv)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_temp_conv04.csv', index=None)
    
    # ice temp.
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(temp_ice)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_temp_ice02.csv', index=None)
    
    # diffusivity
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(diff)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_diff.csv', index=None)
    
    # buoyancy
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(buoyancy)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df = pd.concat([df1, df2], axis = 1)
    df.to_csv('../../lakes/erken/output/py_buoyancy.csv', index=None)
    
    # meteorology
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(meteo)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df2.columns = ["AirTemp_degC", "Longwave_Wm-2",
                      "Latent_Wm-2", "Sensible_Wm-2", "Shortwave_Wm-2",
                      "lightExtinct_m-1","TKE_Jm-1", "ShearStress_Nm-2",
                      "Area_m2", "CC", 'ea', 'Jlw', 'Uw', 'Pa', 'RH', 'PP', 'IceSnowAttCoeff',
                      'iceFlag', 'icemovAvg', 'density_snow', 'ice_prior', 'snow_prior', 
                      'snowice_prior', 'rho_snow_prior', 'IceSnowAttCoeff_prior', 'iceFlag_prior',
                      'dt_iceon_avg_prior', 'icemovAvg_prior']
    df = pd.concat([df1, df2], axis = 1)
    df_airtemp = df['AirTemp_degC']
    df.to_csv('../../lakes/erken/output/py_meteorology_input.csv', index=None)
    
        
    # ice-snow
    df1 = pd.DataFrame(times)
    df1.columns = ['time']
    t1 = np.matrix(icethickness)
    t1 = t1.getT()
    df2 = pd.DataFrame(t1)
    df2.columns = ['ice']
    t1 = np.matrix(snowthickness)
    t1 = t1.getT()
    df3 = pd.DataFrame(t1)
    df3.columns = ['snow']
    t1 = np.matrix(snowicethickness)
    t1 = t1.getT()
    df4 = pd.DataFrame(t1)
    df4.columns = ['snowice']
    df = pd.concat([df1, df2, df3, df4], axis = 1)
    df.to_csv('../../lakes/erken/output/py_icesnow.csv', index=None)
    
    # observed data
    dt = pd.read_csv('../../lakes/erken/observed.csv', index_col=0)
    dt=dt.rename(columns = {'DateTime':'time'})
    dt['time'] = pd.to_datetime(dt['time']) # pd.to_datetime(dt['time'], format='%Y-%m-%d %H')
    dt_red = dt[dt['time'] >= startingDate]
    dt_red = dt_red[dt_red['time'] <= endingDate]
    
    # let's set surface to 0 if airtemp is below 0, assuming we have ice
    temp_flag = df_airtemp <= 0
    wtr_0m = np.array(dt_red['var. 0'])
    wtr_05m = np.array(dt_red['var. 0.5'])
    wtr_0m[temp_flag] = 0
    wtr_05m[temp_flag] = 0
    dt_red['var. 0'] = wtr_0m
    dt_red['var. 0.5'] = wtr_05m 
    dt_red.to_csv('../../lakes/erken/output/py_observed_temp.csv', index=None, na_rep='-999')
    
    dt_notime = dt_red.drop(dt_red.columns[[0]], axis = 1)
    dt_notime = dt_notime.transpose()
    dt_obs = dt_notime.to_numpy()
    dt_obs.shape
    temp.shape
    
    dt_obs=dt_obs[:-1,:]
    #dt_obs=dt_obs[:-1,:]
    dt_obs.shape
    # heatmap of temps  

    
    diff_temp = temp - dt_obs
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(dt_obs, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
    ax.contour(np.arange(.5, dt_obs.shape[1]), np.arange(.5, dt_obs.shape[0]), calc_dens(dt_obs), levels=[999],
               colors=['black', 'gray'],
               linestyles = 'dotted')
    ax.set_ylabel("Depth (m)", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)    
    ax.collections[0].colorbar.set_label("Observed WT ($^\circ$C)")
    xticks_ix = np.array(ax.get_xticks()).astype(int)
    time_label = times[xticks_ix]
    nelement = len(times)//N_pts
    time_label = times[::nelement]
    #time_label = time_label[::nelement]
    #ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
    time_label = times[np.array(ax.get_xticks()).astype(int)]
    ax.set_xticklabels(time_label, rotation=90)
    yticks_ix = np.array(ax.get_yticks()).astype(int)
    depth_label = yticks_ix / 2
    ax.set_yticklabels(depth_label, rotation=0)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0)
    ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
               colors=['black', 'gray'],
               linestyles = 'dotted')
    ax.set_ylabel("Depth (m)", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)    
    ax.collections[0].colorbar.set_label("Modeled WT ($^\circ$C)")
    xticks_ix = np.array(ax.get_xticks()).astype(int)
    time_label = times[xticks_ix]
    nelement = len(times)//N_pts
    time_label = times[::nelement]
    #time_label = time_label[::nelement]
    #ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
    time_label = times[np.array(ax.get_xticks()).astype(int)]
    ax.set_xticklabels(time_label, rotation=90)
    yticks_ix = np.array(ax.get_yticks()).astype(int)
    depth_label = yticks_ix / 2
    ax.set_yticklabels(depth_label, rotation=0)
    plt.show()

    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(diff_temp, cmap=plt.cm.get_cmap('Spectral_r'),  xticklabels=1000, yticklabels=2, vmin = 0, vmax = 10)
    ax.contour(np.arange(.5, temp.shape[1]), np.arange(.5, temp.shape[0]), calc_dens(temp), levels=[999],
               colors=['black', 'gray'],
               linestyles = 'dotted')
    ax.set_ylabel("Depth (m)", fontsize=15)
    ax.set_xlabel("Time", fontsize=15)    
    ax.collections[0].colorbar.set_label("Diff PB-OBS ($^\circ$C)")
    xticks_ix = np.array(ax.get_xticks()).astype(int)
    time_label = times[xticks_ix]
    nelement = len(times)//N_pts
    time_label = times[::nelement]
    #time_label = time_label[::nelement]
    #ax.xaxis.set_major_locator(plt.MaxNLocator(N_pts))
    time_label = times[np.array(ax.get_xticks()).astype(int)]
    ax.set_xticklabels(time_label, rotation=90)
    yticks_ix = np.array(ax.get_yticks()).astype(int)
    depth_label = yticks_ix / 2
    ax.set_yticklabels(depth_label, rotation=0)
    plt.show()
    
    
    fig, axs = plt.subplots(5, figsize = (10,20))
    
    axs[0].plot(times, dt_obs[0,:], color ='red', label = 'observed')
    axs[0].plot(times, temp[0,:], label = 'modeled')
    axs[0].set_title(depth[0])
    axs[0].set_ylim(0,30)
    
    axs[1].plot(times, dt_obs[5,:], color ='red')
    axs[1].plot(times, temp[5,:])
    axs[1].set_title(depth[5])
    axs[1].set_ylim(0,30)
    
    axs[2].plot(times, dt_obs[10,:], color ='red')
    axs[2].plot(times, temp[10,:])
    axs[2].set_title(depth[10])
    axs[2].set_ylim(0,30)
    
    axs[3].plot(times, dt_obs[25,:], color ='red')
    axs[3].plot(times, temp[25,:])
    axs[3].set_title(depth[25])
    axs[3].set_ylim(0,25)
    
    axs[4].plot(times, dt_obs[40,:], color ='red')
    axs[4].plot(times, temp[40,:])
    axs[4].set_title(depth[40])
    axs[4].set_ylim(0,25)
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    plt.show()
