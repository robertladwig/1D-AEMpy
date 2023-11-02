import numpy as np
import pandas as pd
import os
from math import pi, exp, sqrt, log, atan, sin, radians, nan, isinf
from scipy.interpolate import interp1d
from copy import deepcopy
import datetime
from ancillary_functions import calc_cc, buoyancy_freq, center_buoyancy
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from numba import jit
from scipy.linalg import solve_banded
from scipy.stats.stats import pearsonr


## function to calculate density from temperature
def calc_dens(wtemp):
    dens = (999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) +
      (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + 
      (6.536336 * 1e-9 * wtemp**5))
    return dens

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity(rho, depth, g, rho_0, ice, area, T, diff):
    km = 1.4 * 10**(-7)
    
    rho = np.array(rho)
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    if ice == True:
      ak = 0.000898
    else:
      ak = 0.00706 *( max(area)/1E6)**(0.56)
    
    kz = ak * (buoy)**(-0.43)
    
        
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    
    return(kz + km)

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity_hendersonSellers(rho, depth, g, rho_0, ice, area, U10, latitude, T, diff, Cd, km, weight_kz, k0):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    # 1.4 * 10**(-7)
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    rho_a = 1.2

    #depth[0] = depth[1] / 10

    
    U2 = U10 * 10
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    
    if U2 < 2.2:
        Cd = 1.08 * U2**(-0.15)* 10**(-3)
    elif 2.2 <= U2 < 5.0:
        Cd = (0.771 + 0.858 * U2**(-0.15)) *10**(-3)
    elif 5.0 <= U2 < 8.0:
        Cd = (0.867 + 0.0667 * U2**(-0.15)) * 10**(-3)
    elif 8.0 <= U2 < 25:
        Cd = (1.2 + 0.025 * U2**(-0.15)) * 10**(-3)
    elif 25 <= U2 < 50:
        Cd = 0.073 * U2**(-0.15) * 10**(-3)
        
    w_star = Cd * U2
    k_star = 6.6 * (sin(radians(latitude)))**(1/2) * U2**(-1.84)
    

    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy
    
    #breakpoint()
    Ri = (-1 + (1 + 40 * (np.array(buoy) * k**2 * np.array(depth)**2) / 
               (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    kz = (k * w_star * np.array(depth)) / (Pr * (1 + 37 * np.array(Ri)**2)) * np.exp(-k_star * np.array(depth))
    
    tau_w = rho_a * Cd * U2**2
    u_star = sqrt(tau_w / rho_0)
    H_ekman = 0.4 * u_star / f
    
    e_w = xi * sqrt(Cd) * U2
    W_eff = e_w / (xi * sqrt(Cd))
    kz_ekman = 1/f * (rho_a / rho_0 * Cd / kullenberg)**2 * W_eff**2
    
    kz_old = kz
    
    
    # kz[depth < H_ekman] = kz_ekman / 100
    # kz[0:2] = kz_old[0:2]
    
    
    # if (np.mean(T) <= 5):
        # kz = kz * 1000
    # Hongping Gu et al. (2015). Climate Change
    LST = T[0]
    if (LST > 4):
        kz = kz * 10**2
    elif (LST > 0) & (LST <= 4):
        kz = kz * 10**4
    elif LST <= 0:
        kz = kz * 0
    

    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = weight_kz
    
    #kz[0] = kz[1]
        
    kz = weight * kz + (1 - weight) * diff

    
    # kz = ak * (buoy)**(-0.43)
    return(kz +  km)

## this is our attempt for turbulence closure, estimating eddy diffusivity
def eddy_diffusivity_munkAnderson(rho, depth, g, rho_0, ice, area, U10, latitude, Cd, T, diff, k0, km, weight_kz):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    km = 1.4 * 10**(-7)
    rho_a = 1.2
    alpha = 10/3
    beta = 3/2
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    U2 = U10
    
    if U2 < 2.2:
        Cd = 1.08 * U2**(-0.15)* 10**(-3)
    elif 2.2 <= U2 < 5.0:
        Cd = (0.771 + 0.858 * U2**(-0.15)) *10**(-3)
    elif 5.0 <= U2 < 8.0:
        Cd = (0.867 + 0.0667 * U2**(-0.15)) * 10**(-3)
    elif 8.0 <= U2 < 25:
        Cd = (1.2 + 0.025 * U2**(-0.15)) * 10**(-3)
    elif 25 <= U2 < 50:
        Cd = 0.073 * U2**(-0.15) * 10**(-3)
    
    w_star = sqrt(rho_a / rho[0] * Cd * U2**2)
    k_star = 0.51 * (sin(radians(latitude))) / U2
    
    
    
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy
    # (uf./(kappa*z_edge).*exp(-ks*z_edge)).^2; 
    s_wall = (w_star / (k * np.array(depth)) * np.exp(k_star * np.array(depth)))**2
    s_wall = w_star/ (k * np.array(depth) *np.array(rho))
    
    
    X_HS = np.array(buoy)/(s_wall**2 + s_bg + s_seiche)
    Ri=(-1+(1+40*X_HS)**0.5)/20
    
    #breakpoint()
    #Ri = (-1 + (1 + 40 * np.array(buoy) * k**2 * np.array(depth)**2 / 
    #           (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    f_HS = (1.0 / (1 + alpha * Ri)**beta)
    f_HS[Ri == 0] = 1
    
    kz = (k * w_star * np.array(depth)) * np.exp(-k_star * np.array(depth)) * f_HS
    
    kz[0] = kz[1]
    # modify according to Ekman layer depth
    
    tau_w = rho_a * Cd * U2**2
    u_star = sqrt(tau_w / rho_0)
    H_ekman = 0.4 * u_star / f
    
    e_w = xi * sqrt(Cd) * U2
    W_eff = e_w / (xi * sqrt(Cd))
    kz_ekman = 1/f * (rho_a / rho_0 * Cd / kullenberg)**2 * W_eff**2
    
    # kz[depth < H_ekman] = kz_ekman 
    
    if (np.mean(T) <= 5):
        kz = kz * 1000
    
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = 0.5
        
    kz = weight * kz + (1 - weight) * diff

    return(kz +  km)

def eddy_diffusivity_pacanowskiPhilander(rho, depth, g, rho_0, ice, area, U10, latitude, T, diff, Cd, km, weight_kz, k0):
    k = 0.4
    Pr = 1.0
    z0 = 0.0002
    # 1.4 * 10**(-7)
    f = 1 * 10 **(-4)
    xi = 1/3
    kullenberg = 2 * 10**(-2)
    rho_a = 1.2
    K0 = k0 # 10**(-2)
    Kb = 10**(-7)

    #depth[0] = depth[1] / 10
    
    U2 = U10 * 10
    U2 = U10 * (log((2 - 1e-5)/z0)) / (log((10 - 1e-5)/z0))
    
    w_star = Cd * U2
    k_star = 6.6 * (sin(radians(latitude)))**(1/2) * U2**(-1.84)
    

    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
        
    low_values_flags = buoy < 7e-5  # Where values are low
    buoy[low_values_flags] = 7e-5
    
    s_bg = 2 * 10**(-7)
    s_seiche = 0.7 * buoy
    
    #breakpoint()
    Ri = (-1 + (1 + 40 * (np.array(buoy) * k**2 * np.array(depth)**2) / 
               (w_star**2 * np.exp(-2 * k_star * np.array(depth))))**(1/2)) / 20
    
    kz = K0 / (1 + 5 * Ri) + Kb
    
    kz[0] = kz[1]
    
    # modify according to Ekman layer depth
    
    # tau_w = rho_a * Cd * U2**2
    # tau_w = w_star
    # u_star = sqrt(tau_w / rho_0)
    # H_ekman = 0.4 * u_star / f
    
    # e_w = xi * sqrt(Cd) * U2
    # W_eff = e_w / (xi * sqrt(Cd))
    # kz_ekman = 1/f * (rho_a / rho_0 * Cd / kullenberg)**2 * W_eff**2
    
    # kz[depth < H_ekman] = kz_ekman 
    
    # #breakpoint()
    # kz_reduced = kz_ekman * np.exp(-np.array(depth)[depth >= H_ekman])
    # kz_reduced[kz_reduced < k0] = k0
    # kz[depth >= H_ekman] =  kz_reduced / (1 + 5 * Ri[depth >= H_ekman]) + Kb
    
    #breakpoint()

    
    if (np.mean(diff) == 0.0):
        weight = 1
    else:
        weight = weight_kz
        
    kz = weight * kz + (1 - weight) * diff

    
    # kz = ak * (buoy)**(-0.43)
    return(kz)

   
def provide_meteorology(meteofile, secchifile, windfactor):

    meteo = pd.read_csv(meteofile)
    daily_meteo = meteo
    daily_meteo['date'] = pd.to_datetime(daily_meteo['datetime'])
    daily_meteo['Cloud_Cover'] = calc_cc(date = daily_meteo['date'],
                                                airt = daily_meteo['Air_Temperature_celsius'],
                                                relh = daily_meteo['Relative_Humidity_percent'],
                                                swr = daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'],
                                                lat = 43, lon = -89.41,
                                                elev = 258)

    
    daily_meteo['dt'] = (daily_meteo['date'] - daily_meteo['date'][0]).astype('timedelta64[s]') + 1
    daily_meteo['ea'] = (daily_meteo['Relative_Humidity_percent'] * 
      (4.596 * np.exp((17.27*(daily_meteo['Air_Temperature_celsius'])) /
      (237.3 + (daily_meteo['Air_Temperature_celsius']) ))) / 100)
    daily_meteo['ea'] = ((101.325 * np.exp(13.3185 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15))) -
      1.976 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**2 -
      0.6445 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**3 -
      0.1229 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**4)) * daily_meteo['Relative_Humidity_percent']/100)
    daily_meteo['ea'] = (daily_meteo['Relative_Humidity_percent']/100) * 10**(9.28603523 - 2322.37885/(daily_meteo['Air_Temperature_celsius'] + 273.15))
    startDate = pd.to_datetime(daily_meteo.loc[0, 'date']) 
    
    ## calibration parameters
    daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] = daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] 
    daily_meteo['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] = daily_meteo['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] * windfactor # wind speed multiplier
    
    date_time = daily_meteo.date

    daily_meteo['day_of_year_list'] = [t.timetuple().tm_yday for t in date_time]
    daily_meteo['time_of_day_list'] = [t.hour for t in date_time]
    ## light
    # Package ID: knb-lter-ntl.31.30 Cataloging System:https://pasta.edirepository.org.
    # Data set title: North Temperate Lakes LTER: Secchi Disk Depth; Other Auxiliary Base Crew Sample Data 1981 - current.
    if secchifile is not None:
        secview0 = pd.read_csv(secchifile)
        secview0['sampledate'] = pd.to_datetime(secview0['sampledate'])
        secview = secview0.loc[secview0['sampledate'] >= startDate]
        if secview['sampledate'].min() >= startDate:
          firstVal = secview.loc[secview['sampledate'] == secview['sampledate'].min(), 'secnview'].values[0]
          firstRow = pd.DataFrame(data={'sampledate': [startDate], 'secnview':[firstVal]})
          secview = pd.concat([firstRow, secview], ignore_index=True)
      
          
        secview['dt'] = (secview['sampledate'] - secview['sampledate'][0]).astype('timedelta64[s]') + 1
        secview['kd'] = 1.7 / secview['secnview']
        secview['kd'] = secview.set_index('sampledate')['kd'].interpolate(method="linear").values
    else:
        secview = None
    
    return([daily_meteo, secview])
  
    
def provide_phosphorus(tpfile, startingDate, startTime):
    phos = pd.read_csv(tpfile)

    daily_tp = phos
    daily_tp['date'] = pd.to_datetime(daily_tp['datetime'])
    
    daily_tp['ditt'] = abs(daily_tp['date'] - startingDate)
    daily_tp = daily_tp.loc[daily_tp['date'] >= startingDate]
    if startingDate < daily_tp['date'].min():
        daily_tp.loc[-1] = [startingDate, 'epi', daily_tp['tp'].iloc[0], startingDate, daily_tp['ditt'].iloc[0]]  # adding a row
        daily_tp.index = daily_tp.index + 1  # shifting index
        daily_tp.sort_index(inplace=True) 
    daily_tp['dt'] = (daily_tp['date'] - daily_tp['date'][0]).astype('timedelta64[s]') + startTime 

    return(daily_tp)

def initial_profile(initfile, nx, dx, depth, startDate):
  #meteo = processed_meteo
  #startDate = meteo['date'].min()
  obs = pd.read_csv(initfile)
  obs['datetime'] = pd.to_datetime(obs['datetime'])
  obs['ditt'] = abs(obs['datetime'] - startDate)
  init_df = obs.loc[obs['ditt'] == obs['ditt'].min()]
  if max(depth) > init_df.Depth_meter.max():
    lastRow = init_df.loc[init_df.Depth_meter == init_df.Depth_meter.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'Depth_meter'] = max(depth)
    
  profile_fun = interp1d(init_df.Depth_meter.values, init_df.Water_Temperature_celsius.values)
  out_depths = depth # these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  u = profile_fun(out_depths)
  
  # TODO implement warning about profile vs. met start date
  
  return(u)

def wq_initial_profile(initfile, nx, dx, depth, volume, startDate):
  #meteo = processed_meteo
  #startDate = meteo['date'].min()
  obs = pd.read_csv(initfile)
  obs['datetime'] = pd.to_datetime(obs['datetime'])
  
  do_obs = obs.loc[obs['variable'] == 'do']
  do_obs['ditt'] = abs(do_obs['datetime'] - startDate)
  init_df = do_obs.loc[do_obs['ditt'] == do_obs['ditt'].min()]
  if max(depth) > init_df.depth.max():
    lastRow = init_df.loc[init_df.depth == init_df.depth.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'depth'] = max(depth)
    
  profile_fun = interp1d(init_df.depth.values, init_df.observation.values)
  out_depths =depth# these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  do = profile_fun(out_depths)
  
  doc_obs = obs.loc[obs['variable'] == 'doc']
  doc_obs['ditt'] = abs(doc_obs['datetime'] - startDate)
  init_df = doc_obs.loc[doc_obs['ditt'] == doc_obs['ditt'].min()]
  if max(depth) > init_df.depth.max():
    lastRow = init_df.loc[init_df.depth == init_df.depth.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'depth'] = max(depth)
    
  profile_fun = interp1d(init_df.depth.values, init_df.observation.values)
  out_depths = depth# these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  doc = profile_fun(out_depths)
  
  u = np.vstack((do * volume, doc * volume))
  # TODO implement warning about profile vs. met start date
  
  return(u)

def get_hypsography(hypsofile, dx, nx):
  hyps = pd.read_csv(hypsofile)
  out_depths = np.linspace(0, nx*dx, nx+1)
  area_fun = interp1d(hyps.Depth_meter.values, hyps.Area_meterSquared.values)
  area = area_fun(out_depths)
  area[-1] = area[-2] - 1 # TODO: confirm this is correct
  depth = np.linspace(0, nx*dx, nx+1)
  
  volume = area * 1000
  # volume = 0.5 * (area[:-1] + area[1:]) * np.diff(depth)
  # volume = (area[:-1] + area[1:]) * np.diff(depth)
  for d in range(0, (len(depth)-1)):
      volume[d] = np.abs(sum(area[0:(d+1)] * dx) - sum(area[0:d] * dx))

  # volume = (area[:-1] - area[1:]) * np.diff(depth)
  # volume = np.append(volume, 1000)
  
  volume = volume[:-1]
  depth = 1/2 * (depth[:-1] + depth[1:])
  area = 1/2 * (area[:-1] + area[1:])
  
  return([area, depth, volume])

def longwave(cc, sigma, Tair, ea, emissivity, Jlw):  # longwave radiation into
  Tair = Tair + 273.15
  p = (1.33 * ea/Tair)
  Ea = 1.24 * (1 + 0.17 * cc**2) * p**(1/7)
  lw = emissivity * Ea *sigma * Tair**4
  return(lw)

def backscattering(emissivity, sigma, Twater, eps): # backscattering longwave 
  # radiation from the lake
  Twater = Twater + 273.15
  #Twater = Twater
  #Tk = 273.15
  back = -1 * (eps * sigma * (Twater)**4) 
  #back = -1 * (eps * sigma * (Twater)**4 * (1 + 4 * (Twater/Tk) + 6 * (Twater/Tk)**2)) 
  return(back)

def PSIM(zeta):
  # Function to compute stability functions for momentum
  if zeta < 0.0:
    X = (1 - 16*zeta)**0.25
    psim = 2*log((1 + X)/2) + log((1 + X*X)/2)-2*atan(X) + pi/2 
  elif zeta > 0.0:
    if zeta > 0.5:
      if zeta > 10.0:
        psim = log(zeta) - 0.76*zeta - 12.093
      else:
        psim = 0.5/(zeta*zeta) - 4.25/zeta - 7.0*log(zeta) - 0.852
    else:
      psim = -5*zeta
  # Stable case
  else:
    psim = 0.0
  return(psim)

def PSITE(zeta):
  # Function to compute stability functions for sensible and latent heat
  if zeta < 0.0:
    X = (1 - 16*zeta)**0.25
    psite = 2*log((1 + X*X)/2)
  elif zeta > 0.0:# Stable case
    if zeta > 0.5:
      if zeta > 10.0:
        psite = log(zeta) - 0.76*zeta - 12.093
      else:
        psite = 0.5/(zeta*zeta) - 4.25/zeta - 7.0*log(zeta) - 0.852
    else: 
      psite = -5*zeta
  else:
    psite = 0.0
  return(psite)

# def sensible(p2, B, Tair, Twater, Uw): # convection / sensible heat
#   Twater = Twater + 273.15
#   Tair = Tair + 273.15
#   fu = 4.4 + 1.82 * Uw + 0.26 *(Twater - Tair)
#   sensible = -1 * ( p2 * B * fu * (Twater - Tair)) 
#   return(sensible)

def sensible(Tair, Twater, Uw, p2, pa, ea, RH, A, Cd = 0.0013): # evaporation / latent heat
  global H
  # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012839
  
  # Tair =0
  # Twater = 0
  # Uw = 0.01
  # pa = 98393
  # ea = 6.079572
  # A = 31861
  # Cd = 0.0037
  
  const_SpecificHeatAir = 1005;           # Units : J kg-1 K-1
  const_vonKarman = 0.41;                 # Units : none
  const_Gravity = 9.81;                   # Units : m s-2
  const_Charnock = Cd;   
  
  U_Z = Uw
  if Uw <= 0:
    U_Z = 1e-3
  T = Tair
  if Tair == 0:
    T = np.random.uniform(low = 1e-7, high = 1e-5)
  T0 = Twater
  if Twater == 0: 
    T0 = np.random.uniform(low = 1e-7, high = 1e-5)
  Rh = RH
  p = pa/100
  z = 2
  
  # Step 2c - Compute saturated vapour pressure at air temperature
  e_s = 6.11*exp(17.27*T/(237.3+T)) # Units : mb ##REF##
  # Step 2d - Compute vapour pressure
  e_a = Rh*e_s/100 # Units : mb
  ### End step 2
  
  ### Step 3 - Compute other values used in flux calculations
  # Step 3a - Compute specific humidity
  q_z = 0.622*e_a/p # Units: kg kg-1
  # Step 3b - Compute saturated vapour pressure at water temperature
  e_sat = 6.11*exp(17.27*T0/(237.3+T0)) # Units : mb ##REF##
  # Step 3c - Compute humidity at saturation (Henderson-Sellers 1986 eqn 36)
  q_s = 0.622*e_sat/p # Units: kg kg-1
  # Step 3d - Compute latent heat of vaporisation
  L_v = 2.501e6-2370*T0 # Units : J kg-1 ** EQUATION FROM PIET ##REF##
  # Step 3e - Compute gas constant for moist air
  R_a = 287*(1+0.608*q_z) # Units : J kg-1 K-1
  # Step 3f - Compute air density
  rho_a = 100*p/(R_a*(T+273.16)) # Units : kg m-3
  # Step 3g - Compute kinematic viscosity of air 
  v = (1./rho_a)*(4.94e-8*T + 1.7184e-5) # Units : m2 s-1
  # Step 3h - Compute virtual air temperature and virtual air-water temperature difference
  T_v = (T+273.16)*(1+0.61*q_z) # Units - K
  T_ov = (T0+273.16)*(1+0.61*q_s) # Units - K
  del_theta = T_ov - T_v
  # Step 3h - Compute water density 
  rho_w = 1000*(1-1.9549*0.00001*abs(T0-3.84)**1.68)
  ### End step 3
  
  # step 4
  u_star = U_Z *sqrt(0.00104+0.0015/(1+exp((-U_Z+12.5)/1.56))) # Amorocho and DeVries, initialise ustar using U_Z
  
  if u_star == 0: 
    u_star = 1e-6
  
  z_0 = (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
  z_0_prev=z_0*1.1 # To initiate the iteration
  

  
  while (abs((z_0 - z_0_prev))/abs(z_0_prev) > 0.000001): # Converge when z_0 within 0.0001# of previous value 
    u_star=const_vonKarman*U_Z/(log(z/z_0))  # Compute u_star
    dummy = z_0 # Used to control while loop
    z_0=(const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star); # Compute new roughness length
    z_0_prev = dummy # Used to control while loop
  
  # Step 4d - Compute initial neutral drag coefficient
  C_DN = (u_star**2)/(U_Z**2) # Units - none
  # Step 4e - Compute roughness Reynolds number 
  Re_star = u_star*z_0/v # Units - none
  # Step 4f - Compute initial roughness length for temperature
  z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57) # Units - m
  z_T = z_T.real # Get real components, and NaN can create imag component despite no data
  # Step 4g - Compute initial roughness length for vapour 
  z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57); # Units - m 
  z_E = z_E.real # Get real components, and NaN can create imag component despite no data
  # Step 4h - Compute initial neutral sensible heat transfer coefficient 
  C_HN = const_vonKarman*sqrt(C_DN)/(log(z/z_T)) 
  # Step 4i - Compute initial neutral latent heat transfer coefficient
  C_EN = const_vonKarman*sqrt(C_DN)/(log(z/z_E))
  ### End step 4
  
  ### Step 5 - Start iteration to compute corrections for atmospheric stability
  # for (i1 in 1:length(U_Z)){

  # Step 5a - Compute initial sensible heat flux based on neutral coefficients
  H_initial = rho_a*const_SpecificHeatAir*C_HN*U_Z*(T0-T) # Units : W m-2
  # Step 5b - Compute initial latent heat flux based on neutral coefficients
  E_initial = rho_a*L_v*C_EN*U_Z*(q_s-q_z) # Units : W m-2
  # Step 5c - Compute initial Monin-Obukhov length
  L_initial = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H_initial/const_SpecificHeatAir + 0.61*E_initial*(T+273.16)/L_v)) # Units - m
  # Step 5d - Compute initial stability parameter
  zeta_initial = z/L_initial
  # Step 5e - Compute initial stability function
  psim=PSIM(zeta_initial) # Momentum stability function
  psit=PSITE(zeta_initial) # Sensible heat stability function
  psie=PSITE(zeta_initial) # Latent heat stability function
  # Step 5f - Compute corrected coefficients
  C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2
  C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
  C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
  # Step 5g - Start iteration
  L_prev = L_initial
  L = L_prev*1.1 # Initialise while loop
  count=np.zeros(1);
  while (abs((L - L_prev))/abs(L_prev) > 0.000001):
    # Iteration counter
    count=count+1;
    if count > 20:
      break
    # Step 5i - Compute new z_O, roughness length for momentum
    z_0= (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
    # Step 5j - Compute new Re_star
    Re_star = u_star*z_0/v
    # Step 5k - Compute new z_T, roughness length for temperature
    z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5l - Compute new z_E, roughness length for vapour
    z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5p - Compute new stability parameter
    zeta = z/L;
    #fprintf('zeta #g\n',zeta);
    # Step 5q - Check and enforce bounds on zeta
    if zeta > 15:
      zeta = 15
    elif zeta < -15 :
      zeta = -15
    # Step 5r - Compute new stability functions
    psim=PSIM(zeta) # Momentum stability function
    psit=PSITE(zeta) # Sensible heat stability function
    psie=PSITE(zeta) # Latent heat stability function
    # Step 5s - Compute corrected coefficients
    C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2;
    C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
    C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
    # Step 5m - Compute new H (now using corrected coefficients)
    H = rho_a*const_SpecificHeatAir*C_H*U_Z*(T0-T);
    # Step 5n - Compute new E (now using corrected coefficients)
    E = rho_a*L_v*C_E*U_Z*(q_s-q_z);
    # Step 5h - Compute new u_star
    u_star=sqrt(C_D*U_Z**2);
    # Step 5o - Compute new Monin-Obukhov length
    dummy = L; # Used to control while loop
    L = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H/const_SpecificHeatAir + 0.61*E*(T+273.16)/L_v));
    L_prev = dummy; # Used to control while loop

  
  sensible = H
  return sensible* (-1)

# def latent(Tair, Twater, Uw, p2, pa, ea, RH): # evaporation / latent heat
#   Twater = Twater + 273.15
#   Tair = Tair + 273.15
#   Pressure = pa / 100
#   fu = 4.4 + 1.82 * Uw + 0.26 *(Twater - Tair)
#   fw = 0.61 * (1 + 10**(-6) * Pressure * (4.5 + 6 * 10**(-5) * Twater**2))
#   ew = fw * 10 * ((0.7859+0.03477* Twater)/(1+0.00412* Twater))
#   latent = -1* fu * p2 * (ew - ea)# * 1.33) // * 1/6
#   return(latent)
def latent(Tair, Twater, Uw, p2, pa, ea, RH, A, Cd = 0.0013): # evaporation / latent heat
  global E
  # https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009JD012839
   
  # Tair =0
  # Twater = 0
  # Uw = 0.01
  # pa = 98393
  # ea = 6.079572
  # A = 31861
  # Cd = 0.0037
  
  const_SpecificHeatAir = 1005;           # Units : J kg-1 K-1
  const_vonKarman = 0.41;                 # Units : none
  const_Gravity = 9.81;                   # Units : m s-2
  const_Charnock = Cd;   
  
  U_Z = Uw
  if Uw <= 0:
    U_Z = 1e-3
  T = Tair
  if Tair == 0:
    T = np.random.uniform(low = 1e-7, high = 1e-5)
  T0 = Twater
  if Twater == 0: 
    T0 = np.random.uniform(low = 1e-7, high = 1e-5)
  Rh = RH
  p = pa/100
  z = 2
  
  # Step 2c - Compute saturated vapour pressure at air temperature
  e_s = 6.11*exp(17.27*T/(237.3+T)) # Units : mb ##REF##
  # Step 2d - Compute vapour pressure
  e_a = Rh*e_s/100 # Units : mb
  ### End step 2
  
  ### Step 3 - Compute other values used in flux calculations
  # Step 3a - Compute specific humidity
  q_z = 0.622*e_a/p # Units: kg kg-1
  # Step 3b - Compute saturated vapour pressure at water temperature
  e_sat = 6.11*exp(17.27*T0/(237.3+T0)) # Units : mb ##REF##
  # Step 3c - Compute humidity at saturation (Henderson-Sellers 1986 eqn 36)
  q_s = 0.622*e_sat/p # Units: kg kg-1
  # Step 3d - Compute latent heat of vaporisation
  L_v = 2.501e6-2370*T0 # Units : J kg-1 ** EQUATION FROM PIET ##REF##
  # Step 3e - Compute gas constant for moist air
  R_a = 287*(1+0.608*q_z) # Units : J kg-1 K-1
  # Step 3f - Compute air density
  rho_a = 100*p/(R_a*(T+273.16)) # Units : kg m-3
  # Step 3g - Compute kinematic viscosity of air 
  v = (1./rho_a)*(4.94e-8*T + 1.7184e-5) # Units : m2 s-1
  # Step 3h - Compute virtual air temperature and virtual air-water temperature difference
  T_v = (T+273.16)*(1+0.61*q_z) # Units - K
  T_ov = (T0+273.16)*(1+0.61*q_s) # Units - K
  del_theta = T_ov - T_v
  # Step 3h - Compute water density 
  rho_w = 1000*(1-1.9549*0.00001*abs(T0-3.84)**1.68)
  ### End step 3
  
  # step 4
  u_star = U_Z *sqrt(0.00104+0.0015/(1+exp((-U_Z+12.5)/1.56))) # Amorocho and DeVries, initialise ustar using U_Z
  
  if u_star == 0: 
    u_star = 1e-6
  
  z_0 = (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
  z_0_prev=z_0*1.1 # To initiate the iteration
  

  
  while (abs((z_0 - z_0_prev))/abs(z_0_prev) > 0.000001): # Converge when z_0 within 0.0001# of previous value 
    u_star=const_vonKarman*U_Z/(log(z/z_0))  # Compute u_star
    dummy = z_0 # Used to control while loop
    z_0=(const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star); # Compute new roughness length
    z_0_prev = dummy # Used to control while loop
  
  # Step 4d - Compute initial neutral drag coefficient
  C_DN = (u_star**2)/(U_Z**2) # Units - none
  # Step 4e - Compute roughness Reynolds number 
  Re_star = u_star*z_0/v # Units - none
  # Step 4f - Compute initial roughness length for temperature
  z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57) # Units - m
  z_T = z_T.real # Get real components, and NaN can create imag component despite no data
  # Step 4g - Compute initial roughness length for vapour 
  z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57); # Units - m 
  z_E = z_E.real # Get real components, and NaN can create imag component despite no data
  # Step 4h - Compute initial neutral sensible heat transfer coefficient 
  C_HN = const_vonKarman*sqrt(C_DN)/(log(z/z_T)) 
  # Step 4i - Compute initial neutral latent heat transfer coefficient
  C_EN = const_vonKarman*sqrt(C_DN)/(log(z/z_E))
  ### End step 4
  
  ### Step 5 - Start iteration to compute corrections for atmospheric stability
  # for (i1 in 1:length(U_Z)){

  # Step 5a - Compute initial sensible heat flux based on neutral coefficients
  H_initial = rho_a*const_SpecificHeatAir*C_HN*U_Z*(T0-T) # Units : W m-2
  # Step 5b - Compute initial latent heat flux based on neutral coefficients
  E_initial = rho_a*L_v*C_EN*U_Z*(q_s-q_z) # Units : W m-2
  # Step 5c - Compute initial Monin-Obukhov length
  L_initial = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H_initial/const_SpecificHeatAir + 0.61*E_initial*(T+273.16)/L_v)) # Units - m
  # Step 5d - Compute initial stability parameter
  zeta_initial = z/L_initial
  # Step 5e - Compute initial stability function
  psim=PSIM(zeta_initial) # Momentum stability function
  psit=PSITE(zeta_initial) # Sensible heat stability function
  psie=PSITE(zeta_initial) # Latent heat stability function
  # Step 5f - Compute corrected coefficients
  C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2
  C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
  C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
  # Step 5g - Start iteration
  L_prev = L_initial
  L = L_prev*1.1 # Initialise while loop
  count=np.zeros(1);
  while (abs((L - L_prev))/abs(L_prev) > 0.000001):
    # Iteration counter
    count=count+1;
    if count > 20:
      break
    # Step 5i - Compute new z_O, roughness length for momentum
    z_0= (const_Charnock*u_star**2./const_Gravity) + (0.11*v/u_star)
    # Step 5j - Compute new Re_star
    Re_star = u_star*z_0/v
    # Step 5k - Compute new z_T, roughness length for temperature
    z_T = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5l - Compute new z_E, roughness length for vapour
    z_E = z_0*exp(-2.67*(Re_star)**(1/4) + 2.57)
    # Step 5p - Compute new stability parameter
    zeta = z/L;
    #fprintf('zeta #g\n',zeta);
    # Step 5q - Check and enforce bounds on zeta
    if zeta > 15:
      zeta = 15
    elif zeta < -15 :
      zeta = -15
    # Step 5r - Compute new stability functions
    psim=PSIM(zeta) # Momentum stability function
    psit=PSITE(zeta) # Sensible heat stability function
    psie=PSITE(zeta) # Latent heat stability function
    # Step 5s - Compute corrected coefficients
    C_D=const_vonKarman*const_vonKarman/(log(z/z_0)-psim)**2;
    C_H=const_vonKarman*sqrt(C_D)/(log(z/z_T)-psit)
    C_E=const_vonKarman*sqrt(C_D)/(log(z/z_E)-psie)
    # Step 5m - Compute new H (now using corrected coefficients)
    H = rho_a*const_SpecificHeatAir*C_H*U_Z*(T0-T);
    # Step 5n - Compute new E (now using corrected coefficients)
    E = rho_a*L_v*C_E*U_Z*(q_s-q_z);
    # Step 5h - Compute new u_star
    u_star=sqrt(C_D*U_Z**2);
    # Step 5o - Compute new Monin-Obukhov length
    dummy = L; # Used to control while loop
    L = (-rho_a*u_star**3*T_v)/(const_vonKarman*const_Gravity*(H/const_SpecificHeatAir + 0.61*E*(T+273.16)/L_v));
    L_prev = dummy; # Used to control while loop
  # Converge when L within 0.0001# or previous L
    
  # Need to iterate separately for each record
  
  
  ### End step 5
  
  # Take real values to remove any complex values that arise from missing data or NaN.
  # C_D=C_D.real 
  # C_E=C_E.real 
  # C_H=C_H.real 
  # z_0=z_0.real 
  # z_E=z_E.real 
  # z_T=z_T.real
  
  # Compute evaporation [mm/day]
  Evap = 86400*1000*E/(rho_w*L_v)
  
  latent = E
  return latent* (-1)

def heating_module(
        un,
        area,
        volume,
        depth,
        nx,
        dt,
        dx,
        ice,
        kd_ice,
        Tair,
        CC,
        ea,
        Jsw,
        Jlw,
        Uw,
        Pa,
        RH,
        kd_light,
        Hi = 0,
        kd_snow = 0.9,
        rho_fw = 1000,
        rho_snow = 910,
        Hs = 0,
        sigma = 5.67e-8,
        albedo = 0.1,
        eps = 0.97,
        emissivity = 0.97,
        p2 = 1,
        Cd = 0.0013,
        sw_factor = 1.0,
        at_factor = 1.0,
        Hgeo = 0.1,
        turb_factor = 1.0):
    
    if ice and Tair <= 0:
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif (ice and Tair >= 0):
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif not ice:
      albedo = 0.1
      IceSnowAttCoeff = 1
    
    ## (1) HEAT ADDITION
    # surface heat flux
    start_time = datetime.datetime.now()
    
    Tair = Tair * at_factor
    Jsw = Jsw * sw_factor
    
    u = un
    
    Q = (longwave(cc = CC, sigma = sigma, Tair = Tair, ea = ea, emissivity = emissivity, Jlw = Jlw) + #longwave(emissivity = emissivity, Jlw = Jlw) +
            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
            latent(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd)/turb_factor + 
            sensible(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd)/turb_factor)  
    
    # heat addition over depth
    
    
    if ice:
        H =  IceSnowAttCoeff * (Jsw )  * np.exp(-(kd_light) * depth)
    else:
        H =  (1- albedo) * (Jsw )  * np.exp(-(kd_light ) * depth)
    
    Hg = (area[:-1]-area[1:])/dx * Hgeo/(4181 * calc_dens(un[0]))
    
    Hg = np.append(Hg, Hg.min())
    
    # u[0] = (un[0] + 
    #     (Q * area[0]/(1)*1/(4184 * calc_dens(un[0]) ) + abs(H[0+1]-H[0]) * area[0]/(dx) * 1/(4184 * calc_dens(un[0]) ) + # dx to 1
    #     Hg[0]) * dt/area[0])
    #   # all layers in between
    # for i in range(1,(nx-1)):
    #      u[i] = un[i] + (abs(H[i+1]-H[i]) * area[i]/(dx) * 1/(4184 * calc_dens(un[i]) ) + Hg[i])* dt/area[i]
    #   # bottom layer
    # u[(nx-1)] = un[(nx-1)] + (abs(H[(nx-1)]-H[(nx-2)]) * area[(nx-1)]/(area[(nx-1)] * dx) * 1/(4181 * calc_dens(un[(nx-1)])) +Hg[(nx-1)]/area[(nx-1)]) * dt
    
    u[0] = (un[0] + 
        (Q * area[0]/(1)*1/(4184 * calc_dens(un[0]) )/(volume[0]) + (H[0] - H[0+1])/dx * area[0]/(4184 * calc_dens(un[0]) )/(area[0]) + # dx to 1
         Hg[0]/(area[0])) * dt)
      # all layers in between
    for i in range(1,(nx-1)):
         u[i] = un[i] + ( (H[i] - H[i+1])/dx * area[i]/(4184 * calc_dens(un[i]) )/(area[i]) + Hg[i]/(area[i]))* dt
      # bottom layer
    # u[(nx-1)] = un[(nx-1)] + ( H[nx-1] * area[(nx-1)]/(area[(nx-1)] ) * 1/(4181 * calc_dens(un[(nx-1)])) + Hg[(nx-1)]/area[(nx-1)]) * dt
    u[(nx-1)] = un[(nx-1)] + ( (H[nx-2] - H[nx-1])/dx * area[(nx-1)] * 1/(4181 * calc_dens(un[(nx-1)]))/area[(nx-1)] + Hg[(nx-1)]/(area[(nx-1)])) * dt


    end_time = datetime.datetime.now()
    print("heating: " + str(end_time - start_time))
    
    dat = {'temp': u,
           'air_temp': Tair,
           'longwave_flux': (longwave(cc = CC, sigma = sigma, Tair = Tair, ea = ea, emissivity = emissivity, Jlw = Jlw) +
            backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps)),
            'latent_flux': latent(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd),
            'sensible_flux': sensible(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd) ,
            'shortwave_flux': H[0] ,
            'light': kd_light,
            'IceSnowAttCoeff': IceSnowAttCoeff}

    
    return dat

def diffusion_module(
        un,
        kzn,
        Uw,
        depth,
        area,
        dx,
        dt,
        nx,
        g = 9.81,
        ice = 0,
        Cd = 0.013,
        diffusion_method = 'hondzoStefan',
        scheme = 'implicit'):
    
    u = un
    dens_u_n2 = calc_dens(un)
    
    kz = kzn
    
    ## (2) DIFFUSION
    # if diffusion_method == 'hendersonSellers':
    #     kz = eddy_diffusivity_hendersonSellers(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw,  43.100948, u, kzn) / 1
    # elif diffusion_method == 'munkAnderson':
    #     kz = eddy_diffusivity_munkAnderson(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw,  43.100948, Cd, u, kzn) / 1
    # elif diffusion_method == 'hondzoStefan':
    #     kz = eddy_diffusivity(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, u, kzn) / 86400
    
    # kzn = kz
    start_time = datetime.datetime.now()
    if scheme == 'implicit':

      
        # IMPLEMENTATION OF CRANK-NICHOLSON SCHEME

        j = len(un)
        y = np.zeros((len(un), len(un)))

        alpha = (area * kzn * dt) / (2 * dx**2)
        
        az = - alpha # subdiagonal
        bz = (area + 2 * alpha) # diagonal
        cz = - alpha # superdiagonal
        
        bz[0] = 1
        bz[len(bz)-1] = 1
        cz[0] = 0
        
        az =  np.delete(az,0)
        cz =  np.delete(cz,len(cz)-1)
        
        # tridiagonal matrix
        for k in range(j-1):
            y[k][k] = bz[k]
            y[k][k+1] = cz[k]
            y[k+1][k] = az[k]
        

        y[j-1, j-2] = 0
        y[j-1, j-1] = 1


        mn = un * 0.0    
        mn[0] = un[0]
        mn[-1] = un[-1]
        
        for k in range(1,j-1):
            mn[k] = alpha[k] * un[k-1] + (area[k] - 2 * alpha[k]) * un[k] + alpha[k] * un[k+1]

    # DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE
        u = np.linalg.solve(y, mn)

    if scheme == 'explicit':
     
      u[0]= un[0]
      u[-1] = un[-1]
      for i in range(1,(nx-1)):
        u[i] = (un[i] + (kzn[i] * dt / dx**2 * (un[i+1] - 2 * un[i] + un[i-1])))
      

    end_time = datetime.datetime.now()
    print("diffusion: " + str(end_time - start_time))
    
    dat = {'temp': u,
           'diffusivity': kz}
    
    return dat

def mixing_module(
        un,
        depth,
        area,
        volume,
        dx,
        dt,
        nx,
        Uw,
        ice,
        g = 9.81,
        Cd = 0.0013,
        KEice = 1/1000
        ):
    
    u = un
    ## (3) TURBULENT MIXING OF MIXED LAYER
    # the mixed layer depth is determined for each time step by comparing kinetic 
    # energy available from wind and the potential energy required to completely 
    # mix the water column to a given depth
    start_time = datetime.datetime.now()
    Zcv = np.sum(depth * area) / sum(area)  # center of volume
    tau = 1.225 * Cd * Uw ** 2 # wind shear is air density times wind velocity 
    if (Uw <= 15):
      c10 = 0.0005 * sqrt(Uw)
    else:
      c10 = 0.0026
    
    un = u
    shear = sqrt((c10 * calc_dens(un[0]))/1.225) *  Uw # shear velocity
    # coefficient times wind velocity squared
    KE = shear *  tau * dt # kinetic energy as function of wind
    
    if ice:
      KE = KE * KEice
    
    maxdep = 0
    for dep in range(0, nx-1):
      if dep == 0:
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
             # abs(calc_dens(u[dep+1])- calc_dens(u[dep])))
             abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0])))))
      else:
        PEprior = deepcopy(PE)
        PE = (abs(g *   depth[dep] *( depth[dep+1] - Zcv)  *
            # abs(calc_dens(u[dep+1])- calc_dens(u[dep]))) + PEprior
            abs(calc_dens(u[dep+1])- np.mean(calc_dens(u[0:(dep+1)])))) + PEprior)
            
      if PE > KE:
        maxdep = dep - 1
        break
      elif dep > 0 and PE < KE:
          u[(dep - 1):(dep+1)] = np.sum(u[(dep-1):(dep+1)] * volume[(dep-1):(dep+1)])/np.sum(volume[(dep-1):(dep+1)])
      
      maxdep = dep
      

    end_time = datetime.datetime.now()
    print("mixing: " + str(end_time - start_time))
    
    dat = {'temp': u,
           'shear': shear,
           'tau': tau}
    
    return dat

def findPeaks(
        data,
        thresh = 0):
    
    varL = len(data)
    locs = np.zeros( varL, dtype = bool)
    peaks = np.ones(varL)
    
    for i in range(1, (varL -1)):
        dit = data[(i-1):(i+1)]
        pkI = int(np.where(dit == max(dit))[0])
        posPeak = max(dit)
        
        if pkI == 1:
            peaks[i] = posPeak
            locs[i] = True    
            
    inds = np.linspace(0, varL-1, varL)
    locs = inds[locs]
    
    if len(locs) >  0:
        vector = np.vectorize(np.int_)
    
        peaks = peaks[vector(locs)]    

        useI = peaks > thresh
        locs = locs[useI]
    
    
    
    return locs

def thermocline_depth(
        un,
        depth,
        area,
        volume,
        dx,
        dt,
        nx):
    
    #breakpoint()

    
    Smin = 0.1
    seasonal = False
    index = False
    mixed_cutoff = 1
    
    temp_diff = un[1:] - un[:-1]
    if max(temp_diff) - min(temp_diff) < mixed_cutoff:
        therm = float("NaN")  
    
    rho = calc_dens(un)
    
    dRhoPerc = 0.15
    numDepths = len(depth)
    rho_diff = (rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1])
    
    thermoInd = int(np.where(rho_diff == max(rho_diff))[0])
    mDrhoZ = rho_diff[rho_diff == max(rho_diff)]
    thermoD = np.mean(depth[thermoInd:(thermoInd+1)])
    
    
    if (thermoInd > 1) & (thermoInd < (numDepths -2)):
        Sdn = - (depth[thermoInd + 1] - depth[thermoInd]) / (rho_diff[thermoInd + 1] - rho_diff[thermoInd])
        Sup = (depth[thermoInd] - depth[thermoInd -1]) / (rho_diff[thermoInd] - rho_diff[thermoInd -1])
        
        upD = depth[thermoInd]
        dnD = depth[thermoInd + 1]
        
        thermoD = dnD * (Sdn / (Sdn + Sup)) + upD * (Sup /(Sdn + Sup))
    
    dRhoCut = max([dRhoPerc * mDrhoZ, Smin])
    locs = findPeaks(data = rho_diff, thresh = dRhoCut)
    
    vector = np.vectorize(np.int_)
    
    if len(locs) == 0:
        SthermoD = thermoD
        SthermoInd = thermoInd
    else:
        pks = rho_diff[vector(locs)]
        
        mDrhoZ = pks[len(pks)-1]
        SthermoInd = vector(locs[len(pks)-1])
        
        if SthermoInd > (thermoInd +1):
            SthermoD = np.mean(depth[SthermoInd:(SthermoInd+1)])
            
            if (SthermoInd > 1) & (SthermoInd < (numDepths -2)):
                Sdn = - (depth[SthermoInd + 1] - depth[SthermoInd]) / (rho_diff[SthermoInd + 1] - rho_diff[SthermoInd])
                Sup = (depth[SthermoInd] - depth[SthermoInd -1]) / (rho_diff[SthermoInd] - rho_diff[SthermoInd -1])
        
                upD = depth[SthermoInd]
                dnD = depth[SthermoInd + 1]
        
                SthermoD = dnD * (Sdn / (Sdn + Sup)) + upD * (Sup /(Sdn + Sup))
        else:
            SthermoD = thermoD
            SthermoInd = thermoInd
    
    if (SthermoD < thermoD):
        SthermoD = thermoD
        SthermoInd = thermoInd
              
    
    return thermoD

def meta_depths(
        un,
        depth,
        area,
        volume,
        dx,
        dt,
        nx):
    
    #breakpoint()
    slope = 0.1
    seasonal = True
    mxied_cutoff = 1
    
    vector = np.vectorize(np.int_)
    
    thermoD = thermocline_depth(un = un, depth =depth, area = area, volume = volume, dx = dx, dt = dt, nx = nx)
    
    if len([thermoD]) < 1:
        upper = float("NaN")  
        lower = float("NaN")  
    else:
        rho = calc_dens(un)
        
        dRhoPerc = 0.15
        numDepths = len(depth)
        
        rho_diff = (rho[1:] - rho[:-1]) / (depth[1:] - depth[:-1])
        
        metaBot_depth = depth[numDepths-1]
        metaTop_depth = depth[0]
        Tdepth = np.ones(numDepths - 1)
        
        for i in range(0, (numDepths -1)):
            Tdepth[i] = np.mean(depth[i:(i+1)])
        
        if (thermoD in Tdepth) == False:
            tmp = np.append(Tdepth, thermoD)
            tmp.sort()
        else:
            tmp = Tdepth
        
        sortDepth = tmp
        numDepths = len(sortDepth)
        
        profile_fun = interp1d(Tdepth, rho_diff)
        out_depths = sortDepth
        drho_dz = profile_fun(out_depths)
  
        thermo_index = np.where(sortDepth == thermoD)
        thermo_index = np.asarray(thermo_index)
        thermo_index = int(thermo_index)
        
        metaBot_depth_one = thermoD
        for i in range(int(thermo_index), (numDepths-1)):
            if drho_dz[i] < slope:
                metaBot_depth_one = sortDepth[i]
                break
        
        if int(thermo_index) == (numDepths-1):
            metaBot_depth_one = sortDepth[numDepths-1]
        
        if ((i - thermo_index) >= 1) & (drho_dz[thermo_index] > slope):
            profile_fun = interp1d(drho_dz[thermo_index:(i+1)], sortDepth[thermo_index:(i+1)],  fill_value ='extrapolate')
            metaBot_depth = profile_fun(slope)
        
        metaTop_depth_one = thermoD
        for i in range(thermo_index, 0,-1):
            if drho_dz[i] < slope:
                metaTop_depth_one = sortDepth[i]
                break
            
        if ((thermo_index - i) >= 1) & (drho_dz[thermo_index] > slope):
            profile_fun = interp1d(drho_dz[i:(thermo_index+1)], sortDepth[i:(thermo_index+1)], fill_value ='extrapolate')
            metaTop_depth = profile_fun(slope)
            
        if (metaTop_depth > thermoD):
            metaTop_depth = thermoD
        if (metaBot_depth < thermoD):
            metaBot_depth = thermoD
        
        if metaTop_depth <= 0:
            metaTop_depth = metaTop_depth_one
        if metaBot_depth >= max(depth):
            metaBot_depth = metaBot_depth_one
       
    
    return metaTop_depth, metaBot_depth, thermoD

def mixing_module_minlake(
        un,
        o2n,
        docln,
        docrn,
        pocln,
        pocrn,
        depth,
        area,
        volume,
        dx,
        dt,
        nx,
        Uw,
        ice,
        g = 9.81,
        Cd = 0.0013,
        KEice = 1/1000,
        W_str = None):
    
    u = un
    start_time = datetime.datetime.now()
    
    
    if W_str is None:
        W_str = 1.0 - exp(-0.3 * max(area/10**6))
    else:
        W_str = W_str
    tau = 1.225 * Cd * Uw ** 2 # wind shear is air density times wind velocity 
    
    KE = W_str * max(area) * sqrt(tau**3/ calc_dens(u[0]) ) * dt
    
    #meta_top, meta_bot, thermD = meta_depths(un = u,
    #                          depth = depth,area = area, volume = volume, dx = dx, dt = dt, nx = nx)
    
    if ice:
        KE = 0.0
    
    o2 =o2n/volume
    docl =docln/volume
    docr =docrn/volume
    pocl =pocln/volume
    pocr = pocrn/volume
    
    o2n =o2n/volume
    docln =docln/volume
    docrn =docrn/volume
    pocln =pocln/volume
    pocrn = pocrn/volume
    

    #idx = np.where(depth > thermD)
    #idx = idx[0][0]
    
    #breakpoint()
    zb = 0
    WmixIndicator = 1
    while WmixIndicator == 1:
        #breakpoint()
        rho = calc_dens(u)
        d_rho = (rho[1:] - rho[:-1]) 
        inx = np.where(d_rho > 0.0)
        MLD = 0
        
        inx_array = np.array(inx)
        if inx_array.size == 0:
            break
        #print(inx)
        zb = int(inx[0][0])
        
        if zb == (len(d_rho) -1):
            break
        MLD = depth[zb]
        dD = d_rho[zb]
        
        Zg = sum(area[0:(zb+2)] *depth[0:(zb+2)]) / sum(area[0:(zb+2)] )
        if zb==0:
            volume_epi = volume[0]
        else:
            volume_epi = sum(volume[0:zb]) 
        V_weight = volume[zb+2] *volume_epi / (volume[zb+2] + volume_epi) 
        POE = (dD * g * V_weight * (MLD + dx/2 - Zg))
        
        

        KP_ratio = KE/POE
        if KP_ratio > 1:
            Tmix = sum((volume[0:(zb+2)] * u[0:(zb+2)])) / sum(volume[0:(zb+2)])
            u[0:(zb+2)] = Tmix
            
            o2mix = sum((volume[0:(zb+2)] * o2[0:(zb+2)])) / sum(volume[0:(zb+2)])
            docrmix = sum((volume[0:(zb+2)] * docr[0:(zb+2)])) / sum(volume[0:(zb+2)])
            doclmix = sum((volume[0:(zb+2)] * docl[0:(zb+2)])) / sum(volume[0:(zb+2)])
            pocrmix = sum((volume[0:(zb+2)] * pocr[0:(zb+2)])) / sum(volume[0:(zb+2)])
            poclmix = sum((volume[0:(zb+2)] * pocl[0:(zb+2)])) / sum(volume[0:(zb+2)])
            
            #breakpoint()
            o2[0:(zb+2)] = o2mix
            docr[0:(zb+2)] = docrmix
            docl[0:(zb+2)] = doclmix
            pocr[0:(zb+2)] = pocrmix
            pocl[0:(zb+2)] = poclmix
    
            
            KE = KE - POE
        else:
            #breakpoint()
            volume_res = volume[0:(zb+1)]
            volume_res = np.append(volume_res, KP_ratio * volume[(zb+2)])
            temp_res = u[0:(zb+2)]
            Tmix = sum(volume_res * temp_res) / sum(volume_res)
            u[0:(zb +1)] = Tmix
            u[(zb +2)] = KP_ratio * Tmix + (1 - KP_ratio) *u[zb+2]
            
            o2_res = o2[0:(zb+2)]
            o2mix = sum(volume_res * o2_res) / sum(volume_res)
            o2[0:(zb +1)] = o2mix
            o2[(zb +2)] = KP_ratio * o2mix + (1 - KP_ratio) *o2[zb+2]
            
            docl_res = docl[0:(zb+2)]
            doclmix = sum(volume_res * docl_res) / sum(volume_res)
            docl[0:(zb +1)] = doclmix
            docl[(zb +2)] = KP_ratio * doclmix + (1 - KP_ratio) *docl[zb+2]
            
            docr_res = docr[0:(zb+2)]
            docrmix = sum(volume_res * docr_res) / sum(volume_res)
            docr[0:(zb +1)] = docrmix
            docr[(zb +2)] = KP_ratio * docrmix + (1 - KP_ratio) *docr[zb+2]
            
            pocl_res = pocl[0:(zb+2)]
            poclmix = sum(volume_res * pocl_res) / sum(volume_res)
            pocl[0:(zb +1)] = poclmix
            pocl[(zb +2)] = KP_ratio * poclmix + (1 - KP_ratio) *pocl[zb+2]
            
            pocr_res = pocr[0:(zb+2)]
            pocrmix = sum(volume_res * pocr_res) / sum(volume_res)
            pocr[0:(zb +1)] = pocrmix
            pocr[(zb +2)] = KP_ratio * pocrmix + (1 - KP_ratio) *pocr[zb+2]
            
            KE = 0
            WmixIndicator = 0
        #print(KE)
            
    # epi_dens = np.mean(calc_dens(u[0:idx]))
    # layer_dens = calc_dens(u[idx])
    # delta_dens =  (layer_dens - epi_dens) 
    # epi_volume = sum(volume[0:idx])
    # layer_volume = volume[idx]
    # volume_ratio = ((epi_volume * layer_volume)/(epi_volume + layer_volume))
    # epi_zg = (np.matmul(area[0:idx], depth[0:idx]) * calc_dens(u[0:idx])) / (sum(area[0:idx]) *calc_dens(u[0:idx])) 
    # delta_depth = meta_top + (depth[idx] - meta_top) -  epi_zg[0]
    # PE = g *delta_dens * volume_ratio * (delta_depth)
    
    
    o2 =o2n*volume
    docl =docln*volume
    docr =docrn*volume
    pocl =pocln*volume
    pocr = pocrn*volume
    
    energy_ratio = KE
    
    end_time = datetime.datetime.now()
    print("mixing: " + str(end_time - start_time))
    
    dat = {'temp': u,
           'o2': o2,
           'docr': docr,
           'docl': docl,
           'pocr': pocr,
           'pocl':pocl,
           'tau': tau,
           'thermo_dep': zb,
           'energy_ratio': energy_ratio}
    
    return dat
    
def convection_module(
        un,
        nx,
        volume,
        denThresh = 1e-3):
    
    u = un
    ## (4) DENSITY INSTABILITIES
    # convective overturn: Convective mixing is induced by an unstable density 
    # profile. All groups of water layers where the vertical density profile is 
    # unstable are mixed with the first stable layer below the unstable layer(s) 
    # (i.e., a layer volume weighed means of temperature and other variables are 
    # calculated for the mixed water column). This procedure is continued until 
    # the vertical density profile in the whole water column becomes neutral or stable.
    start_time = datetime.datetime.now()
    dens_u = calc_dens(u) 
    diff_dens_u = np.diff(dens_u) 
    diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
    un = u 
    while np.any(diff_dens_u < 0):
      dens_u = calc_dens(u)
      for dep in range(0, nx-1):
        if dens_u[dep+1] < dens_u[dep] and abs(dens_u[dep+1] - dens_u[dep]) >= denThresh:
          u[(dep):(dep+2)] = np.sum(u[(dep):(dep+2)] * volume[(dep):(dep+2)])/np.sum(volume[(dep):(dep+2)])
          dens_u = calc_dens(u)#break
      
      dens_u = calc_dens(u)
      diff_dens_u = np.diff(dens_u)
      diff_dens_u[abs(diff_dens_u) <= denThresh] = 0
      
    
    end_time = datetime.datetime.now()
    print("convection: " + str(end_time - start_time))
    
    dat = {'temp': u}
    
    return dat

def ice_module(
        un,
        dt,
        dx,
        area,
        Tair,
        CC,
        ea,
        Jsw,
        Jlw,
        Uw,
        Pa,
        RH,
        PP,
        IceSnowAttCoeff,
        ice = False,
        dt_iceon_avg = 0.8,
        iceT = 6,
        supercooled = 0,
        rho_snow = 250,
        rho_new_snow = 250,
        rho_max_snow = 450,
        rho_ice = 910,
        rho_fw = 1000,
        Ice_min = 0.1,
        Cw = 4.18E6,
        L_ice = 333500,
        meltP = 1,
        Hi = 0,
        Hs = 0,
        Hsi = 0,
        K_ice = 2.1,
        sigma = 5.67e-8,
        emissivity = 0.97,
        eps = 9.97,
        p2 = 1.0,
        Cd = 0.0013
        ):
    
    
    u = un
    ## (5) ICE FORMATION
    # according to Saloranta & Andersen (2007) and ice growth due to Stefan's law
    # (Leppäranta 1991)
    start_time = datetime.datetime.now()
    icep  = max(dt_iceon_avg,  (dt/86400))
    x = (dt/86400) / icep
    iceT = iceT * (1 - x) + u[0] * x

    
    K_snow = 2.22362 * (rho_snow/1000)**1.885
    Tice = 0
    
    
    if (iceT <= 0) and Hi < Ice_min and Tair <= 0 and ice == False:
      supercooled = u < 0
      initEnergy = np.sum((0-u[supercooled])*area[supercooled] * dx * Cw)
      
      Hi = Ice_min+(initEnergy/(910*L_ice))/np.max(area)
      
      ice = True
      
      if Hi >= 0:
          
        u[supercooled] = 0
        u[0] = 0
        
    elif ice == True and Hi >= Ice_min:
        Q_surf = (u[0] - 0) * Cw * dx
        u[0] = 0
        
        if Tair > 0:
            Tice = 0
            dHsnew = 0
            
            if (Hs > 0):
                dHs = (-1) * np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw + (longwave(cc = CC, sigma = sigma, Tair = Tair, ea = ea, emissivity = emissivity, Jlw = Jlw) + #longwave(emissivity = emissivity, Jlw = Jlw) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd)) ))/ (rho_fw * L_ice)])
                if (Hs + dHs) < 0:
                    Hi_new = Hi + (Hs + dHs) * (rho_fw/rho_ice)
                else:
                    Hi_new = Hi
            else:
                dHs = 0
                
                Hi_new = Hi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw + (longwave(cc = CC, sigma = sigma, Tair = Tair, ea = ea, emissivity = emissivity, Jlw = Jlw) + #longwave(emissivity = emissivity, Jlw = Jlw) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                Hsi = Hsi - np.max([0, meltP * dt * (((1 - IceSnowAttCoeff) * Jsw + (longwave(cc = CC, sigma = sigma, Tair = Tair, ea = ea, emissivity = emissivity, Jlw = Jlw) + #longwave(emissivity = emissivity, Jlw = Jlw) +
                                                                                   backscattering(emissivity = emissivity, sigma = sigma, Twater = un[0], eps = eps) +
                                                                                   latent(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd) + 
                                                                                   sensible(Tair = Tair, Twater = un[0], Uw = Uw, p2 = p2, pa = Pa, ea=ea, RH = RH, A = area, Cd = Cd)) ))/ (rho_ice * L_ice)])
                if Hsi <= 0:
                    Hsi = 0
        else:
            if Hs > 0:
                K_snow = 2.22362 * (rho_snow/1000)**(1.885)
                p = (K_ice/K_snow) * (((rho_fw/rho_snow) * Hs ) / Hi)
                dHsi = np.max([0, Hi * (rho_ice/rho_fw -1) + Hs])
                Hsi = Hsi + dHsi

            else:
                p = 1/(10 * Hi)
                dHsi = 0
            
            Tice = (p * 0 + Tair) / (1 + p)
            Hi_new = np.sqrt((Hi + dHsi)**2 + 2 * K_ice/(rho_ice * L_ice)* (0 - Tice) * dt)
            
            # PRECIPITATION
            dHsnew = PP * 1/(1000 * 86400) * dt

            dHs = dHsnew - dHsi * (rho_ice/rho_fw)
            dHsi = 0   

                
        Hi = Hi_new - np.max([0,(Q_surf/(rho_ice * L_ice))])
        

    
        Q_surf = 0

        Hs = Hs + dHs
        
        

    
    
        if Hi < Hsi:
            Hsi = np.max([0, Hi])
        
        if Hs <= 0:
            Hs = 0
            rho_snow = rho_new_snow
        else:
            rho_snow = rho_snow * (Hs - dHsnew) / Hs + rho_new_snow * dHsnew/Hs
    elif ice and Hi <= Ice_min:
        ice = False
    
    if (ice == False):
        Hi = 0
        Hs = 0
        Hsi = 0
    
    end_time = datetime.datetime.now()
    print("ice: " + str(end_time - start_time))
    dat = {'temp': u,
            'icethickness': Hi,
            'snowthickness': Hs,
            'snowicethickness': Hsi,
            'iceFlag': ice,
            'icemovAvg': iceT,
            'supercooled': supercooled,
            'density_snow': rho_snow}
    
    return dat

def do_sat_calc(temp, baro, altitude = 0, salinity = 0):
    mgL_mlL = 1.42905
    
    mmHg_mb = 0.750061683 # conversion from mm Hg to millibars
    if baro is None:
        mmHg_inHg = 25.3970886 # conversion from inches Hg to mm Hg
        standard_pressure_sea_level = 29.92126 # Pb, inches Hg
        standard_temperature_sea_level = 15 + 273.15 # Tb, 15 C = 288.15 K
        gravitational_acceleration = 9.80665 # g0, m/s^2
        air_molar_mass = 0.0289644 # M, molar mass of Earth's air (kg/mol)
        universal_gas_constant = 8.31447 #8.31432 # R*, N*m/(mol*K)
        
        # estimate pressure by the barometric formula
        baro = (1/mmHg_mb) * mmHg_inHg * standard_pressure_sea_level * exp((-gravitational_acceleration * air_molar_mass * altitude) / (universal_gas_constant * standard_temperature_sea_level))
    
    u = 10 ** (8.10765 - 1750.286 / (235 + temp)) # u is vapor pressure of water; water temp is used as an approximation for water & air temp at the air-water boundary
    press_corr = (baro*mmHg_mb - u) / (760 - u) # pressure correction is ratio of current to standard pressure after correcting for vapor pressure of water. 0.750061683 mmHg/mb
    
    ts = log((298.15 - temp)/(273.15 + temp))
    o2_sat = 2.00907 + 3.22014*ts + 4.05010*ts**2 + 4.94457*ts**3 + -2.56847e-1*ts**4 + 3.88767*ts**5 - salinity*(6.24523e-3 + 7.37614e-3*ts + 1.03410e-2*ts**2 + 8.17083e-3*ts**3) - 4.88682e-7*salinity**2
    return exp(o2_sat) * mgL_mlL * press_corr

def boundary_module(
        un,
        o2n,
        docln,
        docrn,
        pocln,
        pocrn,
        area,
        volume,
        depth,
        nx,
        dt,
        dx,
        ice,
        kd_ice,
        Tair,
        CC,
        ea,
        Jsw,
        Jlw,
        Uw,
        Pa,
        RH,
        kd_light,
        TP,
        Hi = 0,
        kd_snow = 0.9,
        rho_fw = 1000,
        rho_snow = 910,
        Hs = 0,
        sigma = 5.67e-8,
        albedo = 0.1,
        eps = 0.97,
        emissivity = 0.97,
        p2 = 1,
        Cd = 0.0013,
        sw_factor = 1.0,
        at_factor = 1.0,
        turb_factor = 1.0,
        wind_factor = 1.0,
        p_max = 1.0/86400,
        IP = 0.1,
        delta= 1.08,
        conversion_constant = 0.1,
        sed_sink = -1.0 / 86400,
        k_half = 0.5,
        piston_velocity = 1.0,
        sw_to_par = 2.114):
    
    if ice and Tair <= 0:
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif (ice and Tair >= 0):
      albedo = 0.3
      IceSnowAttCoeff = exp(-kd_ice * Hi) * exp(-kd_snow * (rho_fw/rho_snow)* Hs)
    elif not ice:
      albedo = 0.1
      IceSnowAttCoeff = 1
    
    ## (1) HEAT ADDITION
    # surface heat flux
    start_time = datetime.datetime.now()
    
    Tair = Tair * at_factor
    Jsw = Jsw * sw_factor
    Uw = Uw * wind_factor
    
    u = un
    o2 = o2n
    docr = docrn
    docl = docln
    pocr = pocrn
    pocl = pocln

    # light attenuation
    
    if ice:
        H =  IceSnowAttCoeff * (Jsw )  * np.exp(-(kd_light) * depth)
    else:
        H =  (1- albedo) * (Jsw )  * np.exp(-(kd_light ) * depth)
    
    
    if ice:
        piston_velocity = 1e-5 / 86400
        IP_m = IP / 10
    else:
        IP_m = IP
        
    #npp = p_max * (1 - np.exp(-IP * H/p_max)) * TP * conversion_constant * delta**(u - 20) * volume
    npp = H * sw_to_par * IP_m * TP  * delta**(u - 20) * volume
    
    #breakpoint()
    o2 = o2n + dt * npp * 32/12 
    docr = docrn + dt * npp * (0.0)
    docl = docln + dt * npp * (0.2)
    pocr = pocrn + dt * npp * (0.0)
    pocl = pocln + dt * npp * (0.8)
    
    #breakpoint()
    
    
    o2[0] = (o2[0] +  # m/s g/m3 m2
        (piston_velocity * (do_sat_calc(u[0], 982.2, altitude = 258) - o2[0]/volume[0]) * area[0] ) * dt)
    
    o2[(nx-1)] = o2[(nx-1)] + (delta**(u[(nx-1)] - 20) * sed_sink * area[nx-1] * o2[nx-1]/volume[nx-1]/(k_half +  o2[nx-1]/volume[nx-1])) * dt

    #breakpoint()
    end_time = datetime.datetime.now()
    print("wq boundary flux: " + str(end_time - start_time))
    
    dat = {'o2': o2,
           'docr': docr,
           'docl': docl,
           'pocr': pocr,
           'pocl':pocl,
           'npp': npp}

    
    return dat

def prodcons_module(
        un,
        o2n,
        docln,
        docrn,
        pocln,
        pocrn,
        area,
        volume,
        depth,
        nx,
        dt,
        dx,
        delta= 1.08,
        k_half = 0.5,
        resp_docr = -0.001,
        resp_docl = -0.01,
        resp_poc = -0.1): 

    
    ## (1) HEAT ADDITION
    # surface heat flux
    start_time = datetime.datetime.now()
    

    
    u = un
    o2 = o2n
    docr = docrn
    docl = docln
    pocr = pocrn
    pocl = pocln

    # light attenuation
    
    
    def fun(y, a, consumption):
        #"Production and destruction term for a simple linear model."
        o2n, docrn, docln, pocrn, pocln = y
        resp_docr, resp_docln, resp_poc = a
        consumption = consumption
        p = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]]
        d = [[0, 32/12 * (docrn * resp_docr * consumption), 32/12 *(docln * resp_docl * consumption), 32/12 * (pocrn * resp_poc * consumption), 32/12 * (pocln * resp_poc * consumption)],
         [0, (docrn * resp_docr * consumption), 0, 0, 0],
         [0, 0, (docln * resp_docl * consumption), 0, 0],
         [0, 0, 0, (pocrn * resp_poc * consumption), 0],
         [0, 0, 0, 0, (pocln * resp_poc * consumption)]]
        return p,d

    def solve_mprk(fun, y0, dt, resp, delta, u, volume, k_half):
        
        #breakpoint()
        len_y0 = len(y0)
        # t = np.arange(*t_span, step=dt)
        y = np.zeros([len_y0, 1])
        y[:, 0] = y0
        eye = np.identity(len_y0, dtype=bool)
        a = np.zeros_like(eye, dtype=float)
        r = np.zeros_like(a[:, 0], dtype=float)
        
        consumption =  delta**(u-20) * (y[0]/volume)/(k_half +  y[0]/volume)
        
        ci =0
        # Get the production and destruction term:
        p0, d0 = fun(y[:, ci],  resp, consumption)
        p0 = np.asarray(p0)
        d0 = np.asarray(d0)

        # Calculate diagonal:
        a[eye] = dt * d0.sum(1) / y[:, ci] + 1

        # Calculate non-diagonal:
        c_rep = np.broadcast_to(y[:, ci], (len_y0, len_y0))
        a[~eye] = -dt * p0[~eye] / c_rep[~eye]

        # Something:
        r[:] = y[:, ci] + dt*p0[eye]

        # Solve system of equation:
        c0 = np.linalg.solve(a, r)

        # Run the algorithm a second time:
        # Get the production and destruction term:
        p, d = fun(c0,  resp, consumption)
        p = np.asarray(p)
        d = np.asarray(d)

        # Calculate the mean value of the terms:
        p = 0.5 * (p0 + p)
        d = 0.5 * (d0 + d)

        # Calculate diagonal:
        a[eye] = dt * d.sum(1) / c0 + 1

        # Calculate non-diagonal:
        c_rep = np.broadcast_to(c0, (len_y0, len_y0))
        a[~eye] = -dt * p[~eye] / c_rep[~eye]

        # Something:
        r[:] = y[:, ci] + dt*p[eye]

        # Solve system of equation:
        y = np.linalg.solve(a, r)
        # breakpoint()
        return [y, 86400 * resp[0] * consumption, 86400 * resp[1] * consumption, 86400 * resp[2] * consumption]

    docr_respiration = o2 * 0.0
    docl_respiration = o2 * 0.0
    poc_respiration = o2 * 0.0
    
    
    for dep in range(0, nx-1):
        mprk_res = solve_mprk(fun, y0 =  [o2n[dep], docrn[dep], docln[dep], pocrn[dep], pocln[dep]], dt = dt, 
               resp = [resp_docr, resp_docl, resp_poc], delta = delta, u = u[dep],
               volume = volume[dep], k_half = k_half)
        o2[dep], docr[dep], docl[dep], pocr[dep], pocl[dep] = mprk_res[0]
        docr_respiration[dep], docl_respiration[dep], poc_respiration[dep] = [mprk_res[1], mprk_res[2], mprk_res[3]]

    
    # breakpoint()
    # o2 = o2n + dt * consumption * (docrn + docln + pocrn + pocln) * (resp_docr + resp_docl + 2* resp_poc)
    # docr = docrn + dt * consumption * (docrn * resp_docr)
    # docl = docln + dt * consumption * (docln * resp_docl)
    # pocr = pocrn + dt * consumption * (pocrn * resp_poc)
    # pocl = pocln + dt * consumption * (pocln * resp_poc)
    
    end_time = datetime.datetime.now()
    print("wq production and consumption: " + str(end_time - start_time))
    
    dat = {'o2': o2,
           'docr': docr,
           'docl': docl,
           'pocr': pocr,
           'pocl':pocl,
           'docr_respiration': docr_respiration,
           'docl_respiration': docl_respiration,
           'poc_respiration': poc_respiration}

    
    return dat

def transport_module(
        un,
        o2n,
        docrn,
        docln,
        pocrn,
        pocln,
        kzn,
        Uw,
        depth,
        area,
        volume, 
        dx,
        dt,
        nx,
        g = 9.81,
        ice = 0,
        Cd = 0.013,
        diffusion_method = 'hondzoStefan',
        scheme = 'implicit',
        settling_rate = 0.3,
        sediment_rate = 0.03,
        thermo_dep = 0.0):

    u = un
    dens_u_n2 = calc_dens(un)
    
    kz = kzn
    
    zb = thermo_dep
    
  
    # o2mix = sum((volume[0:(zb+2)] * o2n[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # docrmix = sum((volume[0:(zb+2)] * docrn[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # doclmix = sum((volume[0:(zb+2)] * docln[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # pocrmix = sum((volume[0:(zb+2)] * pocrn[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # poclmix = sum((volume[0:(zb+2)] * pocln[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # o2n[0:(zb+2)] = o2mix
    # docrn[0:(zb+2)] = docrmix
    # docln[0:(zb+2)] = doclmix
    # pocrn[0:(zb+2)] = pocrmix
    # pocln[0:(zb+2)] = poclmix
    
    o2n = o2n / volume
    docrn = docrn / volume 
    docln = docln / volume
    pocr = pocrn
    pocl = pocln
    


    start_time = datetime.datetime.now()
    if scheme == 'implicit':

      
        # IMPLEMENTATION OF CRANK-NICHOLSON SCHEME

        j = len(un)
        y = np.zeros((len(un), len(un)))

        alpha = (area * kzn * dt) / (2 * dx**2)
        
        az = - alpha # subdiagonal
        bz = (area + 2 * alpha) # diagonal
        cz = - alpha # superdiagonal
        
        bz[0] = 1
        bz[len(bz)-1] = 1
        cz[0] = 0
        
        az =  np.delete(az,0)
        cz =  np.delete(cz,len(cz)-1)
        
        # tridiagonal matrix
        for k in range(j-1):
            y[k][k] = bz[k]
            y[k][k+1] = cz[k]
            y[k+1][k] = az[k]
        

        y[j-1, j-2] = 0
        y[j-1, j-1] = 1
        
        mn = un * 0.0    
        mn[0] = un[0]
        mn[-1] = un[-1]
        
        for k in range(1,j-1):
            mn[k] = alpha[k] * un[k-1] + (area[k] - 2 * alpha[k]) * un[k] + alpha[k] * un[k+1]

    # DERIVED TEMPERATURE OUTPUT FOR NEXT MODULE
        u = np.linalg.solve(y, mn)

        mn = o2n * 0.0    
        mn[0] = o2n[0]
        mn[-1] = o2n[-1]
        
        for k in range(1,j-1):
            mn[k] = alpha[k] * o2n[k-1] + (area[k] - 2 * alpha[k]) * o2n[k] + alpha[k] * o2n[k+1]
        o2 = np.linalg.solve(y, mn)* volume
        
        mn = docrn * 0.0    
        mn[0] = docrn[0]
        mn[-1] = docrn[-1]
        
        for k in range(1,j-1):
            mn[k] = alpha[k] * docrn[k-1] + (area[k] - 2 * alpha[k]) * docrn[k] + alpha[k] * docrn[k+1]
        docr = np.linalg.solve(y, mn) * volume
        
        mn = docln * 0.0    
        mn[0] = docln[0]
        mn[-1] = docln[-1]
        
        for k in range(1,j-1):
            mn[k] = alpha[k] * docln[k-1] + (area[k] - 2 * alpha[k]) * docln[k] + alpha[k] * docln[k+1]
        docl = np.linalg.solve(y, mn) * volume
        
   
    sinking_loss_pocl = pocln *  settling_rate/dx
    pocl[:-1] = pocln[:-1] - dt * sinking_loss_pocl[:-1]
    pocl[1:] = pocl[1:] + dt * sinking_loss_pocl[:-1]
    pocl[(nx-1)] = pocl[(nx-1)] - dt * pocl[(nx-1)] * sediment_rate/dx
    
    sinking_loss_pocr = pocrn *  settling_rate/dx
    pocr[:-1] = pocrn[:-1] - dt * sinking_loss_pocr[:-1]
    pocr[1:] = pocr[1:] + dt * sinking_loss_pocr[:-1]
    pocr[(nx-1)] = pocr[(nx-1)] - dt * pocr[(nx-1)] * sediment_rate/dx
    
    # o2mix = sum((volume[0:(zb+2)] * o2[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # docrmix = sum((volume[0:(zb+2)] * docr[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # doclmix = sum((volume[0:(zb+2)] * docl[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # pocrmix = sum((volume[0:(zb+2)] * pocr[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # poclmix = sum((volume[0:(zb+2)] * pocl[0:(zb+2)])) / sum(volume[0:(zb+2)])
    # o2[0:(zb+2)] = o2mix
    # docr[0:(zb+2)] = docrmix
    # docl[0:(zb+2)] = doclmix
    # pocr[0:(zb+2)] = pocrmix
    # pocl[0:(zb+2)] = poclmix

    if pocr[(nx-1)] < 0:
        pocr[(nx-1)]  = 0
    if pocl[(nx-1)] < 0:
        pocl[(nx-1)]  = 0

    end_time = datetime.datetime.now()
    print("wq transport: " + str(end_time - start_time))
    
    dat = {'o2': o2,
           'docr': docr,
           'docl': docl,
           'pocr': pocr,
           'pocl':pocl}
    
    return dat

def run_wq_model(
  u, 
  o2,
  docr,
  docl,
  pocr,
  pocl,
  startTime, 
  endTime,
  area,
  volume,
  depth,
  zmax,
  nx,
  dt,
  dx,
  daily_meteo,
  secview,
  phosphorus_data,
  mean_depth,
  ice=False,
  Hi=0,
  iceT=6,
  supercooled=0,
  diffusion_method = 'hendersonSellers',
  scheme='implicit',
  km = 1.4 * 10**(-7),
  k0 = 1 * 10**(-2),
  weight_kz = 0.5, 
  kd_light=None,
  denThresh=1e-3,
  albedo=0.1,
  eps=0.97,
  emissivity=0.97,
  sigma=5.67e-8,
  sw_factor = 1.0,
  wind_factor = 1.0,
  at_factor = 1.0,
  turb_factor = 1.0,
  p2=1,
  B=0.61,
  g=9.81,
  Cd=0.0013, # momentum coeff (wind)
  meltP=1,
  dt_iceon_avg=0.8,
  Hgeo=0.1, # geothermal heat
  KEice=1/1000,
  Ice_min=0.1,
  pgdl_mode='on',
  Hs = 0,
  rho_snow = 250,
  Hsi = 0,
  rho_ice = 910,
  rho_fw = 1000,
  rho_new_snow = 250,
  rho_max_snow = 450,
  K_ice = 2.1,
  Cw = 4.18E6,
  L_ice = 333500,
  kd_snow = 0.9,
  kd_ice = 0.7,p_max = 1.0/86400,
  IP = 0.1,
  delta= 1.08,
  conversion_constant = 0.1,
  sed_sink = -1.0 / 86400,
  k_half = 0.5,
  resp_docr = -0.001,
  resp_docl = -0.01,
  resp_poc = -0.1,
  settling_rate = 0.3,
  sediment_rate = 0.01,
  piston_velocity = 1.0,
  light_water = 0.125,
  light_doc = 0.02,
  light_poc = 0.7,
  W_str = None):
    
  ## linearization of driver data, so model can have dynamic step
  Jsw_fillvals = tuple(daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values[[0, -1]])
  Jsw = interp1d(daily_meteo.dt.values, daily_meteo.Shortwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jsw_fillvals, bounds_error=False)
  Jlw_fillvals = tuple(daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values[[0,-1]])
  Jlw = interp1d(daily_meteo.dt.values, daily_meteo.Longwave_Radiation_Downwelling_wattPerMeterSquared.values, kind = "linear", fill_value=Jlw_fillvals, bounds_error=False)
  Tair_fillvals = tuple(daily_meteo.Air_Temperature_celsius.values[[0,-1]])
  Tair = interp1d(daily_meteo.dt.values, daily_meteo.Air_Temperature_celsius.values, kind = "linear", fill_value=Tair_fillvals, bounds_error=False)
  ea_fillvals = tuple(daily_meteo.ea.values[[0,-1]])
  ea = interp1d(daily_meteo.dt.values, daily_meteo.ea.values, kind = "linear", fill_value=ea_fillvals, bounds_error=False)
  Uw_fillvals = tuple(daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values[[0, -1]])
  Uw = interp1d(daily_meteo.dt.values, wind_factor * daily_meteo.Ten_Meter_Elevation_Wind_Speed_meterPerSecond.values, kind = "linear", fill_value=Uw_fillvals, bounds_error=False)
  CC_fillvals = tuple(daily_meteo.Cloud_Cover.values[[0,-1]])
  CC = interp1d(daily_meteo.dt.values, daily_meteo.Cloud_Cover.values, kind = "linear", fill_value=CC_fillvals, bounds_error=False)
  Pa_fillvals = tuple(daily_meteo.Surface_Level_Barometric_Pressure_pascal.values[[0,-1]])
  Pa = interp1d(daily_meteo.dt.values, daily_meteo.Surface_Level_Barometric_Pressure_pascal.values, kind = "linear", fill_value=Pa_fillvals, bounds_error=False)
  if kd_light is None:
      kd_fillvals = tuple(secview.kd.values[[0,-1]])
      kd = interp1d(secview.dt.values, secview.kd.values, kind = "linear", fill_value=kd_fillvals, bounds_error=False)
  RH_fillvals = tuple(daily_meteo.Relative_Humidity_percent.values[[0,-1]])
  RH = interp1d(daily_meteo.dt.values, daily_meteo.Relative_Humidity_percent.values, kind = "linear", fill_value=RH_fillvals, bounds_error=False)
  PP_fillvals = tuple(daily_meteo.Precipitation_millimeterPerDay.values[[0,-1]])
  PP = interp1d(daily_meteo.dt.values, daily_meteo.Precipitation_millimeterPerDay.values, kind = "linear", fill_value=PP_fillvals, bounds_error=False)
  TP_fillvals = tuple(phosphorus_data.tp.values[[0,-1]])
  TP = interp1d(phosphorus_data.dt.values, phosphorus_data.tp.values, kind = "linear", fill_value=TP_fillvals, bounds_error=False)

  
  step_times = np.arange(startTime*dt, endTime*dt, dt)
  nCol = len(step_times)
  um = np.full([nx, nCol], np.nan)
  kzm = np.full([nx, nCol], np.nan)
  mix_z = np.full([1,nCol], np.nan)
  kd_lightm = np.full([1,nCol], np.nan)
  Him= np.full([1,nCol], np.nan)
  Hsm= np.full([1,nCol], np.nan)
  Hsim= np.full([1,nCol], np.nan)
  thermo_depm = np.full([1,nCol], np.nan)
  energy_ratiom = np.full([1,nCol], np.nan)
  

  um_initial = np.full([nx, nCol], np.nan)
  um_heat = np.full([nx, nCol], np.nan)
  um_diff = np.full([nx, nCol], np.nan)
  um_mix = np.full([nx, nCol], np.nan)
  um_conv = np.full([nx, nCol], np.nan)
  um_ice = np.full([nx, nCol], np.nan)
  n2m = np.full([nx, nCol], np.nan)
  meteo_pgdl = np.full([28, nCol], np.nan)
  
  o2_initial = np.full([nx, nCol], np.nan)
  o2_bc = np.full([nx, nCol], np.nan)
  o2_pd = np.full([nx, nCol], np.nan)
  o2_diff = np.full([nx, nCol], np.nan)
  o2m = np.full([nx, nCol], np.nan)
  
  docl_initial = np.full([nx, nCol], np.nan)
  docl_bc = np.full([nx, nCol], np.nan)
  docl_pd = np.full([nx, nCol], np.nan)
  docl_diff = np.full([nx, nCol], np.nan)
  doclm = np.full([nx, nCol], np.nan)
  
  docr_initial = np.full([nx, nCol], np.nan)
  docr_bc = np.full([nx, nCol], np.nan)
  docr_pd = np.full([nx, nCol], np.nan)
  docr_diff = np.full([nx, nCol], np.nan)
  docrm = np.full([nx, nCol], np.nan)
  
  pocl_initial = np.full([nx, nCol], np.nan)
  pocl_bc = np.full([nx, nCol], np.nan)
  pocl_pd = np.full([nx, nCol], np.nan)
  pocl_diff = np.full([nx, nCol], np.nan)
  poclm = np.full([nx, nCol], np.nan)
  
  pocr_initial = np.full([nx, nCol], np.nan)
  pocr_bc = np.full([nx, nCol], np.nan)
  pocr_pd = np.full([nx, nCol], np.nan)
  pocr_diff = np.full([nx, nCol], np.nan)
  pocrm = np.full([nx, nCol], np.nan)
  
  nppm = np.full([nx, nCol], np.nan)
  docr_respirationm = np.full([nx, nCol], np.nan)
  docl_respirationm = np.full([nx, nCol], np.nan)
  poc_respirationm = np.full([nx, nCol], np.nan)
  
  if not kd_light is None:
    def kd(n): # using this shortcut for now / testing if it works
      return kd_light

  
  #breakpoint()
  #times = np.arange(startTime, endTime, dt)
  times = np.arange(startTime * dt, endTime * dt, dt)
  for idn, n in enumerate(times):
    
    print(idn)
          
    un = deepcopy(u)
    un_initial = un
    #breakpoint()
    
    depth_limit = mean_depth
    # depth_limit = 1
    
    sum_doc = (docr[depth < depth_limit] + docl[depth < depth_limit] )/volume[depth < depth_limit] 
    sum_poc = (pocr[depth < depth_limit]  + pocl[depth < depth_limit] )/volume[depth < depth_limit] 
    kd_light = light_water +  light_doc * np.mean(sum_doc) + light_poc * np.mean(sum_poc)
    

    time_ind = np.where(times == n)
    
    um_initial[:, idn] = u
    o2_initial[:, idn] = o2
    docr_initial[:, idn] = docr
    docl_initial[:, idn] = docl
    pocr_initial[:, idn] = pocr
    pocl_initial[:, idn] = pocl
    

    ## (1) HEATING
    heating_res = heating_module(
        un = u,
        area = area,
        volume = volume,
        depth = depth, 
        nx = nx,
        dt = dt,
        dx = dx,
        ice = ice,
        kd_ice = kd_ice,
        Tair = Tair(n),
        CC = CC(n),
        ea = ea(n),
        Jsw = Jsw(n),
        Jlw = Jlw(n),
        Uw = Uw(n),
        Pa= Pa(n),
        RH = RH(n),
        kd_light = kd_light,
        Hi = Hi,
        rho_snow = rho_snow,
        Hs = Hs,
        at_factor = at_factor,
        sw_factor = sw_factor,
        turb_factor = turb_factor)
    
    u = heating_res['temp']
    IceSnowAttCoeff = heating_res['IceSnowAttCoeff']
    
    plt.plot(u, color = 'red')
    
    um_heat[:, idn] = u
    
    ## (5) ICE AND SNOW
    ice_res = ice_module(
        un = u,
        dt = dt,
        dx = dx,
        area = area,
        Tair = Tair(n),
        CC = CC(n),
        ea = ea(n),
        Jsw = Jsw(n),
        Jlw = Jlw(n),
        Uw = Uw(n),
        Pa= Pa(n),
        RH = RH(n),
        PP = PP(n),
        IceSnowAttCoeff = IceSnowAttCoeff,
        ice = ice,
        dt_iceon_avg = dt_iceon_avg,
        iceT = iceT,
        supercooled = supercooled,
        rho_snow = rho_snow,
        Hi = Hi,
        Hsi = Hsi,
        Hs = Hs)
    
    u = ice_res['temp']
    Hi = ice_res['icethickness']
    Hs = ice_res['snowthickness']
    Hsi = ice_res['snowicethickness']
    ice = ice_res['iceFlag']
    iceT = ice_res['icemovAvg']
    supercooled = ice_res['supercooled']
    rho_snow = ice_res['density_snow']
    
    plt.plot(u, color = 'blue')
    
    um_ice[:, idn] = u
    

    
    dens_u_n2 = calc_dens(u)
    if 'kz' in locals():
        1+1
    else: 
        kz = u * 0.0
        
    if diffusion_method == 'hendersonSellers':
        kz = eddy_diffusivity_hendersonSellers(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw(n),  43.100948, u, kz, Cd, km, weight_kz, k0) / 1
    elif diffusion_method == 'munkAnderson':
        kz = eddy_diffusivity_munkAnderson(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw(n),  43.100948, Cd, u, kz) / 1
    elif diffusion_method == 'hondzoStefan':
        kz = eddy_diffusivity(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, u, kz) / 86400
    elif diffusion_method == 'pacanowskiPhilander':
        kz = eddy_diffusivity_pacanowskiPhilander(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw(n),  43.100948, u, kz, Cd, km, weight_kz, k0) / 1
    
    
    ## (2) DIFFUSION
    diffusion_res = diffusion_module(
        un = u,
        kzn = kz,
        Uw = Uw(n),
        depth= depth,
        dx = dx,
        area = area,
        dt = dt,
        nx = nx,
        ice = ice, 
        diffusion_method = diffusion_method,
        scheme = scheme)
    
    u = diffusion_res['temp']
    kz = diffusion_res['diffusivity']
    
    plt.plot(u, color = 'purple')
    
    kzm[:,idn] = kz
    um_diff[:, idn] = u
    
    
    
    ## (WQ1) BOUNDARY ADDITION
    boundary_res = boundary_module(
        un = u,
        o2n = o2,
        docrn = docr,
        docln = docl,
        pocrn = pocr,
        pocln = pocl,
        area = area,
        volume = volume,
        depth = depth, 
        nx = nx,
        dt = dt,
        dx = dx,
        ice = ice,
        kd_ice = kd_ice,
        Tair = Tair(n),
        CC = CC(n),
        ea = ea(n),
        Jsw = Jsw(n),
        Jlw = Jlw(n),
        Uw = Uw(n),
        Pa= Pa(n),
        RH = RH(n),
        kd_light = kd_light,
        TP = TP(n),
        Hi = Hi,
        rho_snow = rho_snow,
        Hs = Hs,
        at_factor = at_factor,
        sw_factor = sw_factor,
        turb_factor = turb_factor,
        wind_factor = wind_factor,
        p_max = p_max,
        IP = IP,
        delta= delta,
        conversion_constant = conversion_constant,
        sed_sink = sed_sink,
        k_half = k_half,
        piston_velocity = piston_velocity)
    
    o2 = boundary_res['o2']
    docr = boundary_res['docr']
    docl = boundary_res['docl']
    pocr = boundary_res['pocr']
    pocl = boundary_res['pocl']
    npp = boundary_res['npp']

    o2_bc[:, idn] = o2
    docr_bc[:, idn] = docr
    docl_bc[:, idn] = docl
    pocr_bc[:, idn] = pocr
    pocl_bc[:, idn] = pocl
    nppm[:, idn] = npp
    
    ## (WQ2) PRODUCTION CONSUMPTION
    prodcons_res = prodcons_module(
        un = u,
        o2n = o2,
        docrn = docr,
        docln = docl,
        pocrn = pocr,
        pocln = pocl,
        area = area,
        volume = volume,
        depth = depth, 
        nx = nx,
        dt = dt,
        dx = dx,
        delta= delta,
        k_half = k_half,
        resp_docr = resp_docr,
        resp_docl = resp_docl,
        resp_poc = resp_poc)
    
    o2 = prodcons_res['o2']
    docr = prodcons_res['docr']
    docl = prodcons_res['docl']
    pocr = prodcons_res['pocr']
    pocl = prodcons_res['pocl']
    docr_respiration = prodcons_res['docr_respiration']
    docl_respiration = prodcons_res['docl_respiration']
    poc_respiration = prodcons_res['poc_respiration']

    o2_pd[:, idn] = o2
    docr_pd[:, idn] = docr
    docl_pd[:, idn] = docl
    pocr_pd[:, idn] = pocr
    pocl_pd[:, idn] = pocl
    
    docr_respirationm[:, idn] = docr_respiration
    docl_respirationm[:, idn] = docl_respiration
    poc_respirationm[:, idn] = poc_respiration
    
    
    dens_u_n2 = calc_dens(u)
    if 'kz' in locals():
        1+1
    else: 
        kz = u * 0.0
        
    if diffusion_method == 'hendersonSellers':
        kz = eddy_diffusivity_hendersonSellers(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw(n),  43.100948, u, kz, Cd, km, weight_kz, k0) / 1
    elif diffusion_method == 'munkAnderson':
        kz = eddy_diffusivity_munkAnderson(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw(n),  43.100948, Cd, u, kz) / 1
    elif diffusion_method == 'hondzoStefan':
        kz = eddy_diffusivity(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, u, kz) / 86400
    elif diffusion_method == 'pacanowskiPhilander':
        kz = eddy_diffusivity_pacanowskiPhilander(dens_u_n2, depth, g, np.mean(dens_u_n2) , ice, area, Uw(n),  43.100948, u, kz, Cd, km, weight_kz, k0) / 1
    
    ## (WQ3) TRANSPORT
    transport_res = transport_module(
        un = u,
        o2n = o2,
        docrn = docr,
        docln = docl,
        pocrn = pocr,
        pocln = pocl,
        kzn = kz,
        Uw = Uw(n),
        depth= depth,
        dx = dx,
        area = area,
        volume = volume,
        dt = dt,
        nx = nx,
        ice = ice, 
        diffusion_method = diffusion_method,
        scheme = scheme,
        settling_rate = settling_rate,
        sediment_rate = sediment_rate)
    
    o2 = transport_res['o2']
    docr = transport_res['docr']
    docl = transport_res['docl']
    pocr = transport_res['pocr']
    pocl = transport_res['pocl']

    o2_diff[:, idn] = o2
    docr_diff[:, idn] = docr
    docl_diff[:, idn] = docl
    pocr_diff[:, idn] = pocr
    pocl_diff[:, idn] = pocl
    
    # print(o2_bc[:, idn]/volume)
    # print(o2_pd[:, idn]/volume)
    # print(o2_diff[:, idn]/volume)
    # breakpoint()
    # (3) MIXING
    if (idn == 3943):
        print('')
        #breakpoint()
    
    #breakpoint()
    plt.plot(u)
    mixing_res = mixing_module_minlake(
        un = u,
        o2n = o2,
        docrn = docr,
        docln = docl,
        pocrn = pocr,
        pocln = pocl,
        depth = depth,
        area = area,
        volume = volume,
        dx = dx,
        dt = dt,
        nx = nx,
        Uw = Uw(n),
        ice = ice, 
        W_str = W_str)
    
    plt.plot(u, color = 'green')
    
    #breakpoint()
    
    u = mixing_res['temp'] 
    thermo_dep = mixing_res['thermo_dep']
    energy_ratio = mixing_res['energy_ratio']
    o2 = mixing_res['o2']
    docr = mixing_res['docr']
    docl = mixing_res['docl']
    pocr = mixing_res['pocr']
    pocl = mixing_res['pocl']

    um_mix[:, idn] = u
    thermo_depm[0,idn] = thermo_dep
    energy_ratiom[0, idn] = energy_ratio
    
    ## (4) CONVECTION
    convection_res = convection_module(
        un = u,
        nx = nx,
        volume = volume)
    
    u = convection_res['temp']
    
    plt.plot(u, color = 'black')
    #plt.show()
    
    um_conv[:, idn] = u
    
    o2m[:, idn] = o2
    docrm[:, idn] = docr
    doclm[:, idn] = docl
    pocrm[:, idn] = pocr
    poclm[:, idn] = pocl
    
    icethickness_prior = Hi
    snowthickness_prior = Hs
    snowicethickness_prior = Hsi
    rho_snow_prior = rho_snow
    IceSnowAttCoeff_prior = IceSnowAttCoeff
    ice_prior = ice
    dt_iceon_avg_prior = dt_iceon_avg
    iceT_prior = iceT
    

    um[:, idn] = u
    
    Him[0,idn] = Hi
    Hsm[0,idn] = Hs
    Hsim[0,idn] = Hsi
    
    kd_lightm[0,idn] =kd_light
    
    
    meteo_pgdl[0, idn] = heating_res['air_temp']
    meteo_pgdl[1, idn] = heating_res['longwave_flux']
    meteo_pgdl[2, idn] = heating_res['latent_flux']
    meteo_pgdl[3, idn] = heating_res['sensible_flux']
    meteo_pgdl[4, idn] = heating_res['shortwave_flux']
    meteo_pgdl[5, idn] = heating_res['light']
    meteo_pgdl[6, idn] = -999 #mixing_res['shear']
    meteo_pgdl[7, idn] = -999 #mixing_res['tau']
    meteo_pgdl[8, idn] = np.nanmax(area)
    meteo_pgdl[9, idn] = CC(n)
    meteo_pgdl[10, idn] = ea(n)
    meteo_pgdl[11, idn] = Jlw(n)
    meteo_pgdl[12, idn] = Uw(n)
    meteo_pgdl[13, idn] = Pa(n)
    meteo_pgdl[14, idn] = RH(n)
    meteo_pgdl[15, idn] = PP(n)
    meteo_pgdl[16, idn] = IceSnowAttCoeff
    meteo_pgdl[17, idn] = ice
    meteo_pgdl[18, idn] = iceT
    meteo_pgdl[19, idn] = rho_snow
    meteo_pgdl[20, idn] = icethickness_prior 
    meteo_pgdl[21, idn] = snowthickness_prior
    meteo_pgdl[22, idn] = snowicethickness_prior 
    meteo_pgdl[23, idn] = rho_snow_prior 
    meteo_pgdl[24, idn] = IceSnowAttCoeff_prior
    meteo_pgdl[25, idn] = ice_prior
    meteo_pgdl[26, idn] = dt_iceon_avg_prior
    meteo_pgdl[27, idn] = iceT_prior
    
    dens_u_n2 = calc_dens(u)
    rho_0 = np.mean(dens_u_n2)
    buoy = np.ones(len(depth)) * 7e-5
    buoy[:-1] = np.abs(dens_u_n2[1:] - dens_u_n2[:-1]) / (depth[1:] - depth[:-1]) * g / rho_0
    buoy[-1] = buoy[-2]
    # n2 = 9.81/np.mean(dens_u_n2) * (dens_u_n2[1:] - dens_u_n2[:-1])/dx
    n2m[:,idn] = buoy # np.concatenate([n2, np.array([np.nan])])

  bf_sim = np.apply_along_axis(center_buoyancy, axis=1, arr = um.T, depths=depth)
  

  df_z_df_sim = pd.DataFrame({'time': times, 'thermoclineDep': bf_sim})

  df_z_df_sim['epi'] = np.nan
  df_z_df_sim['hypo'] = np.nan
  df_z_df_sim['tot'] = np.nan
  df_z_df_sim['stratFlag'] = np.nan
  for j in range(df_z_df_sim.shape[0]):
    if np.isnan(df_z_df_sim.loc[j, 'thermoclineDep']):
      cur_z = 1
      cur_ind = 0
    else:
      cur_z = df_z_df_sim.loc[j, 'thermoclineDep']
      cur_ind = np.max(np.where(depth < cur_z))
      
    df_z_df_sim.loc[j, 'epi'] = np.sum(um[0:(cur_ind + 1), j] * area[0:(cur_ind+1)]) / np.sum(area[0:(cur_ind+1)])
    df_z_df_sim.loc[j, 'hypo'] = np.sum(um[ cur_ind:, j] * area[cur_ind:]) / np.sum(area[cur_ind:])
    df_z_df_sim.loc[j, 'tot'] = np.sum(um[:,j] * area) / np.sum(area)
    if calc_dens(um[-1,j]) - calc_dens(um[0,j]) >= 0.1 and np.mean(um[:,j]) >= 4:
      df_z_df_sim.loc[j, 'stratFlag'] = 1
    else:
      df_z_df_sim.loc[j, 'stratFlag'] = 0
      
  dat = {'temp' : um,
               'diff' : kzm,
               'icethickness' : Him,
               'snowthickness' : Hsm,
               'snowicethickness' : Hsim,
               'iceflag' : ice,
               'icemovAvg' : iceT,
               'supercooled' : supercooled,
               'endtime' : endTime, 
               'average' : df_z_df_sim,
               'temp_initial' : um_initial,
               'temp_heat' : um_heat,
               'temp_diff' : um_diff,
               'temp_mix' : um_mix,
               'temp_conv' : um_conv,
               'temp_ice' : um_ice,
               'meteo_input' : meteo_pgdl,
               'buoyancy' : n2m,
               'density_snow' : rho_snow,
               'o2': o2m,
               'docr': docrm,
               'docl': doclm,
               'pocr': pocrm,
               'pocl': poclm,
               'npp':nppm,
               'docr_respiration': docr_respirationm,
               'docl_respiration': docl_respirationm,
               'poc_respiration': poc_respirationm,
               'kd_light': kd_lightm,
               'thermo_dep': thermo_depm,
               'energy_ratio': energy_ratiom}
  
  return(dat)
