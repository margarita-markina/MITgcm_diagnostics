import numpy as np
import pandas as pd
import numba
import math

import xarray as xr
import glob
import os

from xgcm import Grid
from xmitgcm import open_mdsdataset
import ecco_v4_py as ecco
import gsw

# these are my functions
from extract_velocity_across_section import *
from process_grid import *
from read_OSNAP_coords import *


import cmocean

import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline


import warnings
warnings.simplefilter("ignore") 


YEAR_NUMBERS = np.arange(81,91)

print(YEAR_NUMBERS[0]-1)
print(YEAR_NUMBERS[-1])


# The main code


#EXPERIMENT = 'WISHBONE/POSITIVE_ANOM_WEST/'
# EXPERIMENT_OUT = 'CONTROL/'
# EXPERIMENT = 'WIND_SCALED_LF_INDIVIDUAL_POINTS'
# EXPERIMENT = 'CONTROL_SUBMESO_FK'
# EXPERIMENT = 'FRESH_EGC'

EXPERIMENT = 'CONTROL_ABS_WINDS_NO_SUBMESO_SSH_CORRECTED'
# EXPERIMENT = 'CONTROL_ABS_WINDS_NO_SUBMESO_SSH_CORRECTED'

PATH_data = '/gws/nopw/j04/snapdragon/markina/DIAGNOSTICS/' + EXPERIMENT + '/MONTHLY_outputs/'

PATH_grid = '/gws/nopw/j04/snapdragon/markina/GRID/'
PATH_plots = '/home/users/markina/Plots/'

PATH_out = '/gws/nopw/j04/snapdragon/markina/DIAGNOSTICS/' + EXPERIMENT + '/MOC_sigma_OSNAP_from_monthly_means/'


PATH_coords = '/home/users/markina/Data/OSNAP_LINES/'

# coords_filename = 'OSNAP_section_grid_2019.txt'


coords_east_filename = 'OSNAP_EAST_002.txt' # these files have been generated on my laptop using matlab code - for 0.02 degree resolution
coords_west_filename = 'OSNAP_WEST_002_1.txt'

sigma_0_min=22. # CHANGE this later (if needed)
sigma_0_max=28.5

sigma_0_division = 27.

sigma_0_larger_bins = np.arange(sigma_0_min, sigma_0_division, 0.05)
sigma_0_smaller_bins = np.arange(sigma_0_division, sigma_0_max, 0.005)

target_sigma_0_levels = np.concatenate((sigma_0_larger_bins, sigma_0_smaller_bins))


jj_min = 700  # these are the same indices as used in state 2d fields
jj_max = 1400

ii_min = 250
ii_max = 1400 


x_smol = np.arange(ii_min,ii_max)
y_smol = np.arange(jj_min,jj_max) 



nx=1440
ny=1440
nz=50

# x = np.arange(0,nx)
# y = np.arange(0,ny)


YC_filename = PATH_grid + 'YC.data'
YC = np.fromfile(YC_filename, dtype='>f').reshape(ny,nx)

XC_filename = PATH_grid + 'XC.data'
XC = np.fromfile(XC_filename, dtype='>f').reshape(ny,nx)

YG_filename = PATH_grid + 'YG.data'
YG = np.fromfile(YG_filename, dtype='>f').reshape(ny,nx)

XG_filename = PATH_grid + 'XG.data'
XG = np.fromfile(XG_filename, dtype='>f').reshape(ny,nx)

YG_smol = YG[jj_min:jj_max, ii_min:ii_max]
XG_smol = XG[jj_min:jj_max, ii_min:ii_max]


dz = [10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.00, 10.01,10.03,\
      10.11, 10.32, 10.80, 11.76, 13.42, 16.04, 19.82, 24.85, \
      31.10, 38.42, 46.50, 55.00, 63.50, 71.58, 78.90, 85.15, 90.18, \
      93.96, 96.58, 98.25, 99.25,100.01,101.33,104.56,111.33,122.83, \
      139.09,158.94,180.83,203.55,226.50,249.50,272.50,295.50,318.50, \
      341.50,364.50,387.50,410.50,433.50,456.50]


Zp1 = np.zeros(len(dz)+1)

Zp1[0] = 0.

for idepth in range(1,len(dz)+1):
    Zp1[idepth] = Zp1[idepth-1]+dz[idepth-1]

Zp1 = -1.*Zp1


Z = np.zeros(len(dz))

for idepth in range(0,len(dz)):
    if idepth==0:
        Z[idepth] = (Zp1[idepth+1]-Zp1[idepth])*0.5 + 0.
    else:
        Z[idepth] = (Zp1[idepth+1]-Zp1[idepth])*0.5 + Zp1[idepth]


YC_new = np.reshape(YC, (1,ny,nx))

drF_filename = PATH_grid + 'DRF.data'
drF_tmp = np.fromfile(drF_filename, dtype='>f') # >f correponds to big endian binary file (that is used by MITgcm)
drF = drF_tmp.reshape(nz) 

dyG_filename = PATH_grid + 'DYG.data'
dyG_tmp = np.fromfile(dyG_filename, dtype='>f')
dyG = dyG_tmp.reshape(ny,nx) 
# dyG = dyG_src[jj_section,ii_min:ii_max]
# del(dyG_src)

# plt.contourf(dyG) # should be a nice smooth plot

dxG_filename = PATH_grid + 'DXG.data'
dxG_tmp = np.fromfile(dxG_filename, dtype='>f') 
dxG = dxG_tmp.reshape(ny,nx) 
# dxG = dxG_src[jj_section,ii_min:ii_max]
# del(dxG_src)

    
# Read variables relevant for computation of density

#Reading hydrostatic pressure and reference density - needed for computing density field to define density surfaces 

PHrefC_filename = PATH_grid + 'PHrefC.data' 
PHrefC_tmp = np.fromfile(PHrefC_filename, dtype='>f') # >f correponds to big endian binary file (that is used by MITgcm)
pressure_m2_s2 = PHrefC_tmp.reshape(nz) # Reference Hydrostatic Pressure [m2 s-2]
# plt.plot(pressure_m2_s2) # should be a nice smooth plot

RhoRef_filename = PATH_grid + 'RhoRef.data'
RhoRef_tmp = np.fromfile(RhoRef_filename, dtype='>f') # >f correponds to big endian binary file (that is used by MITgcm)
density = RhoRef_tmp.reshape(nz) 
pressure_dbar = pressure_m2_s2*density/10000. # 1 dbar = 10 KPa = 10000 Pa

da_pres = xr.DataArray(data=pressure_dbar, dims=["k"],)
da_z = xr.DataArray(data=Z, dims=["k"])
da_zp1 = xr.DataArray(data=Zp1, dims=["k"])

da_dxg = xr.DataArray(data=dxG[jj_min:jj_max, ii_min:ii_max], dims=['j_g', 'i'])
da_dyg = xr.DataArray(data=dyG[jj_min:jj_max, ii_min:ii_max], dims=['j', 'i_g'])

da_lat = xr.DataArray(data=YC[jj_min:jj_max, ii_min:ii_max], dims=["j", "i"])
da_lon = xr.DataArray(data=XC[jj_min:jj_max, ii_min:ii_max], dims=["j", "i"])


# # Reading all grid variables and SNAP coordinates

k=np.arange(0,50)
tile=np.arange(1,2) # since coordinate should be ndarray rather than simple int  

ds_grid, _, _, da_z, da_pres = read_grid(PATH_grid, nx, ny, nz, ii_min, ii_max, jj_min, jj_max)

ds_west_coords, ds_east_coords, ds_full_coords = read_osnap_coordinates(PATH_coords + coords_west_filename, PATH_coords + coords_east_filename)

    
months = ['05', '06', '07', '08', '09', '10', '11', '12', '01', '02', '03', '04']

# %%time

for YEAR in YEAR_NUMBERS:

    for imonth in np.arange(12):

        print(str(YEAR) + '/' + str(months[imonth])+ " in progress...")

        print('reading data (state)...')
        

        ds_temp = xr.open_dataset(PATH_data+str(YEAR)+"_state_3d_temp_monthly.nc").isel(time=imonth)
        ds_sal = xr.open_dataset(PATH_data+str(YEAR)+"_state_3d_sal_monthly.nc").isel(time=imonth)


        print('reading data (trsp)...')

        ds_uvel = xr.open_dataset(PATH_data+str(YEAR)+"_trsp_3d_uvel_monthly.nc").isel(time=imonth)
        ds_vvel = xr.open_dataset(PATH_data+str(YEAR)+"_trsp_3d_vvel_monthly.nc").isel(time=imonth)

        ds = xr.Dataset(data_vars=dict(THETA=(['k','j', 'i'], ds_temp.temp.values),
                                       SALT=(['k','j', 'i'], ds_sal.sal.values),
                                       UVELMASS=(['k','j', 'i_g'], ds_uvel.uvel.values),
                                       VVELMASS=(['k','j_g', 'i'], ds_vvel.vvel.values),),
                         coords=dict(k=(["k"], np.arange(0, nz)),
                                     j=(["j"], y_smol),
                                     j_g=(["j_g"], y_smol),
                                     i_g=(["i_g"], x_smol),
                                     i=(["i"], x_smol),) )

        del ds_temp, ds_sal, ds_uvel, ds_vvel

        ds['pressure_dbar']=da_pres
        ds['Z']=da_z
        ds['DXG']=da_dxg
        ds['DYG']=da_dyg
        # ds['Z']=da_z

        ds['lat']=da_lat
        ds['lon']=da_lon

        print('computing density...')

        potential_temp = ds.THETA
        salin = ds.SALT 

        absolute_salin = gsw.SA_from_SP(salin, ds.pressure_dbar - 10.1325, ds.lon, ds.lat)
        conservative_temp = gsw.CT_from_pt(absolute_salin, potential_temp)
        sigma_2_tmp = gsw.sigma2(absolute_salin, conservative_temp)
        sigma_0_tmp = gsw.sigma0(absolute_salin, conservative_temp)

        sigma_2 = sigma_2_tmp.to_dataset(name='SIGMA_2')
        sigma_0 = sigma_0_tmp.to_dataset(name='SIGMA_0')

        ds['SIGMA_0'] = sigma_0.SIGMA_0
        ds['SIGMA_2'] = sigma_2.SIGMA_2

        del absolute_salin, conservative_temp, sigma_2_tmp, sigma_0_tmp, sigma_0, sigma_2


        level_outer_data = Zp1
        level_outer = xr.DataArray(
            level_outer_data,
            dims=['level_outer'],
            coords={'level_outer': ('level_outer', level_outer_data)}
        )
        ds = ds.assign_coords({'level_outer': level_outer})


        grid = Grid(ds, coords={'Z': {'center': 'k', 'outer': 'level_outer'},
                                'X': {'center': 'i', 'left': 'i_g'},
                                'Y': {'center': 'j', 'left': 'j_g'}
                                },
                    periodic=False,
                    )
        
    
    
        # Compute volume transports in both north and east directions (separately)

        drF_ds = xr.Dataset(data_vars=dict(drF=(['k'], drF), ), 
                    coords=dict(k=(['k'], np.arange(0,50)),) )

        dyU_ds = xr.Dataset(data_vars=dict(dyU=(['j', 'i_g'], ds_grid.DYU.values ),), 
                        coords=dict(j=(['j'], y_smol),
                                    i_g=(['i_g'], x_smol,) ) )

        dxV_ds = xr.Dataset(data_vars=dict(dxV=(['j_g', 'i'], ds_grid.DXV.values ),), 
                        coords=dict(j_g=(['j_g'], y_smol),
                                    i=(['i'], x_smol),) )

    
        print('computing volume transports...')
    
        v_volume_transport = ds['VVELMASS']*drF_ds['drF']*dxV_ds['dxV']
        u_volume_transport = ds['UVELMASS']*drF_ds['drF']*dyU_ds['dyU']
        
        print('done')
    


    
        # Computing MOC across OSNAP West and OSNAP East


        print('computing MOC(OSNAP west)...')

        line_west = get_osnap_section(YG_smol, XG_smol, ds_west_coords.lats.values, ds_west_coords.lons.values)
        uv_points_west = get_uv_points(line_west)

        moc_section_west = compute_moc_sigma_osnap(uv_points_west, 
                                                   u_volume_transport.values, 
                                                   v_volume_transport.values, 
                                                   ds.SIGMA_0.values,
                                                   target_sigma_0_levels)
        del line_west, uv_points_west



        print('computing MOC(OSNAP east)...')

        line_east = get_osnap_section(YG_smol, XG_smol, ds_east_coords.lats.values, ds_east_coords.lons.values)
        uv_points_east = get_uv_points(line_east)

        moc_section_east = compute_moc_sigma_osnap(uv_points_east, 
                                               u_volume_transport.values, 
                                               v_volume_transport.values, 
                                               ds.SIGMA_0.values,
                                               target_sigma_0_levels)
        del line_east, uv_points_east


        print('computing MOC(OSNAP full)...')

        line_full = get_osnap_section(YG_smol, XG_smol, ds_full_coords.lats.values, ds_full_coords.lons.values)
        uv_points_full = get_uv_points(line_full)


        moc_section_full = compute_moc_sigma_osnap(uv_points_full, 
                                               u_volume_transport.values, 
                                               v_volume_transport.values, 
                                               ds.SIGMA_0.values,
                                               target_sigma_0_levels)
        del line_full, uv_points_full

        del u_volume_transport, v_volume_transport

        print("creating output array...")

        moc_sigma_space_ds = xr.Dataset(data_vars=dict(moc_sigma_0_full=(['sigma_0'], moc_section_full),
                                                       moc_sigma_0_west=(['sigma_0'], moc_section_west),
                                                       moc_sigma_0_east=(['sigma_0'], moc_section_east), ),
                                        coords=dict(sigma_0=(['sigma_0'], target_sigma_0_levels), ) )



        moc_sigma_space_ds.to_netcdf(PATH_out+"MOC_sigma_OSNAP_timestep_yr_"+str(YEAR)+"_month_"+str(months[imonth])+".nc")

        del moc_sigma_space_ds, moc_section_full, moc_section_west, moc_section_east

        del ds

        print('done!')
