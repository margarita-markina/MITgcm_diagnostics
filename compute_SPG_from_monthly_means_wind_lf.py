import numpy as np
import pandas as pd

import glob
import os

from xgcm import Grid
# import ecco_v4_py as ecco # this depends on cartopy which is not properly working with this environment for some reason

from xmitgcm import open_mdsdataset

import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl


import cmocean

import warnings
warnings.simplefilter("ignore") 

# EXPERIMENT = 'WIND_LF/'

EXPERIMENT = 'WIND_LF_fluxes_FULL_mom'

PATH_data = '/gws/nopw/j04/snapdragon/markina/DIAGNOSTICS/SENSITIVITY_TESTS/'+EXPERIMENT+'/MONTHLY_outputs/'

PATH_grid = '/gws/nopw/j04/snapdragon/markina/GRID/'

PATH_plots = '/home/users/markina/Plots/'

PATH_out = '/gws/nopw/j04/snapdragon/markina/DIAGNOSTICS/SENSITIVITY_TESTS/' + EXPERIMENT + '/SPG_BSF_from_monthly_means/'

YEAR_NUMBERS = np.arange(51,52)

jj_min = 700  # these are the same indices as used in state 2d fields
jj_max = 1400

ii_min = 250
ii_max = 1400 


x_smol = np.arange(ii_min,ii_max)
y_smol = np.arange(jj_min,jj_max) 

nx_smol = len(x_smol)
ny_smol = len(y_smol)

nx=1440
ny=1440
nz=50

# x = np.arange(0,nx)
# y = np.arange(0,ny)




YC_filename = PATH_grid + 'YC.data'
YC_tmp = np.fromfile(YC_filename, dtype='>f') 
YC = YC_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]

XC_filename = PATH_grid + 'XC.data'
XC_tmp = np.fromfile(XC_filename, dtype='>f') 
XC = XC_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]

# da_lat = xr.DataArray(data=YC[jj_min:jj_max, ii_min:ii_max], dims=["j", "i"])
# da_lon = xr.DataArray(data=XC[jj_min:jj_max, ii_min:ii_max], dims=["j", "i"])



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

drF_filename = PATH_grid + 'DRF.data'
drF_tmp = np.fromfile(drF_filename, dtype='>f') # >f correponds to big endian binary file (that is used by MITgcm)
drF = drF_tmp.reshape(nz) 

dyG_filename = PATH_grid + 'DYG.data'
dyG_tmp = np.fromfile(dyG_filename, dtype='>f')
dyG = dyG_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# dyG = dyG_src[jj_section,ii_min:ii_max]
# del(dyG_src)

# plt.contourf(dyG) # should be a nice smooth plot

dxG_filename = PATH_grid + 'DXG.data'
dxG_tmp = np.fromfile(dxG_filename, dtype='>f') 
dxG = dxG_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
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

# da_dxg = xr.DataArray(data=dxG[jj_min:jj_max, ii_min:ii_max], dims=['j_g', 'i'])
# da_dyg = xr.DataArray(data=dyG[jj_min:jj_max, ii_min:ii_max], dims=['j', 'i_g'])



# Computing vertical cell sizes - needed for metrics

maskW_filename = PATH_grid + 'maskInW.data'
maskW_tmp = np.fromfile(maskW_filename, dtype='>f') 
maskW = maskW_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# plt.contourf(maskW) 

maskS_filename = PATH_grid + 'maskInS.data'
maskS_tmp = np.fromfile(maskS_filename, dtype='>f') 
maskS = maskS_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(maskS) 
    
hFacW_filename = PATH_grid + 'hFacW.data'
hFacW_tmp = np.fromfile(hFacW_filename, dtype='>f') 
hFacW = hFacW_tmp.reshape(nz,ny,nx)[:,jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(hFacW[0,:,:])

hFacC_filename = PATH_grid + 'hFacC.data'
hFacC_tmp = np.fromfile(hFacC_filename, dtype='>f')
hFacC = hFacC_tmp.reshape(nz,ny,nx)[:,jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(hFacC[0,:,:]) 

hFacS_filename = PATH_grid + 'hFacS.data'
hFacS_tmp = np.fromfile(hFacS_filename, dtype='>f')
hFacS = hFacS_tmp.reshape(nz,ny,nx)[:,jj_min:jj_max, ii_min:ii_max]
# plt.contourf(hFacS[0,:,:]) 

XC_filename = PATH_grid + 'XC.data'
XC_tmp = np.fromfile(XC_filename, dtype='>f') 
XC = XC_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(XC) 

XG_filename = PATH_grid + 'XG.data'
XG_tmp = np.fromfile(XG_filename, dtype='>f') 
XG = XG_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(XG) 

YG_filename = PATH_grid + 'YG.data'
YG_tmp = np.fromfile(YG_filename, dtype='>f') 
YG = YG_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(YG) 

dxC_filename = PATH_grid + 'DXC.data'
dxC_tmp = np.fromfile(dxC_filename, dtype='>f') 
dxC = dxC_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# plt.contourf(dxC) 

dyC_filename = PATH_grid + 'DYC.data'
dyC_tmp = np.fromfile(dyC_filename, dtype='>f') 
dyC = dyC_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# plt.contourf(dyC) 

rA_filename = PATH_grid + 'RAC.data'
rA_tmp = np.fromfile(rA_filename, dtype='>f') 
rA = rA_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max] 
# plt.contourf(rA)

rAz_filename = PATH_grid + 'RAZ.data'
rAz_tmp = np.fromfile(rAz_filename, dtype='>f') 
rAz = rAz_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# plt.contourf(rAz) 

rAs_filename = PATH_grid + 'RAS.data'
rAs_tmp = np.fromfile(rAs_filename, dtype='>f') 
rAs = rAs_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# plt.contourf(rAs) 

rAw_filename = PATH_grid + 'RAW.data'
rAw_tmp = np.fromfile(rAw_filename, dtype='>f') 
rAw = rAw_tmp.reshape(ny,nx)[jj_min:jj_max, ii_min:ii_max]
# plt.contourf(rAw) 
    
    
    
drW = hFacW * np.reshape(drF,(nz,1,1)) #vertical cell size at u point
drS = hFacS * np.reshape(drF,(nz,1,1)) #vertical cell size at v point
drC = hFacC * np.reshape(drF,(nz,1,1)) #vertical cell size at tracer point

# k=np.arange(0,50)
# tile=np.arange(1,2) # since coordinate should be ndarray rather than simple int

drW_new = np.reshape(drW, (1, nz, ny_smol, nx_smol))
drS_new = np.reshape(drS, (1, nz, ny_smol, nx_smol))
drC_new = np.reshape(drC, (1, nz, ny_smol, nx_smol))


hFacW_new = np.reshape(hFacW, (1, nz, ny_smol, nx_smol))
hFacS_new = np.reshape(hFacS, (1, nz, ny_smol, nx_smol))
hFacC_new = np.reshape(hFacC, (1, nz, ny_smol, nx_smol))

dyC_new = np.reshape(dyC, (1,ny_smol,nx_smol))
dxC_new = np.reshape(dxC, (1,ny_smol,nx_smol))

rA_new = np.reshape(rA, (1,ny_smol,nx_smol))
rAw_new = np.reshape(rAw, (1,ny_smol,nx_smol))
rAs_new = np.reshape(rAs, (1,ny_smol,nx_smol))
rAz_new = np.reshape(rAz, (1,ny_smol,nx_smol))

dyG_new = np.reshape(dyG, (1,ny_smol,nx_smol))
dxG_new = np.reshape(dxG, (1,ny_smol,nx_smol))
maskW_new = np.reshape(maskW, (1,ny_smol,nx_smol))
maskS_new = np.reshape(maskS, (1,ny_smol,nx_smol))
YC_new = np.reshape(YC, (1,ny_smol,nx_smol))

k=np.arange(0,50)
tile=np.arange(1,2) # since coordinate should be ndarray rather than simple int


# Computing barotropic stream function in a loop (for memory efficiency)


months = ['05', '06', '07', '08', '09', '10', '11', '12', '01', '02', '03', '04']


for YEAR in YEAR_NUMBERS:

    for imonth in np.arange(len(months)):

        print(str(YEAR) + '/' + str(months[imonth])+ " in progress...")

        print('reading data (trsp)...')

        ds_uvel = xr.open_dataset(PATH_data+str(YEAR)+"_trsp_3d_uvel_monthly.nc").isel(time=imonth)
        ds_vvel = xr.open_dataset(PATH_data+str(YEAR)+"_trsp_3d_vvel_monthly.nc").isel(time=imonth)

        ds = xr.Dataset(data_vars=dict(UVELMASS=(['k','j', 'i'], ds_uvel.uvel.values),
                                       VVELMASS=(['k','j', 'i'], ds_vvel.vvel.values),),
                         coords=dict(k=(["k"], np.arange(0, nz)),
                                     j=(["j"], y_smol),
                                     i=(["i"], x_smol),) )


    
        del ds_uvel, ds_vvel
    
    
    #     psi_barotrop_all = np.zeros([1, len(y), len(x)])

        uvel_smol = ds['UVELMASS'].values / hFacW_new[0,:,:,:]
        vvel_smol = ds['VVELMASS'].values / hFacS_new[0,:,:,:]

        uvel_new = np.reshape(uvel_smol, (1,nz,ny_smol,nx_smol))
        vvel_new = np.reshape(vvel_smol, (1,nz,ny_smol,nx_smol))

        ds_for_barotrop = xr.Dataset(data_vars=dict(UVEL=(["tile", "k", "j", "i_g"], uvel_new),
                                           VVEL=(["tile", "k", "j_g", "i"], vvel_new),
                                           drW=(["tile", "k", "j", "i_g"], drW_new),
                                           drS=(["tile", "k", "j_g", "i"], drS_new),
                                           drC=(["tile", "k", "j", "i"], drC_new),
                                           hFacW=(["tile", "k", "j", "i_g"], hFacW_new),
                                           hFacS=(["tile", "k", "j_g", "i"], hFacS_new),
                                           hFacC=(["tile", "k", "j", "i"], hFacC_new),
                                           drF = (["k"], drF),
                                           dyG = ([ "tile", "j", "i_g"], dyG_new),
                                           dxG = ([ "tile", "j_g", "i"], dxG_new),
                                           dxC = ([ "tile", "j_g", "i"], dxC_new),
                                           dyC = ([ "tile", "j_g", "i"], dyC_new),
                                           rA = ([ "tile", "j", "i"], rA_new),
                                           rAw = ([ "tile", "j", "i_g"], rAw_new),
                                           rAs = ([ "tile", "j_g", "i"], rAs_new),
                                           rAz = ([ "tile", "j_g", "i_g"], rAz_new), ) ,
                                       coords=dict(tile=(["tile"], tile),
                                            k=(["k"], k),
                                            j=(["j"], y_smol),
                                            j_g=(["j_g"], y_smol),
                                            i=(["i"], x_smol),
                                            i_g=(["i_g"], x_smol), ) )

        del uvel_smol, vvel_smol

        # creating grid for conservative interpolation # CHANGE later (probably)
        metrics = {
            ('X',): ['dxC', 'dxG'], # X distances
            ('Y',): ['dyC', 'dyG'], # Y distances
            ('Z',): ['drW', 'drS', 'drC'], # Z distances
            ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw'] # Areas
        }

        grid_barotrop = Grid(ds_for_barotrop, metrics = metrics, coords={'Z': {'center': 'k'},
                                    'X': {'center': 'i', 'left': 'i_g'},
                                    'Y': {'center': 'j', 'left': 'j_g'}}, periodic=False)

        print('computing diagnostics...')

        psi_barotrop = grid_barotrop.cumint(grid_barotrop.integrate(ds_for_barotrop.VVEL,'Z'),'X', boundary='fill')
    #     psi_barotrop = grid_barotrop.cumint(grid_barotrop.integrate(ds_for_barotrop.UVEL,'Z'),'Y', boundary='fill')
        psi_barotrop_nans = np.where(ds_for_barotrop.hFacC.values[0,0,:,:]==0., np.NaN, psi_barotrop.sel(tile=1).values/ 1e6) # mask continents as NaNs and convert to Sverdrups



        del uvel_new, vvel_new


        print('writing output to file...')

        psi_barotrop_north_atl = xr.DataArray(np.array(psi_barotrop_nans, dtype=float), dims=('y', 'x'),
                                             coords={'x' : x_smol, 'y' : y_smol})

        psi_barotrop_north_atl.to_netcdf(PATH_out+'BSF_yr_'+str(YEAR)+"_month_"+str(months[imonth])+'.nc')


        del psi_barotrop_nans, psi_barotrop_north_atl, psi_barotrop


