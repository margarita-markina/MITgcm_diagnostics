# functions needed for getting the model grid and writing it to easily digestible file
# (initially taken from compute_MOC_section_smaller_bins.py code

import numpy as np
import xarray as xr



def read_grid(PATH_IN, nx, ny, nz, ii_min, ii_max, jj_min, jj_max):
    '''
    read all grid variebles and write it to xarray datasets/data arrays - 
    to use it later for computing various diagnostics (i.e. MOC along the line)
    
    input:
    PATH_IN - path to the files with vertical grid variables (DRF, RhoRef, PHrefC) 
              and parameters of the horizontal grid tile001_f32.mitgrid
              
    nx, ny, nz - sizes of the grid - should be constants (1440,1440,50) - but just in case
              
    ii_min, ii_max - jj indices; the same number as we used for cutting sections - 
                     just to make code a bit   lighter; 
                     can be optimised in a more clever way later   
              
    jj_min, jj_max - jj indices; the same number as we used for cutting sections - 
                     just to make code a bit   lighter; 
                     can be optimised in a more clever way later
                     

              
    output:
    ds_grid - dataset with parameters of horizontal grid (from tile001_f32.mitgrid); corresponds to FULL grid
    da_lat, da_lon, da_z, da_pres - data arrays with coordinates and vertical grid (needed for computing potential densities) and coordinates; corresponds to CUTTED grid
    
    
    '''
    
              
    # this dz values has been taken directly from mitgcm configuration file (data)
              
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


    drF_filename = PATH_IN + 'DRF.data'
    drF = np.fromfile(drF_filename, dtype='>f').reshape(nz)
              
              
    # Read variable relevant for computation of density

    #Reading hydrostatic pressure and reference density - needed for computing density field to define density surfaces 

    PHrefC_filename = PATH_IN + 'PHrefC.data' 
    pressure_m2_s2 = np.fromfile(PHrefC_filename, dtype='>f').reshape(nz) # Reference Hydrostatic Pressure [m2 s-2]
    # plt.plot(pressure_m2_s2) # should be a nice smooth plot

    RhoRef_filename = PATH_IN + 'RhoRef.data'
    density = np.fromfile(RhoRef_filename, dtype='>f').reshape(nz) 
    pressure_dbar = pressure_m2_s2*density/10000. # 1 dbar = 10 KPa = 10000 Pa



    da_pres = xr.DataArray(data=pressure_dbar, dims=["k"],)
    da_z = xr.DataArray(data=Z, dims=["k"])
    
    
    
    x_smol = np.arange(ii_min,ii_max)
    y_smol = np.arange(jj_min,jj_max)


    grid_all = np.fromfile(PATH_IN + 'tile001_f32.mitgrid', dtype='>f') #np.dtype(np.float64)) #
    grid_all = grid_all.reshape(16, ny+1, nx+1) 

    XC = grid_all[0,:,:]
    YC = grid_all[1,:,:]

    DXF = grid_all[2,:,:]
    DYF = grid_all[3,:,:]

    RAC = grid_all[4,:,:]

    XG = grid_all[5,:,:]
    YG = grid_all[6,:,:]

    DXV = grid_all[7,:,:]
    DYU = grid_all[8,:,:]

    RAZ = grid_all[9,:,:]

    DXC = grid_all[10,:,:]
    DYC = grid_all[11,:,:]

    RAW = grid_all[12,:,:]

    RAS = grid_all[13,:,:]

    DXG = grid_all[14,:,:]
    DYG = grid_all[15,:,:]

    ds_grid = xr.Dataset(data_vars=dict(XC=(['j', 'i'], XC[jj_min:jj_max, ii_min:ii_max]), 
                                        YC=(['j', 'i'], YC[jj_min:jj_max, ii_min:ii_max]), 
                                        DXF=(['j', 'i'], DXF[jj_min:jj_max, ii_min:ii_max]), 
                                        DYF=(['j', 'i'], DYF[jj_min:jj_max, ii_min:ii_max]), 
                                        RAC=(['j', 'i'], RAC[jj_min:jj_max, ii_min:ii_max]), 
                                        XG=(['j_g', 'i_g'], XG[jj_min:jj_max, ii_min:ii_max]), 
                                        YG=(['j_g', 'i_g'], YG[jj_min:jj_max, ii_min:ii_max]), 
                                        DXV=(['j_g', 'i'], DXV[jj_min:jj_max, ii_min:ii_max]), 
                                        DYU=(['j', 'i_g'], DYU[jj_min:jj_max, ii_min:ii_max]), 
                                        RAZ=(['j_g', 'i_g'], RAZ[jj_min:jj_max, ii_min:ii_max]),
                                        DXC=(['j', 'i_g'], DXC[jj_min:jj_max, ii_min:ii_max]), 
                                        DYC=(['j_g', 'i'], DYC[jj_min:jj_max, ii_min:ii_max]),
                                        RAW=(['j', 'i_g'], RAW[jj_min:jj_max, ii_min:ii_max]),
                                        RAS=(['j_g', 'i'], RAS[jj_min:jj_max, ii_min:ii_max]),
                                        DXG=(['j_g', 'i'], DXG[jj_min:jj_max, ii_min:ii_max]), 
                                        DYG=(['j', 'i_g'], DYG[jj_min:jj_max, ii_min:ii_max]),
                                           ),

                              coords=dict(i=(["i"], x_smol),
                                          i_g=(["i_g"], x_smol),
                                          j=(["j"], y_smol ),
                                          j_g=(["j_g"], y_smol ),) ) 


    da_lat = xr.DataArray(data=YC[:-1,:-1], dims=["i", "j"])
    da_lon = xr.DataArray(data=XC[:-1,:-1], dims=["i", "j"])
    # da_lat_new = xr.DataArray(data=YC, dims=["i", "j_g"])
    
    
    return ds_grid, da_lat, da_lon, da_z, da_pres