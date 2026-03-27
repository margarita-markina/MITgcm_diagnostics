# functions needed for reading OSNAP coordinates and writing them to the file

import numpy as np
import xarray as xr
import pandas as pd


def read_osnap_coordinates(path_to_WEST, path_to_EAST):
    '''
    reading coordinates of OSNAP arrays (initially generated on my local machine using matlab code from Tyllis - updated for my needs) 
    
    input: 
    path_to_WEST, path_to_EAST - paths plus names of the files containing coordinates for OSNAP WEST and OSNAP EAST
    
    output:
    ds_west, ds_east, ds_full - xarray datasets with coordinates of OSNAP WEST, EAST and FULL
    
    '''
    
    data_east = pd.read_csv(path_to_EAST, header=None, sep=",")
    lats_snap_east = np.zeros(len(data_east.values))
    lons_snap_east = np.zeros(len(data_east.values))

    for ii in range(0, len(data_east.values)):
        lats_snap_east[ii] = data_east.values[ii][1]
        lons_snap_east[ii] = data_east.values[ii][0]



    data_west = pd.read_csv(path_to_WEST, header=None, sep=",")
    lats_snap_west = np.zeros(len(data_west.values))
    lons_snap_west = np.zeros(len(data_west.values))

    for ii in range(0, len(data_west.values)):
        lats_snap_west[ii] = data_west.values[ii][1]
        lons_snap_west[ii] = data_west.values[ii][0]


    lats_snap_full = np.zeros(len(data_west.values)+len(data_east.values))
    lons_snap_full = np.zeros(len(data_west.values)+len(data_east.values))

    icount = 0
    for ii in range(0, len(data_west.values)+len(data_east.values)):
        if ii < len(data_west.values):
            lats_snap_full[icount] = data_west.values[ii][1]
            lons_snap_full[icount] = data_west.values[ii][0]
        else:
            lats_snap_full[icount] = data_east.values[ii-len(data_west.values)][1]
            lons_snap_full[icount] = data_east.values[ii-len(data_west.values)][0]

        icount+=1
        
        
    ds_east = xr.Dataset(data_vars=dict(lats=(['ix'], lats_snap_east), 
                                               lons=(['ix'], lons_snap_east), ),

                              coords=dict(ix=(["ix"], np.arange(0, len(lats_snap_east)),) ) )

    ds_west = xr.Dataset(data_vars=dict(lats=(['ix'], lats_snap_west), 
                                           lons=(['ix'], lons_snap_west), ),

                          coords=dict(ix=(["ix"], np.arange(0, len(lats_snap_west)),) ) )
    
    
    ds_full = xr.Dataset(data_vars=dict(lats=(['ix'], lats_snap_full), 
                                       lons=(['ix'], lons_snap_full), ),

                      coords=dict(ix=(["ix"], np.arange(0, len(lats_snap_full)),) ) )
    
    return ds_west, ds_east, ds_full