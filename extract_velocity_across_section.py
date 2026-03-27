# this file is identical to the file extract_velocity_across_section.py stored in
# stored in /home/users/markina/Code/RYF_diagnostics_during_spiniup/

import numpy as np


# These functions serving fot extraction of velocities across OSNAP line have been taken (and adapted in non-meaninful way) from Laura Jackson's code

def dist_from_line(a,b,point):
    ''' Distance of point (lat0,lon0) from line lat=a*lon+b'''
    lat0,lon0=point
    dist = abs(lat0 - a*lon0 - b)/(a**2+1)**(0.5)
    return dist

def define_line(point0,point1):
    ''' Find line between two points lat = a*lon+b'''
    a = (point1[1]-point0[1])/(point1[0]-point0[0])
    b = point1[1]-a*point1[0]
    return a,b

def nearest_ji(lat,lon,pos):
    '''
    Returns the (j,i) tuple of the grid point closest to the lat and lon given
    
    input:
        lat, lon  2D arrays of lat and lon
        pos       tuple of (lat0,lon0) to find
    
    output:
        (j,i)     nearest index (tuple)
       
    '''
    lat0,lon0 = tuple(pos)
    
#     print('nearest ji')
#     print('lat='+str(lat0))
#     print('lon='+str(lon0))
    if lon0 < lon.min():
        lon0+=360.0
    if lon0 > lon.max():
        lon0-=360.0
    weight = (lat - lat0)**2 + (lon - lon0)**2
    tmp = np.where(weight == weight.min())
    j,i = tmp[0][0],tmp[1][0]
#     print('j='+str(j))
#     print('i='+str(i))
    return (j,i)
   
def get_line_segment(point0,point1,latf,lonf,dirn0): 
    ''' Get line segment between two points '''
    # Define line in terms of intercept and gradient  
    a,b=define_line(point0,point1)
    #Find ji coordinates of end points of line segment (ie neighbouring obs points)
    ji0 = nearest_ji(latf,lonf,[point0[1],point0[0]])
    ji1 = nearest_ji(latf,lonf,[point1[1],point1[0]])
    ji_list = [ji0]
    j,i=ji0
#     print('ji0='+str(ji0))
#     print('ji1='+str(ji1))
    line_end = False
    if ji0 == ji1:  # Stop if start and end points are the same
        line_end = True
    while not line_end:
        # Possible directions could be N,S,E from this point (not W)
        # Note that N,S,E refer to directions in j,i coords not lat,lon
        #
        # Find distance of each possible direction from line between start and end points
        ip=i+1
        if ip>=lonf.shape[1]:
            ip=0
#         print('back to get_line_segment')
#         print('i='+str(i))
#         print('j='+str(j))
        distN = dist_from_line(a,b,[latf[j+1,i],lonf[j+1,i]])
        distE = dist_from_line(a,b,[latf[j,ip],lonf[j,ip]])
        distS = dist_from_line(a,b,[latf[j-1,i],lonf[j-1,i]])
        # If N is closest and previous dirn wasn't S
        if distN < distE and distN < distS and dirn0!='S':
            dirn0='N'
        # If S is closest and previous dirn wasn't N         
        elif distS < distE and distS < distN and dirn0!='N':   
            dirn0='S'
        # Otherwise go E
        else:
            dirn0='E'
        # If neighbouring point is end point then go there          
        if (j+1,i)==ji1:
            dirn0='N'
        elif (j-1,i)==ji1:
            dirn0='S'
        elif (j,ip)==ji1:
            dirn0='E'
        # Set ji for new point                   
        if dirn0=='N':
            j+=1
        elif dirn0=='S':
            j-=1
        elif dirn0=='E':
            i=ip                              
        ji = (j,i)   
        ji_list.append(ji)
        if ji == ji1:
            line_end=True # Stop if reach end     
        if len(ji_list)>10000:
            line_end=True        # In case of problem don't create infinite loop! 
    return ji_list,dirn0


# # This is adapted from Laura's section for getting the j,i coordinates from lat/lon of the section - this is old and working - but treating different coordinate files weirdly

def get_osnap_section(latf, lonf, lat_section, lon_section):
    '''
    Step from one obs point to the next using a path on q grid. This path is chosen to minimise 
    the distance from the direct line between each two points in lat/lon space
   
    inputs:
        latf - array with coordinates specifying latitudes of SW corner of model grid (i.e. YG)
        lonf - array with coordinates specifying longitudes of SW corner of model grid (i.e. XG)
        lat_section - array containing latitudes of the section 
        lon_section - array containing longitudes of the section 
        opts        dictionary of options read from configfile
      
    outputs:
        line        list of (j,i) points (tuples) for coordinates of the line   
    '''


    if len(lat_section)!=len(lon_section):
        print('Error! arrays with lats and lons of the section have different size!')
        return
    
    if np.shape(latf)!=np.shape(lonf):
        print('Error! arrays with model lats and lons have different size!')
        return
    
    
    npoints = len(lat_section)
    
    
    # Get line segment between each neighbouring obs points and append to line. Keep track of direction last travelled 
    line=[]
    dirn='E' # Assume previous direction was E
    
    for i in range(npoints-1):  
#         print('input for get_line_segment:')
#         print('point0')
#         print((lon_section[i], lat_section[i]))
#         print('point1')
#         print((lon_section[i+1], lat_section[i+1]))
        
        line_segment,dirn = get_line_segment((lon_section[i], lat_section[i]) , (lon_section[i+1], lat_section[i+1]),latf,lonf,dirn)  #CAREFUL here - I assumed that lon_section should be first in the tuple; change if not  
        if len(line)==0:   
            line = line_segment
        else:
            line += line_segment[1:]   #Excludes duplication of first point of segment/last point of previous segment    
            
#     # Display 
#     if opts['display']:
#         display_obs(line,obs_points,opts)
#     #   

    return line


# Laura Jaskson's function from uv_points_Cgrid.py; slighly adapted to remove opts
def get_uv_points(line):
    '''
    From line work out which dirn and whether U or V velocity is required 
    and in which direction (for going northwards into section)
   
    inputs:
        line   list of (j,i) tuples of the line on q grid
        opts   dictionary of options read from configfile
      
    outputs:
        uvpoints  tuple of (j,i,*) where * is U/V +/-  
    '''
    n=len(line)
    un_points = []
    up_points = []
    vp_points = []
    all_points = []
    # Between each two points work out direction travelled
    for i in range(n-1):
        p0 = line[i]
        p1 = line[i+1]
        dirnj = p1[0]-p0[0]
        dirni = p1[1]-p0[1]
        if dirnj==1 and dirni==0: # if N then -U
            un_points.append(p1)
            all_points.append((p1[0],p1[1],'U-'))
        elif dirnj==-1 and dirni==0: # if S then U
            up_points.append(p0)
            all_points.append((p0[0],p0[1],'U+'))
        elif dirnj==0 and dirni==1: # if E then V
            vp_points.append(p1)
            all_points.append((p1[0],p1[1],'V+'))
        else:
            print(i,'Unknown dirn', dirni,dirnj)
#    # Display
#    if opts['display']:
#       display_obs(un_points,up_points,vp_points,line,opts)
   #
    return all_points


# These are my functions for commputing MOC, MHT and MFT across OSNAP lines (utilising i, j and flags for velocities that can be obtained using the functions above)


def compute_moc_sigma_osnap(uv_coords_and_falgs, volume_transport_u, volume_transport_v, sigma_input, target_sigma_levels_input):
    '''
    input:
    uv_coords_and_falgs - list containing jj indices, ii indices and flags 
                          obtained as output from Laura's function get_uv_points (that itslef uses get_osnap_sections)
                          now these are indices of the smaller reduced array
                          
    volume_transport_u - volume transport across cell surface parallel to y axis (u_east*drF*dyG);
                         np array with dimensions [depth, lat, lon]
                        
    volume_transport_v - volume transport across cell surface parallel to x axis (v_north*drF*dxG);
                         np array with dimensions [depth, lat, lon]
                     
    sigma_input - np array with dimensions [depth, lat, lon]
    target_sigma_levels_input - target sigma levels
    
    output:
    moc_sigma_section_out - np array with dimensions [sigma]
    '''
    
    moc_sigma_section_out = np.zeros(len(target_sigma_levels_input))
    
    if np.shape(volume_transport_u) != np.shape(volume_transport_v):
        print('Error! Sizes of volume transport arrays do not match!')
        return
    
    if np.shape(volume_transport_u) != np.shape(sigma_input):
        print('Error! Sizes of volume transport arrays and sigmas do not match!')
        return
    
    volume_transport_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
    sigma_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
        
        
    # First we fill in arrays with volume fluxes and sigmas across the section with sizes [depth, len_section]
    
#     print('creating an array with volume transports and sigmas across section...')
    
    for isection in range(len(uv_coords_and_falgs)): 
        
        jj_tmp = uv_coords_and_falgs[isection][0]
        ii_tmp = uv_coords_and_falgs[isection][1]
        flag_tmp = uv_coords_and_falgs[isection][2]
        
        if flag_tmp=='U+':
            transport_tmp = volume_transport_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='U-': 
            transport_tmp = -1.*volume_transport_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='V+': 
            transport_tmp = volume_transport_v[:, jj_tmp, ii_tmp]
            
        sigma_section[:, isection] = sigma_input[:, jj_tmp, ii_tmp]
        
        volume_transport_section[:, isection] = transport_tmp
        
        del jj_tmp, ii_tmp, flag_tmp, transport_tmp
        
        
    transports_in_sigma_0_bins = np.zeros(len(target_sigma_levels_input))

#     print('computing transports in different sigma bins...')
    
    for ibin in np.arange(0, len(transports_in_sigma_0_bins)-1):

        sigma_0_min_tmp = target_sigma_levels_input[ibin]
        sigma_0_max_tmp = target_sigma_levels_input[ibin+1]

        v_transport_sum = 0

        rows, cols = np.where((sigma_section >= sigma_0_min_tmp) & (sigma_section < sigma_0_max_tmp))
        v_transport_sum = volume_transport_section[rows, cols].sum()

#             v_transport_sum = v_transports_tmp_lat.where((sigmas_0_tmp_lat>= sigma_0_min_tmp) & (sigmas_0_tmp_lat < sigma_0_max_tmp)).sum()

        transports_in_sigma_0_bins[ibin] = v_transport_sum


    transports_in_sigma_0_cumcum = -1.*np.cumsum(transports_in_sigma_0_bins[::-1])[::-1]

    moc_sigma_section_out = transports_in_sigma_0_cumcum/1e6
        
    return moc_sigma_section_out


def compute_moc_therm_osnap(uv_coords_and_falgs, volume_transport_u, volume_transport_v, temperature_input, target_temperature_levels_input):
    '''
    input:
    uv_coords_and_falgs - list containing jj indices, ii indices and flags 
                          obtained as output from Laura's function get_uv_points (that itslef uses get_osnap_sections)
                          now these are indices of the smaller reduced array
                          
    volume_transport_u - volume transport across cell surface parallel to y axis (u_east*drF*dyG);
                         np array with dimensions [depth, lat, lon]
                        
    volume_transport_v - volume transport across cell surface parallel to x axis (v_north*drF*dxG);
                         np array with dimensions [depth, lat, lon]
                     
    temperature_input - np array with dimensions [depth, lat, lon]
    target_temperature_levels_input - target temperature levels
    
    output:
    moc_therm_section_out - np array with dimensions [target_temperature_levels_input]
    '''
    
    moc_therm_section_out = np.zeros(len(target_temperature_levels_input))
    
    if np.shape(volume_transport_u) != np.shape(volume_transport_v):
        print('Error! Sizes of volume transport arrays do not match!')
        return
    
    if np.shape(volume_transport_u) != np.shape(temperature_input):
        print('Error! Sizes of volume transport arrays and temperatures do not match!')
        return
    
    volume_transport_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
    temperature_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
        
        
    # First we fill in arrays with volume fluxes and sigmas across the section with sizes [depth, len_section]
    
#     print('creating an array with volume transports and sigmas across section...')
    
    for isection in range(len(uv_coords_and_falgs)): 
        
        jj_tmp = uv_coords_and_falgs[isection][0]
        ii_tmp = uv_coords_and_falgs[isection][1]
        flag_tmp = uv_coords_and_falgs[isection][2]
        
        if flag_tmp=='U+':
            transport_tmp = volume_transport_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='U-': 
            transport_tmp = -1.*volume_transport_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='V+': 
            transport_tmp = volume_transport_v[:, jj_tmp, ii_tmp]
            
        temperature_section[:, isection] = temperature_input[:, jj_tmp, ii_tmp]
        
        volume_transport_section[:, isection] = transport_tmp
        
        del jj_tmp, ii_tmp, flag_tmp, transport_tmp
        
        
    transports_in_temp_bins = np.zeros(len(target_temperature_levels_input))

#     print('computing transports in different temp bins...')
    
    for ibin in np.arange(0, len(transports_in_temp_bins)-1):

        temp_min_tmp = target_temperature_levels_input[ibin]
        temp_max_tmp = target_temperature_levels_input[ibin+1]

        v_transport_sum = 0

        rows, cols = np.where((temperature_section >= temp_min_tmp) & (temperature_section < temp_max_tmp))
        v_transport_sum = volume_transport_section[rows, cols].sum()

#             v_transport_sum = v_transports_tmp_lat.where((sigmas_0_tmp_lat>= sigma_0_min_tmp) & (sigmas_0_tmp_lat < sigma_0_max_tmp)).sum()

        transports_in_temp_bins[ibin] = v_transport_sum


    transports_in_temp_cumcum = -1.*np.cumsum(transports_in_temp_bins)

    moc_therm_section_out = transports_in_temp_cumcum/1e6
        
    return moc_therm_section_out



def compute_moc_salin_osnap(uv_coords_and_falgs, volume_transport_u, volume_transport_v, salinity_input, target_salinity_levels_input):
    '''
    input:
    uv_coords_and_falgs - list containing jj indices, ii indices and flags 
                          obtained as output from Laura's function get_uv_points (that itslef uses get_osnap_sections)
                          now these are indices of the smaller reduced array
                          
    volume_transport_u - volume transport across cell surface parallel to y axis (u_east*drF*dyG);
                         np array with dimensions [depth, lat, lon]
                        
    volume_transport_v - volume transport across cell surface parallel to x axis (v_north*drF*dxG);
                         np array with dimensions [depth, lat, lon]
                     
    salinity_input - np array with dimensions [depth, lat, lon]
    target_salinity_levels_input - target salinity levels
    
    output:
    moc_salin_section_out - np array with dimensions [target_salinity_levels_input]
    '''
    
    moc_salin_section_out = np.zeros(len(target_salinity_levels_input))
    
    if np.shape(volume_transport_u) != np.shape(volume_transport_v):
        print('Error! Sizes of volume transport arrays do not match!')
        return
    
    if np.shape(volume_transport_u) != np.shape(salinity_input):
        print('Error! Sizes of volume transport arrays and temperatures do not match!')
        return
    
    volume_transport_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
    salinity_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
        
        
    # First we fill in arrays with volume fluxes and sigmas across the section with sizes [depth, len_section]
    
#     print('creating an array with volume transports and sigmas across section...')
    
    for isection in range(len(uv_coords_and_falgs)): 
        
        jj_tmp = uv_coords_and_falgs[isection][0]
        ii_tmp = uv_coords_and_falgs[isection][1]
        flag_tmp = uv_coords_and_falgs[isection][2]
        
        if flag_tmp=='U+':
            transport_tmp = volume_transport_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='U-': 
            transport_tmp = -1.*volume_transport_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='V+': 
            transport_tmp = volume_transport_v[:, jj_tmp, ii_tmp]
            
        salinity_section[:, isection] = salinity_input[:, jj_tmp, ii_tmp]
        
        volume_transport_section[:, isection] = transport_tmp
        
        del jj_tmp, ii_tmp, flag_tmp, transport_tmp
        
        
    transports_in_salinity_bins = np.zeros(len(target_salinity_levels_input))

#     print('computing transports in different salinity bins...')
    
    for ibin in np.arange(0, len(transports_in_salinity_bins)-1):

        salin_min_tmp = target_salinity_levels_input[ibin]
        salin_max_tmp = target_salinity_levels_input[ibin+1]

        v_transport_sum = 0

        rows, cols = np.where((salinity_section >= salin_min_tmp) & (salinity_section < salin_max_tmp))
        v_transport_sum = volume_transport_section[rows, cols].sum()

#             v_transport_sum = v_transports_tmp_lat.where((sigmas_0_tmp_lat>= sigma_0_min_tmp) & (sigmas_0_tmp_lat < sigma_0_max_tmp)).sum()

        transports_in_salinity_bins[ibin] = v_transport_sum


    transports_in_salinity_cumcum = np.cumsum(transports_in_salinity_bins[::-1])[::-1]

    moc_salin_section_out = transports_in_salinity_cumcum/1e6
        
    return moc_salin_section_out




def compute_heat_transport_sigma_osnap(uv_coords_and_falgs, heat_transport_x, heat_transport_y): 
    '''
    input:
    uv_coords_and_falgs - list containing jj indices, ii indices and flags 
                          obtained as output from Laura's function get_uv_points (that itslef uses get_osnap_sections)
                          now these are indices of the smaller reduced array
                          
    heat_transport_x    -  heat transport computed by MITgcm (ADVx_TH+DFxE_TH)
                           np array with dimensions [depth, lat, lon]
    
    heat_transport_y    -  heat transport computed by MITgcm (ADVy_TH+DFyE_TH)
                           np array with dimensions [depth, lat, lon]

    
    output:
    heat_transport_section_out - array consisting of one number 
    '''
    
    WATTS_TO_PETAWATTS = 10**-15
    RHO_CONST = 1029
    HEAT_CAPACITY = 4000
    
    if np.shape(heat_transport_x) != np.shape(heat_transport_y):
        print('Error! Sizes of heat transport arrays do not match!')
        return

    heat_transport_section = np.zeros([np.shape(heat_transport_x)[0], len(uv_coords_and_falgs)]) # [depth, len_section]
    
    # Select values across OSNAP and choose x+/y+/x- depending on the mask
    
#     print('creating an array with heat transports and sigmas across section...')
    transport_tmp=[]
    
    for isection in range(len(uv_coords_and_falgs)): 
        
        jj_tmp = uv_coords_and_falgs[isection][0]
        ii_tmp = uv_coords_and_falgs[isection][1]
        flag_tmp = uv_coords_and_falgs[isection][2]
        
        if flag_tmp=='U+':
            transport_tmp = heat_transport_x[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='U-': 
            transport_tmp = -1.*heat_transport_x[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='V+': 
            transport_tmp = heat_transport_y[:, jj_tmp, ii_tmp]
            
        heat_transport_section[:, isection] = transport_tmp
        
        del jj_tmp, ii_tmp, flag_tmp, transport_tmp
        
    # sum across all depths and length of the section and multiply by Cp, Rho and convert to PW
        
    heat_transport_sum = heat_transport_section.sum()

    heat_transport_section_out = WATTS_TO_PETAWATTS * RHO_CONST * HEAT_CAPACITY * heat_transport_sum
        
    return heat_transport_section_out


def compute_salt_transport_sigma_osnap(uv_coords_and_falgs, volume_transport_u, volume_transport_v, salt_u, salt_v, mean_salinity): 
    '''
    input:
    uv_coords_and_falgs - list containing jj indices, ii indices and flags 
                          obtained as output from Laura's function get_uv_points (that itslef uses get_osnap_sections)
                          now these are indices of the smaller reduced array
                          
    volume_transport_u - volume transport across cell surface parallel to y axis (u_east*drF*dyG);
                         np array with dimensions [depth, lat, lon]
                        
    volume_transport_v - volume transport across cell surface parallel to x axis (v_north*drF*dxG);
                         np array with dimensions [depth, lat, lon]
                           
                           
    salt_u    -  salinity from model out interpolated to u grid
                           np array with dimensions [depth, lat, lon]
    
    salt_v    -  salinity from model out interpolated to v grid
                           np array with dimensions [depth, lat, lon]
                           
                           
    mean_salinity - mean salinity across the current array of interest (computed initially in compute_mean_salinities_for_MFT notebook)

    
    output:
    salt_transport_section_out - array consisting of one number 
    '''
    
    CUBIC_METERS_TO_SVERDRUPS = 10**-6

    
    if np.shape(volume_transport_u) != np.shape(volume_transport_v):
        print('Error! Sizes of velocity arrays do not match!')
        return
    
    if np.shape(salt_u) != np.shape(salt_v):
        print('Error! Sizes of salinity arrays do not match!')
        return
    
    if np.shape(salt_u) != np.shape(salt_v):
        print('Error! Sizes of velocity arrays do not match sizes of salinity arrays!')
        return

    salt_transport_section = np.zeros([np.shape(volume_transport_u)[0], len(uv_coords_and_falgs)]) 
    
    # Select values across OSNAP and choose x+/y+/x- depending on the mask
    
#     print('creating an array with heat transports and sigmas across section...')
    transport_tmp=[]
    
    for isection in range(len(uv_coords_and_falgs)): 
        
        jj_tmp = uv_coords_and_falgs[isection][0]
        ii_tmp = uv_coords_and_falgs[isection][1]
        flag_tmp = uv_coords_and_falgs[isection][2]
        
        if flag_tmp=='U+':
            vol_u_tmp = volume_transport_u[:, jj_tmp, ii_tmp]
            s_tmp = salt_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='U-': 
            vol_u_tmp = -1.*volume_transport_u[:, jj_tmp, ii_tmp]
            s_tmp = salt_u[:, jj_tmp, ii_tmp]
            
        if flag_tmp=='V+': 
            vol_u_tmp = volume_transport_v[:, jj_tmp, ii_tmp]
            s_tmp = salt_v[:, jj_tmp, ii_tmp]
            
        salt_transport_tmp = vol_u_tmp * (s_tmp - mean_salinity) / mean_salinity
        
        del vol_u_tmp, s_tmp
            
        salt_transport_section[:, isection] = salt_transport_tmp 
        
        del jj_tmp, ii_tmp, flag_tmp, salt_transport_tmp
        
    # sum across all depths and length of the section and 
        
    salt_transport_sum = salt_transport_section.sum()

    salt_transport_section_out = -1.* CUBIC_METERS_TO_SVERDRUPS * salt_transport_sum
        
    return salt_transport_section_out

# if __name__ == '__main__':
#     try:
#         main()
#     except KeyboardInterrupt as err:
#         print(err)
#         sys.exit() 