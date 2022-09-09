import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import cartopy
import cartopy.crs as ccrs
import glob
import sys
import datetime

dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

file_var = 'dp_mean'
orig_var = 'd2m'
crop = 'Maize'

era5_max_deciles = xr.open_dataset('%s/era5_tw_max_deciles.nc'%dirHeatData)
lat = era5_max_deciles.latitude.values
lon = era5_max_deciles.longitude.values


sacksMaizeNc = xr.open_dataset('%s/sacks/%s.crop.calendar.fill.nc'%(dirAgData, crop))
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan
sacksEnd = sacksMaizeNc['harvest'].values
sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
sacksEnd[sacksEnd < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)

# decile bins
bins = np.arange(0, 101, 1)

n = 0

num_chunks = 32
n_chunk = int(sys.argv[1])

lat_chunk = round(lat.size/num_chunks)
lon_chunk = round(lon.size/num_chunks)

# loop over all latitudes
for xlat_ind, xlat in enumerate(range(n_chunk*lat_chunk, (n_chunk+1)*lat_chunk, 1)):

    cur_lat_1 = lat[xlat]
    
    if xlat+lat_chunk < lat.size:
        cur_lat_2 = lat[xlat+lat_chunk]
    else:
        cur_lat_2 = lat[-1]
    
    cur_huss_deciles = np.full([lat_chunk, lon.size, bins.size], np.nan)
    
    print('opening era5 full time series for lat = (%.2f, %.2f)...'%(cur_lat_1, cur_lat_2))

    # leave in K
    dp_era5 = xr.open_mfdataset('%s/daily/%s*.nc'%(dirERA5, file_var))
    print('selecting dp...')
    dp_era5 = dp_era5.sel(latitude=slice(cur_lat_1, cur_lat_2))
    print('loading dp...')
    dp_era5.load()

    # in hpa
    sp_era5 = xr.open_mfdataset('%s/daily/%s*.nc'%(dirERA5, 'sp'))
    print('selecting sp...')
    sp_era5 = sp_era5.sel(latitude=slice(cur_lat_1, cur_lat_2))
    print('loading sp...')
    sp_era5.load()
    sp_era5['sp'] /= 100
    
    print('processing huss deciles lat = %.2f...'%cur_lat_1)
    
    # loop over all longitudes
    for ylon in range(0, lon.size):
        
        
        
        cur_dp = dp_era5.d2m[:, xlat_ind, ylon]
        cur_sp = sp_era5.sp[:, xlat_ind, ylon]

        e0 = 6.113
        c_water = 5423

        cur_q = (622 * e0 * np.exp(c_water * (cur_dp - 273.15)/(cur_dp * 273.15)))/cur_sp # g/kg 

        cur_huss_deciles[xlat_ind, ylon, :] = np.nanpercentile(cur_q, bins)
        

        
print('writing files...')
with open('huss_percentiles_%d.dat'%(n_chunk), 'wb') as f:
    pickle.dump(cur_huss_deciles, f)




# sys.exit()
# sp_era5['sp'] /= 100

# # decile bins
# bins = np.arange(0, 101, 1)
            
# # this will hold the decile cutoffs for every grid cell for the current year
# yearly_grow_huss_deciles = np.full([dp_era5.lat.size, dp_era5.lon.size, bins.size], np.nan)



#        # if crop calendar is not nan at this grid cell 
#         if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
    
#             # print out progress through loop
#             if n % 25000 == 0:
#                 print('%.2f%%'%(n/ngrid*100))

#             curDp = dp_era5[orig_var][:, xlat, ylon]
# #             curTx = temp_era5['mx2t'][:, xlat, ylon]
#             curSp = sp_era5['sp'][:, xlat, ylon] # should be in hpa
            
#             e0 = 6.113
#             c_water = 5423
            
#             curQ = (622 * e0 * np.exp(c_water * (curDp - 273.15)/(curDp * 273.15)))/curSp # g/kg 

            
#             yearly_huss_deciles[:, xlat, ylon] = np.nanpercentile(curDp, bins)
            
#             n += 1

# print('renaming dims...')

# da_grow_tmax_deciles = xr.DataArray(data   = yearly_grow_tmax_deciles, 
#                       dims   = ['lat', 'lon', 'bin'],
#                       coords = {'bin':bins, 'lat':temp_era5.lat, 'lon':temp_era5.lon},
#                       attrs  = {'units'     : 'C'
#                         })
# ds_grow_tmax_deciles = xr.Dataset()
# ds_grow_tmax_deciles['%s_grow_deciles'%file_var] = da_grow_tmax_deciles


# print('saving netcdf...')
# ds_grow_tmax_deciles.to_netcdf('decile_bins/era5_%s_%s_growing_season_deciles_%d.nc'%(crop, file_var, year))
