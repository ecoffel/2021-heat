import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import cartopy
import cartopy.crs as ccrs
import glob
import sys, pickle
import datetime

dirERA5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

crop = 'Maize'
file_var = 'volumetric_soil_water_layer_1_monthly'


# decile bins
bins = np.arange(0, 101, 1)

n = 0

xlat = int(sys.argv[1])





print('opening era5 full time series')

# leave in K
sm_era5 = xr.open_mfdataset('%s/monthly/%s*.nc'%(dirERA5Land, file_var))

lat = sm_era5.latitude.values
lon = sm_era5.longitude.values

# this is run for EACH latitude band
cur_lat_1 = lat[xlat]

print('selecting sm for lat = (%.2f)...'%(cur_lat_1))
sm_era5 = sm_era5.sel(latitude=cur_lat_1)
print('loading sm...')
sm_era5.load()



cur_sm_deciles = np.full([lon.size, bins.size], np.nan)


print('processing soil water deciles lat = %.2f...'%cur_lat_1)
# loop over all longitudes
for ylon in range(0, lon.size):

    cur_sm = sm_era5.swvl1[:, ylon]
    
    cur_sm_deciles[ylon, :] = np.nanpercentile(cur_sm, bins)
        
print('writing files...')
with open('sm_percentiles_%d.dat'%(xlat), 'wb') as f:
    pickle.dump(cur_sm_deciles, f)



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
