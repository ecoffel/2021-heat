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

dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

file_var = 'sm'
orig_var = 'swvl1'
crop = 'Maize'

# decile bins
bins = np.arange(0, 101, 1)

n = 0

xlat = int(sys.argv[1])

# leave in K
sm_era5 = xr.open_mfdataset('%s/daily/%s*.nc'%(dirERA5, file_var))


lat = sm_era5.latitude.values
lon = sm_era5.longitude.values

cur_sm_deciles = np.full([lon.size, bins.size], np.nan)

# this is run for EACH latitude band
cur_lat_1 = lat[xlat]
print('opening era5 full time series for lat = (%.2f)...'%(cur_lat_1))


print('selecting sm...')
sm_era5 = sm_era5.sel(latitude=cur_lat_1)
print('loading sm...')
sm_era5.load()


print('processing sm deciles lat = %.2f...'%cur_lat_1)

# loop over all longitudes
for ylon in range(0, lon.size):

    cur_sm = sm_era5.swvl1[:, ylon] 

    cur_sm_deciles[ylon, :] = np.nanpercentile(cur_sm, bins)
    
print('writing files...')
with open('decile_bins/era5-land-sm-daily/sm_percentiles_%d.dat'%(xlat), 'wb') as f:
    pickle.dump(cur_sm_deciles, f)



