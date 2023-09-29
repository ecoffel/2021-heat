import rasterio as rio
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
import numpy as np
import numpy.matlib
from scipy import interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import scipy
import os, sys, pickle, gzip
import datetime
import geopy.distance
import xarray as xr
import pandas as pd
import rasterio
import geopandas as gpd
import shapely.geometry
import shapely.ops
import xesmf as xe
import cartopy
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import itertools
import random
import metpy
from metpy.plots import USCOUNTIES
import warnings
warnings.filterwarnings('ignore')

dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirSacks = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

file_var = 'tasmax'
orig_var = 'mx2t'
crop = 'Soybeans'

year = int(sys.argv[1])

latRange = [-90, 90]
lonRange = [0, 360]

sacksMaizeNc = xr.open_dataset('%s/sacks/%s.crop.calendar.fill.nc'%(dirSacks, crop))
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan
sacksEnd = sacksMaizeNc['harvest'].values
sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
sacksEnd[sacksEnd < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)

regridMesh_cur_model = xr.Dataset()

n = 0

# print('opening era5 %d...'%year)
temp_era5 = xr.open_mfdataset(['%s/daily/%s_%d.nc'%(dirERA5, file_var, year-1), '%s/daily/%s_%d.nc'%(dirERA5, file_var, year)], concat_dim='time')
temp_era5[orig_var] -= 273.15
temp_era5 = temp_era5.rename({'latitude':'lat', 'longitude':'lon'})

lat = temp_era5.lat
lon = temp_era5.lon

# LOAD ET
et_era5_deciles = xr.open_dataset('%s/era5_evaporation_deciles.nc'%dirHeatData)
et_era5_deciles.load()

et_era5 = xr.open_mfdataset(['%s/daily/evaporation_%d.nc'%(dirERA5, year-1), '%s/daily/evaporation_%d.nc'%(dirERA5, year)], concat_dim='time')
et_era5 = et_era5.rename({'latitude':'lat', 'longitude':'lon'})
et_era5 = et_era5['e']
et_era5_deciles_values = et_era5_deciles.e.values.copy()


# # LOAD SOIL MOISTURE
sm_deciles = np.full([lat.size, lon.size, 101], np.nan)

# print('opening sm datasets for %d'%year)
sm_era5 = xr.open_mfdataset(['%s/daily/sm_%d.nc'%(dirEra5Land, year-1), '%s/daily/sm_%d.nc'%(dirEra5Land, year)], concat_dim='time')
sm_era5 = sm_era5.rename({'latitude':'lat', 'longitude':'lon'})

# # load sm deciles
sm_deciles = np.full([101, 1801, 3600], np.nan)

print('loading sm deciles')
for xlat in range(0, sm_deciles.shape[1]):
    with open('decile_bins/era5-land-sm-daily/sm_percentiles_%d.dat'%(xlat), 'rb') as f:
        tmp = pickle.load(f)
        sm_deciles[:, xlat, :] = tmp.T

da_sm_deciles = xr.DataArray(data   = sm_deciles, 
                          dims   = ['percentile', 'lat', 'lon'],
                          coords = {'percentile':np.arange(0, 101, 1), 'lat':sm_era5.lat, 'lon':sm_era5.lon})



percentile_bins = np.arange(0, 101, 1)

# regrid sacks data to current model res
regridMesh = xr.Dataset({'lat': (['lat'], temp_era5.lat),
                                   'lon': (['lon'], temp_era5.lon)})

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)
regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)

regridder_sm_era5 = xe.Regridder(xr.DataArray(data=sm_era5.swvl1, dims=['time', 'lat', 'lon'], coords={'time':sm_era5.time, 'lat':sm_era5.lat, 'lon':sm_era5.lon}), regridMesh, 'bilinear', reuse_weights=True)
regridder_sm_deciles = xe.Regridder(xr.DataArray(data=da_sm_deciles, dims=['percentile', 'lat', 'lon'], coords={'percentile':np.arange(0,101,1), 'lat':sm_era5.lat, 'lon':sm_era5.lon}), regridMesh, 'bilinear', reuse_weights=True)

sm_era5_regrid = regridder_sm_era5(sm_era5.swvl1)
sm_deciles_regrid = regridder_sm_deciles(da_sm_deciles)

sacksStart_regrid = regridder_start(sacksStart)
sacksEnd_regrid = regridder_end(sacksEnd)

planting_dates = xr.DataArray(sacksStart_regrid, dims=["lat", "lon"], coords=[temp_era5.lat, temp_era5.lon])
harvesting_dates = xr.DataArray(sacksEnd_regrid, dims=["lat", "lon"], coords=[temp_era5.lat, temp_era5.lon])

# Create a DataArray with the same size as the temperature data, filled with the year of each observation
year_da, _ = xr.broadcast(temp_era5['time.year'], temp_era5.mx2t)

# Create masks for the growing season
same_year_mask = ((year_da == year) & (year_da['time.dayofyear'] >= planting_dates) & 
                  (year_da['time.dayofyear'] <= harvesting_dates) & (planting_dates <= harvesting_dates))

last_year_mask = ((year_da == year-1) & (year_da['time.dayofyear'] >= planting_dates) & (planting_dates > harvesting_dates)) | \
                 ((year_da == year) & (year_da['time.dayofyear'] <= harvesting_dates) & (planting_dates > harvesting_dates))

# Combine the masks and select the temperature data for the growing season
growing_season_mask = same_year_mask | last_year_mask
growing_season_temp = temp_era5.mx2t.where(growing_season_mask)

# Find the maximum temperature in the growing season for each grid cell
max_growing_season_temp = growing_season_temp.max(dim="time", skipna=True)
mean_growing_season_temp = growing_season_temp.mean(dim="time", skipna=True)


# Stack the spatial dimensions into a single dimension
growing_season_temp_stacked = growing_season_temp.stack(location=['lat', 'lon'])
sm_era5_regrid_stacked = sm_era5_regrid.stack(location=['lat', 'lon'])
et_era5_stacked = et_era5.stack(location=['lat', 'lon'])

# Find the date of maximum temperature for each 'location'
date_of_max_temp_stacked = growing_season_temp_stacked.idxmax(dim="time", skipna=True)

# Create a mask for valid (non-NaT) dates
valid_dates_mask = ~date_of_max_temp_stacked.isnull()

# Apply the mask to select only valid dates
date_of_max_temp_stacked_valid = date_of_max_temp_stacked.where(valid_dates_mask)

# Create a new coordinate for 'time' in the soil moisture DataArray that matches the 'location' dimension
sm_era5_regrid_stacked['time_matched'] = ('location', date_of_max_temp_stacked_valid)

# Select soil moisture values using the new 'time_matched' coordinate
sm_hottest_day_stacked = sm_era5_regrid_stacked.sel(time=sm_era5_regrid_stacked.time_matched, method='nearest')

# Unstack the 'location' dimension to retrieve the original spatial dimensions
sm_hottest_day = sm_hottest_day_stacked.unstack('location')
sm_hottest_day.load()

# now for ET
et_era5_stacked['time_matched'] = ('location', date_of_max_temp_stacked_valid)
et_hottest_day_stacked = et_era5_stacked.sel(time=et_era5_stacked.time_matched, method='nearest')
et_hottest_day = et_hottest_day_stacked.unstack('location')
et_hottest_day.load()


# If sm_deciles_regrid is a numpy array, convert it to a DataArray
if isinstance(sm_deciles_regrid, np.ndarray):
    sm_deciles_regrid = xr.DataArray(
        sm_deciles_regrid, 
        dims=['percentile', 'lat', 'lon'], 
        coords={'percentile': np.arange(0, 101, 1), 'lat': sm_hottest_day.lat, 'lon': sm_hottest_day.lon}
    )

# Find the nearest percentile value for each grid cell of sm_hottest_day
sm_hottest_day_percentiles = xr.apply_ufunc(
    np.digitize, 
    sm_hottest_day, 
    sm_deciles_regrid, 
    input_core_dims=[[], ['percentile']], 
    vectorize=True
) - 1  # subtract 1 because np.digitize bins start at 1


if isinstance(et_era5_deciles_values, np.ndarray):
    et_era5_deciles_values = xr.DataArray(
        et_era5_deciles_values, 
        dims=['percentile', 'lat', 'lon'], 
        coords={'percentile': np.arange(0, 101, 1), 'lat': et_hottest_day.lat, 'lon': et_hottest_day.lon}
    )

# Find the nearest percentile value for each grid cell of sm_hottest_day
et_hottest_day_percentiles = xr.apply_ufunc(
    np.digitize, 
    et_hottest_day, 
    et_era5_deciles_values, 
    input_core_dims=[[], ['percentile']], 
    vectorize=True
) - 1  # subtract 1 because np.digitize bins start at 1




print('saving netcdf...')
max_growing_season_temp.to_netcdf('era5_txx_tmean/era5_%s_txx_%d_vector.nc'%(crop, year))
mean_growing_season_temp.to_netcdf('era5_txx_tmean/era5_%s_t_mean_%d_vector.nc'%(crop, year))
sm_hottest_day_percentiles.to_netcdf('era5_txx_tmean/era5_%s_sm_on_txx_%d_vector.nc'%(crop, year))
et_hottest_day_percentiles.to_netcdf('era5_txx_tmean/era5_%s_et_on_txx_%d_vector.nc'%(crop, year))
# ds_grow_et_on_tmax.to_netcdf('era5_txx_tmean/era5_%s_et_on_txx_fixed_sign_%d.nc'%(crop, year))