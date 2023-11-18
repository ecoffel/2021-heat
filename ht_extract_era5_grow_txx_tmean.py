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

print('opening era5 %d...'%year)
temp_era5 = xr.open_dataset('%s/daily/%s_%d.nc'%(dirERA5, file_var, year))
temp_era5.load()
temp_era5[orig_var] -= 273.15

print('opening era5 %d...'%(year-1))
temp_era5_last_year = xr.open_dataset('%s/daily/%s_%d.nc'%(dirERA5, file_var, year-1))
temp_era5_last_year.load()
temp_era5_last_year[orig_var] -= 273.15

temp_era5 = temp_era5.rename_dims(latitude='lat', longitude='lon')
temp_era5 = temp_era5.rename({'latitude':'lat', 'longitude':'lon'})

temp_era5_last_year = temp_era5_last_year.rename_dims(latitude='lat', longitude='lon')
temp_era5_last_year = temp_era5_last_year.rename({'latitude':'lat', 'longitude':'lon'})

lat = temp_era5.lat
lon = temp_era5.lon

# LOAD SOIL MOISTURE
sm_deciles = np.full([lat.size, lon.size, 101], np.nan)

print('opening sm datasets for %d'%year)
sm1_era5 = xr.open_dataset('%s/daily/sm_layer_1_%d.nc'%(dirEra5Land, year))
sm2_era5 = xr.open_dataset('%s/daily/sm_layer_2_%d.nc'%(dirEra5Land, year))
sm_era5 = sm1_era5.swvl1 + sm2_era5.swvl2
sm_era5.load()

sm1_era5_last_year = xr.open_dataset('%s/daily/sm_layer_1_%d.nc'%(dirEra5Land, year-1))
sm2_era5_last_year = xr.open_dataset('%s/daily/sm_layer_2_%d.nc'%(dirEra5Land, year-1))
sm_era5_last_year = sm1_era5_last_year.swvl1 + sm2_era5_last_year.swvl2
sm_era5_last_year.load()

sm_era5 = sm_era5.rename({'latitude':'lat', 'longitude':'lon'})
sm_era5_last_year = sm_era5_last_year.rename({'latitude':'lat', 'longitude':'lon'})

# load sm deciles
sm_deciles = np.full([101, 1801, 3600], np.nan)

for xlat in range(0, sm_deciles.shape[1]):
    with open('decile_bins/era5-land-sm-daily/sm_percentiles_layer_1_and_2_%d.dat'%(xlat), 'rb') as f:
        tmp = pickle.load(f)
        sm_deciles[:, xlat, :] = tmp.T

da_sm_deciles = xr.DataArray(data   = sm_deciles, 
                          dims   = ['percentile', 'lat', 'lon'],
                          coords = {'percentile':np.arange(0, 101, 1), 'lat':sm_era5.lat, 'lon':sm_era5.lon})


# load et
# print('loading et')
# era5_et_deciles = xr.open_dataset('%s/era5_evaporation_deciles.nc'%dirHeatData)
# era5_et_deciles.load()

# era5_et = xr.open_dataset('%s/daily/evaporation_%d.nc'%(dirERA5, year))
# era5_et.load()

# era5_et_last_year = xr.open_dataset('%s/daily/evaporation_%d.nc'%(dirERA5, year-1))
# era5_et_last_year.load()

# era5_et_deciles_values = era5_et_deciles.e.values.copy()

percentile_bins = np.arange(0, 101, 1)


# regrid sacks data to current model res
regridMesh = xr.Dataset({'lat': (['lat'], temp_era5.lat),
                                   'lon': (['lon'], temp_era5.lon)})

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)
regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)

sacksStart_regrid = regridder_start(sacksStart)
sacksEnd_regrid = regridder_end(sacksEnd)


regridder_sm_era5 = xe.Regridder(xr.DataArray(data=sm_era5, dims=['time', 'lat', 'lon'], coords={'time':sm_era5.time, 'lat':sm_era5.lat, 'lon':sm_era5.lon}), regridMesh, 'bilinear', reuse_weights=True)

regridder_sm_era5_last_year = xe.Regridder(xr.DataArray(data=sm_era5_last_year, dims=['time', 'lat', 'lon'], coords={'time':sm_era5_last_year.time, 'lat':sm_era5_last_year.lat, 'lon':sm_era5_last_year.lon}), regridMesh, 'bilinear', reuse_weights=True)

regridder_sm_deciles = xe.Regridder(xr.DataArray(data=da_sm_deciles, dims=['percentile', 'lat', 'lon'], coords={'percentile':np.arange(0,101,1), 'lat':sm_era5.lat, 'lon':sm_era5.lon}), regridMesh, 'bilinear', reuse_weights=True)

sm_era5_regrid = regridder_sm_era5(sm_era5)
sm_era5_last_year_regrid = regridder_sm_era5_last_year(sm_era5_last_year)
sm_deciles_regrid = regridder_sm_deciles(da_sm_deciles)


# count up all non-nan grid cells so we can estimate percent complete
ngrid = 0
for xlat in range(temp_era5.lat.size):
    for ylon in range(temp_era5.lon.size):
        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
            ngrid += 1


yearly_grow_tmax = np.full([temp_era5.lat.size, temp_era5.lon.size], np.nan)
yearly_grow_sm_on_tmax = np.full([temp_era5.lat.size, temp_era5.lon.size], np.nan)
# yearly_grow_et_on_tmax = np.full([temp_era5.lat.size, temp_era5.lon.size], np.nan)
yearly_grow_tmean = np.full([temp_era5.lat.size, temp_era5.lon.size], np.nan)
            
            
for xlat in range(temp_era5.lat.size):

    for ylon in range(temp_era5.lon.size):

#         if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
    
#             if n % 1000 == 0:
#                 print('%.2f%%'%(n/ngrid*100))

#             cur_sm_deciles = sm_deciles_regrid.values[:, xlat, ylon]
#             cur_et_deciles = era5_et_deciles['e'].values[:, xlat, ylon]
            
#             curTmax = temp_era5[orig_var][:, xlat, ylon].values
#             curSm = sm_era5_regrid[:, xlat, ylon].values
#             curEt = era5_et['e'][:, xlat, ylon].values

#             if len(curTmax) > 0:
#                 yearly_grow_tmax[xlat, ylon] = np.nanmax(curTmax)

#                 max_ind = np.argmax(curTmax)

#                 if ~np.isnan(curSm[max_ind]):
#                     cur_p_ind = abs(cur_sm_deciles-curSm[max_ind]).argmin()
#                     yearly_grow_sm_on_tmax[xlat, ylon] = percentile_bins[cur_p_ind]

#                 if ~np.isnan(curEt[max_ind]):
#                     cur_p_ind = abs(cur_et_deciles-curEt[max_ind]).argmin()
#                     yearly_grow_et_on_tmax[xlat, ylon] = 1-percentile_bins[cur_p_ind]

#                 yearly_grow_tmean[xlat, ylon] = np.nanmean(curTmax)
                
#             n += 1
            
        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
            if n % 1000 == 0:
                print('%.2f%%'%(n/ngrid*100))
            
            if sacksStart_regrid[xlat, ylon] > sacksEnd_regrid[xlat, ylon]:
                
                cur_sm_deciles = sm_deciles_regrid.values[:, xlat, ylon]

                # start loop on 2nd year to allow for growing season that crosses jan 1
                curTmax1 = temp_era5_last_year[orig_var][int(sacksStart_regrid[xlat, ylon]):, xlat, ylon].values
                curTmax2 = temp_era5[orig_var][:int(sacksEnd_regrid[xlat, ylon]), xlat, ylon].values
                curTmax = np.concatenate([curTmax1, curTmax2])
                
                curSm1 = sm_era5_last_year_regrid[int(sacksStart_regrid[xlat, ylon]):, xlat, ylon].values
                curSm2 = sm_era5_regrid[:int(sacksEnd_regrid[xlat, ylon]), xlat, ylon].values
                curSm = np.concatenate([curSm1, curSm2])
                
#                 curEt1 = era5_et_last_year['e'][int(sacksStart_regrid[xlat, ylon]):, xlat, ylon].values
#                 curEt2 = era5_et['e'][:int(sacksEnd_regrid[xlat, ylon]), xlat, ylon].values
#                 curEt = np.concatenate([curEt1, curEt2])

                if len(curTmax) > 0:
                    yearly_grow_tmax[xlat, ylon] = np.nanmax(curTmax)
                    
                    max_ind = np.argmax(curTmax)
                    
                    if ~np.isnan(curSm[max_ind]):
                        cur_p_ind = abs(cur_sm_deciles-curSm[max_ind]).argmin()
                        yearly_grow_sm_on_tmax[xlat, ylon] = percentile_bins[cur_p_ind]
                        
#                     if ~np.isnan(curEt[max_ind]):
#                         cur_p_ind = abs(cur_et_deciles-curEt[max_ind]).argmin()
#                         yearly_grow_et_on_tmax[xlat, ylon] = percentile_bins[cur_p_ind]

                    yearly_grow_tmean[xlat, ylon] = np.nanmean(curTmax)
                n += 1
                

            else:
                cur_sm_deciles = sm_deciles_regrid.values[:, xlat, ylon]
                
                curTmax = temp_era5[orig_var][int(sacksStart_regrid[xlat, ylon]):int(sacksEnd_regrid[xlat, ylon]), xlat, ylon].values
                curSm = sm_era5_regrid[int(sacksStart_regrid[xlat, ylon]):int(sacksEnd_regrid[xlat, ylon]), xlat, ylon].values
#                 curEt = era5_et['e'][:int(sacksEnd_regrid[xlat, ylon]), xlat, ylon].values
                
                if len(curTmax) > 0:
                    yearly_grow_tmax[xlat, ylon] = np.nanmax(curTmax)
                    
                    max_ind = np.argmax(curTmax)
                    
                    if ~np.isnan(curSm[max_ind]):
                        cur_p_ind = abs(cur_sm_deciles-curSm[max_ind]).argmin()
                        yearly_grow_sm_on_tmax[xlat, ylon] = percentile_bins[cur_p_ind]
                        
#                     if ~np.isnan(curEt[max_ind]):
#                         cur_p_ind = abs(cur_et_deciles-curEt[max_ind]).argmin()
#                         yearly_grow_et_on_tmax[xlat, ylon] = percentile_bins[cur_p_ind]
                    
                    yearly_grow_tmean[xlat, ylon] = np.nanmean(curTmax)
                n += 1

print('renaming dims...')

da_grow_tmax = xr.DataArray(data   = yearly_grow_tmax, 
                      dims   = ['lat', 'lon'],
                      coords = {'time':year, 'lat':temp_era5.lat, 'lon':temp_era5.lon},
                      attrs  = {'units'     : 'C'
                        })
ds_grow_tmax = xr.Dataset()
ds_grow_tmax['%s_grow_max'%file_var] = da_grow_tmax


da_grow_sm_on_tmax = xr.DataArray(data   = yearly_grow_sm_on_tmax, 
                      dims   = ['lat', 'lon'],
                      coords = {'time':year, 'lat':temp_era5.lat, 'lon':temp_era5.lon},
                      attrs  = {'units'     : 'C'
                        })
ds_grow_sm_on_tmax = xr.Dataset()
ds_grow_sm_on_tmax['sm_grow_on_tmax'] = da_grow_sm_on_tmax


# da_grow_et_on_tmax = xr.DataArray(data   = yearly_grow_et_on_tmax, 
#                       dims   = ['lat', 'lon'],
#                       coords = {'time':year, 'lat':temp_era5.lat, 'lon':temp_era5.lon},
#                       attrs  = {'units'     : 'C'
#                         })
# ds_grow_et_on_tmax = xr.Dataset()
# ds_grow_et_on_tmax['et_grow_on_tmax'] = da_grow_et_on_tmax


da_grow_tmean = xr.DataArray(data   = yearly_grow_tmean, 
                      dims   = ['lat', 'lon'],
                      coords = {'time': year, 'lat':temp_era5.lat, 'lon':temp_era5.lon},
                      attrs  = {'units'     : 'C'
                        })
ds_grow_tmean = xr.Dataset()
ds_grow_tmean['%s_grow_mean'%file_var] = da_grow_tmean


print('saving netcdf...')
# ds_grow_tmax.to_netcdf('era5_txx_tmean/era5_%s_txx_%d.nc'%(crop, year))
# ds_grow_tmean.to_netcdf('era5_txx_tmean/era5_%s_t_mean_%d.nc'%(crop, year))
ds_grow_sm_on_tmax.to_netcdf('era5_txx_tmean/era5_%s_sm_on_txx_2_layers_%d.nc'%(crop, year))
# ds_grow_et_on_tmax.to_netcdf('era5_txx_tmean/era5_%s_et_on_txx_fixed_sign_%d.nc'%(crop, year))