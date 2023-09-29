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
from scipy import signal
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


dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirEra5Land = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5-Land'
dirAg6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/research/2020-ag-cmip6'
dirSacks = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

# dirEra5 = '/dartfs-hpc/rc/lab/C/CMIG/ERA5'

crop = 'Maize'

#1981-1990
#1990-2000
#2000-2010
#2010-2018
# years = [1961, 1981]
# years = [1981, 2001]
# years = [2001, 2021]
# years = [1961, 1980]
years = [1981, 2016]

sacksMaizeNc = xr.open_dataset('%s/sacks/%s.crop.calendar.fill.nc'%(dirSacks, crop))
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan
sacksEnd = sacksMaizeNc['harvest'].values
sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
sacksEnd[sacksEnd < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)


# calc era5 quantiles
ds_sm = xr.open_mfdataset('%s/monthly/volumetric_soil_water_layer_1_monthly_*.nc'%(dirEra5Land))

ds_evap = xr.open_mfdataset('%s/monthly/total_evaporation_monthly_*.nc'%(dirEra5Land))
ds_evap['e'] *= -1

ds_sm = ds_sm.sel(time=slice('%d'%years[0],'%d'%years[1]))
ds_evap = ds_evap.sel(time=slice('%d'%years[0],'%d'%years[1]))

print('loading data...')
ds_sm.load()
ds_evap.load()

ds_evap = ds_evap.rename({'latitude':'lat', 'longitude':'lon'})
ds_sm = ds_sm.rename({'latitude':'lat', 'longitude':'lon'})

ds_tasmax = xr.open_dataset('%s/monthly/tasmax_1981.nc'%(dirEra5))
ds_tasmax = ds_tasmax.sel(time=slice('%d'%years[0],'%d'%years[1]))

lat = ds_tasmax.latitude
lon = ds_tasmax.longitude

lat.load()
lon.load()

print('regridding...')
# regrid sacks data
regridMesh = xr.Dataset({'lat': (['lat'], lat),
                         'lon': (['lon'], lon),})

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)
regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)

sacksStart_regrid = regridder_start(sacksStart)
sacksEnd_regrid = regridder_end(sacksEnd)

regridder_e_era5land = xe.Regridder(xr.DataArray(data=ds_evap.e, dims=['time', 'lat', 'lon'], \
                                                 coords={'time': ds_evap.time, 'lat':ds_evap.lat, 'lon':ds_evap.lon}), regridMesh, 'bilinear', reuse_weights=True)
ds_evap_regrid = regridder_e_era5land(ds_evap)
ds_sm_regrid = regridder_e_era5land(ds_sm)


sacksStart_regrid[sacksStart_regrid<1]=1
sacksEnd_regrid[sacksEnd_regrid<1]=1

def is_growing_season(month, sacks_start, sacks_end):
    if sacks_start < sacks_end:
        return (month >= sacks_start) & (month <= sacks_end)
    else:
        return (month >= sacks_start) | (month <= sacks_end)
    
with open('%s/cropped_area/crop_land_regrid_era5.dat'%(dirAg6), 'rb') as f:
    crop_ha_regrid = pickle.load(f)

nnLen = lat.size*lat.size

n = 0

r_sm_et = np.full([len(lat), len(lon)], np.nan)

print('processing...')
for xlat in range(len(lat)):

    if lat[xlat] < -60 or lat[xlat] > 60: continue
    
    for ylon in range(len(lon)):

        if n % 20000 == 0:
                print('%.1f %% complete'%(n/(nnLen)*100))
                
        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):# and crop_ha_regrid[xlat, ylon] > 0:

            
            sacks_start_month = datetime.datetime.strptime('2020%d'%int(sacksStart_regrid[xlat,ylon]), '%Y%j').date().month
            sacks_end_month = datetime.datetime.strptime('2020%d'%int(sacksEnd_regrid[xlat,ylon]), '%Y%j').date().month
            
            # select growing seasons for all years
            ds_sm_growing = ds_sm_regrid.swvl1[:,xlat,ylon].sel(time=is_growing_season(ds_sm_regrid['time.month'], sacks_start_month, sacks_end_month))
            ds_evap_growing = ds_evap_regrid.e[:,xlat,ylon].sel(time=is_growing_season(ds_evap_regrid['time.month'], sacks_start_month, sacks_end_month))
            

            # resample to group growing seasons (1 year frequency, offset starting at 1 month before start of growing season)
            ds_sm_growing_1y = ds_sm_growing.resample(time='1Y', loffset='%dM'%(sacks_start_month-1)).mean()
            ds_evap_growing_1y = ds_evap_growing.resample(time='1Y', loffset='%dM'%(sacks_start_month-1)).mean()
            
            nn = np.where((~np.isnan(ds_sm_growing_1y.values)) & (~np.isnan(ds_evap_growing_1y.values)))[0]
            
            if nn.size > 10:
                ds_sm_growing_1y_detrend = signal.detrend(ds_sm_growing_1y.values[nn])
                ds_evap_growing_1y_detrend = signal.detrend(ds_evap_growing_1y.values[nn])

    #             r_t_et[xlat, ylon] = np.corrcoef(ds_sm_growing_1y.values,ds_evap_growing_1y.values)[0,1]
                r_sm_et[xlat, ylon] = np.corrcoef(ds_sm_growing_1y_detrend,ds_evap_growing_1y_detrend)[0,1]

        n += 1
                
da_grow_r_sm_et = xr.DataArray(data   = r_sm_et, 
                      dims   = ['latitude', 'longitude'],
                      coords = {'latitude':lat, 'longitude':lon},
                      attrs  = {'units'     : 'correlation'
                        })
ds_grow_r_sm_et = xr.Dataset()
ds_grow_r_sm_et['r_sm_et'] = da_grow_r_sm_et


print('saving netcdf...')
# ds_grow_r_t_et.to_netcdf('r_t_et_era5_crop_restricted_%d_%d.nc'%(years[0], years[1]))
ds_grow_r_sm_et.to_netcdf('r_sm_et_era5_%d_%d_detrend-%s.nc'%(years[0], years[1], crop))
