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

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'




# year = int(sys.argv[1])

# txx = np.full([lat.size, lon.size], np.nan)
# tx_mean_grow = np.full([lat.size, lon.size], np.nan)

# print(year)

print('loading tx')
era5_tx = xr.open_mfdataset('%s/daily/tasmax_*.nc'%(dirEra5), combine='by_coords')
# era5_tx['mx2t'] -= 273.15

# lat = era5_tx.latitude.values
# lon = era5_tx.longitude.values

# lat_inds = np.arange(lat.size)
# lon_inds = np.arange(lon.size)

# N_gridcell = lat.size*lon.size

# sacksMaizeNc = xr.open_dataset('%s/sacks/Maize.crop.calendar.fill.nc'%dirAgData)
# sacksStart = sacksMaizeNc['plant'].values
# sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
# sacksStart[sacksStart < 0] = np.nan
# sacksEnd = sacksMaizeNc['harvest'].values
# sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
# sacksEnd[sacksEnd < 0] = np.nan

# sacksLat = np.linspace(90, -90, 360)
# sacksLon = np.linspace(0, 360, 720)

# # regrid sacks data
# regridMesh = xr.Dataset({'lat': (['lat'], lat),
#                          'lon': (['lon'], lon),})

# regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)
# regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)

# sacksStart_regrid = regridder_start(sacksStart)
# sacksEnd_regrid = regridder_end(sacksEnd)

# txx = era5_tx.resample(time='1Y').max()
# txx = txx.rename({'mx2t':'txx'})
# txx.to_netcdf('%s/txx/era5_txx.nc'%(dirHeatData), encoding={'txx': {"dtype": "float32", "zlib": True, 'complevel':9}})

txx = era5_tx.resample(time='1Y').mean()
txx = txx.rename({'mx2t':'t_mean'})
txx.to_netcdf('%s/txx/era5_tmean.nc'%(dirHeatData), encoding={'t_mean': {"dtype": "float32", "zlib": True, 'complevel':9}})

# print('writing files...')
# with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tw_%d_%d.dat'%(dirHeatData, threshold_perc, year), 'wb') as f:
#     pickle.dump(tx_during_tw, f)
# with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tx_%d_%d.dat'%(dirHeatData, threshold_perc, year), 'wb') as f:
#     pickle.dump(tw_during_tx, f)
    
# with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tx_%d_%d.dat'%(dirHeatData, threshold_perc, year), 'wb') as f:
#     pickle.dump(tx_during_tx, f)
# with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tw_%d_%d.dat'%(dirHeatData, threshold_perc, year), 'wb') as f:
#     pickle.dump(tw_during_tw, f)
    
# with open('%s/heat-wave-days/tx-on-tw/era5_tx_diff_from_tw_on_tw_%d.dat'%(dirHeatData, year), 'wb') as f:
#     pickle.dump(tx_diff_from_tw_during_tw, f)
# with open('%s/heat-wave-days/tw-on-tx/era5_tw_diff_from_tx_on_tx_%d.dat'%(dirHeatData, year), 'wb') as f:
#     pickle.dump(tw_diff_from_tx_during_tx, f)
    


