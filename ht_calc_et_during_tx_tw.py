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

# tx or tw 
ref_var = 'tx'

year = int(sys.argv[1])
print(year)

print('loading percentile data')
if ref_var == 'tw':
    era5_hw_deciles = xr.open_dataset('%s/era5_tw_max_deciles.nc'%dirHeatData)
    era5_hw_deciles.load()
    era5_hw_deciles_values = era5_hw_deciles.tw.values.copy()
    
    print('loading tw')
    era5_hw = xr.open_dataset('%s/daily/tw_max_%d.nc'%(dirEra5, year))
    era5_hw.load()
elif ref_var == 'tx':
    era5_hw_deciles = xr.open_dataset('%s/era5_tasmax_deciles.nc'%dirHeatData)
    era5_hw_deciles.load()
    era5_hw_deciles['mx2t'] -= 273.15
    era5_hw_deciles_values = era5_hw_deciles.mx2t.values.copy()
    
    print('loading tx')
    era5_hw = xr.open_dataset('%s/daily/tasmax_%d.nc'%(dirEra5, year))
    era5_hw.load()
    era5_hw['mx2t'] -= 273.15

percentile_bins = era5_hw_deciles['quantile'].values

lat = era5_hw_deciles.latitude.values
lon = era5_hw_deciles.longitude.values

# load et
print('loading et')
era5_et_deciles = xr.open_dataset('%s/era5_evaporation_deciles.nc'%dirHeatData)
era5_et_deciles.load()


era5_et = xr.open_dataset('%s/daily/evaporation_%d.nc'%(dirEra5, year))
era5_et.load()


era5_et_deciles_values = era5_et_deciles.e.values.copy()

# find tx when tw > 95p
threshold_perc = 50

et_during_hw = np.full([lat.size, lon.size], np.nan)

lat_inds = np.arange(lat.size)
lon_inds = np.arange(lon.size)

N_gridcell = lat.size*lon.size

sacksMaizeNc = xr.open_dataset('%s/sacks/Maize.crop.calendar.fill.nc'%dirAgData)
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan
sacksEnd = sacksMaizeNc['harvest'].values
sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
sacksEnd[sacksEnd < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)

# regrid sacks data
regridMesh = xr.Dataset({'lat': (['lat'], lat),
                         'lon': (['lon'], lon),})

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)
regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)

sacksStart_regrid = regridder_start(sacksStart)
sacksEnd_regrid = regridder_end(sacksEnd)


n = 0

for xlat in lat_inds:

    for ylon in lon_inds:

        if n % 10000 == 0:
            print('%.1f%%'%(n/(lat.size*lon.size)*100))

        n += 1
        
        if np.isnan(sacksStart_regrid[xlat, ylon]):
            continue
        
        # get current grid cell cutoffs for tw and tx
        hw_deciles = era5_hw_deciles_values[:, xlat, ylon]
        et_deciles = era5_et_deciles_values[:, xlat, ylon]

        # find tw cutoff for current grid cell
        perc_thresh_ind = np.where(100*percentile_bins == threshold_perc)[0]
        perc_thresh_hw_value = hw_deciles[perc_thresh_ind]

        if ref_var == 'tx':
            cur_hw_values = era5_hw.mx2t.values[:, xlat, ylon].copy()
        elif ref_var == 'tw':
            cur_hw_values = era5_hw.tw.values[:, xlat, ylon].copy()
        
        cur_et_values = era5_et.e.values[:, xlat, ylon].copy()
        
        # convert all tx/tw values into percentiles
        cur_hw_p = np.full([cur_hw_values.size], np.nan)
        cur_et_p = np.full([cur_et_values.size], np.nan)
        
        for d in range(cur_hw_values.size):
            hw_val = cur_hw_values[d]
            et_val = cur_et_values[d]
            
            cur_p_ind = abs(hw_deciles-hw_val).argmin()
            cur_hw_p[d] = percentile_bins[cur_p_ind]
            
            cur_p_ind = abs(et_deciles-et_val).argmin()
            cur_et_p[d] = 1-percentile_bins[cur_p_ind]
            
            
        
        # find days when tw exceeds
        hw_exceed_ind = np.where((100*cur_hw_p >= threshold_perc))[0]
        # mean tx on those days
        et_during_hw[xlat, ylon] = np.nanmean(cur_et_p[hw_exceed_ind])
        
                
print('writing files...')
with open('%s/heat-wave-days/et-on-%s/era5_et_on_%s_%d_%d.dat'%(dirHeatData, ref_var, ref_var, threshold_perc, year), 'wb') as f:
    pickle.dump(et_during_hw, f)
    


