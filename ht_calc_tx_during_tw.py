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

print('loading percentile data')
era5_tw_deciles = xr.open_dataset('%s/era5_tw_max_deciles.nc'%dirHeatData)
era5_tw_deciles.load()
era5_tw_deciles_values = era5_tw_deciles.tw.values.copy()

era5_tx_deciles = xr.open_dataset('%s/era5_tasmax_deciles.nc'%dirHeatData)
era5_tx_deciles.load()
era5_tx_deciles['mx2t'] -= 273.15
era5_tx_deciles_values = era5_tx_deciles.mx2t.values.copy()

percentile_bins = era5_tw_deciles['quantile'].values

lat = era5_tw_deciles.latitude.values
lon = era5_tw_deciles.longitude.values

year = int(sys.argv[1])

# find tx when tw > 95p
threshold_perc = 95

tx_during_tw = np.full([lat.size, lon.size], np.nan)
tw_during_tx = np.full([lat.size, lon.size], np.nan)

print(year)

print('loading tx')
era5_tx = xr.open_dataset('%s/daily/tasmax_%d.nc'%(dirEra5, year))
era5_tx.load()
era5_tx['mx2t'] -= 273.15

print('loading tw')
era5_tw = xr.open_dataset('%s/daily/tw_max_%d.nc'%(dirEra5, year))
era5_tw.load()
    
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
        tw_deciles = era5_tw_deciles_values[:, xlat, ylon]
        tx_deciles = era5_tx_deciles_values[:, xlat, ylon]

        # find tw cutoff for current grid cell
        perc_thresh_ind = np.where(100*percentile_bins == threshold_perc)[0]
        perc_thresh_tw_value = tw_deciles[perc_thresh_ind]
        perc_thresh_tx_value = tx_deciles[perc_thresh_ind]

        cur_tx_values = era5_tx.mx2t.values[:, xlat, ylon].copy()
        cur_tw_values = era5_tw.tw.values[:, xlat, ylon].copy()
        
        # convert all tx/tw values into percentiles
        cur_tx_p = np.full([cur_tx_values.size], np.nan)
        cur_tw_p = np.full([cur_tw_values.size], np.nan)
        
        for d in range(cur_tx_values.size):
            tx_val = cur_tx_values[d]
            tw_val = cur_tw_values[d]
            
            cur_p_ind = abs(tx_deciles-tx_val).argmin()
            cur_tx_p[d] = percentile_bins[cur_p_ind]
            
            cur_p_ind = abs(tw_deciles-tw_val).argmin()
            cur_tw_p[d] = percentile_bins[cur_p_ind]
            
        
        # find days when tw exceeds
        tw_exceed_ind = np.where((100*cur_tw_p >= threshold_perc))[0]
        # mean tx on those days
        tx_during_tw[xlat, ylon] = np.nanmean(cur_tx_p[tw_exceed_ind])
        
        
        # find days when tx exceeds
        tx_exceed_ind = np.where((100*cur_tx_p >= threshold_perc))[0]
        tw_during_tx[xlat, ylon] = np.nanmean(cur_tw_p[tx_exceed_ind])
        
        
print('writing files...')
with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tw_%d.dat'%(dirHeatData, year), 'wb') as f:
    pickle.dump(tx_during_tw, f)
with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tx_%d.dat'%(dirHeatData, year), 'wb') as f:
    pickle.dump(tw_during_tx, f)
    


