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

year = int(sys.argv[1])

print(year)

print('loading tx')
era5_tx = xr.open_dataset('%s/daily/tasmax_%d.nc'%(dirEra5, year))
era5_tx['mx2t'] -= 273.15
era5_tx_weekly = era5_tx.mx2t.resample(time='M').max()

print('loading tw')
era5_tw = xr.open_dataset('%s/daily/tw_max_%d.nc'%(dirEra5, year))
era5_tw_weekly = era5_tw.tw.resample(time='M').max()
        
da_tx_weekly = xr.DataArray(data   = era5_tx_weekly, 
                      dims   = ['time', 'latitude', 'longitude'],
                      coords = {'time':era5_tx_weekly.time, 'latitude':era5_tx_weekly.latitude, 'longitude':era5_tx_weekly.longitude},
                      attrs  = {'units'     : 'C'
                        })
ds_tx_weekly = xr.Dataset()
ds_tx_weekly['mx2t'] = da_tx_weekly

da_tw_weekly = xr.DataArray(data   = era5_tw_weekly, 
                      dims   = ['time', 'latitude', 'longitude'],
                      coords = {'time':era5_tw_weekly.time, 'latitude':era5_tw_weekly.latitude, 'lon':era5_tw_weekly.longitude},
                      attrs  = {'units'     : 'C'
                        })
ds_tw_weekly = xr.Dataset()
ds_tw_weekly['tw'] = da_tw_weekly

print('saving netcdf...')
ds_tx_weekly.to_netcdf('weekly_tx_tw/tx_monthly_%d.nc'%(year))
ds_tw_weekly.to_netcdf('weekly_tx_tw/tw_monthly_%d.nc'%(year))