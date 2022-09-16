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


era5_max_deciles = xr.open_dataset('%s/era5_tw_max_deciles.nc'%dirHeatData)
lat = era5_max_deciles.latitude.values
lon = era5_max_deciles.longitude.values


yearRange = np.arange(1981, 2014+1)


tx_during_tw = np.full([len(yearRange), lat.size, lon.size], np.nan)
tw_during_tx = np.full([len(yearRange), lat.size, lon.size], np.nan)

tx_during_tx = np.full([len(yearRange), lat.size, lon.size], np.nan)
tw_during_tw = np.full([len(yearRange), lat.size, lon.size], np.nan)

tx_diff_from_tw_during_tw = np.full([len(yearRange), lat.size, lon.size], np.nan)
tw_diff_from_tx_during_tx = np.full([len(yearRange), lat.size, lon.size], np.nan)

et_during_tw = np.full([len(yearRange), lat.size, lon.size], np.nan)
et_during_tx = np.full([len(yearRange), lat.size, lon.size], np.nan)

huss_during_tw = np.full([len(yearRange), lat.size, lon.size], np.nan)
huss_during_tx = np.full([len(yearRange), lat.size, lon.size], np.nan)

tx_during_tw_trend = np.full([lat.size, lon.size], np.nan)
tx_during_tw_pval = np.full([lat.size, lon.size], np.nan)

tw_during_tx_trend = np.full([lat.size, lon.size], np.nan)
tw_during_tx_pval = np.full([lat.size, lon.size], np.nan)

tx_during_tx_trend = np.full([lat.size, lon.size], np.nan)
tx_during_tx_pval = np.full([lat.size, lon.size], np.nan)

tw_during_tw_trend = np.full([lat.size, lon.size], np.nan)
tw_during_tw_pval = np.full([lat.size, lon.size], np.nan)

tx_diff_from_tw_during_tw_trend = np.full([lat.size, lon.size], np.nan)
tx_diff_from_tw_during_tw_pval = np.full([lat.size, lon.size], np.nan)

tw_diff_from_tx_during_tx_trend = np.full([lat.size, lon.size], np.nan)
tw_diff_from_tx_during_tx_pval = np.full([lat.size, lon.size], np.nan)

et_during_tw_trend = np.full([lat.size, lon.size], np.nan)
et_during_tw_pval = np.full([lat.size, lon.size], np.nan)

et_during_tx_trend = np.full([lat.size, lon.size], np.nan)
et_during_tx_pval = np.full([lat.size, lon.size], np.nan)

huss_during_tw_trend = np.full([lat.size, lon.size], np.nan)
huss_during_tw_pval = np.full([lat.size, lon.size], np.nan)

huss_during_tx_trend = np.full([lat.size, lon.size], np.nan)
huss_during_tx_pval = np.full([lat.size, lon.size], np.nan)

# tx_during_tw_cmip6 = xr.Dataset()
# tw_during_tx_cmip6 = xr.Dataset()

for y_ind, y in enumerate(yearRange):
    with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tx_%d.dat'%(dirHeatData, y), 'rb') as f:
        tw_during_tx[y_ind, :, :] = pickle.load(f)
    with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tw_%d.dat'%(dirHeatData, y), 'rb') as f:
        tx_during_tw[y_ind, :, :] = pickle.load(f)
        
    with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tw_%d.dat'%(dirHeatData, y), 'rb') as f:
        tw_during_tw[y_ind, :, :] = pickle.load(f)
    with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tx_%d.dat'%(dirHeatData, y), 'rb') as f:
        tx_during_tx[y_ind, :, :] = pickle.load(f)
        
    with open('%s/heat-wave-days/tx-on-tw/era5_tx_diff_from_tw_on_tw_%d.dat'%(dirHeatData, y), 'rb') as f:
        tx_diff_from_tw_during_tw[y_ind, :, :] = pickle.load(f)
    with open('%s/heat-wave-days/tw-on-tx/era5_tw_diff_from_tx_on_tx_%d.dat'%(dirHeatData, y), 'rb') as f:
        tw_diff_from_tx_during_tx[y_ind, :, :] = pickle.load(f)
        
    with open('%s/heat-wave-days/et-on-tx/era5_et_on_tx_%d.dat'%(dirHeatData, y), 'rb') as f:
        et_during_tx[y_ind, :, :] = pickle.load(f)
    with open('%s/heat-wave-days/et-on-tw/era5_et_on_tw_%d.dat'%(dirHeatData, y), 'rb') as f:
        et_during_tw[y_ind, :, :] = pickle.load(f)
        
    with open('%s/heat-wave-days/huss-on-tx/era5_huss_on_tx_%d.dat'%(dirHeatData, y), 'rb') as f:
        huss_during_tx[y_ind, :, :] = pickle.load(f)
    with open('%s/heat-wave-days/huss-on-tw/era5_huss_on_tw_%d.dat'%(dirHeatData, y), 'rb') as f:
        huss_during_tw[y_ind, :, :] = pickle.load(f)
        
for xlat in np.arange(0, lat.size):
    print(xlat)
    for ylon in np.arange(0, lon.size):

        tx_during_tw_cur = tx_during_tw[:, xlat, ylon]
        tx_during_tw_cur_1d = tx_during_tw_cur.reshape([tx_during_tw_cur.size, 1])
        
        tw_during_tx_cur = tw_during_tx[:, xlat, ylon]
        tw_during_tx_cur_1d = tw_during_tx_cur.reshape([tw_during_tx_cur.size, 1])
        
        tx_during_tx_cur = tx_during_tx[:, xlat, ylon]
        tx_during_tx_cur_1d = tx_during_tx_cur.reshape([tx_during_tx_cur.size, 1])
        
        tw_during_tw_cur = tw_during_tw[:, xlat, ylon]
        tw_during_tw_cur_1d = tw_during_tw_cur.reshape([tw_during_tw_cur.size, 1])
        
        tx_diff_from_tw_during_tw_cur = tx_diff_from_tw_during_tw[:, xlat, ylon]
        tx_diff_from_tw_during_tw_cur_1d = tx_diff_from_tw_during_tw_cur.reshape([tx_diff_from_tw_during_tw_cur.size, 1])
        
        tw_diff_from_tx_during_tx_cur = tw_diff_from_tx_during_tx[:, xlat, ylon]
        tw_diff_from_tx_during_tx_cur_1d = tw_diff_from_tx_during_tx_cur.reshape([tw_diff_from_tx_during_tx_cur.size, 1])
        
        et_during_tw_cur = et_during_tw[:, xlat, ylon]
        et_during_tw_cur_1d = et_during_tw_cur.reshape([et_during_tw_cur.size, 1])
        
        et_during_tx_cur = et_during_tx[:, xlat, ylon]
        et_during_tx_cur_1d = et_during_tx_cur.reshape([et_during_tx_cur.size, 1])
        
        huss_during_tw_cur = huss_during_tw[:, xlat, ylon]
        huss_during_tw_cur_1d = huss_during_tw_cur.reshape([huss_during_tw_cur.size, 1])
        
        huss_during_tx_cur = huss_during_tx[:, xlat, ylon]
        huss_during_tx_cur_1d = huss_during_tx_cur.reshape([huss_during_tx_cur.size, 1])


        nn = np.where((~np.isnan(tx_during_tw_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(tx_during_tw_cur_1d[nn].size))
            mdl = sm.OLS(tx_during_tw_cur_1d[nn], X).fit()
            tw_during_tx_trend[xlat, ylon] = mdl.params[1]
            tw_during_tx_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            tw_during_tx_trend[xlat, ylon] = np.nan
            tw_during_tx_pval[xlat, ylon] = np.nan

        nn = np.where((~np.isnan(tw_during_tx_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(tw_during_tx_cur_1d[nn].size))
            mdl = sm.OLS(tw_during_tx_cur_1d[nn], X).fit()
            tx_during_tw_trend[xlat, ylon] = mdl.params[1]
            tx_during_tw_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            tx_during_tw_trend[xlat, ylon] = np.nan
            tx_during_tw_pval[xlat, ylon] = np.nan
            
            
            
            
            
        nn = np.where((~np.isnan(tx_during_tx_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(tx_during_tx_cur_1d[nn].size))
            mdl = sm.OLS(tx_during_tx_cur_1d[nn], X).fit()
            tx_during_tx_trend[xlat, ylon] = mdl.params[1]
            tx_during_tx_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            tx_during_tx_trend[xlat, ylon] = np.nan
            tx_during_tx_pval[xlat, ylon] = np.nan

        nn = np.where((~np.isnan(tw_during_tw_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(tw_during_tw_cur_1d[nn].size))
            mdl = sm.OLS(tw_during_tw_cur_1d[nn], X).fit()
            tw_during_tw_trend[xlat, ylon] = mdl.params[1]
            tw_during_tw_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            tw_during_tw_trend[xlat, ylon] = np.nan
            tw_during_tw_pval[xlat, ylon] = np.nan
            
            
            
            
        nn = np.where((~np.isnan(tx_diff_from_tw_during_tw_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(tx_diff_from_tw_during_tw_cur_1d[nn].size))
            mdl = sm.OLS(tx_diff_from_tw_during_tw_cur_1d[nn], X).fit()
            tx_diff_from_tw_during_tw_trend[xlat, ylon] = mdl.params[1]
            tx_diff_from_tw_during_tw_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            tx_diff_from_tw_during_tw_trend[xlat, ylon] = np.nan
            tx_diff_from_tw_during_tw_pval[xlat, ylon] = np.nan

        nn = np.where((~np.isnan(tw_diff_from_tx_during_tx_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(tw_diff_from_tx_during_tx_cur_1d[nn].size))
            mdl = sm.OLS(tw_diff_from_tx_during_tx_cur_1d[nn], X).fit()
            tw_diff_from_tx_during_tx_trend[xlat, ylon] = mdl.params[1]
            tw_diff_from_tx_during_tx_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            tw_diff_from_tx_during_tx_trend[xlat, ylon] = np.nan
            tw_diff_from_tx_during_tx_pval[xlat, ylon] = np.nan


            
            
        nn = np.where((~np.isnan(et_during_tx_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(et_during_tx_cur_1d[nn].size))
            mdl = sm.OLS(et_during_tx_cur_1d[nn], X).fit()
            et_during_tx_trend[xlat, ylon] = mdl.params[1]
            et_during_tx_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            et_during_tx_trend[xlat, ylon] = np.nan
            et_during_tx_pval[xlat, ylon] = np.nan


        nn = np.where((~np.isnan(et_during_tw_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(et_during_tw_cur_1d[nn].size))
            mdl = sm.OLS(et_during_tw_cur_1d[nn], X).fit()
            et_during_tw_trend[xlat, ylon] = mdl.params[1]
            et_during_tw_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            et_during_tw_trend[xlat, ylon] = np.nan
            et_during_tw_pval[xlat, ylon] = np.nan
            
            
            
        nn = np.where((~np.isnan(huss_during_tx_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(huss_during_tx_cur_1d[nn].size))
            mdl = sm.OLS(huss_during_tx_cur_1d[nn], X).fit()
            huss_during_tx_trend[xlat, ylon] = mdl.params[1]
            huss_during_tx_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            huss_during_tx_trend[xlat, ylon] = np.nan
            huss_during_tx_pval[xlat, ylon] = np.nan


        nn = np.where((~np.isnan(huss_during_tw_cur_1d)))[0]
        if nn.size > 10:
            X = sm.add_constant(np.arange(huss_during_tw_cur_1d[nn].size))
            mdl = sm.OLS(huss_during_tw_cur_1d[nn], X).fit()
            huss_during_tw_trend[xlat, ylon] = mdl.params[1]
            huss_during_tw_pval[xlat, ylon] = mdl.pvalues[1]
        else:
            huss_during_tw_trend[xlat, ylon] = np.nan
            huss_during_tw_pval[xlat, ylon] = np.nan
        

print('writing files...')
with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tw_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tx_during_tw_trend, f)
with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tw_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tx_during_tw_pval, f)
    
with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tx_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tw_during_tx_trend, f)
with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tx_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tw_during_tx_pval, f)
    
    
with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tx_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tx_during_tx_trend, f)
with open('%s/heat-wave-days/tx-on-tw/era5_tx_on_tx_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tx_during_tx_pval, f)
    
with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tw_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tw_during_tw_trend, f)
with open('%s/heat-wave-days/tw-on-tx/era5_tw_on_tw_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tw_during_tw_pval, f)
    
    
    
with open('%s/heat-wave-days/tx-on-tw/era5_tx_diff_from_tw_during_tw_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tx_diff_from_tw_during_tw_trend, f)
with open('%s/heat-wave-days/tx-on-tw/era5_tx_diff_from_tw_during_tw_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tx_diff_from_tw_during_tw_pval, f)
    
with open('%s/heat-wave-days/tw-on-tx/era5_tw_diff_from_tx_during_tx_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tw_diff_from_tx_during_tx_trend, f)
with open('%s/heat-wave-days/tw-on-tx/era5_tw_diff_from_tx_during_tx_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(tw_diff_from_tx_during_tx_pval, f)
    
    
with open('%s/heat-wave-days/et-on-tw/era5_et_on_tw_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(et_during_tw_trend, f)
with open('%s/heat-wave-days/et-on-tw/era5_et_on_tw_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(et_during_tw_pval, f)
    
with open('%s/heat-wave-days/et-on-tx/era5_et_on_tx_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(et_during_tx_trend, f)
with open('%s/heat-wave-days/et-on-tx/era5_et_on_tx_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(et_during_tx_pval, f)
    
    
with open('%s/heat-wave-days/huss-on-tw/era5_huss_on_tw_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(huss_during_tw_trend, f)
with open('%s/heat-wave-days/huss-on-tw/era5_huss_on_tw_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(huss_during_tw_pval, f)
    
with open('%s/heat-wave-days/huss-on-tx/era5_huss_on_tx_trend.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(huss_during_tx_trend, f)
with open('%s/heat-wave-days/huss-on-tx/era5_huss_on_tx_trend_pval.dat'%(dirHeatData), 'wb') as f:
    pickle.dump(huss_during_tx_pval, f)
    


