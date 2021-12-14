import matplotlib.pyplot as plt 
import numpy as np
import numpy.matlib
from scipy import interpolate
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import scipy
import os, sys, pickle, gzip
import datetime
import xarray as xr
import pandas as pd
import xesmf as xe
import random

import warnings
warnings.filterwarnings('ignore')

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

year = int(sys.argv[1])

era5_max_deciles = xr.open_dataset('%s/era5_tw_max_deciles.nc'%dirHeatData)
lat = era5_max_deciles.latitude.values.copy()
lon = era5_max_deciles.longitude.values.copy()

del era5_max_deciles

wave_max_len = 10

for q in [1]:
    for l in [3]:
    
        print('q=%d, l=%d'%(q,l))

        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_et_during_tx_global_%d_%d_%d_%d'%(dirHeatData, l, wave_max_len, q, year), 'rb') as f:
            dayoffset_et_during_tx = pickle.load(f)
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_et_during_tw_global_%d_%d_%d_%d'%(dirHeatData, l, wave_max_len, q, year), 'rb') as f:
            dayoffset_et_during_tw = pickle.load(f)

        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_perc_et_during_tx_global_%d_%d_%d_%d'%(dirHeatData, l, wave_max_len, q, year), 'rb') as f:
            dayoffset_perc_et_during_tx = pickle.load(f)
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_perc_et_during_tw_global_%d_%d_%d_%d'%(dirHeatData, l, wave_max_len, q, year), 'rb') as f:
            dayoffset_perc_et_during_tw = pickle.load(f)

        cur_hw_mean_et_during_tx = np.full([lat.size, lon.size], np.nan)
        cur_hw_mean_et_during_tw = np.full([lat.size, lon.size], np.nan)

        cur_hw_perc_et_during_tx = np.full([lat.size, lon.size], np.nan)
        cur_hw_perc_et_during_tw = np.full([lat.size, lon.size], np.nan)
        
        cur_all_hw_mean_et_during_tx = {}
        cur_all_hw_mean_et_during_tw = {}

        cur_all_hw_perc_et_during_tx = {}
        cur_all_hw_perc_et_during_tw = {}

        n = 0

        for xlat in range(lat.size):
            
            cur_all_hw_mean_et_during_tx[xlat] = {}
            cur_all_hw_mean_et_during_tw[xlat] = {}

            cur_all_hw_perc_et_during_tx[xlat] = {}
            cur_all_hw_perc_et_during_tw[xlat] = {}
            
            for ylon in range(lon.size):

                if n % 100000 == 0:
                    print('%.1f%%'%(n/(lat.size*lon.size)*100))

                n += 1

                if dayoffset_et_during_tx[xlat][ylon].size > 0:
                    cur_hw_mean_et_during_tx[xlat, ylon] = np.nanmean(np.nanmean(-dayoffset_et_during_tx[xlat][ylon][:,1:], axis=1))
                    cur_all_hw_mean_et_during_tx[xlat][ylon] = np.nanmean(-dayoffset_et_during_tx[xlat][ylon][:,1:], axis=1)

                if dayoffset_et_during_tw[xlat][ylon].size > 0:
                    cur_hw_mean_et_during_tw[xlat, ylon] = np.nanmean(np.nanmean(-dayoffset_et_during_tw[xlat][ylon][:,1:], axis=1))
                    cur_all_hw_mean_et_during_tw[xlat][ylon] = np.nanmean(-dayoffset_et_during_tw[xlat][ylon][:,1:], axis=1)

                # percentile thresholds
                if dayoffset_perc_et_during_tx[xlat][ylon].size > 0:
                    cur_hw_perc_et_during_tx[xlat, ylon] = np.nanmean(np.nanmean(1 - dayoffset_perc_et_during_tx[xlat][ylon][:,1:], axis=1))
                    cur_all_hw_perc_et_during_tx[xlat][ylon] = np.nanmean(1 - dayoffset_perc_et_during_tx[xlat][ylon][:,1:], axis=1)

                if dayoffset_perc_et_during_tw[xlat][ylon].size > 0:
                    cur_hw_perc_et_during_tw[xlat, ylon] = np.nanmean(np.nanmean(1 - dayoffset_perc_et_during_tw[xlat][ylon][:,1:], axis=1))
                    cur_all_hw_perc_et_during_tw[xlat][ylon] = np.nanmean(1 - dayoffset_perc_et_during_tw[xlat][ylon][:,1:], axis=1)

        hw_stats = {
                   'hw_mean_et_during_tx':cur_hw_mean_et_during_tx,
                    'all_hw_mean_et_during_tx':cur_all_hw_mean_et_during_tx,
                   'hw_perc_et_during_tx':cur_hw_perc_et_during_tx,
                    'all_hw_perc_et_during_tx':cur_all_hw_perc_et_during_tx,
                   'hw_mean_et_during_tw':cur_hw_mean_et_during_tw,
                    'all_hw_mean_et_during_tw':cur_all_hw_mean_et_during_tw,
                    'hw_perc_et_during_tw':cur_hw_perc_et_during_tw,
                   'all_hw_perc_et_during_tw':cur_all_hw_perc_et_during_tw}
        
        with open('%s/heat-wave-days/hw-dayoffset-stats/dayoffset_stats_wx_vars_global_%d_%d_%d_%d.dat'%(dirHeatData, l, wave_max_len, q, year), 'wb') as f:
            pickle.dump(hw_stats, f)
        