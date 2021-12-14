import numpy as np
import numpy.matlib
from scipy import interpolate
import os, sys, pickle, gzip
import datetime
import xarray as xr
import pandas as pd
import itertools
import random

import warnings
warnings.filterwarnings('ignore')

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

def findConsec(data):
    # find longest consequtative sequence of years with yield data
    ptList = []
    ptMax = (-1, -1)
    ptCur = (-1, -1)
    for i in range(len(data)):
        # start sequence
        if ptCur[0] == -1:
            ptCur = (i, -1)
        #end sequence
        elif ((data[i]-data[i-1]) > 1 and ptCur[0] >= 0):
            ptCur = (ptCur[0], i-1)
            if ptCur[1]-ptCur[0] > ptMax[1]-ptMax[0] or ptMax == (-1, -1):
                ptMax = ptCur
            ptList.append(ptCur)
            ptCur = (i, -1)
        # reached end of sequence
        elif i >= len(data)-1 and ptCur[0] >= 0:
            ptCur = (ptCur[0], i)
            if ptCur[1]-ptCur[0] > ptMax[1]-ptMax[0] or ptMax == (-1, -1):
                ptMax = ptCur
            ptList.append(ptCur)
    return ptList

era5_et_deciles = xr.open_dataset('%s/era5_evaporation_deciles.nc'%dirHeatData)
era5_et_deciles.load()
era5_et_deciles_values = era5_et_deciles.e.values.copy()

percentile_bins = era5_et_deciles['quantile'].values

lat = era5_et_deciles.latitude.values
lon = era5_et_deciles.longitude.values

year = int(sys.argv[1])

wave_days_tw_max = np.full([lat.size, lon.size, 3], np.nan)
wave_days_inds_tw_max = np.nan
wave_days_values_tw_max = np.nan

wave_days_tx_max = np.full([lat.size, lon.size, 3], np.nan)
wave_days_inds_tx_max = np.nan
wave_days_values_tx_max = np.nan

print(year)

with open('%s/heat-wave-days/full-year/era5_mx2t_heat_wave_days_ind_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    wave_days_inds_tx_max = pickle.load(f)

with open('%s/heat-wave-days/full-year/era5_tw_heat_wave_days_ind_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    wave_days_inds_tw_max = pickle.load(f)

dsEt = xr.open_dataset('%s/daily/evaporation_%d.nc'%(dirEra5, year))
dsEt.load()
dsEt_values = dsEt.e.values.copy()
    
lat_inds = np.arange(lat.size)
lon_inds = np.arange(lon.size)

N_gridcell = lat.size*lon.size

for wave_len in [3]:

    wave_min_len = wave_len
    wave_max_len = 10

    for q in [1]:
        print('len = %d / q = %d'%(wave_len, q))

        waveVals_et_during_tw = {}
        wavePerc_et_during_tw = {}

        waveVals_et_during_tx = {}
        wavePerc_et_during_tx = {}

        dayoffset_et_during_tw = {} 
        dayoffset_et_during_tx = {} 
        
        dayoffset_perc_et_during_tw = {} 
        dayoffset_perc_et_during_tx = {} 

        n = 0
        
        for xlat in lat_inds:
            waveVals_et_during_tw[xlat] = {}
            wavePerc_et_during_tw[xlat] = {}

            waveVals_et_during_tx[xlat] = {}
            wavePerc_et_during_tx[xlat] = {}
            
            dayoffset_et_during_tw[xlat] = {} 
            dayoffset_et_during_tx[xlat] = {} 

            dayoffset_perc_et_during_tw[xlat] = {} 
            dayoffset_perc_et_during_tx[xlat] = {} 
            
            for ylon in lon_inds:
                
                waveVals_et_during_tw[xlat][ylon] = []
                wavePerc_et_during_tw[xlat][ylon] = []

                waveVals_et_during_tx[xlat][ylon] = []
                wavePerc_et_during_tx[xlat][ylon] = []
                
                # get current grid cell cutoffs for tw and tx
                et_deciles = era5_et_deciles_values[:, xlat, ylon]
                
                if n % 100000 == 0:
                    print('%.1f%%'%(n/(lat.size*lon.size)*100))
                
                n += 1
                
                curInds_tw = wave_days_inds_tw_max[xlat, ylon, q]
                
                if len(curInds_tw) > 0:
                    waves_tw = findConsec(curInds_tw[0])
                    for wave in waves_tw:
                        if wave[1]-wave[0]+1 >= wave_min_len and wave[1]-wave[0]+1 <= wave_max_len:
                            waveVals_et_during_tw[xlat][ylon].append(dsEt_values[curInds_tw[0][wave[0]:wave[1]+1], xlat, ylon])
                                                            
                            wave_et_during_tw_perc_vals = []
                                                        
                            for d, day_val in enumerate(dsEt_values[curInds_tw[0][wave[0]:wave[1]+1], xlat, ylon]):
                                dec_ind = np.where((abs(day_val-et_deciles) == np.nanmin(abs(day_val-et_deciles))))[0]
                                if dec_ind.size > 0:
                                    wave_et_during_tw_perc_vals.append(percentile_bins[dec_ind[0]])
                                    
                            wavePerc_et_during_tw[xlat][ylon].append(wave_et_during_tw_perc_vals)
                            
                            
                curInds_tx = wave_days_inds_tx_max[xlat, ylon, q]
                if len(curInds_tx) > 0:
                    waves_tx = findConsec(curInds_tx[0])
                    for wave in waves_tx:
                        if wave[1]-wave[0]+1 >= wave_min_len and wave[1]-wave[0]+1 <= wave_max_len:
                            waveVals_et_during_tx[xlat][ylon].append(dsEt_values[curInds_tx[0][wave[0]:wave[1]+1], xlat, ylon])
                            
                            wave_et_during_tx_perc_vals = []
                            
                            
                            for d, day_val in enumerate(dsEt_values[curInds_tx[0][wave[0]:wave[1]+1], xlat, ylon]):
                                dec_ind = np.where((abs(day_val-et_deciles) == np.nanmin(abs(day_val-et_deciles))))[0]
                                if dec_ind.size > 0:
                                    wave_et_during_tx_perc_vals.append(percentile_bins[dec_ind[0]])
                            wavePerc_et_during_tx[xlat][ylon].append(wave_et_during_tx_perc_vals)

                
                waveVals_et_during_tw[xlat][ylon] = np.array([np.array(x) for x in waveVals_et_during_tw[xlat][ylon]])
                wavePerc_et_during_tw[xlat][ylon] = np.array([np.array(x) for x in wavePerc_et_during_tw[xlat][ylon]])
                waveVals_et_during_tx[xlat][ylon] = np.array([np.array(x) for x in waveVals_et_during_tx[xlat][ylon]])
                wavePerc_et_during_tx[xlat][ylon] = np.array([np.array(x) for x in wavePerc_et_during_tx[xlat][ylon]])

                dayoffset_et_during_tw[xlat][ylon] = np.full([waveVals_et_during_tw[xlat][ylon].shape[0], wave_max_len], np.nan)
                dayoffset_et_during_tx[xlat][ylon] = np.full([waveVals_et_during_tx[xlat][ylon].shape[0], wave_max_len], np.nan)
                dayoffset_perc_et_during_tw[xlat][ylon] = np.full([waveVals_et_during_tw[xlat][ylon].shape[0], wave_max_len], np.nan)
                dayoffset_perc_et_during_tx[xlat][ylon] = np.full([waveVals_et_during_tx[xlat][ylon].shape[0], wave_max_len], np.nan)
                
                # loop over all heat waves
                for w in range(waveVals_et_during_tw[xlat][ylon].shape[0]):
                    
                    # subtract value for 1st day of heat wave to get departures from the 1st day val
                    waveVals_et_during_tw[xlat][ylon][w] = np.array([x-waveVals_et_during_tw[xlat][ylon][w][0] for x in waveVals_et_during_tw[xlat][ylon][w]])
                    
                    # loop over all days of current heat wave
                    for i in range(waveVals_et_during_tw[xlat][ylon][w].shape[0]):
                        
                        if len(waveVals_et_during_tw[xlat][ylon][w]) > i:
                            dayoffset_et_during_tw[xlat][ylon][w, i] = waveVals_et_during_tw[xlat][ylon][w][i]
                            dayoffset_perc_et_during_tw[xlat][ylon][w, i] = wavePerc_et_during_tw[xlat][ylon][w][i]

                for w in range(waveVals_et_during_tx[xlat][ylon].shape[0]):
                    waveVals_et_during_tx[xlat][ylon][w] = np.array([x-waveVals_et_during_tx[xlat][ylon][w][0] for x in waveVals_et_during_tx[xlat][ylon][w]])
                    
                    for i in range(dayoffset_et_during_tx[xlat][ylon][w].shape[0]):
                        if len(waveVals_et_during_tx[xlat][ylon][w]) > i:
                            dayoffset_et_during_tx[xlat][ylon][w, i] = waveVals_et_during_tx[xlat][ylon][w][i]
                            dayoffset_perc_et_during_tx[xlat][ylon][w, i] = wavePerc_et_during_tx[xlat][ylon][w][i]

                dayoffset_et_during_tx[xlat][ylon] = np.array(dayoffset_et_during_tx[xlat][ylon])
                dayoffset_perc_et_during_tx[xlat][ylon] = np.array(dayoffset_perc_et_during_tx[xlat][ylon])
                dayoffset_et_during_tw[xlat][ylon] = np.array(dayoffset_et_during_tw[xlat][ylon])
                dayoffset_perc_et_during_tw[xlat][ylon] = np.array(dayoffset_perc_et_during_tw[xlat][ylon])
        
        print('writing et during tx...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_et_during_tx_global_%d_%d_%d_%d'%(dirHeatData, wave_min_len, wave_max_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_et_during_tx, f)
        print('writing et during tw...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_et_during_tw_global_%d_%d_%d_%d'%(dirHeatData, wave_min_len, wave_max_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_et_during_tw, f)
            
        print('writing et during tx perc...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_perc_et_during_tx_global_%d_%d_%d_%d'%(dirHeatData, wave_min_len, wave_max_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_perc_et_during_tx, f)
        print('writing et during tw perc...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_perc_et_during_tw_global_%d_%d_%d_%d'%(dirHeatData, wave_min_len, wave_max_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_perc_et_during_tw, f)

