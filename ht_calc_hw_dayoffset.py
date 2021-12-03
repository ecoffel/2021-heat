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

era5_max_quantiles = xr.open_dataset('%s/era5_tw_max_quantiles.nc'%dirHeatData)
lat = era5_max_quantiles.latitude.values
lon = era5_max_quantiles.longitude.values

year = int(sys.argv[1])

# region = 'eu'

wave_days_tw_max = np.full([lat.size, lon.size, 3], np.nan)
wave_days_inds_tw_max = np.nan
wave_days_values_tw_max = np.nan

wave_days_tx_max = np.full([lat.size, lon.size, 3], np.nan)
wave_days_inds_tx_max = np.nan
wave_days_values_tx_max = np.nan

print(year)

with open('%s/heat-wave-days/full-year/era5_mx2t_heat_wave_days_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    cur_tx_heatwave_days = pickle.load(f)
    wave_days_tx_max = cur_tx_heatwave_days
with open('%s/heat-wave-days/full-year/era5_mx2t_heat_wave_days_ind_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    wave_days_inds_tx_max = pickle.load(f)
with open('%s/heat-wave-days/full-year/era5_mx2t_heat_wave_days_all_values_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    cur_tx_heatwave_days_values = pickle.load(f)
    wave_days_values_tx_max = np.array(cur_tx_heatwave_days_values, dtype=object)

with open('%s/heat-wave-days/full-year/era5_tw_heat_wave_days_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    cur_tw_heatwave_days = pickle.load(f)
    wave_days_tw_max = cur_tw_heatwave_days
with open('%s/heat-wave-days/full-year/era5_tw_heat_wave_days_ind_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    wave_days_inds_tw_max = pickle.load(f)
with open('%s/heat-wave-days/full-year/era5_tw_heat_wave_days_all_values_full_year_%d.dat'%(dirHeatData, year), 'rb') as f:
    cur_tw_heatwave_days_values = pickle.load(f)
    wave_days_values_tw_max = np.array(cur_tw_heatwave_days_values, dtype=object)



# if region == 'eu':
#     lat_inds = np.where((lat < 55) & (lat >= 30))[0]
#     lon_inds = np.where((lon < 60) & (lon >= 0))[0]
# elif region == 'nh-midlat':
#     lat_inds = np.where((lat < 55) & (lat >= 30))[0]
#     lon_inds = np.where((lon <= 360) & (lon >= 0))[0]
# elif region == 'tropics':
#     lat_inds = np.where((lat < 15) & (lat >= -15))[0]
#     lon_inds = np.where((lon <= 360) & (lon >= 0))[0]
    
lat_inds = np.arange(lat.size)
lon_inds = np.arange(lon.size)
    
for wave_len in [3, 5, 7]:

    wave_min_len = wave_len
    wave_max_len = wave_len

    for q in [0, 1, 2]:
        print('len = %d / q = %d'%(wave_len, q))

        waveInds_tw = {}
        waveVals_tw = {}
        waveVals_tx_during_tw = {}

        waveInds_tx = {}
        waveVals_tx = {}
        waveVals_tw_during_tx = {}

        dayoffset_tw = {} 
        dayoffset_tx_during_tw = {} 
        dayoffset_tx = {} 
        dayoffset_tw_during_tx = {} 

        for xlat in lat_inds:
            waveInds_tw[xlat] = {}
            waveVals_tw[xlat] = {}
            waveVals_tx_during_tw[xlat] = {}

            waveInds_tx[xlat] = {}
            waveVals_tx[xlat] = {}
            waveVals_tw_during_tx[xlat] = {}
            
            dayoffset_tw[xlat] = {} 
            dayoffset_tx_during_tw[xlat] = {} 
            dayoffset_tx[xlat] = {} 
            dayoffset_tw_during_tx[xlat] = {} 

            for ylon in lon_inds:
                
                waveInds_tw[xlat][ylon] = []
                waveVals_tw[xlat][ylon] = []
                waveVals_tx_during_tw[xlat][ylon] = []

                waveInds_tx[xlat][ylon] = []
                waveVals_tx[xlat][ylon] = []
                waveVals_tw_during_tx[xlat][ylon] = []
                
                curInds_tw = wave_days_inds_tw_max[xlat, ylon, q]
                if len(curInds_tw) > 0:
                    waves_tw = findConsec(curInds_tw[0])
                    for wave in waves_tw:
                        if wave[1]-wave[0]+1 >= wave_min_len and wave[1]-wave[0]+1 <= wave_max_len:
                            waveVals_tw[xlat][ylon].append(wave_days_values_tw_max[xlat, ylon, q][0][curInds_tw[0][wave[0]:wave[1]+1]])
                            waveVals_tx_during_tw[xlat][ylon].append(wave_days_values_tx_max[xlat, ylon, q][0][curInds_tw[0][wave[0]:wave[1]+1]])

                curInds_tx = wave_days_inds_tx_max[xlat, ylon, q]
                if len(curInds_tx) > 0:
                    waves_tx = findConsec(curInds_tx[0])
                    for wave in waves_tx:
                        if wave[1]-wave[0]+1 >= wave_min_len and wave[1]-wave[0]+1 <= wave_max_len:
                            waveVals_tx[xlat][ylon].append(wave_days_values_tx_max[xlat, ylon, q][0][curInds_tx[0][wave[0]:wave[1]+1]])
                            waveVals_tw_during_tx[xlat][ylon].append(wave_days_values_tw_max[xlat, ylon, q][0][curInds_tx[0][wave[0]:wave[1]+1]])

                
                waveVals_tw[xlat][ylon] = np.array([np.array(x) for x in waveVals_tw[xlat][ylon]])
                waveVals_tx_during_tw[xlat][ylon] = np.array([np.array(x) for x in waveVals_tx_during_tw[xlat][ylon]])
                waveVals_tx[xlat][ylon] = np.array([np.array(x) for x in waveVals_tx[xlat][ylon]])
                waveVals_tw_during_tx[xlat][ylon] = np.array([np.array(x) for x in waveVals_tw_during_tx[xlat][ylon]])


                dayoffset_tw[xlat][ylon] = np.full([waveVals_tw[xlat][ylon].shape[0], wave_max_len], np.nan)
                dayoffset_tx_during_tw[xlat][ylon] = np.full([waveVals_tw[xlat][ylon].shape[0], wave_max_len], np.nan)
                dayoffset_tx[xlat][ylon] = np.full([waveVals_tx[xlat][ylon].shape[0], wave_max_len], np.nan)
                dayoffset_tw_during_tx[xlat][ylon] = np.full([waveVals_tx[xlat][ylon].shape[0], wave_max_len], np.nan)

                for w in range(waveVals_tw[xlat][ylon].shape[0]):
                    waveVals_tw[xlat][ylon][w] = np.array([x-waveVals_tw[xlat][ylon][w][0] for x in waveVals_tw[xlat][ylon][w]])
                    waveVals_tx_during_tw[xlat][ylon][w] = np.array([x-waveVals_tx_during_tw[xlat][ylon][w][0] for x in waveVals_tx_during_tw[xlat][ylon][w]])
                    for i in range(dayoffset_tw[xlat][ylon].shape[1]):
                        if len(waveVals_tw[xlat][ylon][w]) > i:
                            dayoffset_tw[xlat][ylon][w, i] = waveVals_tw[xlat][ylon][w][i]
                            dayoffset_tx_during_tw[xlat][ylon][w, i] = waveVals_tx_during_tw[xlat][ylon][w][i]

                for w in range(waveVals_tx[xlat][ylon].shape[0]):
                    waveVals_tx[xlat][ylon][w] = np.array([x-waveVals_tx[xlat][ylon][w][0] for x in waveVals_tx[xlat][ylon][w]])
                    waveVals_tw_during_tx[xlat][ylon][w] = np.array([x-waveVals_tw_during_tx[xlat][ylon][w][0] for x in waveVals_tw_during_tx[xlat][ylon][w]])
                    for i in range(dayoffset_tx[xlat][ylon].shape[1]):
                        if len(waveVals_tx[xlat][ylon][w]) > i:
                            dayoffset_tx[xlat][ylon][w, i] = waveVals_tx[xlat][ylon][w][i]
                            dayoffset_tw_during_tx[xlat][ylon][w, i] = waveVals_tw_during_tx[xlat][ylon][w][i]

                dayoffset_tx[xlat][ylon] = np.array(dayoffset_tx[xlat][ylon])
                dayoffset_tw_during_tx[xlat][ylon] = np.array(dayoffset_tw_during_tx[xlat][ylon])

                dayoffset_tw[xlat][ylon] = np.array(dayoffset_tw[xlat][ylon])
                dayoffset_tx_during_tw[xlat][ylon] = np.array(dayoffset_tx_during_tw[xlat][ylon])

                
        print('writing tx...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tx_global_%d_%d_%d'%(dirHeatData, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tx, f)
        print('writing tw during tx...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tw_during_tx_global_%d_%d_%d'%(dirHeatData, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tw_during_tx, f)
        print('writing tw...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tw_global_%d_%d_%d'%(dirHeatData, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tw, f)
        print('writing tx during tw...')
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tx_during_tw_global_%d_%d_%d'%(dirHeatData, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tx_during_tw, f)

