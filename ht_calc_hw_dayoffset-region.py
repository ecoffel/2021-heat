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
    
lat_inds = range(lat.size)
lon_inds = range(lon.size)
    

    
for wave_len in [3, 5, 7]:

    wave_min_len = wave_len
    wave_max_len = wave_len

    for q in [0, 1, 2]:
        print('len = %d / q = %d'%(wave_len, q))

        waveInds_tw = []
        waveVals_tw = []
        waveVals_tx_during_tw = []

        waveInds_tx = []
        waveVals_tx = []
        waveVals_tw_during_tx = []


        for xlat in lat_inds:
            for ylon in lon_inds:
                
                curInds_tw = wave_days_inds_tw_max[xlat, ylon, q]
                if len(curInds_tw) > 0:
                    waves_tw = findConsec(curInds_tw[0])
                    for wave in waves_tw:
                        if wave[1]-wave[0]+1 >= wave_min_len and wave[1]-wave[0]+1 <= wave_max_len:
                            waveVals_tw.append(wave_days_values_tw_max[xlat, ylon, q][0][curInds_tw[0][wave[0]:wave[1]+1]])
                            waveVals_tx_during_tw.append(wave_days_values_tx_max[xlat, ylon, q][0][curInds_tw[0][wave[0]:wave[1]+1]])

                curInds_tx = wave_days_inds_tx_max[xlat, ylon, q]
                if len(curInds_tx) > 0:
                    waves_tx = findConsec(curInds_tx[0])
                    for wave in waves_tx:
                        if wave[1]-wave[0]+1 >= wave_min_len and wave[1]-wave[0]+1 <= wave_max_len:
                            waveVals_tx.append(wave_days_values_tx_max[xlat, ylon, q][0][curInds_tx[0][wave[0]:wave[1]+1]])
                            waveVals_tw_during_tx.append(wave_days_values_tw_max[xlat, ylon, q][0][curInds_tx[0][wave[0]:wave[1]+1]])

        waveVals_tw = np.array([np.array(x) for x in waveVals_tw])
        waveVals_tx_during_tw = np.array([np.array(x) for x in waveVals_tx_during_tw])
        waveVals_tx = np.array([np.array(x) for x in waveVals_tx])
        waveVals_tw_during_tx = np.array([np.array(x) for x in waveVals_tw_during_tx])


        dayoffset_tw = np.full([waveVals_tw.shape[0], wave_max_len], np.nan)
        dayoffset_tx_during_tw = np.full([waveVals_tw.shape[0], wave_max_len], np.nan)
        dayoffset_tx = np.full([waveVals_tx.shape[0], wave_max_len], np.nan)
        dayoffset_tw_during_tx = np.full([waveVals_tx.shape[0], wave_max_len], np.nan)

        lens_tw = []
        for w in range(waveVals_tw.shape[0]):
            waveVals_tw[w] = np.array([x-waveVals_tw[w][0] for x in waveVals_tw[w]])
            waveVals_tx_during_tw[w] = np.array([x-waveVals_tx_during_tw[w][0] for x in waveVals_tx_during_tw[w]])
            lens_tw.append(len(waveVals_tw[w]))
            for i in range(dayoffset_tw.shape[1]):
                if len(waveVals_tw[w]) > i:
                    dayoffset_tw[w, i] = waveVals_tw[w][i]
                    dayoffset_tx_during_tw[w, i] = waveVals_tx_during_tw[w][i]

        lens_tx = []
        for w in range(waveVals_tx.shape[0]):
            waveVals_tx[w] = np.array([x-waveVals_tx[w][0] for x in waveVals_tx[w]])
            waveVals_tw_during_tx[w] = np.array([x-waveVals_tw_during_tx[w][0] for x in waveVals_tw_during_tx[w]])
            lens_tx.append(len(waveVals_tx[w]))
            for i in range(dayoffset_tx.shape[1]):
                if len(waveVals_tx[w]) > i:
                    dayoffset_tx[w, i] = waveVals_tx[w][i]
                    dayoffset_tw_during_tx[w, i] = waveVals_tw_during_tx[w][i]

        dayoffset_tx = np.array(dayoffset_tx)
        dayoffset_tw_during_tx = np.array(dayoffset_tw_during_tx)

        dayoffset_tw = np.array(dayoffset_tw)
        dayoffset_tx_during_tw = np.array(dayoffset_tx_during_tw)


        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tx_%s_%d_%d_%d'%(dirHeatData, region, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tx, f)
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tw_during_tx_%s_%d_%d_%d'%(dirHeatData, region, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tw_during_tx, f)
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tw_%s_%d_%d_%d'%(dirHeatData, region, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tw, f)
        with open('%s/heat-wave-days/hw-dayoffset/dayoffset_tx_during_tw_%s_%d_%d_%d'%(dirHeatData, region, wave_len, q, year), 'wb') as f:
            pickle.dump(dayoffset_tx_during_tw, f)

