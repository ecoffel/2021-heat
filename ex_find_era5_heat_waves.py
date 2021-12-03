import rasterio as rio
import matplotlib.pyplot as plt 
import numpy as np
from scipy import interpolate
import statsmodels.api as sm
import scipy.stats as st
import os, sys, pickle, gzip
import datetime
import geopy.distance
import xarray as xr
import xesmf as xe
import cartopy.crs as ccrs
import glob

dirAgData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirHeat = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'
dirEra5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'

maxVar = 'mx2t'

year = int(sys.argv[1])

# if os.path.isfile('heat_wave_days/era5_heat_wave_days_%d.dat'%year):
#     sys.exit()

if maxVar == 'tw':
    era5_max_quantiles = xr.open_dataset('%s/era5_tw_max_quantiles.nc'%dirHeat)
    era5_max_quantiles.load()
elif maxVar == 'mx2t':
    era5_max_quantiles = xr.open_dataset('%s/era5_mx2t_quantiles.nc'%dirHeat)
    era5_max_quantiles.load()

lat = era5_max_quantiles.latitude.values
lon = era5_max_quantiles.longitude.values

regridMesh = xr.Dataset({'lat': (['lat'], lat),
                         'lon': (['lon'], lon),})


sacksMaizeNc = xr.open_dataset('%s/sacks/Maize.crop.calendar.fill.nc'%dirAgData)
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh, 'bilinear', reuse_weights=True)
sacksStart_regrid = regridder_start(sacksStart)

print('year %d'%year)

if maxVar == 'mx2t':
    dsMax = xr.open_dataset('%s/daily/tasmax_%d.nc'%(dirEra5, year))
    dsMax.load()

    dsMax['mx2t'] -= 273.15
elif maxVar == 'tw':
    dsMax = xr.open_dataset('%s/daily/tw_max_%d.nc'%(dirEra5, year))
    dsMax.load()

heatwave_days = np.full([lat.size, lon.size, 3], np.nan)
heatwave_days_ind = []#np.full([lat.size, lon.size, 3, dsMax[maxVar].values.shape[0]], np.nan)
heatwave_days_values = []
    
n = 0

for xlat in range(len(lat)):

    heatwave_days_ind.append([])
    heatwave_days_values.append([])
    
    for ylon in range(len(lon)):

        heatwave_days_ind[xlat].append([])
        heatwave_days_values[xlat].append([])
        
        if n % 30000 == 0:
            print('%.0f %% complete'%(100*n/((len(lat)*len(lon)))))


        if not np.isnan(sacksStart_regrid[xlat, ylon]):
            curTmax = dsMax[maxVar][:, xlat, ylon]

        # the quantiles have the first 3 less than 50th percentile - only look at heat waves, so > than 50th percentile
        # that is why range starts at 3 here
        for q in range(3, era5_max_quantiles[maxVar].shape[0]):
            
            heatwave_days_ind[xlat][ylon].append([])
            heatwave_days_values[xlat][ylon].append([])
            
            if not np.isnan(sacksStart_regrid[xlat, ylon]):
                cur_ind = np.where(curTmax.values > era5_max_quantiles[maxVar][q, xlat, ylon].values)[0]
                
                heatwave_days[xlat, ylon, q-3] = cur_ind.size
                heatwave_days_ind[xlat][ylon][q-3].append(cur_ind)
                heatwave_days_values[xlat][ylon][q-3].append(curTmax.values)
            
        n += 1
heatwave_days_ind = np.array(heatwave_days_ind, dtype=object)
heatwave_days_values = np.array(heatwave_days_values, dtype=object)
with open('%s/heat-wave-days/full-year/era5_%s_heat_wave_days_full_year_%d.dat'%(dirHeat, maxVar, year), 'wb') as f:
    pickle.dump(heatwave_days, f)
with open('%s/heat-wave-days/full-year/era5_%s_heat_wave_days_ind_full_year_%d.dat'%(dirHeat, maxVar, year), 'wb') as f:
    pickle.dump(heatwave_days_ind, f)
with open('%s/heat-wave-days/full-year/era5_%s_heat_wave_days_all_values_full_year_%d.dat'%(dirHeat, maxVar, year), 'wb') as f:
    pickle.dump(heatwave_days_values, f)
