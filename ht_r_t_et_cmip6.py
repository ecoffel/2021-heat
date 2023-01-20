import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import xesmf as xe
import cartopy
import cartopy.crs as ccrs
import cartopy.util
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import glob
import sys, os
import datetime

dirCmip6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Data-edcoffel-F20/ERA5'
dirSacks = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'


cmip6_models = ['bcc-csm2-mr', 'bcc-esm1', 'canesm5', \
                'kace-1-0-g', 'ipsl-cm6a-lr', 'miroc6', \
                'mri-esm2-0', 'noresm2-lm']

region = 'global'
crop = 'Maize'
model = sys.argv[1]
var_soil = 'evspsblsoi'
var_canopy = 'evspsblveg'
var_tran = 'tran'

if region == 'global':
    latRange = [-90, 90]
    lonRange = [0, 360]
elif region == 'us':
    latRange = [20, 55]
    lonRange = [220, 300]

sacksMaizeNc = xr.open_dataset('%s/sacks/%s.crop.calendar.fill.nc'%(dirSacks, crop))
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan
sacksEnd = sacksMaizeNc['harvest'].values
sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
sacksEnd[sacksEnd < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)

newLat = np.arange(-90, 90, 1.5)
newLon = np.arange(0, 360, 1.5)

# regrid sacks data
regridMesh = xr.Dataset({'lat': (['lat'], newLat),
                         'lon': (['lon'], newLon),})

    
year_range = [1981,2000]
    
def in_time_range(y):
    return (y >= year_range[0]) & (y <= year_range[1])


print('opening %s...'%model)
cmip6_evapsoi_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/%s/%s_Lmon_*.nc'%(dirCmip6, model, var_soil, var_soil), concat_dim='time')
cmip6_evapveg_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/%s/%s_Lmon_*.nc'%(dirCmip6, model, var_canopy, var_canopy), concat_dim='time')
cmip6_tran_hist = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/%s/%s_Lmon_*.nc'%(dirCmip6, model, var_tran, var_tran), concat_dim='time')

print('opening temp')
cmip6_temp_hist = xr.open_mfdataset('%s/%s/%s/%s/%s/*_day_*.nc'%(dirCmip6, model, 'r1i1p1f1', 'historical', 'tasmax'), concat_dim='time')



print('selecting data for %s...'%model)
cmip6_evapsoi_hist = cmip6_evapsoi_hist.sel(time=in_time_range(cmip6_evapsoi_hist['time.year']))
cmip6_evapveg_hist = cmip6_evapveg_hist.sel(time=in_time_range(cmip6_evapveg_hist['time.year']))
cmip6_tran_hist = cmip6_tran_hist.sel(time=in_time_range(cmip6_tran_hist['time.year']))

print('selecting temp')
cmip6_temp_hist = cmip6_temp_hist.sel(time=in_time_range(cmip6_temp_hist['time.year']))

print('resampling temp')
cmip6_temp_hist = cmip6_temp_hist.resample(time='1M').mean()

print('loading')
cmip6_evapsoi_hist.load()
cmip6_evapveg_hist.load()
cmip6_tran_hist.load()
cmip6_temp_hist.load()
cmip6_temp_hist['tasmax'] -= 273.15


cmip6_ET_hist = cmip6_evapsoi_hist.evspsblsoi + cmip6_evapveg_hist.evspsblveg + cmip6_tran_hist.tran

# regrid sacks data to current model res
regridMesh_cur_model = xr.Dataset({'lat': (['lat'], cmip6_evapsoi_hist.lat),
                                   'lon': (['lon'], cmip6_evapsoi_hist.lon)})

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh_cur_model, 'bilinear', reuse_weights=True)
regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh_cur_model, 'bilinear', reuse_weights=True)

sacksStart_regrid = regridder_start(sacksStart)
sacksEnd_regrid = regridder_end(sacksEnd)

# convert sacks day to month
for xlat in range(sacksStart_regrid.shape[0]):
    for ylon in range(sacksStart_regrid.shape[1]):
        
        if not np.isnan(sacksStart_regrid[xlat, ylon]):
            curStart = datetime.datetime.strptime('2020%d'%(round(sacksStart_regrid[xlat, ylon])+1), '%Y%j').date().month
            sacksStart_regrid[xlat, ylon] = curStart-1
            
        if not np.isnan(sacksEnd_regrid[xlat, ylon]):
            curEnd = datetime.datetime.strptime('2020%d'%(round(sacksEnd_regrid[xlat, ylon])+1), '%Y%j').date().month
            sacksEnd_regrid[xlat, ylon] = curEnd-1



# count up all non-nan grid cells so we can estimate percent complete 
ngrid = 0
for xlat in range(cmip6_evapsoi_hist.lat.size):
    for ylon in range(cmip6_evapsoi_hist.lon.size):
        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
            ngrid += 1

            
            
yearly_groups = cmip6_evapsoi_hist.groupby('time.year').groups
yearly_grow_evap = np.full([2014-1981+1, cmip6_evapsoi_hist.lat.size, cmip6_evapsoi_hist.lon.size], np.nan)
yearly_grow_temp = np.full([2014-1981+1, cmip6_evapsoi_hist.lat.size, cmip6_evapsoi_hist.lon.size], np.nan)
cmip6_r_t_et = np.full([cmip6_evapsoi_hist.lat.size, cmip6_evapsoi_hist.lon.size], np.nan)
            
n = 0
for xlat in range(cmip6_evapsoi_hist.lat.size):
    
    for ylon in range(cmip6_evapsoi_hist.lon.size):

        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
            
            if n % 100 == 0:
                print('%.2f%%'%(n/ngrid*100))

            if sacksStart_regrid[xlat, ylon] > sacksEnd_regrid[xlat, ylon]:

                # start loop on 2nd year to allow for growing season that crosses jan 1
                for y,year in enumerate(np.array(list(yearly_groups.keys()))[1:]):

                    cur_evap1 = cmip6_ET_hist[np.array(yearly_groups[year-1])[int(sacksStart_regrid[xlat, ylon]):], xlat, ylon].values
                    cur_evap2 = cmip6_ET_hist[np.array(yearly_groups[year])[:int(sacksEnd_regrid[xlat, ylon])], xlat, ylon].values
                    
                    cur_evap = np.nanmean(np.concatenate([cur_evap1, cur_evap2]))

                    if not np.isnan(cur_evap):
                        yearly_grow_evap[y, xlat, ylon] = cur_evap*60*60*24
                        
                        
                    cur_temp1 = cmip6_temp_hist['tasmax'][np.array(yearly_groups[year-1])[int(sacksStart_regrid[xlat, ylon]):], xlat, ylon].values
                    cur_temp2 = cmip6_temp_hist['tasmax'][np.array(yearly_groups[year])[:int(sacksEnd_regrid[xlat, ylon])], xlat, ylon].values
                    
                    cur_temp = np.nanmean(np.concatenate([cur_temp1, cur_temp2]))

                    if not np.isnan(cur_temp):
                        yearly_grow_temp[y, xlat, ylon] = cur_temp
                        
                n += 1

            else:

                for y,year in enumerate(np.array(list(yearly_groups.keys()))):

                    cur_evap = np.nanmean(cmip6_ET_hist[np.array(yearly_groups[year])[int(sacksStart_regrid[xlat, ylon]):int(sacksEnd_regrid[xlat, ylon])], xlat, ylon])
                    
                    if not np.isnan(cur_evap):
                        yearly_grow_evap[y, xlat, ylon] = cur_evap*60*60*24
                        
                        
                    cur_temp = np.nanmean(cmip6_temp_hist['tasmax'][np.array(yearly_groups[year])[int(sacksStart_regrid[xlat, ylon]):int(sacksEnd_regrid[xlat, ylon])], xlat, ylon].values)
                    
                    if not np.isnan(cur_temp):
                        yearly_grow_temp[y, xlat, ylon] = cur_temp
                n += 1

# now calc corr
for xlat in range(cmip6_evapsoi_hist.lat.size):
    for ylon in range(cmip6_evapsoi_hist.lon.size):
        e = yearly_grow_evap[:, xlat, ylon]
        t = yearly_grow_temp[:, xlat, ylon]
        nn = np.where(~np.isnan(e) & ~np.isnan(t))[0]
        r_t_et = np.corrcoef(e[nn], t[nn])[0,1]
        cmip6_r_t_et[xlat, ylon] = r_t_et
                

# add cyclic point before regridding
lon_data_cyc = cartopy.util.add_cyclic_point(cmip6_evapsoi_hist.lon)
cmip6_r_t_et_data_cyc = cartopy.util.add_cyclic_point(cmip6_r_t_et)
            
da_grow_r_t_et = xr.DataArray(data   = cmip6_r_t_et_data_cyc, 
                      dims   = ['lat', 'lon'],
                      coords = {'lat':cmip6_evapsoi_hist.lat, 'lon':lon_data_cyc},
                      attrs  = {'units'     : 'Correlation'
                        })
ds_grow_r_t_et = xr.Dataset()
ds_grow_r_t_et['r_t_et'] = da_grow_r_t_et


regridder = xe.Regridder(ds_grow_r_t_et, regridMesh, 'bilinear', reuse_weights=True)
regridder.clean_weight_file()
ds_grow_r_t_et_regrid = regridder(ds_grow_r_t_et)
    
    
ds_grow_r_t_et_regrid = ds_grow_r_t_et_regrid.assign_coords({'model':model})


print('saving netcdf...')
ds_grow_r_t_et_regrid.to_netcdf('r_t_et/cmip6_r_t_et_grow_%s_%s_%s_%d_%d_fixed_sh.nc'%(crop, region, model, year_range[0], year_range[1]))
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    