import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import cartopy
import cartopy.crs as ccrs
import glob
import sys
import datetime

dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/ERA5'
dirSacks = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/obbudelm-F21/sacks'

file_var = 'tasmax'
orig_var = 'mx2t'
crop = 'Maize'

year = int(sys.argv[1])

latRange = [-90, 90]
lonRange = [0, 360]

sacksMaizeNc = xr.open_dataset('%s/sacks/%s.crop.calendar.fill.nc'%(dirSacks, crop))
sacksStart = sacksMaizeNc['plant'].values
sacksStart = np.roll(sacksStart, -int(sacksStart.shape[1]/2), axis=1)
sacksStart[sacksStart < 0] = np.nan
sacksEnd = sacksMaizeNc['harvest'].values
sacksEnd = np.roll(sacksEnd, -int(sacksEnd.shape[1]/2), axis=1)
sacksEnd[sacksEnd < 0] = np.nan

sacksLat = np.linspace(90, -90, 360)
sacksLon = np.linspace(0, 360, 720)

regridMesh_cur_model = xr.Dataset()

n = 0

print('opening era5 %d...'%year)
temp_era5 = xr.open_dataset('%s/daily/%s_%d.nc'%(dirERA5, file_var, year))
temp_era5.load()
temp_era5[orig_var] -= 273.15

print('opening era5 %d...'%(year-1))
temp_era5_last_year = xr.open_dataset('%s/daily/%s_%d.nc'%(dirERA5, file_var, year-1))
temp_era5_last_year.load()
temp_era5_last_year[orig_var] -= 273.15

temp_era5 = temp_era5.rename_dims(latitude='lat', longitude='lon')
temp_era5 = temp_era5.rename({'latitude':'lat', 'longitude':'lon'})

temp_era5_last_year = temp_era5_last_year.rename_dims(latitude='lat', longitude='lon')
temp_era5_last_year = temp_era5_last_year.rename({'latitude':'lat', 'longitude':'lon'})

# regrid sacks data to current model res
regridMesh_cur_model = xr.Dataset({'lat': (['lat'], temp_era5.lat),
                                   'lon': (['lon'], temp_era5.lon)})

regridder_start = xe.Regridder(xr.DataArray(data=sacksStart, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh_cur_model, 'bilinear', reuse_weights=True)
regridder_end = xe.Regridder(xr.DataArray(data=sacksEnd, dims=['lat', 'lon'], coords={'lat':sacksLat, 'lon':sacksLon}), regridMesh_cur_model, 'bilinear', reuse_weights=True)

sacksStart_regrid = regridder_start(sacksStart)
sacksEnd_regrid = regridder_end(sacksEnd)

# count up all non-nan grid cells so we can estimate percent complete
ngrid = 0
for xlat in range(temp_era5.lat.size):
    for ylon in range(temp_era5.lon.size):
        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
            ngrid += 1

# decile bins
bins = np.arange(0, 101, 10)
            
# this will hold the decile cutoffs for every grid cell for the current year
yearly_grow_tmax_deciles = np.full([temp_era5.lat.size, temp_era5.lon.size, bins.size], np.nan)

# loop over all latitudes
for xlat in range(temp_era5.lat.size):

    # loop over all longitudes
    for ylon in range(temp_era5.lon.size):

       # if crop calendar is not nan at this grid cell 
        if ~np.isnan(sacksStart_regrid[xlat, ylon]) and ~np.isnan(sacksEnd_regrid[xlat, ylon]):
    
            # print out progress through loop
            if n % 25000 == 0:
                print('%.2f%%'%(n/ngrid*100))

            # if in southern hemisphere
            if sacksStart_regrid[xlat, ylon] > sacksEnd_regrid[xlat, ylon]:

                # start loop on 2nd year to allow for growing season that crosses jan 1
                curTmax1 = temp_era5_last_year[orig_var][int(sacksEnd_regrid[xlat, ylon]):, xlat, ylon]
                curTmax2 = temp_era5[orig_var][:int(sacksStart_regrid[xlat, ylon]), xlat, ylon]

                curTmax = np.concatenate([curTmax1, curTmax2])

                if len(curTmax) > 0:
                    yearly_grow_tmax_deciles[xlat, ylon, :] = np.nanpercentile(curTmax, bins)
                n += 1
                
            # if in northern hemisphere
            else:
                curTmax = temp_era5[orig_var][int(sacksStart_regrid[xlat, ylon]):int(sacksEnd_regrid[xlat, ylon]), xlat, ylon]
                if len(curTmax) > 0:
                    yearly_grow_tmax_deciles[xlat, ylon] = np.nanpercentile(curTmax, bins)
                n += 1

print('renaming dims...')

da_grow_tmax_deciles = xr.DataArray(data   = yearly_grow_tmax_deciles, 
                      dims   = ['lat', 'lon', 'bin'],
                      coords = {'bin':bins, 'lat':temp_era5.lat, 'lon':temp_era5.lon},
                      attrs  = {'units'     : 'C'
                        })
ds_grow_tmax_deciles = xr.Dataset()
ds_grow_tmax_deciles['%s_grow_deciles'%file_var] = da_grow_tmax_deciles


print('saving netcdf...')
ds_grow_tmax_deciles.to_netcdf('decile_bins/era5_%s_%s_growing_season_deciles_%d.nc'%(crop, file_var, year))
