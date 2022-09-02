import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm
import cartopy
import cartopy.crs as ccrs
import cartopy.util
import glob
import sys, os, pickle
import datetime

dirCmip6 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-02/Data-02-edcoffel-F20/CMIP6'
dirERA5 = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Data-edcoffel-F20/ERA5'
dirSacks = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/ag-land-climate'
dirHeatData = '/home/edcoffel/drive/MAX-Filer/Research/Climate-01/Personal-F20/edcoffel-F20/data/projects/2021-heat'

cmip6_models = ['bcc-csm2-mr', 'bcc-esm1', 'canesm5', \
                'kace-1-0-g', 'ipsl-cm6a-lr', 'miroc6', \
                'mri-esm2-0', 'noresm2-lm']

model = sys.argv[1]

yearRange = np.arange(2014, 2014+1)

threshold_perc = 95

def in_time_range(y, year_target):
    return (y == year_target)

print('opening tw')
cmip6_tw = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tw/tw_*.nc'%(dirCmip6, model), concat_dim='time')

print('opening temp')
cmip6_tx = xr.open_mfdataset('%s/%s/r1i1p1f1/historical/tasmax/tasmax*day*.nc'%(dirCmip6, model), concat_dim='time')

print('loading percentile data')
cmip6_tw_deciles = xr.open_dataset('cmip6_quantiles/tw_quantiles_%s.nc'%(model))
cmip6_tw_deciles.load()
cmip6_tw_deciles_values = cmip6_tw_deciles.tw.values

cmip6_tx_deciles = xr.open_dataset('cmip6_quantiles/tasmax_quantiles_%s.nc'%(model))
cmip6_tx_deciles.load()
cmip6_tx_deciles_values = cmip6_tx_deciles.tasmax.values


newLat = np.arange(-90, 90, 1.5)
newLon = np.arange(0, 360, 1.5)

# regrid sacks data
regridMesh = xr.Dataset({'lat': (['lat'], newLat),
                         'lon': (['lon'], newLon),})


for year in yearRange:

    cmip6_tx_cur_year = cmip6_tx.tasmax.sel(time=in_time_range(cmip6_tx['time.year'], year))
    cmip6_tw_cur_year = cmip6_tw.tw.sel(time=in_time_range(cmip6_tw['time.year'], year))

    print('loading year %d'%year)
    cmip6_tx_cur_year.load()
    cmip6_tx_cur_year -= 273.15
    cmip6_tw_cur_year.load()
    
    percentile_bins = cmip6_tw_deciles['quantile'].values

    tx_during_tw = np.full([cmip6_tx.lat.size, cmip6_tx.lon.size], np.nan)
    tw_during_tx = np.full([cmip6_tx.lat.size, cmip6_tx.lon.size], np.nan)
    
    tw_during_tx_exceed = np.full([cmip6_tx.lat.size, cmip6_tx.lon.size], np.nan)

    lat_inds = np.arange(cmip6_tx.lat.size)
    lon_inds = np.arange(cmip6_tx.lon.size)

    N_gridcell = cmip6_tx.lat.size*cmip6_tx.lon.size

    n = 0

    for xlat in lat_inds:

        for ylon in lon_inds:

            if n % 10000 == 0:
                print('%.1f%%'%(n/(cmip6_tx.lat.size*cmip6_tx.lon.size)*100))

            n += 1

            if np.isnan(np.nanmean(cmip6_tw_cur_year.values[:, xlat, ylon])):
                continue

            # get current grid cell cutoffs for tw and tx
            tw_deciles = cmip6_tw_deciles_values[:, xlat, ylon]
            tx_deciles = cmip6_tx_deciles_values[:, xlat, ylon]

            # find tw cutoff for current grid cell
            perc_thresh_ind = np.where(100*percentile_bins == threshold_perc)[0]
            perc_thresh_tw_value = tw_deciles[perc_thresh_ind]
            perc_thresh_tx_value = tx_deciles[perc_thresh_ind]

            cur_tx_values = cmip6_tx_cur_year.values[:, xlat, ylon].copy()
            cur_tw_values = cmip6_tw_cur_year.values[:, xlat, ylon].copy()

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
            tw_during_tx_exceed[xlat, ylon] = tw_exceed_ind.size
            # mean tx on those days
            if tw_exceed_ind.size > 0:
                tx_during_tw[xlat, ylon] = np.nanmean(cur_tx_p[tw_exceed_ind])


            # find days when tx exceeds
            tx_exceed_ind = np.where((100*cur_tx_p >= threshold_perc))[0]
            if tx_exceed_ind.size > 0:
                tw_during_tx[xlat, ylon] = np.nanmean(cur_tw_p[tx_exceed_ind])

                
                
    dir_path_tx_on_tw = '%s/heat-wave-days/tx-on-tw/cmip6/%s/'%(dirHeatData, model)
    dir_path_tw_on_tx = '%s/heat-wave-days/tw-on-tx/cmip6/%s/'%(dirHeatData, model)
    
    if not os.path.isdir(dir_path_tx_on_tw):
        os.mkdir(dir_path_tx_on_tw)
        
    if not os.path.isdir(dir_path_tw_on_tx):
        os.mkdir(dir_path_tw_on_tx)
    
    print('regriding output')    

    # add cyclic point before regridding
    lon_data_cyc = cartopy.util.add_cyclic_point(cmip6_tx.lon)
    tx_during_tw_data_cyc = cartopy.util.add_cyclic_point(tx_during_tw)

    da_tx_during_tw_cyc = xr.DataArray(data   = tx_during_tw_data_cyc, 
                      dims   = ['lat', 'lon'],
                      coords = {'lat':cmip6_tx.lat, 'lon':lon_data_cyc},
                      attrs  = {'units'     : 'Percentile'
                        })

    ds_tx_during_tw_cyc = xr.Dataset()
    ds_tx_during_tw_cyc['tx_during_tw'] = da_tx_during_tw_cyc

    regridder = xe.Regridder(ds_tx_during_tw_cyc, regridMesh, 'bilinear', reuse_weights=True)
    regridder.clean_weight_file()
    ds_tx_during_tw_cyc_regrid = regridder(ds_tx_during_tw_cyc)
    
    
    ds_tx_during_tw_cyc_regrid = ds_tx_during_tw_cyc_regrid.assign_coords({'model':model})
    ds_tx_during_tw_cyc_regrid.to_netcdf('%s/cmip6_tx_on_tw_%d_%s.nc'%(dir_path_tx_on_tw, year, model))
    
    
    
    # add cyclic point before regridding
    lon_data_cyc = cartopy.util.add_cyclic_point(cmip6_tx.lon)
    tw_during_tx_data_cyc = cartopy.util.add_cyclic_point(tw_during_tx)

    da_tw_during_tx_cyc = xr.DataArray(data   = tw_during_tx_data_cyc, 
                      dims   = ['lat', 'lon'],
                      coords = {'lat':cmip6_tx.lat, 'lon':lon_data_cyc},
                      attrs  = {'units'     : 'Percentile'
                        })

    ds_tw_during_tx_cyc = xr.Dataset()
    ds_tw_during_tx_cyc['tw_during_tx'] = da_tw_during_tx_cyc

    regridder = xe.Regridder(ds_tw_during_tx_cyc, regridMesh, 'bilinear', reuse_weights=True)
    regridder.clean_weight_file()
    ds_tw_during_tx_cyc_regrid = regridder(ds_tw_during_tx_cyc)
    
    
    ds_tw_during_tx_cyc_regrid = ds_tw_during_tx_cyc_regrid.assign_coords({'model':model})
    ds_tw_during_tx_cyc_regrid.to_netcdf('%s/cmip6_tw_on_tx_%d_%s.nc'%(dir_path_tw_on_tx, year, model))
    
    
    
    
    
    
    
    
    
    
#     regridder_tx_on_tw = xe.Regridder(xr.DataArray(data=tx_during_tw, dims=['lat', 'lon'], coords={'lat':cmip6_tx_deciles.lat, 'lon':cmip6_tx_deciles.lon}), regridMesh, 'bilinear', reuse_weights=True)
#     regridder_tw_on_tx = xe.Regridder(xr.DataArray(data=tw_during_tx, dims=['lat', 'lon'], coords={'lat':cmip6_tw_deciles.lat, 'lon':cmip6_tw_deciles.lon}), regridMesh, 'bilinear', reuse_weights=True)

#     tx_during_tw_regrid = regridder_tx_on_tw(tx_during_tw)
#     tw_during_tx_regrid = regridder_tw_on_tx(tw_during_tx)
    
#     print('writing files...')
    
#     dir_path_tx_on_tw = '%s/heat-wave-days/tx-on-tw/cmip6/%s/'%(dirHeatData, model)
#     dir_path_tw_on_tx = '%s/heat-wave-days/tw-on-tx/cmip6/%s/'%(dirHeatData, model)
    
#     if not os.path.isdir(dir_path_tx_on_tw):
#         os.mkdir(dir_path_tx_on_tw)
        
#     if not os.path.isdir(dir_path_tw_on_tx):
#         os.mkdir(dir_path_tw_on_tx)
    
#     with open('%s/cmip6_tx_on_tw_%d_%s.dat'%(dir_path_tx_on_tw, year, model), 'wb') as f:
#         pickle.dump(tx_during_tw_regrid, f)
#     with open('%s/cmip6_tw_on_tx_%d_%s.dat'%(dir_path_tw_on_tx, year, model), 'wb') as f:
#         pickle.dump(tw_during_tx_regrid, f)


