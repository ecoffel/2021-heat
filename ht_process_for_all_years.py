
import sys, os, time

for y, year in enumerate(range(1981,2021+1)):
    
    print('running %d'%year)
    os.system('screen -d -m ipython ht_extract_era5_grow_txx_tmean_vector.py %d'%(year))
    time.sleep(240)

    