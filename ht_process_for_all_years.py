
import sys, os, time

for y, year in enumerate(range(1988,2021+1)):
    
    print('running %d'%year)
    os.system('screen -d -m ipython ht_extract_era5_grow_txx_tmean.py %d'%(year))
    time.sleep(800)

    