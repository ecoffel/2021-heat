
import sys, os, time

for d in range(260,721):
    print('running %d'%d)
    os.system('screen -d -m ipython ht_extract_era5_growing_season_huss_deciles.py %d'%(d))
    time.sleep(5)
    

    