
import sys, os, time

for y, year in enumerate(range(1981,2020+1)):
    
    print('running %d'%year)
    os.system('screen -d -m ipython ht_calc_sm_during_tx_tw.py %d'%(year))
    time.sleep(1000)

    