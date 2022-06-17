
import sys, os

for y, year in enumerate(range(1981,1991)):
    
    print('running %d'%year)
    os.system('screen -d -m ipython ht_calc_weekly_tx_tw.py %d'%(year))\

    