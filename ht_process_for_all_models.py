
import sys, os

cmip6_models = ['bcc-csm2-mr', 'bcc-esm1', 'canesm5', \
                'kace-1-0-g', 'ipsl-cm6a-lr', 'miroc6', \
                'mri-esm2-0', 'noresm2-lm']

for model in cmip6_models:
    print('running %s'%model)
    os.system('screen -d -m ipython ht_r_t_et_cmip6.py %s'%(model))

    