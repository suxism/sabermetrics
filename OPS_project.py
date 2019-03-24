#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:17:28 2019
@author: Sux
"""

import pandas as pd

# load info
data = pd.read_csv(r'/Users/Sux/Google Drive/Regular_Season_Batter.csv')
proj = pd.read_csv(r'/Users/Sux/Google Drive/submission.csv', index_col='batter_id')

# target year
tgt_yr = 2019

# set age
data['age'] = data.year - data.year_born.str[:4].astype(int)

#%% projection

# get info
for n in [1, 2, 3]:
    proj[['PA_'+str(n),'OPS_'+str(n),'age_'+str(n)]] = data[data.year == tgt_yr - n][['batter_id','AB','OPS','age']].set_index('batter_id')
proj = proj.fillna(0)

# get age
tgt_age = data.groupby('batter_id').first()
proj['age'] = tgt_age.age + tgt_yr - tgt_age.year

# weighting
proj['wPA'] = 5 * proj['PA_1'] + 3 * proj['PA_2'] + 2 * proj['PA_3']
proj['wOPS'] = (5 * proj['PA_1'] * proj['OPS_1'] + 3 * proj['PA_2'] * proj['OPS_2'] + 2 * proj['PA_3'] * proj['OPS_3'])/proj['wPA']
proj['wOPS'] = proj['wOPS'].fillna(0.8)

# regression to the mean
proj['r'] = proj['wPA']/(proj['wPA'] + 1000)
proj['rOPS'] = proj['wOPS'] * proj['r'] + 0.8 * (1 - proj['r'])

# aging effect
proj['OPS_est'] = proj['rOPS'] - 0.004 * (proj['age'] - 29)
