#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:15:38 2019

@author: Sux
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# load info
data0 = pd.read_csv(r'/Users/Sux/Google Drive/Regular_Season_Batter.csv')
proj = pd.read_csv(r'/Users/Sux/Google Drive/submission.csv').set_index('batter_id')

# target year?
tgt_yr = 2016
# training data
data = data0[data0.year >= 2000][data0.year <= tgt_yr]
# define age
data['age'] = data.year - data.year_born.str[:4].astype(int)


#%% previous OPS

th_AB = 100
coef = pd.DataFrame(index=range(2000, tgt_yr-1), columns=[0, 1, 2])

for yr in range(2000, tgt_yr-3):
    # recent 4yr data
    s0 = data[data.AB >= th_AB][data.year==yr].set_index('batter_id')['OPS']
    s1 = data[data.AB >= th_AB][data.year==yr+1].set_index('batter_id')['OPS']
    s2 = data[data.AB >= th_AB][data.year==yr+2].set_index('batter_id')['OPS']
    s3 = data[data.AB >= th_AB][data.year==yr+3].set_index('batter_id')['OPS']
    # sample data
    sample = pd.concat([s0,s1,s2,s3], axis=1).dropna()
    sample.columns = [['OPS_0','OPS_1','OPS_2','OPS_3']]
    # data to regress
    x = sample[['OPS_0', 'OPS_1','OPS_2']]
    y = sample[['OPS_3']]
    # linear regression
    model = sm.OLS(y, x).fit()  
    coef.loc[yr+3, 0] = model.params[0]
    coef.loc[yr+3, 1] = model.params[1]
    coef.loc[yr+3, 2] = model.params[2]
    
# weights for each year
w1 = coef[0].mean()
w2 = coef[1].mean()
w3 = coef[2].mean()


#%% y2y correlation

reg_PA = 10
corr = pd.Series(index=range(2000, tgt_yr-1))
PAs = pd.Series(index=range(2000, tgt_yr-1))

for yr in range(2000, tgt_yr-1):
    # recent data
    s0 = data[data.AB >= reg_PA][data.year==yr].set_index('batter_id')[['AB','OPS']]
    s1 = data[data.AB >= reg_PA][data.year==yr+1].set_index('batter_id')[['AB','OPS']]
    # sample data
    sample = pd.concat([s0,s1], axis=1).dropna()
    sample.columns = [['PA_0','OPS_0','PA_1','OPS_1']]
    # data to regress
    corr.loc[yr] = sample.corr().loc['OPS_0','OPS_1'].values[0]
    PAs.loc[yr] = min(sample.mean().loc['PA_0'].values[0], sample.mean().loc['PA_1'].values[0])

# how many PAs needed to get high corr
corr_avg = corr.mean()


#%% aging pattern

th_AB = 20
diff = pd.Series(index=range(20, 39))

for ag in range(20, 39):
    # target batters
    data0 = data[data.AB >= th_AB].set_index('batter_id')
    batters = data0[data0.age==ag].index.intersection(data0[data0.age==ag+1].index)
    # sample data
    s0 = data0[data0.age==ag].loc[batters]
    s1 = data0[data0.age==ag+1].loc[batters]
    # OPS changes
    diff.loc[ag] = (s1.OPS - s0.OPS).mean()

# moving average    
diff_avg = diff.rolling(3, center=True).mean().dropna()
# aging parameters
age_c = diff_avg.cumsum().rolling(5, center=True).mean().idxmax()
age_r = (diff_avg.iloc[-1] - diff_avg.iloc[0])/(diff_avg.index[-1] - diff_avg.index[0])

#%% league average

data['AB_OPS'] = data.AB * data.OPS
OPS_avg = data.groupby('year').sum().AB_OPS/data.groupby('year').sum().AB
OPS_avg = OPS_avg.ewm(alpha=w3).mean().iloc[-1]


#%% projection

data['OPS'] = data.OPS.fillna(0)
proj['age'], proj['wPA'], proj['wOPS'], proj['r'] = np.nan, np.nan, np.nan, np.nan
proj['OPS_predict'] = np.nan

for player in proj.index:
    # plyed 3 yrs ago?
    if player in data[data.year==tgt_yr-3].batter_id.values:
        age = (data[data.year==tgt_yr-3][data.batter_id==player].age + 3).values[0]
        AB1 = (data[data.year==tgt_yr-3][data.batter_id==player].AB).values[0]
        OPS1 = (data[data.year==tgt_yr-3][data.batter_id==player].OPS).values[0]
    else:
        AB1, OPS1 = 0, 0
     # plyed 2 yrs ago?    
    if player in data[data.year==tgt_yr-2].batter_id.values:
        age = (data[data.year==tgt_yr-2][data.batter_id==player].age + 2).values[0]
        AB2 = (data[data.year==tgt_yr-2][data.batter_id==player].AB).values[0]
        OPS2 = (data[data.year==tgt_yr-2][data.batter_id==player].OPS).values[0]
    else:
        AB2, OPS2 = 0, 0
     # plyed a year ago?
    if player in data[data.year==tgt_yr-1].batter_id.values:
        age = (data[data.year==tgt_yr-1][data.batter_id==player].age + 1).values[0]
        AB3 = (data[data.year==tgt_yr-1][data.batter_id==player].AB).values[0]
        OPS3 = (data[data.year==tgt_yr-1][data.batter_id==player].OPS).values[0]
    else:
        AB3, OPS3 = 0, 0    
    
    # not plyed in recent years?
    if not (player in data[data.year==tgt_yr-3].batter_id.values or player in data[data.year==tgt_yr-2].batter_id.values or player in data[data.year==tgt_yr-1].batter_id.values):
        proj['r'].loc[player] = 0
        proj['OPS_predict'].loc[player] = OPS_avg
        continue
    
    # weighted average of OPS
    proj['age'].loc[player] = age
    proj['wPA'].loc[player] = w1*AB1 + w2*AB2 + w3*AB3
    proj['wOPS'].loc[player] = (w1*OPS1*AB1 + w2*OPS2*AB2 + w3*OPS3*AB3)/proj.wPA.loc[player]
    
#%% regression to the mean
   
reg_PA = 10
proj['r'] = proj.wPA/(proj.wPA + reg_PA)
proj['rOPS'] = proj.wOPS * proj.r + OPS_avg * (1 - proj.r)
proj['OPS_predict'] = proj.rOPS + age_r * (proj.age - age_c)
    

#%% comparison with actual results
    
th_AB = 10
th_r = 0

proj['OPS_actual'] = data0[data0.year==tgt_yr].OPS
proj['PA'] = data0[data0.year==tgt_yr].AB
result = proj[proj.PA >= th_AB][proj.r >= th_r]

plt.figure()
result.plot.scatter('OPS_predict','OPS_actual', grid=True)
print(result.OPS_predict.corr(result.OPS_actual))
print(((result.OPS_predict - result.OPS_actual)**2).mean()**0.5)
