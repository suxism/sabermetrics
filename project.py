

import pandas as pd
import numpy as np
import statsmodels.api as sm

# load info
data = pd.read_csv(r'C:\Users\USER\Downloads\Batter.csv')
proj = pd.read_csv(r'C:\Users\USER\Downloads\Submission.csv').set_index('batter_id')

# clean data
data.year.astype(int)
data = data[data.year >= 2000]
data['born'] = data.year_born.str[:4].astype(int)
data['age'] = data.year - data.born


#%% previous OPS

th_AB = 100
coef = pd.DataFrame(index=range(2000, 2019), columns=[0, 1, 2])

for yr in range(2000, 2016):

    # recent data
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


#%% regression to the mean

th_AB = 11
corr = pd.Series(index=range(2000, 2018))
PAs = pd.Series(index=range(2000, 2018))

for yr in range(2000, 2018):

    # recent data
    s0 = data[data.AB >= th_AB][data.year==yr].set_index('batter_id')[['AB','OPS']]
    s1 = data[data.AB >= th_AB][data.year==yr+1].set_index('batter_id')[['AB','OPS']]
    # sample data
    sample = pd.concat([s0,s1], axis=1).dropna()
    sample.columns = [['PA_0','OPS_0','PA_1','OPS_1']]
    # data to regress
    corr.loc[yr] = sample.corr().loc['OPS_0','OPS_1'].values[0]
    PAs.loc[yr] = min(sample.mean().loc['PA_0'].values[0], sample.mean().loc['PA_1'].values[0])

# how many PAs needed to get high corr    
corr_avg = corr.mean()
PA_avg = PAs.mean()


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


#%% projection

th_AB = 20
OPS_avg = data[data.AB >= th_AB].groupby('year').mean().OPS.ewm(alpha=0.1).mean().iloc[-1]
proj['OPS'] = np.nan

for player in proj.index:
    
    # plyed in 2016?
    if player in data[data.year==2016].batter_id.values:
        age = (data[data.year==2016][data.batter_id==player].age + 3).values[0]
        AB1 = (data[data.year==2016][data.batter_id==player].AB).values[0]
        OPS1 = (data[data.year==2016][data.batter_id==player].OPS).values[0]
    else:
        AB1, OPS1 = 0, 0
     # plyed in 2017?    
    if player in data[data.year==2017].batter_id.values:
        age = (data[data.year==2017][data.batter_id==player].age + 2).values[0]
        AB2 = (data[data.year==2017][data.batter_id==player].AB).values[0]
        OPS2 = (data[data.year==2017][data.batter_id==player].OPS).values[0]
    else:
        AB2, OPS2 = 0, 0
     # plyed in 2018?    
    if player in data[data.year==2018].batter_id.values:
        age = (data[data.year==2018][data.batter_id==player].age + 1).values[0]
        AB3 = (data[data.year==2018][data.batter_id==player].AB).values[0]
        OPS3 = (data[data.year==2018][data.batter_id==player].OPS).values[0]
    else:
        AB3, OPS3 = 0, 0    
    
    # not plyed in recent years?
    if not (player in data[data.year==2016].batter_id.values or player in data[data.year==2017].batter_id.values or player in data[data.year==2018].batter_id.values):
        proj['OPS'].loc[player] = OPS_avg
        continue
 
    # weighted average of OPS  
    wPA = w1*AB1 + w2*AB2 + w3*AB3
    wOPS = (w1*OPS1*AB1 + w2*OPS2*AB2 + w3*OPS3*AB3)/wPA
    # regression to the mean
    r = wPA/(wPA + PA_avg)
    rOPS = wOPS * r + OPS_avg * (1 - r)
    # aging
    proj['OPS'].loc[player] = rOPS + age_r * (age - age_c)
    
