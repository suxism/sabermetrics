
@author: Ji-Hoon Park

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_csv(r'C:\Users\USER\Downloads\Fangraphs Leaderboard (2).csv').set_index('playerid')
coeff = pd.DataFrame(index=range(1920, 2018), columns=['x1','x2','x3'])

for year in range(1923, 2018):
    x1 = data[data.Season == year-3].WAR
    x2 = data[data.Season == year-2].WAR
    x3 = data[data.Season == year-1].WAR
    y1 = data[data.Season == year].WAR
    subset = pd.concat([x1,x2,x3,y1], axis=1).dropna()
    
    x = subset.iloc[:,:3]
    y = subset.iloc[:,-1]
    
    model = sm.OLS(y, x).fit()  
    coeff['x1'].loc[year] = model.params[0]
    coeff['x2'].loc[year] = model.params[1]
    coeff['x3'].loc[year] = model.params[2]
