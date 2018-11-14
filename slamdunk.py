# -*- coding: utf-8 -*-
"""
@author: Ji-Hoon Park
"""

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\USER\Downloads\stats.csv').set_index(['player']).replace(np.nan,0)

#%% team stats

data['MP'] = data['Min_1ST'] + data['Min_2nd']
data['PTS'] = data['FG']*2 + data['FT']*1 + data['3P']*1
team_data = data.groupby(['team','game']).sum()

for stat in data.columns[3:20]:
    data['Tm ' + stat] = pd.Series(index=data.index)
    data['Opp ' + stat] = pd.Series(index=data.index)

for player in data.index:
    for stat in data.columns[3:20]:
        tm_stat = 'Tm ' + stat
        opp_stat = 'Opp ' + stat
        data[tm_stat].loc[player] = team_data.loc[data.loc[player].team,:][stat].sum()
        
        if data.loc[player].team == 'booksan':
            data[opp_stat].loc[player] = team_data.loc['sangyang',:][stat].sum()
        elif data.loc[player].team == 'sangyang':
            data[opp_stat].loc[player] = team_data.loc['booksan',:][stat].sum()      

data['Tm TS%'] = data['Tm PTS'] / (2 * (data['Tm FGA'] + 0.44 * data['Tm FTA']))

#%% efficiency

data['TS%'] = data['PTS'] / (2 * (data['FGA'] + 0.44 * data['FTA']))
data['AST%'] = 100 * data.AST/(((data.MP/(data['Tm MP']/5)) * data['Tm FG']) - data.FG)
data['BLK%'] = 100 * (data.BLK * (data['Tm MP']/5))/(data.MP * (data['Opp FGA'] - data['Opp 3PA']))
data['DRB%'] = 100 * (data.DRB * (data['Tm MP']/5)) / (data.MP * (data['Tm DRB'] + data['Opp ORB']))
data['TRB%'] = 100 * ((data.DRB + data.ORB) * (data['Tm MP'] / 5)) / (data['MP'] * (data['Tm DRB'] + data['Tm ORB'] + data['Opp DRB'] + data['Opp ORB']))
data['eFG%'] = (data.FG + 0.5 * data['3P'])/data.FGA
data['FG%'] = data.FG / data.FGA
data['FT%'] = data.FT / data.FTA
data['GmSc'] = (11./65)*10. * (data.PTS + 0.4 * data.FG - 0.7 * data.FGA - 0.4*(data.FTA - data.FT) + 0.7 * data.ORB + 0.3 * data.DRB + data.STL + 0.7 * data.AST + 0.7 * data.BLK - 0.4 * data.PF - data.TOV)
data['ORB%'] = 100 * (data.ORB * (data['Tm MP']/5)) / (data.MP * (data['Tm ORB'] + data['Opp DRB']))
data['TOV%'] = 100 * data.TOV / (data.FGA + 0.44 * data.FTA + data.TOV)
data['Usg%'] = 100 * ((data.FGA + 0.44 * data.FTA + data.TOV) * (data['Tm MP'] / 5)) / (data.MP * (data['Tm FGA'] + 0.44 * data['Tm FTA'] + data['Tm TOV']))

#%% Possesions

data['Tm Poss'] = data['Tm FGA'] + 0.4 * data['Tm FTA'] - 1.07 * (data['Tm ORB']/(data['Tm ORB'] + data['Opp DRB'])) * (data['Tm FGA'] - data['Tm FG']) + data['Tm TOV']
data['Opp Poss'] = data['Opp FGA'] + 0.4 * data['Opp FTA'] - 1.07 * (data['Opp ORB']/(data['Opp ORB'] + data['Tm DRB'])) * (data['Opp FGA'] - data['Opp FG']) + data['Opp TOV']
data['Pace'] = 48 * ((data['Tm Poss'] + data['Opp Poss']) / (2 * (data['Tm MP'] / 5)))
lg_Pace = data.groupby('team').Pace.first().mean()

data['Poss adj'] = lg_Pace / data['Pace']

data['STL%'] = 100 * (data['STL'] * (data['Tm MP'] / 5)) / (data['MP'] * data['Opp Poss'])

#%% PER

lg_data = team_data.sum()

factor = (2. / 3) - (0.5 * (lg_data.AST / lg_data.FG)) / (2 * (lg_data.FG / lg_data.FT))
VOP = lg_data.PTS / (lg_data.FGA - lg_data.ORB + lg_data.TOV + 0.44 * lg_data.FTA)
DRBr = lg_data.DRB / (lg_data.DRB + lg_data.ORB)

data['uPER'] = (1 / data.MP) * ( data['3P'] + (2./3) * data.AST + (2 - factor * (data['Tm AST']/ data['Tm FG'])) * data.FG + (data.FT *0.5 * (1 + (1 - (data['Tm AST'] / data['Tm FG'])) + (2./3) * (data['Tm AST']/data['Tm FG']))) \
     - VOP * data.TOV - VOP * DRBr * (data.FGA - data.FG) - VOP * 0.44 * (0.44 + (0.56 * DRBr)) * (data.FTA - data.FT) + VOP * (1 - DRBr) * (data.DRB) + VOP * DRBr * data.ORB + VOP * data.STL \
     + VOP * DRBr * data.BLK - data.PF * ((lg_data.FT/lg_data.PF) - 0.44 * (lg_data.FTA / lg_data.PF) * VOP))

lg_uPER = (1 / lg_data.MP) * ( lg_data['3P'] + (2./3) * lg_data.AST + (2 - factor * (lg_data['AST']/ lg_data['FG'])) * lg_data.FG + (lg_data.FT *0.5 * (1 + (1 - (lg_data['AST'] / lg_data['FG'])) + (2./3) * (lg_data['AST']/lg_data['FG']))) \
     - VOP * lg_data.TOV - VOP * DRBr * (lg_data.FGA - lg_data.FG) - VOP * 0.44 * (0.44 + (0.56 * DRBr)) * (lg_data.FTA - lg_data.FT) + VOP * (1 - DRBr) * (lg_data.DRB) + VOP * DRBr * lg_data.ORB + VOP * lg_data.STL \
     + VOP * DRBr * lg_data.BLK - lg_data.PF * ((lg_data.FT/lg_data.PF) - 0.44 * (lg_data.FTA / lg_data.PF) * VOP))

data['aPER'] = data['Poss adj'] * data['uPER']
data['PER'] = data['aPER'] * (15 / lg_uPER)

print(data['PER'].sort_values(ascending=False))

#%% BPM

a = 0.123391
b = 0.119597
c = -0.151287
d = 1.255644
e = 0.531838
f = -0.305868
g = 0.921292
h = 0.711217
i = 0.017022
j = 0.297639
k = 0.213485
l = 0.725930

data['MPG'] = data['MP']/(1)
data['ReMPG'] = data['MP']/(5)
data['3PAr'] = data['3PA']/data['FGA']
Lg3PAr = data['3PA'].sum()/data['FGA'].sum()

data['Raw BPM'] = a*data['ReMPG'] + b*data['ORB%'] + c*data['DRB%'] + d*data['STL%'] + e*data['BLK%'] + f*data['AST%'] - g*data['Usg%']*data['TOV%'] + \
h*data['Usg%']*(1-data['TOV%'])*(2*(data['TS%'] - data['Tm TS%']) + i*data['AST%'] + j*(data['3PAr'] - Lg3PAr) - k) + l*np.sqrt(data['AST%']*data['TRB%'])

print(data['Raw BPM'].sort_values(ascending=False))

#%% ASPM

a = 0.08033
b = 0.16984
c = 0.27982
d = 1.26329
e = 0.66443
f = 0.53342
g = 1.47832
h = 0.00794
i = 0.01160

data['ASPM'] = a*data['MPG'] + b*data['TRB%'] + c*data['BLK%'] + d*data['STL%'] + e*data['Usg%']*( data['TS%']*2*(1-data['TOV%']) - f*data['TOV%'] - g + h*data['AST%'] + i*data['Usg%'] )

print(data['ASPM'].sort_values(ascending=False))
