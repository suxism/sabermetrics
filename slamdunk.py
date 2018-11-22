# -*- coding: utf-8 -*-
"""
@author: Ji-Hoon Park
"""

import pandas as pd
import numpy as np

data = pd.read_csv(r'C:\Users\USER\Downloads\stats.csv').replace(np.nan,0)

#%% team stats

data['MP'] = data['Min_1ST'] + data['Min_2nd']
data['PTS'] = data['FG']*2 + data['FT']*1 + data['3P']*1
team_data = data.groupby(['team','game']).sum().reset_index()

for stat in data.columns[4:21]:
    data['Tm ' + stat] = pd.Series(index=data.index)
    data['Opp ' + stat] = pd.Series(index=data.index)

for i in data.index:    
    p = data.loc[i].player
    t = data.loc[i].team
    g = data.loc[i].game   
    for stat in data.columns[4:21]:
        tm_stat = 'Tm ' + stat
        data[tm_stat].loc[i] = team_data[team_data.team==t][team_data.game==g][stat].sum()

for i in data.index:    
    p = data.loc[i].player
    t = data.loc[i].team
    g = data.loc[i].game   
    for stat in data.columns[4:21]:
        opp_stat = 'Opp ' + stat
        data[opp_stat].loc[i] = team_data[team_data.team!=t][team_data.game==g][stat].sum()

data.to_csv(r'C:\Users\USER\Downloads\data_raw.csv', index=False)


#%% Stats

data = pd.read_csv(r'C:\Users\USER\Downloads\data_raw.csv')
data['G'] = 1
lg_data = data.groupby(['team','game']).first().drop(columns=['player']).sum().drop(index=['Pos','AST','STL','DRB','ORB','BLK','TOV','PF','FGA','FG','3PA','3P','FTA','FT','Min_1ST','Min_2nd','MP','PTS'])

data['Tm Poss'] = (- 1.07 * (data['Tm ORB']/(data['Tm ORB'] + data['Opp DRB'])) * (data['Tm FGA'] - data['Tm FG']) + data['Tm TOV']).replace(np.nan,0)
data['Tm Poss'] = data['Tm FGA'] + 0.4 * data['Tm FTA'] + data['Tm Poss']
data['Opp Poss'] = (- 1.07 * (data['Opp ORB']/(data['Opp ORB'] + data['Tm DRB'])) * (data['Opp FGA'] - data['Opp FG']) + data['Opp TOV']).replace(np.nan,0)
data['Opp Poss'] = data['Opp FGA'] + 0.4 * data['Opp FTA'] + data['Opp Poss'] 
data['Pace'] = 40 * ((data['Tm Poss'] + data['Opp Poss']) / (2 * (data['Tm MP'] / 5)))
lg_Pace = 40*(data.groupby(['team','game']).first()['Tm Poss'].sum() + data.groupby(['team','game']).first()['Opp Poss'].sum())/ (2 * (data.groupby(['team','game']).first()['Tm MP'].sum()/ 5))

data = data.groupby('player').sum()
data['Tm Poss'] = (- 1.07 * (data['Tm ORB']/(data['Tm ORB'] + data['Opp DRB'])) * (data['Tm FGA'] - data['Tm FG']) + data['Tm TOV']).replace(np.nan,0)
data['Tm Poss'] = data['Tm FGA'] + 0.4 * data['Tm FTA'] + data['Tm Poss']
data['Opp Poss'] = (- 1.07 * (data['Opp ORB']/(data['Opp ORB'] + data['Tm DRB'])) * (data['Opp FGA'] - data['Opp FG']) + data['Opp TOV']).replace(np.nan,0)
data['Opp Poss'] = data['Opp FGA'] + 0.4 * data['Opp FTA'] + data['Opp Poss'] 
data['Pace'] = 40 * ((data['Tm Poss'] + data['Opp Poss']) / (2 * (data['Tm MP'] / 5)))

#data['Poss adj'] = lg_Pace / data['Pace']
data['Poss adj'] = 0.5 + 0.5*(lg_Pace / data['Pace'])
#data['Poss adj'] = 1

data['Tm TS%'] = data['Tm PTS'] / (2 * (data['Tm FGA'] + 0.44 * data['Tm FTA']))
data['TS%'] = data['PTS'] / (2 * (data['FGA'] + 0.44 * data['FTA']))
data['AST%'] = 100 * data.AST/(((data.MP/(data['Tm MP']/5)) * data['Tm FG']) - data.FG)
data['BLK%'] = 100 * (data.BLK * (data['Tm MP']/5))/(data.MP * (data['Opp FGA'] - data['Opp 3PA']))
data['DRB%'] = 100 * (data.DRB * (data['Tm MP']/5)) / (data.MP * (data['Tm DRB'] + data['Opp ORB']))
data['TRB%'] = 100 * ((data.DRB + data.ORB) * (data['Tm MP'] / 5)) / (data['MP'] * (data['Tm DRB'] + data['Tm ORB'] + data['Opp DRB'] + data['Opp ORB']))
data['eFG%'] = (data.FG + 0.5 * data['3P'])/data.FGA
data['FG%'] = data.FG / data.FGA
data['FT%'] = data.FT / data.FTA
data['GmSc'] = data.PTS + 0.4 * data.FG - 0.7 * data.FGA - 0.4*(data.FTA - data.FT) + 0.7 * data.ORB + 0.3 * data.DRB + data.STL + 0.7 * data.AST + 0.7 * data.BLK - 0.4 * data.PF - data.TOV
data['ORB%'] = 100 * (data.ORB * (data['Tm MP']/5)) / (data.MP * (data['Tm ORB'] + data['Opp DRB']))
data['TOV%'] = 100 * data.TOV / (data.FGA + 0.44 * data.FTA + data.TOV)
data['Usg%'] = 100 * ((data.FGA + 0.44 * data.FTA + data.TOV) * (data['Tm MP'] / 5)) / (data.MP * (data['Tm FGA'] + 0.44 * data['Tm FTA'] + data['Tm TOV']))
data['GmSc/G'] = data['GmSc']/data['G']
data['STL%'] = 100 * (data['STL'] * (data['Tm MP'] / 5)) / (data['MP'] * data['Opp Poss'])

#%% PER

factor = (2./3) - (0.5 * (lg_data['Tm AST'] / lg_data['Tm FG'])) / (2 * (lg_data['Tm FG'] / lg_data['Tm FT']))
VOP = lg_data['Tm PTS'] / (lg_data['Tm FGA'] - lg_data['Tm ORB'] + lg_data['Tm TOV'] + 0.44 * lg_data['Tm FTA'])
DRBr = lg_data['Tm DRB'] / (lg_data['Tm DRB'] + lg_data['Tm ORB'])

data['uPER'] = (1./data.MP) * ( data['3P'] + (2./3) * data.AST + (2 - factor * (data['Tm AST']/ data['Tm FG'])) * data.FG + (data.FT *0.5 * (1 + (1 - (data['Tm AST'] / data['Tm FG'])) + (2./3) * (data['Tm AST']/data['Tm FG']))) \
     - VOP * data.TOV - VOP * DRBr * (data.FGA - data.FG) - VOP * 0.44 * (0.44 + (0.56 * DRBr)) * (data.FTA - data.FT) + VOP * (1 - DRBr) * (data.DRB) + VOP * DRBr * data.ORB + VOP * data.STL \
     + VOP * DRBr * data.BLK - data.PF * ((lg_data['Tm FT']/lg_data['Tm PF']) - 0.44 * (lg_data['Tm FTA'] / lg_data['Tm PF']) * VOP))

lg_uPER = (1./lg_data['Tm MP']) * ( lg_data['Tm 3P'] + (2./3) * lg_data['Tm AST'] + (2 - factor * (lg_data['Tm AST']/ lg_data['Tm FG'])) * lg_data['Tm FG'] + (lg_data['Tm FT'] *0.5 * (1 + (1 - (lg_data['Tm AST'] / lg_data['Tm FG'])) + (2./3) * (lg_data['Tm AST']/lg_data['Tm FG']))) \
     - VOP * lg_data['Tm TOV'] - VOP * DRBr * (lg_data['Tm FGA'] - lg_data['Tm FG']) - VOP * 0.44 * (0.44 + (0.56 * DRBr)) * (lg_data['Tm FTA'] - lg_data['Tm FT']) + VOP * (1 - DRBr) * (lg_data['Tm DRB']) + VOP * DRBr * lg_data['Tm ORB'] + VOP * lg_data['Tm STL'] \
     + VOP * DRBr * lg_data['Tm BLK'] - lg_data['Tm PF'] * ((lg_data['Tm FT']/lg_data['Tm PF']) - 0.44 * (lg_data['Tm FTA'] / lg_data['Tm PF']) * VOP))

data['aPER'] = data['Poss adj'] * data['uPER']
lg_aPER = (data['aPER']*data['MP']).replace(np.nan,0).sum()/data['MP'].sum()
data['PER'] = data['aPER'] * (15./ lg_aPER)

print(data[data.MP>10]['PER'].sort_values(ascending=False))

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

data['MPG'] = data['MP']/data['G']
#data['ReMPG'] = data['MP']/(data['G']+4)
data['3PAr'] = data['3PA']/data['FGA']
Lg3PAr = data['3PA'].sum()/data['FGA'].sum()

data['Raw BPM'] = a*data['MPG'] + b*data['ORB%'] + c*data['DRB%'] + d*data['STL%'] + e*data['BLK%'] + f*data['AST%'] - g*data['Usg%']*data['TOV%'] + \
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

#%% output

data.to_csv(r'C:\Users\USER\Downloads\data_output.csv')
