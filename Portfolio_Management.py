#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:35:44 2021

@author: fabianschweinzer
"""
import pandas as pd



#we insert first our dataset, where we get our tickers
list_1 = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.csv')
list_2 = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv')

#we merged both Dataframes
list_merged = pd.concat([list_1, list_2], axis=0).dropna()

#we create a ticker list out of the merged dataframe
tickers = list_merged['ticker'].to_list()

#we need to remove (or test for) duplicates in the ticker list
#we use the set() function to do this
tickers = list(set(tickers))

import pandas_datareader as data 
import datetime as datetime

def data_get(tickers, s ,e ):
    database = data.DataReader(tickers, data_source='yahoo', 
                               start = s, 
                               end = e)['Adj Close']
    return database

#initiate dates, lookback period
s = datetime.datetime(2020,2,5)
e = datetime.datetime.today()

#pull database 
database = data_get(tickers, s, e)

#calculate returns based on the pulled database and drop NaN
#database.dropna(subset='LGVW',how='any', inplace=True)
#delete assets which have not enough data to calculate one year, annualized return
del database['BEKE']
del database['BEAM']
del database['MASS']
del database['HIMS']
del database['U']
del database['ACCD']
del database['RPTX']
del database['BLI']
del database['SDGR']
del database['LGVW']

returns = database.pct_change().dropna()

#returns = returns.reset_index()
#returns = returns.drop([0,1109], axis=0)

#we calculate the mean
#returns = returns.set_index('Date')
mu = returns.mean() *252


#we calculate the standard deviation and variance
sigma = returns.std()
variance = returns.var()

#we calculate the covariance matrix
cov_matrix = returns.cov() *252

#calculate correlation matrix
corr_matrix = returns.corr() 

######################
import matplotlib.pyplot as plt 
import numpy as np

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns, objective_functions
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage 
from pypfopt import CLA, plotting

#calculate mu and sigma using the built-in functions
mu_historical = mean_historical_return(database, returns_data=False, frequency=252, compounding=True)

Cov_shrinkage = risk_models.CovarianceShrinkage(database, returns_data=False).ledoit_wolf()

#calculate semi-covariance for sortino
semi_cov = risk_models.semicovariance(database, returns_data=False, benchmark = 0)

#Efficient Frontier optimization
#either use own calculated mu or mu_historical from built-in function
#either use own calculated cov_matrix, built-in Cov_shrinkage or semi-covariance
#either maximise sharpe or maximise return given a pre-specified volatility
ef_pm= EfficientFrontier(mu, semi_cov, weight_bounds=(0, 0.15), verbose=True)
ef_pm.add_objective(objective_functions.L2_reg, gamma=1)
weights_pm = ef_pm.max_sharpe()

cleaned_weights = ef_pm.clean_weights()

ef_pm.portfolio_performance(verbose=True)

###filter it down to 25 stocks
df_pm = pd.DataFrame.from_dict(data=cleaned_weights, orient='index')
df_pm.columns = ['Weight']
df_pm = df_pm.nlargest(25, 'Weight')
df_pm.to_excel('List_PM_25Stocks.xlsx')
ticker_list_pm = df_pm.index.to_list()

#create a dictionary which can be used for the Discrete Allocation as an input
weight_pm = df_pm.to_dict(orient='dict')
weight_pm = {}
values = df_pm['Weight']
for ticker in ticker_list_pm:
    weight_pm[ticker] = values[ticker]
print(weight_pm)
    

#plot efficient frontier
cla_pm = CLA(mu, semi_cov)
cla_pm.max_sharpe()

frontier = plotting.plot_efficient_frontier(cla_pm, ef_param='risk', show_assets=True)

#Discrete Allocation
#initialize Discrete Allocation to get a full picture what you could buy with a given amount
from pypfopt import DiscreteAllocation
from datetime import datetime

funds_pm = 100000

da_pm = data.DataReader(ticker_list_pm, data_source='yahoo', start=datetime(2020,1,28), end=datetime.today())['Adj Close']

latest_price_pm = da_pm.iloc[-1,:]

alloc_pm = DiscreteAllocation(weight_pm,latest_prices=latest_price_pm, total_portfolio_value=funds_pm)
allocation, leftover = alloc_pm.lp_portfolio()
print(allocation)
print(leftover)

pie_alloc_pm = pd.Series(allocation).plot.pie(figsize=(10,10))
plt.show(pie_alloc_pm)

#after we can add the monte carlo simulation given the ticker list from the optimizer
#after we can do the backtesting (either here or in portfolio visualizer)
#






    