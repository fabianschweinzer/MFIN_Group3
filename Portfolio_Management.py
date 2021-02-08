#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:35:44 2021

@author: fabianschweinzer
"""
import pandas as pd
import seaborn as sns



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

#due to cutting the dataset to the 25 largest we lose some weights
#create a for loop for getting the weights to a sum of 1
for ticker in df_pm:
    df_pm[ticker] = df_pm['Weight'] / np.sum(df_pm['Weight']) * 1


df_pm.to_excel('List_PM_25.xlsx', sheet_name='Equities')
ticker_list_pm = df_pm.index.to_list()


#plot efficient frontier
cla_pm = CLA(mu, semi_cov)
cla_pm.max_sharpe()

frontier = plotting.plot_efficient_frontier(cla_pm, ef_param='risk', show_assets=True)

###############################################################

#read in excel file for the ETFs
etf_data = pd.read_excel('List_PM_ETF.xlsx', sheet_name='ETF').set_index('Unnamed: 0')


#we give etfs a weight of 30% for the total portfolio
for i in etf_data:
    etf_data[i] = etf_data[i]*0.3

#we give equities a weight of 70% for the total portfolio
for i in df_pm:
    df_pm[i] = df_pm[i]*0.7
      
#merge df_pm dataset with etf_data dataset
portfolio_total = pd.concat([df_pm, etf_data])
#
#np.sum(portfolio_total['Weight'])

#create a list of the total portfolio tickers
portfolio_tickers = portfolio_total.index.to_list()

####################################################################
    


###calculate portfolio_return and portfolio standard deviation
portfolio =  data.DataReader(portfolio_tickers, data_source='yahoo', 
                               start = s, 
                               end = e)['Adj Close']

portfolio.sort_index(inplace=True)


portfolio_return = portfolio.pct_change().dropna()

mean_portfolio_return = portfolio_return.mean()

return_std = portfolio_return.std()
portfolio_cov = portfolio_return.cov()

#portfolio return added to the portfolio_return
portfolio_return['Portfolio'] = portfolio_return.mean(axis=1)


stock_portfolio_return = round(np.sum(mean_portfolio_return * portfolio_total['Weight']) * 252,2)

portfolio_std= round(np.sqrt(np.dot(portfolio_total['Weight'].T,np.dot(portfolio_cov, portfolio_total['Weight']))) * np.sqrt(252),2)

print('Portfolio expected annualised return is ' + str(stock_portfolio_return) + ' with a standard deviation of ' + str(portfolio_std))#' and volatility is {}').format(stock_portfolio_return,portfolio_std)


####################################################
#get a monte carlo simulation for the portfolio return
# Parametric
# Initialize Monte Carlo parameters

monte_carlo_runs = 1000
days_to_simulate = 5
loss_cutoff      = 0.95         # count any losses larger than 5% (or -5%)


compound_returns  = return_std.copy()
total_simulations = 0
bad_simulations   = 0

for run_counter in range(0,monte_carlo_runs):   # Loop over runs    
    for i in portfolio_tickers:                      # loop over tickers, below is done once per ticker
        
        # Loop over simulated days:
        compounded_temp = 1
        
        for simulated_day_counter in range(0,days_to_simulate): # loop over days
            
            # Draw from 𝑁~(𝜇,𝜎)
            ######################################################
            simulated_return = np.random.normal(mean_portfolio_return[i],return_std[i],1)
            ######################################################
            
            compounded_temp = compounded_temp * (simulated_return + 1)        
        
        compound_returns[i] = compounded_temp     # store compounded returns
    
    # Now see if those returns are bad by combining with weights
    portfolio_return_mc = compound_returns.dot(portfolio_total['Weight']) # dot product
    
    if(portfolio_return_mc < loss_cutoff):
        bad_simulations = bad_simulations + 1
    
    total_simulations = total_simulations + 1

print("Your portfolio will lose", round((1-loss_cutoff)*100,3), "%",
      "over", days_to_simulate, "days", 
      bad_simulations/total_simulations, "of the time.")

# Plot Returns + VaR

# setting figure size
fig, ax = plt.subplots(figsize = (13, 5))

# histogram for returns
sns.histplot(data  = portfolio_return['Portfolio'],  # data set - index Facebook (or AAPL or GOOG)
             bins  = 'fd',          # number of bins ('fd' = Freedman-Diaconis Rule) 
             kde   = True,          # kernel density plot (line graph)
             alpha = 0.2,           # transparency of colors
             stat  = 'count')     # can be set to 'count', 'frequency', or 'probability'


# this adds a title
plt.title(label = "Distribution of Portfolio Return")


# this adds an x-label
plt.xlabel(xlabel = 'Returns')


# this add a y-label
plt.ylabel(ylabel = 'Count')


# instantiate VaR with 95% confidence level
VaR_95 = np.percentile(portfolio_return, 5)


# this adds a line to signify VaR
plt.axvline(x         = VaR_95,         # x-axis location
            color     = 'r',            # line color
            linestyle = '--')           # line style


# this adds a label to the line
plt.text(VaR_95,                         # x-axis location
         30,                             # y-axis location
         'VaR',                          # text
         horizontalalignment = 'right',  # alignment ('center' | 'left')
         fontsize = 'x-large')           # fontsize


# these compile and display the plot so that it is formatted as expected
plt.tight_layout()
plt.show()


#create a dictionary which can be used for the Discrete Allocation as an input
weight_portfolio = portfolio_total.to_dict(orient='dict')
weight_portfolio = {}
values = portfolio_total['Weight']
for ticker in portfolio_tickers:
    weight_portfolio[ticker] = values[ticker]
print(weight_portfolio)



#Discrete Allocation
#initialize Discrete Allocation to get a full picture what you could buy with a given amount
from pypfopt import DiscreteAllocation
from datetime import datetime

money_available = 100000

da_portfolio = data.DataReader(portfolio_tickers, data_source='yahoo', start=datetime(2020,1,28), end=datetime.today())['Adj Close']

latest_price_portfolio = da_portfolio.iloc[-1,:]

alloc_portfolio = DiscreteAllocation(weight_portfolio,latest_prices=latest_price_portfolio, 
                                     total_portfolio_value=money_available)
allocation, leftover = alloc_portfolio.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

pie_alloc_portfolio = pd.Series(allocation).plot.pie(figsize=(10,10))
plt.show(pie_alloc_portfolio)

#after we can add the monte carlo simulation given the ticker list from the optimizer
#after we can do the backtesting (either here or in portfolio visualizer)
#






    