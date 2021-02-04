#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:23:50 2021

@author: fabianschweinzer
"""

import pandas as pd
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt


sp500_data = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/S&P_Database_Final')
# Welcome message for our Portfolio-Builiding Tool

print("Let's build a portfolio together!")
print("Answer the following questions to get recommendations for your individualized portfolio:")

# Ethical constraints: Weapons

weapon_yes_no = input("Do you feel comfortable investing in companies that make their money with weapons? (Yes/No) ")

while weapon_yes_no != 'Yes' and weapon_yes_no != 'No':
    weapon_yes_no = input("Please enter a valid input: (Yes/No) ")

if weapon_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have something to do with weapons
    filter_array = sp500_data.loc[:, 'Weapons'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Ethical constraints: Gambling

gambling_yes_no = input("Do you feel comfortable investing in companies that operate in the gambling sector? (Yes/No) ")

while gambling_yes_no != 'Yes' and gambling_yes_no != 'No':
    gambling_yes_no = input("Please enter a valid input: (Yes/No) ")

if gambling_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have something to do with gambling
    filter_array = sp500_data.loc[:, 'Gambling'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Ethical constraints: Tobacco

tobacco_yes_no = input("Do you feel comfortable investing in companies that sell tobacco-products? (Yes/No) ")

while tobacco_yes_no != 'Yes' and tobacco_yes_no != 'No':
    tobacco_yes_no = input("Please enter a valid input: (Yes/No) ")

if tobacco_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have something to do with tobacco
    filter_array = sp500_data.loc[:, 'Tobacco'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Ethical constraints: Animal Testing

animal_testing_yes_no = input("Do you feel comfortable investing in companies that test their products on animals? (Yes/No) ")

while animal_testing_yes_no != 'Yes' and animal_testing_yes_no != 'No':
    animal_testing_yes_no = input("Please enter a valid input: (Yes/No) ")

if animal_testing_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have test their products on animals
    filter_array = sp500_data.loc[:, 'Animal Testing'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Find out risk tolerance of investor


# Find out 25th, 50th and 75th percentile

sp500_std_25th_percentile = np.percentile(sp500_data.loc[:, "Annualized Std"], 25)
sp500_std_50th_percentile = np.percentile(sp500_data.loc[:, "Annualized Std"], 50)
sp500_std_75th_percentile = np.percentile(sp500_data.loc[:, "Annualized Std"], 75)

print(len(sp500_data))

# Find out importance of ESG for investor

esg_importance = input("How important is the ESG-criteria for you as investor? (Low/Medium/High) ")

# Find out 25th, 50th and 75th percentile

sp500_esg_25th_percentile = np.percentile(sp500_data.loc[:, "ESG Score"], 25)
sp500_esg_50th_percentile = np.percentile(sp500_data.loc[:, "ESG Score"], 50)


while esg_importance != 'Low' and esg_importance != 'Medium' and esg_importance != 'High':
    esg_importance = input("Please enter a valid input: (Low/Medium/High) ")

if esg_importance == 'Low':
    # Do nothing
    None
elif esg_importance == 'Medium':
    filter_array = sp500_data.loc[:, "ESG Score"] > sp500_std_25th_percentile
    sp500_data = sp500_data[filter_array]
else:
    filter_array = sp500_data.loc[:, "Annualized Std"] > sp500_std_50th_percentile
    sp500_data = sp500_data[filter_array]


test = input("test")

"""
risk_tolerance = input("Which attitude towards risk matches your character traits the most? (Low/Medium/High) ")

while risk_tolerance != 'Low' and risk_tolerance != 'Medium' and risk_tolerance != 'High':
    risk_tolerance = input("Please enter a valid input: (Low/Medium/High) ")
if risk_tolerance == 'Low':
    filter_array = sp500_data.loc[:, "Annualized Std"] < sp500_std_25th_percentile
    sp500_data = sp500_data[filter_array]
elif risk_tolerance == 'Medium':
    filter_array_1 = sp500_data.loc[:, "Annualized Std"] > sp500_std_25th_percentile
    sp500_data = sp500_data[filter_array_1]
    filter_array_2 = sp500_data.loc[:, "Annualized Std"] < sp500_std_75th_percentile
    sp500_data = sp500_data[filter_array_2]
else:
    filter_array = sp500_data.loc[:, "Annualized Std"] > sp500_std_75th_percentile
    sp500_data = sp500_data[filter_array]
 """


database = sp500_data
database.set_index('Symbol', inplace=True)


##first steps project
#getting data from yahoo finance


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import yahoo_fin as yf
from pandas_datareader import data
#import seaborn as sns

#initialize dataset 

##read cleaned S&P 500 dataframe
database = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/S&P_Database_Final')
database.set_index('Symbol', inplace=True)

#extract expected return calculated with CAPM model

mu_target = database.iloc[:,-2]

#initiate Cov Matrix 

cov_matrix = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/Cov_Matrix_SP500')

cov_matrix.set_index('Unnamed: 0', inplace = True)

#we need to filter the returns based on the tickers of the cov_matrix
#update: calculated now the cov matrix for the whole database, so no need to slice anymore

#cov_matrix_list = cov_matrix.index.to_list()
#mu_capm_sliced = mu_capm[mu_capm.index.isin(cov_matrix.index)]


##Start using pyopft in order to optimize Portfolio according to certain constraints
##pypfopt optimisation

from pypfopt import EfficientFrontier
#from pypfopt import risk_models
from pypfopt import expected_returns, objective_functions
#from pypfopt.risk_models import CovarianceShrinkage 
from pypfopt import CLA, plotting

#starting optimizer by defining mu and sigma 
#calculate expected returns and sample covariance
#we already calculated mu and sigma in a previos step, so we can use these variables


#calculate the EfficientFrontier using calculated mu, sigma
ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0.0,0.2))
ef.add_objective(objective_functions.L2_reg, gamma=1)
target_volatility = 0.6
#put in conditional arguments regarding the investors risk tolerance
risk_tolerance = input("Which attitude towards risk matches your character traits the most? (Low/Medium/High) ")

while risk_tolerance != 'Low' and risk_tolerance != 'Medium' and risk_tolerance != 'High':
    risk_tolerance = input("Please enter a valid input: (Low/Medium/High) ")
if risk_tolerance == 'Low':
    weights = ef.min_volatility()
elif risk_tolerance == 'Medium':
    weights = ef.max_sharpe()
else:
    weights = ef.efficient_risk(target_volatility)

#clean weights for a better visualization
clean_weights = ef.clean_weights()
ef.portfolio_performance(verbose = True)

##visualize the efficient frontier line
#there is an issue here, if I want to run it, I will get a timeout
#cla= CLA(mu_capm_sliced, cov_matrix)
#cla.min_volatility()
#cla.portfolio_performance(verbose=True)

#there is some issue here regarding the cla return (probably to many requests a time)
#ax = plotting.plot_efficient_frontier(cla, show_assets=True, show_fig=True )

###################################################################
#step 2, as we can't specifiy exactly 25 stocks in the first round, we need to optimize again
#we subset the outcome of the first optimization to the 25 stocks with the highest weights
#reduce the size to 25 stocks in order to go to the next round of optimization
df = pd.DataFrame.from_dict(data=clean_weights, orient='index')
df.columns = ['Weight']
df_filter = df['Weight'] > 0
df_1 = df[df_filter].sort_values(by='Weight', ascending=False)
df_1 = df_1.head(25)


#slice the database to get only the 25 stocks and perform a second round of optimization
#subseting the expected return
mu_target_25 = mu_target[mu_target.index.isin(df_1.index)]

#subsetting the cov matrix is more worrysome
#I used a transpose technique to do the subsetting on both axis
cov_matrix_25 = cov_matrix[cov_matrix.index.isin(df_1.index)]
cov_matrix_25 = cov_matrix_25.T
cov_matrix_25 = cov_matrix_25[cov_matrix_25.index.isin(df_1.index)]

#Effficient Frontier optimization for the top25 stocks
#we insert minimum weight limit, so every stocks gets a minimum weight of 1%
#we also include a maximum weight limit, so to make sure there is not too much importance on one particular stock

ef_25 = EfficientFrontier(mu_target_25, cov_matrix_25, weight_bounds=(0.01, 0.2))
if risk_tolerance == 'Low':
    weights_25 = ef_25.min_volatility()
elif risk_tolerance == 'Medium':
    weights_25 = ef_25.max_sharpe()
else:
    weights_25 = ef_25.efficient_risk(target_volatility)

#clean weights and and print the performance indicators
clean_weights_25 = ef_25.clean_weights()
ef_25.portfolio_performance(verbose = True)

print(clean_weights_25)

#create piechart 
piechart = pd.Series(clean_weights_25).plot.pie(figsize=(10,10))
plt.show(piechart)

#create barchart
barchart = pd.Series(clean_weights_25).sort_values(ascending=True).plot.barh(figsize=(10,6))
plt.show(barchart)

#covariance heatmap
plotting.plot_covariance(cov_matrix_25, plot_correlation = True)

##create the Efficient frontier line and visualize it

cla_25 = CLA(mu_target_25, cov_matrix_25)
if risk_tolerance == 'Low':
    cla_25.min_volatility()
elif risk_tolerance == 'Medium':
    cla_25.max_sharpe()
else:
    cla_25.efficient_risk(target_volatility)

cla_25.portfolio_performance(verbose=True)

#plot the efficient frontier line
ax_25 = plotting.plot_efficient_frontier(cla_25, show_assets=False, show_fig=False )


####plot with ef function
#risk_range = np.linspace(0.1, 0.4, 100)
#ax_ef = plotting.plot_efficient_frontier(ef_25, ef_param='risk', ef_param_range=risk_range, show_assets=True, show_fig=True)

#initialize Discrete Allocation to get a full picture what you could buy with a given amount
from pypfopt import DiscreteAllocation
from datetime import datetime

tickers = pd.DataFrame.from_dict(data=clean_weights_25, orient='index')
tickers_list = tickers.index.to_list()

funds = 10000

da = data.DataReader(tickers_list, data_source='yahoo', start=datetime(2021,1,28), end=datetime.today())['Adj Close']

latest_price = da.iloc[-1,:]

alloc = DiscreteAllocation(clean_weights_25,latest_prices=latest_price, total_portfolio_value=funds)
allocation, leftover = alloc.lp_portfolio()
print(allocation)
print(leftover)

pie_alloc = pd.Series(allocation).plot.pie(figsize=(10,10))
plt.show(pie_alloc)

######
#backtesting with pyfolio from quantopian
#import pyfolio as pf
#initialize 
#returns_da = da.pct_change().dropna()
#returns_da.index = pd.to_datetime(returns_da.index)

#tear_sheet_25 = pf.create_returns_tear_sheet(returns_da)
#tear_sheet_25 













    



   
    
    
    
