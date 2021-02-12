#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:23:50 2021

@author: fabianschweinzer
"""

import pandas as pd
import numpy as np
#import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns


sp500_data = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/S&P_Database_Final')
# Welcome message for our Portfolio-Builiding Tool

print("Let's build a portfolio together!")
print("Answer the following questions to get recommendations for your individualized portfolio:")

#Funds available

funds_q = float(input("How many dollars do you want to invest? (Specify an amount)"))


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
    
#Fin out risk tolerance

#put in conditional arguments regarding the investors risk tolerance
risk_tolerance = input("Which attitude towards risk matches your character traits the most? (Low/Medium/High) ")

while risk_tolerance != 'Low' and risk_tolerance != 'Medium' and risk_tolerance != 'High':
    risk_tolerance = input("Please enter a valid input: (Low/Medium/High) ")



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

#get the database into the framework
database = sp500_data
database.set_index('Symbol', inplace=True)


##first steps project
#getting data from yahoo finance


from pandas_datareader import data
#import seaborn as sns

#extract expected return calculated with CAPM model

mu_target = database.iloc[:,-2]
target_list = mu_target.index.to_list()

#initiate Cov Matrix 

cov_matrix = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/Cov_Matrix_SP500')
cov_matrix.set_index('Unnamed: 0', inplace = True)


#in order to match the target return dataframe with the cov matrix we need to subset the cov matrix
#I used a transpose technique to do the subsetting on both axis
cov_matrix = cov_matrix[cov_matrix.index.isin(mu_target.index)]
cov_matrix = cov_matrix.T
cov_matrix = cov_matrix[cov_matrix.index.isin(mu_target.index)]
cov_matrix = cov_matrix * 252


##Start using pypopft in order to optimize Portfolio according to certain constraints
##pypfopt optimization

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns, objective_functions
from pypfopt.risk_models import CovarianceShrinkage 
from pypfopt import CLA, plotting

#starting optimizer by defining mu and sigma 
#calculate expected returns and sample covariance
#we already calculated mu and sigma in a previos step, so we can use these variables


#calculate the EfficientFrontier using calculated mu, sigma
#ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
#ef.add_objective(objective_functions.L2_reg, gamma=0.05)
target_volatility = 0.3
#put in conditional arguments regarding the investors risk tolerance
#risk_tolerance = input("Which attitude towards risk matches your character traits the most? (Low/Medium/High) ")

#while risk_tolerance != 'Low' and risk_tolerance != 'Medium' and risk_tolerance != 'High':
#    risk_tolerance = input("Please enter a valid input: (Low/Medium/High) ")
if risk_tolerance == 'Low':
    ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
    weights = ef.min_volatility()
elif risk_tolerance == 'Medium':
    ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.max_sharpe()
elif risk_tolerance == 'High':
    ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.25)
    weights = ef.efficient_risk(target_volatility)

#clean weights for a better visualization
clean_weights = ef.clean_weights()
ef.portfolio_performance(verbose = True)

#####get the ticker list from the created dictionary
tickers = pd.DataFrame.from_dict(data=clean_weights, orient='index')
#we need to slize the tickers_list so we get only tickers with non-zero weights
tickers.columns = ['Weight']
tickers = tickers[tickers['Weight']!= 0]
tickers.sort_values(by='Weight', ascending=False)

#create list out of sliced tickers dataframe
tickers_list = tickers.index.to_list()

#get sliced covariance matrix
cov_matrix_tickers = cov_matrix[cov_matrix.index.isin(tickers.index)]
cov_matrix_tickers = cov_matrix_tickers.T
cov_matrix_tickers = cov_matrix_tickers[cov_matrix_tickers.index.isin(tickers.index)]

#get the sliced mu_target dataframe
mu_target_tickers = mu_target[mu_target.index.isin(tickers.index)]


###################################################################
#step 2, as we can't specifiy exactly 25 stocks in the first round, we need to optimize again
#we subset the outcome of the first optimization to the 25 stocks with the highest weights
#reduce the size to 25 stocks in order to go to the next round of optimization
#df = pd.DataFrame.from_dict(data=clean_weights, orient='index')
#df.columns = ['Weight']
#df = df.nlargest(25, 'Weight')


#slice the database to get only the 25 stocks and perform a second round of optimization
#subseting the expected return
#mu_target_25 = mu_target[mu_target.index.isin(df.index)]

#subsetting the cov matrix is more worrysome
#I used a transpose technique to do the subsetting on both axis
#cov_matrix_25 = cov_matrix[cov_matrix.index.isin(df.index)]
#cov_matrix_25 = cov_matrix_25.T
#cov_matrix_25 = cov_matrix_25[cov_matrix_25.index.isin(df.index)]

#Effficient Frontier optimization for the top25 stocks
#we insert minimum weight limit, so every stocks gets a minimum weight of 1%
#we also include a maximum weight limit, so to make sure there is not too much importance on one particular stock

#ef_25 = EfficientFrontier(mu_target_25, cov_matrix_25, weight_bounds=(0.01, 0.2))
#if risk_tolerance == 'Low':
   # weights_25 = ef_25.min_volatility()
#elif risk_tolerance == 'Medium':
#    weights_25 = ef_25.max_sharpe()
#else:
#    weights_25 = ef_25.efficient_risk(target_volatility)

#clean weights and and print the performance indicators
#clean_weights_25 = ef_25.clean_weights()
#ef_25.portfolio_performance(verbose = True)


#create piechart 
piechart = pd.Series(tickers['Weight']).plot.pie(figsize=(10,10))
plt.show(piechart)


#create barchart
barchart = pd.Series(tickers['Weight']).sort_values(ascending=True).plot.barh(figsize=(10,6))
plt.show(barchart)

#plotting.plot_weights(tickers['Weight'])

#covariance heatmap

plotting.plot_covariance(cov_matrix_tickers, plot_correlation = True)

##create the Efficient frontier line and visualize it

cla = CLA(mu_target_tickers, cov_matrix_tickers)
if risk_tolerance == 'Low':
    cla.min_volatility()
else:
    cla.max_sharpe()

#plot the efficient frontier line
ax_portfolio = plotting.plot_efficient_frontier(cla, ef_param='utility', ef_param_range=np.linspace(0.2,0.6,100), points=1000)


#initialize Discrete Allocation to get a full picture what you could buy with a given amount
from pypfopt import DiscreteAllocation
from datetime import datetime

#create a dictionary which can be used for the Discrete Allocation as an input
weight_pf = {}
values_pf = tickers['Weight']
for ticker in tickers_list:
    weight_pf[ticker] = values_pf[ticker]
print(weight_pf)

funds = funds_q

da = data.DataReader(tickers_list, data_source='yahoo', start=datetime(2020,1,28), end=datetime.today())['Adj Close']

latest_price = da.iloc[-1,:]

alloc = DiscreteAllocation(weight_pf,latest_prices=latest_price, total_portfolio_value=funds)
allocation, leftover = alloc.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

#pie_alloc = pd.Series(allocation).plot.pie(figsize=(10,10))
#plt.show()

#barchart of the Discrete Allocation
barchart_alloc = pd.Series(allocation).sort_values(ascending=True).plot.barh(figsize=(10,6))
plt.show()

######
#monte carlo simulation to get VaR95

###calculate portfolio_return and portfolio standard deviation
s = datetime(2018,2,5)
e = datetime.today()

portfolio_data =  data.DataReader(tickers_list, data_source='yahoo',
                               start = s ,
                             end= e )['Adj Close']



portfolio_return_historic = portfolio_data.pct_change().dropna()
portfolio_return_historic_mean = portfolio_return_historic.mean()
portfolio_return_historic_std = portfolio_return_historic.std()


#annualised_portfolio_return = mean_portfolio_return *252
#annualised_portfolio_cov = portfolio.pct_change().dropna().cov() *252

return_std = database['Annualized Std'].reset_index()
return_std = return_std[return_std['Symbol'].isin(mu_target_tickers.index)]
return_std = return_std.set_index('Symbol')
return_std = return_std.iloc[:,-1]


#portfolio_cov = portfolio_return.cov()
#portfolio_shrinkage = risk_models.CovarianceShrinkage(portfolio, returns_data=False).ledoit_wolf()

#portfolio return added to the portfolio_return
portfolio_return_historic['Portfolio'] = portfolio_return_historic.mean(axis=1)

portfolio_return = mu_target_tickers.dot(tickers['Weight'])

portfolio_std = np.sqrt(np.dot(tickers['Weight'].T,np.dot(cov_matrix_tickers, tickers['Weight'])))


print('Portfolio expected annualised return is ' + str(portfolio_return) + ' with a standard deviation of ' + str(portfolio_std))#


################
#plotting efficient frontier
#plot efficient frontier
#cla_portfolio = CLA(annualised_portfolio_return, annualised_portfolio_cov)
#cla_portfolio.max_sharpe()

#portfolio_frontier = plotting.plot_efficient_frontier(cla_portfolio, ef_param='risk', show_assets=True)
#cla_portfolio.portfolio_performance(verbose=True)


####################################################
#get a monte carlo simulation for the portfolio return
# Parametric
# Initialize Monte Carlo parameters

monte_carlo_runs = 1000
days_to_simulate = 5
loss_cutoff      = 0.95        # count any losses larger than 5% (or -5%)


compound_returns  = return_std.copy()
total_simulations = 0
bad_simulations   = 0

for run_counter in range(0,monte_carlo_runs):   # Loop over runs    
    for i in tickers_list:                      # loop over tickers, below is done once per ticker
        
        # Loop over simulated days:
        compounded_temp = 1
        
        for simulated_day_counter in range(0,days_to_simulate): # loop over days
            
            # Draw from 𝑁~(𝜇,𝜎)
            ######################################################
            simulated_return = np.random.normal(portfolio_return_historic_mean[i],portfolio_return_historic_std[i],1)
            ######################################################
            
            compounded_temp = compounded_temp * (simulated_return + 1)        
        
        compound_returns[i] = compounded_temp     # store compounded returns
    
    # Now see if those returns are bad by combining with weights
    portfolio_return_mc = compound_returns.dot(tickers['Weight']) # dot product
    
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
sns.histplot(data  = portfolio_return_historic['Portfolio'], # data set - index Facebook (or AAPL or GOOG)
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
VaR_95 = np.percentile(portfolio_return_historic, 5)


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

#after we can do the backtesting (either here or in portfolio visualizer)
#backtesting with bt backtrader 
#incorporate backtesting in our project

########################################################
# Import the bt package so we can use the backtesting functions
import bt

# Intead of using a date to get the data in every call, I set up a variable here
# It is usually better to use variables so you can change things later in ONE place
# rather than many places. 
beginning = '2015-01-01'

# Import data
data_bt = bt.get(tickers_list, start=beginning, end= e)

# We will need the risk-free rate to get correct Sharpe Ratios 
riskfree =  bt.get('^IRX', start=beginning)
# Take the average of the risk free rate over entire time period
riskfree_rate = float(riskfree.mean()) / 100
# Print out the risk free rate to make sure it looks good
print(riskfree_rate)

s_mark = bt.Strategy('Portfolio', 
                       [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighEqually(),
                       bt.algos.Rebalance()])

#s_mark = bt.Strategy('Portfolio', 
                     #  [bt.algos.RunEveryNPeriods(9, 3),
                    #   bt.algos.SelectAll(),
                    #   bt.algos.WeighMeanVar(),
                    #   bt.algos.Rebalance()])

b_mark = bt.Backtest(s_mark, data_bt)


# Fetch some data
data_sp500 = bt.get('spy,agg', start=beginning, end= s)

# Recreate the strategy named First_Strat
b_sp500 = bt.Strategy('SP500', [bt.algos.RunMonthly(),
                                     bt.algos.SelectAll(),
                                     bt.algos.WeighEqually(),
                                     bt.algos.Rebalance()])

# Create a backtest named test
sp500_test = bt.Backtest(b_sp500, data_sp500)

#run the backtest
result = bt.run(b_mark, sp500_test)

#create the run only for the b_mark
result_1 = bt.run(b_mark)

#result = bt.run(b_mark, b_inv, b_random, b_best, b_sp500)
result.set_riskfree_rate(riskfree_rate)
result.plot()

#show histogram
#result_1.plot_histogram()


# Show some performance metrics
result.display()






    



   
    
    
    
