#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:23:50 2021

@author: fabianschweinzer
"""

##first steps project
#getting data from yahoo finance


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
#import seaborn as sns

#initialize dataset 

##read cleaned S&P 500 dataframe
database = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/S&P_Database_Final')
database.set_index('Symbol', inplace=True)

#extract expected return calculated with CAPM model

mu_capm = database.iloc[:,-1]

#initiate Cov Matrix 

cov_matrix = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/Cov_Matrix_SP500')
print(cov_matrix.head())
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
ef = EfficientFrontier(mu_capm, cov_matrix, weight_bounds=(0.0,0.2))
ef.add_objective(objective_functions.L2_reg, gamma=1)
weights = ef.max_sharpe()

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
mu_capm_25 = mu_capm[mu_capm.index.isin(df_1.index)]

#subsetting the cov matrix is more worrysome
#I used a transpose technique to do the subsetting on both axis
cov_matrix_25 = cov_matrix[cov_matrix.index.isin(df_1.index)]
cov_matrix_25 = cov_matrix_25.T
cov_matrix_25 = cov_matrix_25[cov_matrix_25.index.isin(df_1.index)]

#Effficient Frontier optimization for the top25 stocks
#we insert minimum weight limit, so every stocks gets a minimum weight of 1%
#we also include a maximum weight limit, so to make sure there is not too much importance on one particular stock

ef_25 = EfficientFrontier(mu_capm_25, cov_matrix_25, weight_bounds=(0.01, 0.2))
weights_25 = ef_25.max_sharpe()

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

cla_25 = CLA(mu_capm_25, cov_matrix_25)
cla_25.max_sharpe()

cla_25.portfolio_performance(verbose=True)

#plot the efficient frontier line
ax_25 = plotting.plot_efficient_frontier(cla_25, show_assets=False, show_fig=False )


####plot with ef function
risk_range = np.linspace(0.1, 0.4, 100)
ax_ef = plotting.plot_efficient_frontier(ef_25, ef_param='risk', ef_param_range=risk_range, show_assets=True, show_fig=True)








    



   
    
    
    
