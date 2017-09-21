# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:43:30 2016

@author: Su Danyang
"""


import numpy as np
import pandas as pd
from scipy.optimize import minimize
import auc_function as f
# import data
with open('auction_data.csv') as fp:
    auction_data = pd.read_csv(fp,header=None,dtype={'period': np.int, 'win_b': np.float32, 'second_b': np.float32, 'num_bidder': np.int})

auction_data.columns = ['period','win_b','second_b','num_bidder']
auction_data.loc[auction_data.win_b == 10000000,'win_b'] = np.inf
auction_data.loc[auction_data.second_b == 10000000,'second_b'] = np.inf
rho = 0.8
T = 10000
T_burn_in = 500

num_grid = 1000
num_auction = 10
num_entrant = 15

M_data = f.get_M(auction_data)

 
mu = M_data[0]
sigma = M_data[1]

grid = np.linspace(mu-5*sigma,mu+5*sigma,num_grid)
bid = grid
bid,M_norm = f.bid_eq(mu,sigma,rho,T,T_burn_in,grid,bid,num_auction,num_entrant)
theta_opt = minimize(f.obj_function, np.array([100,20]), args=(M_norm,M_data,M_data[0],M_data[1]))

