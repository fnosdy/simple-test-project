# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:48:42 2016

@author: Su Danyang
"""

import numpy as np
import random as rnd
from scipy.interpolate import InterpolatedUnivariateSpline as extrap
from itertools import compress

def bid_eq(mu,sigma,rho,T,T_burn_in,grid,bid,num_auction,num_entrant):
    bid_old = bid
    dist = 1
    while dist > 0.05:
        bid_data,M = get_bid_info(mu,sigma,rho,T,T_burn_in,grid,bid_old,num_auction,num_entrant)
        bid_new = np.array(bid_optimizer(bid_data,grid,bid_old,rho))
        dist = np.mean(np.abs(bid_old-bid_new))
        bid_old = 1/2*bid_old + 1/2*bid_new
        print('Distance is {my_dist}.'.format(my_dist=dist))
    return bid, M
    
def obj_function(theta,M_norm,M_data,mu,sigma):
    mu_norm = theta[0];
    sigma_norm = theta[1];
    M_theta = [0 for ct in range(4)];
    M_theta[0] = mu_norm + sigma_norm/sigma*(M_norm[0]-mu);
    M_theta[1] = sigma_norm*M_norm[1]/sigma;
    M_theta[2] = mu_norm + sigma_norm/sigma*(M_norm[2]-mu);
    M_theta[3] = sigma_norm*M_norm[3]/sigma;
    G = sum([(M_theta[ct]-M_data[ct])**2 for ct in range(4)])
    return G
    
def get_M(auction_data):
    M_data = np.zeros(4)
    auction_data = auction_data.replace([np.inf, -np.inf], np.nan)
    exec('M_data[0] = auction_data.{name}.mean(0)'.format(name=auction_data.columns.values[1]))
    exec('M_data[1] = auction_data.{name}.std(0)'.format(name=auction_data.columns.values[1]))
    exec('M_data[2] = auction_data.{name}.mean(0)'.format(name=auction_data.columns.values[2]))
    exec('M_data[3] = auction_data.{name}.std(0)'.format(name=auction_data.columns.values[2]))
    return M_data
    
def get_bid_info(mu,sigma,rho,T,T_burn_in,grid,bid,num_auction,num_entrant):
    valuations = [rnd.normalvariate(mu,sigma) for ct in range(1000)]
    bids = extrap(grid, bid, k=1)(valuations)
    
    #initialize data
    bid_data = []
    for i in range(T_burn_in+T):
        num_bidder = len(valuations)
        b_type = [[valuations[ct], bids[ct]] for ct in range(num_bidder)]
        match = [rnd.randint(0,num_auction-1) for ct in range(num_bidder)]
        
        num_active_auction = [0 for ct in range(num_auction)]
        # Enter their bids into data
        current_bids = [[0,0] for ct in range(num_auction)]
        keep_flag = [1 for ct in range(num_bidder)]
        for auc in range(num_auction):
            match_flag = [1 if ct==auc else 0 for ct in match]
            current_type = sorted(compress(b_type,match_flag),key=lambda y:y[1],reverse=True)     
            num_active_auction[auc] = len(current_type)
            if num_active_auction[auc] > 1:
                current_bids[auc] = [current_type[0][1],current_type[1][1]]
                keep_flag = [0 if (ct==auc and ct2[1]==current_bids[auc][0]) else ct3 for ct,ct2,ct3 in zip(match,b_type,keep_flag)] #winner exit flag
            else:
                current_bids[auc] = [np.inf,np.inf]
            
        keep_flag = [0 if (rnd.uniform(0,1)>rho) else keep_flag[ct] for ct in range(num_bidder)]
        b_type = list(compress(b_type,keep_flag))
        entrant_vals = [rnd.normalvariate(mu,sigma) for ct in range(num_entrant)]
        entrant_bids = extrap(grid, bid, k=1)(entrant_vals)
        valuations = [b_type[ct][0] for ct in range(len(b_type))]
        valuations.extend(entrant_vals)
        bids = [b_type[ct][1] for ct in range(len(b_type))]
        bids.extend(entrant_bids)
        bid_data.extend([[i,current_bids[ct][0],current_bids[ct][1],num_active_auction[ct]] for ct in range(num_auction)])
    no_drop_flag = [0 if (bid_data[ct][0]<T_burn_in or bid_data[ct][3]==0) else 1 for ct in range(len(bid_data))]
    M = []
    M.append(np.mean([x[1] for x in bid_data if x[1]<np.inf]))
    M.append(np.std([x[1] for x in bid_data if x[1]<np.inf]))
    M.append(np.mean([x[2] for x in bid_data if x[2]<np.inf]))
    M.append(np.std([x[2] for x in bid_data if x[2]<np.inf]))
    return compress(bid_data,no_drop_flag),M

def bid_optimizer(bid_data,grid,bid,rho):
    period,first_bid,second_bid,num_active_auction = list(zip(*bid_data))
    counter_bid = list(first_bid + second_bid)
    auction_weight = [ct - 1 for ct in num_active_auction] + [1 for ct in num_active_auction]
    grid_length = len(grid)
    total_weight = sum(num_active_auction)
    dist = 1
    bid_old = list(bid)
    bid_new = [0 for ct in bid]
    gamma = 2
    p_w = [0 for ct in bid]
    ec_w = [0 for ct in bid]
    for i in range(grid_length):
        flag_win = [grid[i]>=ct for ct in counter_bid]
        p_w[i] = sum(compress(auction_weight,flag_win))/total_weight
        ec_w[i] = sum(compress([ct1*ct2 for ct1,ct2 in zip(counter_bid,auction_weight)],flag_win))/total_weight
    
    while dist>0.01:
        p_win = extrap(grid, p_w, k=1)(bid_old)
        ec_win = extrap(grid, ec_w, k=1)(bid_old)
        bid_new = [ct1 -rho*(ct1*ct2-ct3)/(1-(1-ct2)*rho) for ct1,ct2,ct3 in zip(grid,p_win,ec_win)]       
        dist = max([abs(ct1 - ct2) for ct1,ct2 in zip(bid_new,bid_old)])
        bid_old = [(1-1/gamma)*ct1 + 1/gamma*ct2 for ct1,ct2 in zip(bid_new,bid_old)]
    

    return bid_old                   
    