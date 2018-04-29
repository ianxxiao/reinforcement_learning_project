#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:44:36 2018

@author: Ian
"""

def user_input():
    
    '''
    This function creates all initial parameter for the training based on 
    user inputs.
    '''
    
    episode_list = [eps for eps in range(100, 150, 100)]
    
    # ---- Bike Stock Parameters ----
    # linear: a linear increasing bike stock with 3 additional bikes per hour
    # random: a linear increasing bike stock with random fluctuation
    # actual_1: randomly pick traffic from one citibike stations
    # -------------------------------
    
    data = input("Linear, Random, or Actual?: ").lower()
    
    ID = 392
    
    brain = input("Enter agent type (all, q or dqn): ").lower()
    
    model_based = None
    
    if brain == 'q':
        modeled = input("Model-based? Y or N: ").upper()
        if modeled == 'Y':
            model_based = True
        else:
            model_based = False
    
    if brain == 'dqn':
        model_based = False
    
    if data == 'actual':
        citi_df = citi_data_processing()
        
    else:
        citi_df = None
    
    return episode_list, data, ID, brain, model_based, citi_df


def citi_data_processing():
    
    # TO DO: Brenton to add
    
    # Dummy Value for Testing
    
    citi_df = None
    
    return citi_df