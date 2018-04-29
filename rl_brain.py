#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:26:48 2018

@author: Ian

This script is for creating a RL agent class object. This object has the 
following method:
    
    1) choose_action: this choose an action based on Q(s,a) and greedy eps
    2) learn: this updates the Q(s,a) table
    3) check_if_state_exist: this check if a state exist based on env feedback

"""

import numpy as np
import pandas as pd

class agent():
    
    
    def __init__(self, epsilon, lr, gamma, current_stock, debug):
        
        print("Created an Agent ...")
        self.actions = [-10, -3, -1, 0]
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.debug = debug
        self.current_stock = current_stock
        
        # performance metric
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)
        self.hourly_action_history = []
        self.hourly_stock_history = []
       
    def choose_action(self, s):
        
        '''
        This funciton choose an action based on Q Table. It also does 
        validation to ensure stock will not be negative after moving bikes.
        Input: 
            - s: current bike stock
        Output:
            - action: number of bikes to move
        
        '''
        
        self.check_state_exist(s)
        self.current_stock = s
        
        # find valid action based on current stock 
        # cannot pick an action that lead to negative stock
        
        # !!!! remove action validation; only rely on reward/penalty !!!
        # valid_state_action = self.find_valid_action(self.q_table.loc[s, :])
        
        valid_state_action = self.q_table.loc[s, :]
                
        if np.random.uniform() < self.epsilon:
                        
            try:
                # find the action with the highest expected reward
                
                valid_state_action = valid_state_action.reindex(np.random.permutation(valid_state_action.index))
                action = valid_state_action.idxmax()
            
            except:
                # if action list is null, default to 0
                action = 0
                        
            if self.debug == True:
                print("Decided to Move: {}".format(action))
                        
        else:
            
            # randomly choose an action
            # re-pick if the action leads to negative stock
            try:
                action = np.random.choice(valid_state_action.index)
            except:
                action = 0
            
            if self.debug == True:
                print("Randomly Move: {}".format(action))
        
        self.hourly_action_history.append(action)
        self.hourly_stock_history.append(s)
        
        return action
 
    
    def learn(self, s, a, r, s_, g):
        
        '''
        This function updates Q tables after each interaction with the
        environment.
        Input: 
            - s: current bike stock
            - a: current action (number of bikes to move)
            - r: reward received from current state
            - s_: new bike stock based on bike moved and new stock
        Output: None
        '''
        
        if self.debug == True:
            print("Moved Bikes: {}".format(a))
            print("Old Bike Stock: {}".format(s))
            print("New Bike Stock: {}".format(s_))
            print("---")
        
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        
        if g == False:
            
            # Updated Q Target Value if it is not end of day  
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        
        else:
            
            # Update Q Target Value as Immediate reward if end of day
            q_target = r
            
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        
        return

    
    def check_state_exist(self, state):
        
        # Add a new row with state value as index if not exist
        
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.actions), 
                        index = self.q_table.columns,
                        name = state
                        )
                )
        
        return
    

    def find_valid_action(self, state_action):
        
        '''
        This function check the validity acitons in a given state.
        Input: 
            - state_action: the current state under consideration
        Output:
            - state_action: a pandas Series with only the valid actions that
                            will not cause negative stock
        '''
        
        # remove action that will stock to be negative
        
        for action in self.actions:
            if self.current_stock + action < 0:
                
                if self.debug == True:
                    print("Drop action {}, current stock {}".format(action, self.current_stock))
                
                state_action.drop(index = action, inplace = True)
        
        return state_action
        
    
    def print_q_table(self):
        
        print(self.q_table)


    def get_q_table(self):
        
        return self.q_table

    
    def get_hourly_actions(self):
        
        return self.hourly_action_history
    
    def get_hourly_stocks(self):
        
        return self.hourly_stock_history

    
    def reset_hourly_history(self):
        
        self.hourly_action_history = []
        self.hourly_stock_history = []