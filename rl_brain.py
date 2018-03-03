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
    
    
    def __init__(self, epsilon, lr, gamma, debug):
        
        print("Created an Agent ...")
        self.actions = [-3, -1, 0]
        self.reward = 0
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma
        self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)
        self.debug = debug
        
    def choose_action(self, observation):
        
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            # choose the best action given a state
            state_action = self.q_table.loc[observation, :]
            #state_action = state_action.reindex(np.random.permutation(state_action))
            action = state_action.idxmax()  
            
            if self.debug == True:
                print(state_action)
                print("Decided to Move: {}".format(action))
                        
        else:
            # randomly choose an action
            action = np.random.choice(self.actions)
            
            if self.debug == True:
                print("Randomly Move: {}".format(action))
        
        return action
    
    def learn(self, s, a, r, s_):
        
        if self.debug == True:
            print("Moved Bikes: {}".format(a))
            print("Old Bike Stock: {}".format(s))
            print("New Bike Stock: {}".format(s_))
            print("---")
        
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        
        if s_ != 'terminal':
           
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r
            
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)
        
        return 0
    
    def check_state_exist(self, state):
        
        if state not in self.q_table.index:
            
            self.q_table = self.q_table.append(
                pd.Series(
                        [0]*len(self.actions), 
                        index = self.q_table.columns,
                        name = state
                        )
                )
        
        return


    def print_q_table(self):
        
        print(self.q_table)