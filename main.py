#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:27:09 2018

@author: Ian

This is the main workflow of initializing and training a RL agent and 
collecting analytics insights for reporting.

"""

from env import env
from rl_brain import agent
import matplotlib.pyplot as plt
import numpy as np
import datetime

def get_timestampe():
    
    return str(datetime.datetime.now()).replace(" ", "").replace(":", "").replace(".", "")

def save_training_results(reward_list, episodes):
    plt.plot(reward_list) 

def train_operator(episodes, logging):
    
    '''
    This function trains an RL agent by interacting with the bike station 
    environment. It also tracks and reports performance stats.
    Input:
        - episodes: a list of episodes in each training session
        (e.g. [50, 100, 200])
    Output:
        - success ratio: cnt of success / total number of episode
    '''
    
    print("Start training the Agent ...")
    rewards = 0
    reward_list = []
    
    for eps in range(episodes):
        
        bike_station.reset()
            
        while True:
            
            # Pick an Action, Collect Reward and New State, Decide a New Action, Repeat
            action = operator.choose_action(bike_station.get_old_stock())
            current_hour, old_stock, new_stock, reward, done = bike_station.ping(action)
            operator.learn(old_stock, action, reward, new_stock)
            
            rewards += reward
            
            if done == True:
                
                print("Episode: {} | Final Stock: {} |Final Reward: {}".format(eps, 
                      old_stock, rewards))
                
                reward_list.append(rewards)
                rewards = 0
                                
                break
    
    if logging == True:
                
        operator.export_q_table(episodes, get_timestampe())
            
    #plot_rewards(reward_list, episodes)
    
    cnt_success = np.count_nonzero(np.array(reward_list) > 0)
    cnt_fail = np.count_nonzero(np.array(reward_list) <= 0)
        
    return cnt_success / (cnt_success+cnt_fail)

if __name__ == "__main__":

    ratios = []
    success_ratio = 0
    
    # Experiment different training episodes in each Training Session
    
    for eps in [10000, 15000, 20000, 25000, 50000, 100000]:
        
        # Initiate new evironment and RL agent
        bike_station = env("linear", debug = False)
        operator = agent(epsilon = 0.9, lr = 0.01, gamma = 0.9, debug = False)
        
        # Train the RL agent and collect performance stats
        success_ratio = train_operator(eps, logging = True)
        ratios.append(success_ratio)
        
        # Destroy the environment and agent objects
        del bike_station
        del operator
        
    print(ratios)