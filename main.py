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

def plot_rewards(reward_list, episodes):
    plt.plot(reward_list) 

def train_operator(episodes):
    
    print("Start training the Agent ...")
    reward_list = []
    rewards = 0
    
    for eps in range(episodes):
        
        bike_station.reset()
            
        while True:
            
            action = operator.choose_action(bike_station.get_old_stock())
            current_hour, old_stock, new_stock, reward, done = bike_station.ping(action)
            operator.learn(old_stock, action, reward, new_stock)
            rewards += reward
            
            if done == True:
                
                print("Episode: {} | Final Stock: {} |Final Reward: {}".format(eps, 
                      old_stock, reward))
                reward_list.append(rewards)
                rewards = 0
                
                break
    
    #plot_rewards(reward_list, episodes)
    
    cnt_success = np.count_nonzero(np.array(reward_list) == 5)
    cnt_fail = np.count_nonzero(np.array(reward_list) == -1)
    
    return cnt_success / (cnt_success+cnt_fail)

if __name__ == "__main__":

    ratios = []
    success_ratio = 0
    
    for eps in (15000, 30000, 50000, 80000, 100000):
        
        bike_station = env("linear", debug = False)
        operator = agent(epsilon = 0.8, lr = 0.01, gamma = 0.9, debug = False)
        
        success_ratio = train_operator(eps)
        ratios.append(success_ratio)
        
        del bike_station
        del operator
        
    print(ratios)