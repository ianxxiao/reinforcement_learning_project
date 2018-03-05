#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:35:23 2018

@author: Ian

This creates a class for training session.

"""

import numpy as np
import matplotlib.pyplot as plt
from env import env
from rl_brain import agent
import datetime

class trainer():
    
    def __init__(self):
        
        # Session Properties
        self.episodes = []
        self.stock_type = ""
        self.logging = False
        self.env_debug = False
        self.rl_debug = False
        self.bike_station = None
        self.operator = None
        
        # Performance Metric
        self.success_ratio = 0
        self.rewards = []  # [[r from session 1], [r from session 2] ...]
        self.final_stocks = [] # [[stock from session 1], [stock from session 2] ...]
    
    def start(self, episodes, stock_type, logging, env_debug, rl_debug):
        
        self.episodes = episodes
        self.stock_type = stock_type
        self.logging = logging
        self.env_debug = env_debug
        self.rl_debug = rl_debug
        
        for eps in self.episodes:
        
            # Initiate new evironment and RL agent
            self.bike_station = env(self.stock_type, debug = self.env_debug)
            self.operator = agent(epsilon = 0.9, lr = 0.01, gamma = 0.9, 
                                  current_stock = self.bike_stateion.current_stock(), 
                                  debug = self.rl_debug)
            
            # Train the RL agent and collect performance stats
            rewards, final_stocks = self.train_operator(eps, logging = self.logging)
            
            # Log the results from this training session
            self.rewards.append(rewards)
            self.final_stocks.append(final_stocks)
            
            # Destroy the environment and agent objects
            self.bike_station = None
            self.operator = None
        
        if logging == True:
            
            self.save_session_results(self.get_timestampe())
            
        return
    
    
    def train_operator(self, episodes, logging):
    
        '''
        This function trains an RL agent by interacting with the bike station 
        environment. It also tracks and reports performance stats.
        Input:
            - episodes: a int of episode to be trained in this session (e.g. 500)
        Output:
            - reward_list: a list of reward per episode in this sesison
            - final_stocks: a list of final stocks per episode in this session
        '''
        
        print("Start training the Agent ...")
        rewards = 0
        reward_list = []
        final_stocks = []
        
        for eps in range(episodes):
            
            self.bike_station.reset()
                
            while True:
                
                # Pick an Action, Collect Reward and New State, Decide a New Action, Repeat
                action = self.operator.choose_action(self.bike_station.get_old_stock())
                current_hour, old_stock, new_stock, reward, done = self.bike_station.ping(action)
                self.operator.learn(old_stock, action, reward, new_stock)
                
                rewards += reward
                
                if done == True:
                    
                    print("Episode: {} | Final Stock: {} |Final Reward: {}".format(eps, 
                          old_stock, rewards))
                    
                    reward_list.append(rewards)
                    final_stocks.append(old_stock)
                    rewards = 0
                                    
                    break
        
        if logging == True:
                    
            self.operator.export_q_table(episodes, self.get_timestampe())
                            
        return reward_list, final_stocks
    
    def get_timestampe(self):
        
        
        return str(datetime.datetime.now()).replace(" ", "").replace(":", "").\
                    replace(".", "").replace("-", "")
    
    
    def cal_performance(self):
        
        successful_stocking = []
        
        print("===== Performance =====")
        for session in range(len(self.final_stocks)):
            
            num_overstock = np.count_nonzero(np.array(self.final_stocks[session]) > 50)
            num_understock = np.count_nonzero(np.array(self.final_stocks[session]) <= 50)
            ratio = num_understock*100 / (num_overstock + num_understock)
            
            print("Session {} | Overstock {} Times | Understock {} Times | {}% Successful".format(session, num_overstock, 
                  num_understock, ratio))
            
            successful_stocking.append(ratio)
        
        return successful_stocking
    
    def save_session_results(self, timestamp):
        
        # Plot Overall Success Rate by Episode
        
        successful_stocking = self.cal_performance()
        
        title = "% of Successful Rebalancing - " + timestamp
        
        fig1 = plt.figure()
        plt.plot(self.episodes, successful_stocking)
        plt.xlabel("Episodes")
        plt.ylabel("% Success Rate")
        plt.title(title)
        fig1.savefig("./training_results/"+"session_success_rate_" + timestamp)
        
        # Plot Reward History by Training Session
        
        for session in range(len(self.rewards)):
            
            fig = plt.figure()
            
            title = "Reward History by Training Session " + str(session) + " - " + timestamp
            
            x_axis = [x for x in range(self.episodes[session])]
            plt.plot(x_axis, self.rewards[session], label = "Session "+str(session))
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(title)
            fig.savefig("./reward_history/" + "reward_history_" + timestamp)
        
        return
    
    
