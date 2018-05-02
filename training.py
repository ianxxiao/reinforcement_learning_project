#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:35:23 2018

@author: Ian, Prince, Brenton, Alex

This creates a class for training session with the following methods:
    - start()
    - train_operator()
    - get_timestamp()
    - cal_performance()
    - save_session_results()
    - reset_episode_action_history()

"""

import numpy as np
import matplotlib.pyplot as plt
from env import env
from rl_brain import agent
from dqn import DeepQNetwork
import datetime
import os

class trainer():
    
    def __init__(self, station_history):
        
        # Session Properties
        self.episodes = []
        self.stock_type = ""
        self.logging = False
        self.env_debug = False
        self.rl_debug = False
        self.bike_station = None
        self.operator = None
        self.sim_stock = []
        self.model_based = False
        self.ID = None
        self.method = None
        self.station_history = station_history
        
        # Performance Metric
        self.success_ratio = 0
        self.rewards = []  # [[r from session 1], [r from session 2] ...]
        self.avg_rewards = [] #[np.mean([r from session 1]), np.mean([r from session 2])...]
        self.final_stocks = [] # [[stock from session 1], [stock from session 2] ...]
        self.episode_action_history = []
        self.episode_stock_history = []
        self.session_action_history = []
        self.session_stock_history = []
        self.q_tables = []
        self.actions = [-10, -3, -1, 0]
        
    
    def start(self, episodes, stock_type, logging, env_debug, rl_debug, brain, ID, model_based):
        #brain: which method to use. Q learning vs DQN
        
        self.episodes = episodes
        self.stock_type = stock_type
        self.logging = logging
        self.env_debug = env_debug
        self.rl_debug = rl_debug
        self.brain = brain
        self.ID = ID
        self.model_based = model_based
        
        if brain == 'q' and model_based == False:
            self.method = 'QLN'
        elif brain == 'q' and model_based == True:
            self.method = 'FCT'
        else:
            self.method = 'DQN'
        
        idx = 0
        
        for eps in self.episodes:
        
            # Initiate new evironment and RL agent
            self.bike_station = env(self.stock_type, debug = self.env_debug, ID = self.ID,
                                    station_history = self.station_history)
            self.sim_stock.append(self.bike_station.get_sim_stock())

            if self.brain == 'q':
                self.operator = agent(epsilon = 0.9, lr = 0.01, gamma = 0.9, 
                                  current_stock = self.bike_station.current_stock(), 
                                  debug = self.rl_debug,
                                  expected_stock = self.bike_station.get_expected_stock(),
                                  model_based = model_based)
            elif self.brain == 'dqn':
                self.operator = DeepQNetwork(self.bike_station.n_actions, self.bike_station.n_features, 0.01, 0.9)
            else:
                print("Error: pick correct brain")
                break
            
            # Train the RL agent and collect performance stats
            rewards, final_stocks = self.train_operator(idx, len(self.episodes), eps,
            logging = self.logging, brain = self.brain, model_based = self.model_based)
            
            # Log the results from this training session
            self.rewards.append(rewards)
            self.avg_rewards.append(np.mean(rewards))
            self.final_stocks.append(final_stocks)
            #self.q_tables.append(self.operator.get_q_table())
            self.session_action_history.append(self.episode_action_history)
            self.session_stock_history.append(self.episode_stock_history)
            self.reset_episode_history()
            
            # Destroy the environment and agent objects
            self.bike_station = None
            self.operator = None
            
            idx += 1
        
        if logging == True:
            if self.brain == 'q':
                self.save_session_results(self.get_timestamp(replace = True))
            else:
                self.save_session_results_dqn(self.get_timestamp(replace = True))
            
        return
    
    
    def train_operator(self, idx, num_sessions, episodes, logging, brain, model_based):
    
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
        step = 0
        
        for eps in range(episodes):
            
            self.bike_station.reset()
                
            while True:
                
                # Agent picks an action (number of bikes to move)
                # Agent sends the action to bike station environment
                # Agent gets feedback from the environment (e.g. reward of the action, new bike stock after the action, etc.)
                # Agent "learn" the feedback by updating its Q-Table (state, action, reward)
                # Repeat until end of day (23 hours)
                # Reset bike station environment to start a new day, repeat all
                
                
                if self.brain == 'q':
                    action = self.operator.choose_action(self.bike_station.get_old_stock(),
                                                     self.bike_station.get_expected_stock())
                    current_hour, old_stock, new_stock, expected_stock, _, reward, done, game_over = self.bike_station.ping(action)

                else:
                    action = self.operator.choose_action(self.bike_station.get_old_stock())
                    current_hour, old_stock, new_stock, reward, done = self.bike_station.ping_dqn(action)
                    self.operator.store_transition(old_stock, action, reward, new_stock)
                    if step > 50 and (step % 10 == 0):
                        self.operator.learn()

                #observation_, reward, done = self.bike_station.ping(action)
                if done == True:
                    
                    print("{} of {} Session | Episode: {} | Final Stock: {} |Final Reward: {:.2f}".format(idx, 
                          num_sessions, eps, old_stock, rewards))

                    
                    reward_list.append(rewards)
                    final_stocks.append(old_stock)
                    rewards = 0
                    
                    # Log session action history by episode
                    if brain == 'q':
                        self.episode_action_history.append(self.operator.get_hourly_actions())
                        self.episode_stock_history.append(self.operator.get_hourly_stocks())
                        self.operator.reset_hourly_history()
                    else:
                        self.episode_stock_history.append(self.operator.get_hourly_stocks());
                        self.operator.reset_hourly_history()
                                    
                    break


                if brain == 'q':

                    self.operator.learn(old_stock, action, reward, new_stock, expected_stock, game_over)


                step +=1
                rewards += reward
                
                # Log hourly action history by each episode


            with open('dqn_log.txt', 'a') as f:
                f.write("{} of {} Session | Episode: {} | Final Stock: {} |Final Reward: {:.2f} \n".format(idx, 
                    num_sessions, eps, old_stock, rewards))

                            
        return reward_list, final_stocks
    
    def get_timestamp(self, replace):
        
        if replace == True:
        
            return str(datetime.datetime.now()).replace(" ", "").replace(":", "").\
                        replace(".", "").replace("-", "")
        
        else:
            
            return str(datetime.datetime.now())
    
    
    def reset_episode_history(self):
        
        self.episode_action_history = []
        self.episode_stock_history = []
        
    
    def cal_performance(self):
        
        successful_stocking = []
        
        print("===== Performance =====")
        
        for session in range(len(self.final_stocks)):
            length = len(self.final_stocks[session])
            num_overstock = np.count_nonzero(np.array(self.final_stocks[session]) > 50)
            num_understock = np.count_nonzero(np.array(self.final_stocks[session]) <= 0)
            ratio = (length - num_understock - num_overstock)*100 / length
            
            print("Session {} | Overstock {} Times | Understock {} Times | {}% Successful".format(session, num_overstock, 
                  num_understock, ratio))

            average_reward = round(self.avg_rewards[session], 2)
            print("Average Episode Reward for Session: {}".format(average_reward))
            
            successful_stocking.append(ratio)
        
        return successful_stocking
    
    
    def save_session_results(self, timestamp):
        
        '''
        This function logs the following: 
            - overall success ratio of each session
            - line chart of success ratio by session
            - line chart of reward history by session
            - Q Table of each session
            - Comparison Line Chart of First and Last Episode Hourly Actions
        '''
        
        # --- create a session folder ---
        dir_path = "./performance_log/" + timestamp
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        successful_stocking = self.cal_performance()
        
        # --- Write Success Rate to File ---
        fname = dir_path + "/success_rate - " + timestamp + ".txt"
        
        with open(fname, 'w') as f:
            
            f.write("Logged at {}".format(self.get_timestamp(replace = False)))
            f.write("\n")
            f.write("This training session ran episodes: {}".format(self.episodes))
            f.write("\n")
        
            for session in range(len(successful_stocking)):
                f.write("Session {} | Episodes: {} | Success Rate: {:.2f}%".format(session, 
                        self.episodes[session], successful_stocking[session]))
                f.write("\n")
        
        # --- Plot Overall Success Rate by Episode ---
        
        title = "% of Successful Rebalancing - " + timestamp
        
        fig1 = plt.figure()
        plt.plot(self.episodes, successful_stocking)
        plt.xlabel("Episodes")
        plt.ylabel("% Success Rate")
        plt.title(title)
        fig1.savefig(dir_path + "/session_success_rate_" + timestamp)
        
        # --- Plot Reward History by Training Session ---
        
        for session in range(len(self.rewards)):
            
            fig = plt.figure(figsize=(10, 8))
            
            title = "Reward History by Training Session " + str(session) + " - " + timestamp
            
            x_axis = [x for x in range(self.episodes[session])]
            plt.plot(x_axis, self.rewards[session], label = "Session "+str(session))
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title(title)
            fig.savefig(dir_path + "/reward_history_session_" + \
                        str(session) + timestamp)
            
        # --- Save Q tables --- 
        
        for session in range(len(self.q_tables)):
            
            self.q_tables[session].to_csv(dir_path + "/q_table_session_" + \
                        str(session) + timestamp + ".csv")
        
        # --- Comparison Line Chart of First and Last Episode for each Session ---
        
        file_path = dir_path + "/action_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        
        for session in range(len(self.session_action_history)):
            
            first_eps_idx = 0
            last_eps_idx = len(self.session_action_history[session])-1
            
            fig = plt.figure(figsize=(10, 8))
            title = "Session " + str(session) + " - Hourly Action of Eps " + str(first_eps_idx) + " and Eps " + str(last_eps_idx)
            
            x_axis = [x for x in range(len(self.session_action_history[session][0]))]
            plt.plot(x_axis, self.session_action_history[session][0], label = "Eps 0")
            plt.plot(x_axis, self.session_action_history[session][-1], 
                     label = "Eps " + str(last_eps_idx))
            
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bikes Moved")
            plt.title(title)
            
            fig.savefig(file_path + "/action_history_" + str(session) + timestamp)
        
        
        # --- Comparison Line Chart of Simulated and Rebalaned Bike Stock --- #
        file_path = dir_path + "/stock_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        
        for session in range(len(self.session_stock_history)):
            
            first_eps_idx = 0
            last_eps_idx = len(self.session_action_history[session])-1
            
            fig = plt.figure(figsize=(10, 8))
            title = "[" + self.method + "]" + "Session " + str(session) + " - Original vs. Balanced Bike Stock after " + str(first_eps_idx) + " and Eps " + str(last_eps_idx)
            
            x_axis = [x for x in range(len(self.session_stock_history[session][0]))]
            plt.plot(x_axis, self.sim_stock[session], label = "Original without Balancing")
            plt.plot(x_axis, self.session_stock_history[session][0], label = "Balanced Bike Stock - Eps 0")
            plt.plot(x_axis, self.session_stock_history[session][-1], 
                     label = "Balanced Bike Stock - Eps " + str(last_eps_idx))
            
            plt.axhline(y = 50, c = "r", ls = "--", label = "Upper Stock Limit")
            plt.axhline(y = 0, c = "r", ls = "--", label = "Lower Stock Limit")
            
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bike Stock")
            plt.title(title)
            
            fig.savefig(file_path + "/stock_history_" + str(session) + timestamp)
        
        return

    def save_session_results_dqn(self, timestamp):
        dir_path = "./performance_log/" + timestamp
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
        # --- Comparison Line Chart of Simulated and Rebalaned Bike Stock --- #
        file_path = dir_path + "/stock_history"
        
        if not os.path.exists(file_path):
            os.makedirs(file_path)       
        
        
        successful_stocking = self.cal_performance()
        
        # --- Write Success Rate to File ---
        fname = dir_path + "/success_rate - " + timestamp + ".txt"
        
        with open(fname, 'w') as f:
            
            f.write("Logged at {}".format(self.get_timestamp(replace = False)))
            f.write("\n")
            f.write("This training session ran episodes: {}".format(self.episodes))
            f.write("\n")
        
            for session in range(len(successful_stocking)):
                f.write("Session {} | Episodes: {} | Success Rate: {:.2f}%".format(session, 
                        self.episodes[session], successful_stocking[session]))
                f.write("\n")

            # --- Plot Overall Success Rate by Episode ---
        
        title = "% of Successful Rebalancing - " + timestamp
        
        fig1 = plt.figure()
        plt.plot(self.episodes, successful_stocking)
        plt.xlabel("Episodes")
        plt.ylabel("% Success Rate")
        plt.title(title)
        fig1.savefig(dir_path + "/session_success_rate_" + timestamp)

        for session in range(len(self.session_stock_history)):
            
            first_eps_idx = 0
            last_eps_idx = len(self.session_stock_history[session])-1
            
            fig = plt.figure(figsize=(10, 8))
            title = "[" + self.method + "]" + " Session " + str(session) + " - Original vs. Balanced Bike Stock after " + str(first_eps_idx) + " and Eps " + str(last_eps_idx)
            
            x_axis = [x for x in range(len(self.session_stock_history[session][0]))]
            plt.plot(x_axis, self.sim_stock[session], label = "Original without Balancing")
            plt.plot(x_axis, self.session_stock_history[session][0], label = "Balanced Bike Stock - Eps 0")
            plt.plot(x_axis, self.session_stock_history[session][-1], 
                     label = "Balanced Bike Stock - Eps " + str(last_eps_idx))
            
            plt.axhline(y = 50, c = "r", ls = "--", label = "Upper Stock Limit")
            plt.axhline(y = 0, c = "r", ls = "--", label = "Lower Stock Limit")
            
            plt.legend()
            plt.xlabel("Hours")
            plt.ylabel("Number of Bike Stock")
            plt.title(title)
            
            fig.savefig(file_path + "/stock_history_" + "DQN" + str(session) + timestamp)
        
        return
    
    
