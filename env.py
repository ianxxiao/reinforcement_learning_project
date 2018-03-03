#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:26:58 2018

@author: Ian

This script is for creating an Environment class. Each environment represents
a bike stateion with the following methods:
    1) generate: this initialize bike station with stock characteristics
    2) ping: this communicates with RL Agent with current stock info, reward,
                and episode termination status; iterate to new hour
    3) update: this updates the bike stock based on RL Agent Action
    4) reset: reset all environment properties for new episode training

"""

class env():
    
    def __init__(self, mode, debug):
        
        print("Creating A Bike Environment...")
        
        self.mode = mode
        self.num_hours = 23
        self.current_hour = 0
        self.bike_stock = self.generate_stock(mode)
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.debug = debug
        
        if self.debug == True:
            print("Generating Bike Stock: {}".format(self.mode))
            print("Bike Stock: {}".format(self.bike_stock))
        
    def generate_stock(self, mode):
        
        # generate a list of 24 hourly bike stock based on mode
        # mode: linear, random, real based on citiBike Data
        
        bike_stock = [20]
        
        if mode == "linear":
            for i in range(1, 24):
                bike_stock.append(bike_stock[i-1]+3)
        
        return bike_stock
    
    def ping(self, action):
        
        # share back t+1 stock, reward of t, and termination status
        if self.debug == True:
            print("Current Hour: {}".format(self.current_hour))
            print("Current Stock: {}".format(self.bike_stock[self.current_hour]))
            print("Bikes Moved in Last Hour: {}".format(self.bike_moved))
            print("Will move {} bikes".format(action))
            print("---")
        
        if action != 0:
            self.update_stock(action)
        
        if self.current_hour == 23:
            if self.bike_stock[self.current_hour] <= 50:
                self.reward = 5
            else: 
                self.reward = -1
            self.done = True
            self.new_stock = 'terminal'
        
        else: 
            # update to next hour
            self.update_hour()
            self.old_stock = self.bike_stock[self.current_hour - 1]
            self.new_stock = self.bike_stock[self.current_hour]
            
        return self.current_hour, self.old_stock, self.new_stock, self.reward, self.done
    
    def get_old_stock(self):
        
        return self.old_stock
    
    def update_stock(self, num_bike):
        
        # update bike stock based on RL Agent action at t
        if self.current_hour != 23:
            for hour in range(self.current_hour+1, len(self.bike_stock)):
                self.bike_stock[hour] += num_bike
                
            self.bike_moved = num_bike
        
        else:
            if self.debug == True:
                print("Last Hour. Cannot Move Bikes.")
            pass
        
        return
    
    def update_hour(self):
        
        # update current_hour 
        self.current_hour += 1
        
        if self.debug == True:
            print("Tick... Forwarded Current Hour")
                
        return
    
    def reset(self):
        
        if self.debug == True:
            print("Reset Environment ...")
        
        self.num_hours = 23
        self.current_hour = 0
        self.bike_stock = self.generate_stock(self.mode)
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0