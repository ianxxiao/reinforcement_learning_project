#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 12:26:58 2018

@author: Ian, Prince, Brenton, Alex

This script is for creating an Environment class. Each environment represents
a bike stateion with the following methods:
    1) generate: this initialize bike station with stock characteristics
    2) ping: this communicates with RL Agent with current stock info, reward,
                and episode termination status; iterate to new hour
    3) update: this updates the bike stock based on RL Agent Action
    4) reset: reset all environment properties for new episode training

"""

import numpy as np
import pandas as pd

class env():
    
    def __init__(self, mode, debug):
        
        print("Creating A Bike Environment...")
        
        self.mode = mode
        self.seed = np.random.random_integers(0, 10)
        self.num_hours = 23
        self.current_hour = 0
        self.bike_stock_sim = self.generate_stock(mode) # original copy
        self.bike_stock = self.bike_stock_sim.copy() # to be reset to original copy every episode
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.debug = debug

        self.actions = [-10, -3, -1, 0]
        self.n_actions = len(self.actions)
        #features of the observation: hour, old stock, new stock
        self.n_features = 1

        self.citibike_df = 0

        
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
                
        if mode == "random":
            for i in range(1, 24):
                bike_stock.append(bike_stock[i-1] + 3 + np.random.random_integers(-5, 5))
                
        if mode == "actual_1":
            
            print("loading and processing citiBike Data ...")
            self.citibike_df = self.process_citibike(20)
            
            # randomly pick a bike station
            bike_stock = np.array(self.citibike_df.sample(n = 1))[0][4:28].tolist()
            
            print(bike_stock)
            
        return bike_stock
    
    
    def ping(self, action):
        
        # share back t+1 stock, reward of t, and termination status
        if self.debug == True:
            print("Current Hour: {}".format(self.current_hour))
            print("Current Stock: {}".format(self.bike_stock[self.current_hour]))
            print("Bikes Moved in Last Hour: {}".format(self.bike_moved))
            print("Collect {} rewards".format(self.reward))
            print("Will move {} bikes".format(action))
            print("---")
        
        if action != 0:
            self.update_stock(action)
            self.reward = 0.5*action
            
        if self.bike_stock[self.current_hour] > 50:
            self.reward = -30
            
        if self.bike_stock[self.current_hour] < 0:
            self.reward = -30
        
        if self.current_hour == 23:
            if self.bike_stock[self.current_hour] <= 50:
                self.reward = 20
            else: 
                self.reward = -20
            self.done = True
            self.new_stock = 'terminal'

        # update to next hour
        if self.current_hour != 23:
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
        self.bike_stock = self.bike_stock_sim.copy()
        self.done = False
        self.reward = 0
        self.bike_moved = 0
        self.old_stock = self.bike_stock[0]
        self.new_stock = 0
        #return (self.current_hour, self.old_stock, self.new_stock)
        
    def current_stock(self):
        
        return self.bike_stock[self.current_hour]
    
    def get_sim_stock(self):
        
        return self.bike_stock

    
    def process_citibike(self, starting_bal):
        
        # process real citi bike data from Sept 2017
        # calculate bike stock based on inflow and outflow trips
        # return a pandas dataframe of 
        
        print("Loading data from CitiBike...")
        bike = pd.read_csv("https://s3.amazonaws.com/tripdata/201709-citibike-tripdata.csv.zip")
        bike['starttime'] = pd.to_datetime(bike['starttime'], infer_datetime_format= True)
        bike['stoptime'] = pd.to_datetime(bike['stoptime'], infer_datetime_format= True)
        
        bike['day'] = bike['starttime'].dt.day
        bike['start_hour'] = bike['starttime'].dt.hour
        bike['end_hour'] = bike['stoptime'].dt.hour
        bike['DOW'] = bike['starttime'].dt.dayofweek
        
        # Create a dataset with all unique station id, name, and lat/lon
        
        uni_dep_stations = bike[['start station id', 'start station name', 
                         'start station latitude', 'start station longitude']].drop_duplicates()

        uni_arv_stations = bike[['end station id', 'end station name', 
                                 'end station latitude', 'end station longitude']].drop_duplicates()
        
        uni_dep_stations.columns = ["id", "name", "lat", "lon"]
        uni_arv_stations.columns = ["id", "name", "lat", "lon"]
        uni_station = pd.concat([uni_dep_stations, uni_arv_stations], axis = 0).drop_duplicates()
        uni_station.head()
        
        # Create hourly departure count by day across the month
        print("Calculating Departure and Arrivals ...")
        
        monthDep = pd.pivot_table(bike[['start station id', 'day','start_hour']],
                                     index = "start station id", columns = ['day', "start_hour"], 
                                     aggfunc = np.size, fill_value= 0).reset_index()
        
        monthDep.columns = ["dep_" + str(day) + "_" + str(hour) for day, hour in monthDep.columns]
        
        
        # Create hourly arrival count by day across the month

        monthArv = pd.pivot_table(bike[['end station id', 'day','end_hour']],
                                     index = "end station id", columns = ['day', "end_hour"], 
                                     aggfunc = np.size, fill_value= 0).reset_index()
        
        monthArv.columns = ["arv_" + str(day) + "_" + str(hour) for day, hour in monthArv.columns]
        
        # Create a hourly net flow count by day across the month 

        monthNet = uni_station.merge(monthDep, how = "left", left_on = "id", right_on = "dep_start station id_").\
                              merge(monthArv, how = "left", left_on = "id", right_on = "arv_end station id_").fillna(0)
        
        for day in range(1, 31):
                
            for hour in range(0, 24):
                
                try:
                    net_col = "net_" + str(day) + "_" + str(hour)
                    dep_col = "dep_" + str(day) + "_" + str(hour)
                    arv_col = "arv_" + str(day) + "_" + str(hour)
                    monthNet[net_col] = monthNet[arv_col] - monthNet[dep_col]
                except (KeyError):
                    print("Missing day: {} | Missing hour: {}".format(day, hour))
                    pass
        
        # Create a dataframe of bike stock amount based on starting balance
        df_citibike = self.calHourlyBal(monthNet, starting_bal)
        
                
        return df_citibike
    
    
    
    def calHourlyBal(self, df, starting_bal):
        
        print("Calculating Hourly Bike Stock for Each Station ...")
        hourBal = df
        
        # Calculate hourly bike balance based on starting stock
        for day in range(1, 31):
            for hour in range(0, 24):
                try:
                    
                    if day == 1 and hour == 0:
                        bal_col = "bal_1_0"
                        hourBal["bal_1_0"] = starting_bal
                        
                    elif day > 1 and hour == 0:
                        
                        bal_col = "bal_" + str(day) + "_" + str(hour)
                        last_bal_col = "bal_" + str(day-1) + "_23"
                        net_col = "net_" + str(day) + "_0"
                        
                        hourBal[bal_col] = hourBal[last_bal_col] + hourBal[net_col]
                    
                    else:
                        
                        bal_col = "bal_" + str(day) + "_" + str(hour)
                        last_bal_col = "bal_" + str(day) + "_" + str(hour-1)
                        net_col = "net_" + str(day) + "_" + str(hour)
                                            
                        hourBal[bal_col] = hourBal[last_bal_col] + hourBal[net_col]
                
                except (KeyError) as ex:
                    # use previous balance for missing time slot
                    print("Missing net flow at day {} hour {}".format(day, hour))
                    
                    #hourBal[bal_col] = hourBal[last_bal_col]
                    pass
        
        # Only keep balance and change columns
        bal_col = hourBal.columns[hourBal.columns.str.contains("bal_")]
        hourBal[bal_col] = hourBal[bal_col].astype('int')
        final_bal = pd.concat([hourBal[["id", "name", "lat", "lon"]], hourBal[bal_col]], axis = 1) 
        
        return final_bal

