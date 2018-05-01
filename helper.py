#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 17:44:36 2018

@author: Ian, Brenton, Prince, Alex
"""

import pandas as pd
import numpy as np

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
    
    ID = 497
    
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
        station_history = citi_data_processing(ID)
        print(station_history)
        
    else:
        station_history = None
    
    return episode_list, data, ID, brain, model_based, station_history


def citi_data_processing(ID):
    
    citi_df = process_citibike(20)
    station_history = list(np.array(citi_df[citi_df['id'] == ID])[0][4:28])
    return station_history


def process_citibike(starting_bal):
        
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
        
    monthDep = pd.pivot_table(bike[['start station id', 'day','start_hour', 'starttime']],
                                     index = "start station id", columns = ['day', "start_hour"], 
                                     aggfunc = np.size, fill_value= 0).reset_index()
        
    monthDep.columns = ["dep_" + str(day) + "_" + str(hour) for _, day, hour in monthDep.columns]
        
        
    # Create hourly arrival count by day across the month

    monthArv = pd.pivot_table(bike[['end station id', 'day','end_hour', 'stoptime']],
                                     index = "end station id", columns = ['day', "end_hour"], 
                                     aggfunc = np.size, fill_value= 0).reset_index()
        
    monthArv.columns = ["arv_" + str(day) + "_" + str(hour) for _, day, hour in monthArv.columns]
        
    # Create a hourly net flow count by day across the month 

    monthNet = uni_station.merge(monthDep, how = "left", left_on = "id", right_on = "dep__").\
                              merge(monthArv, how = "left", left_on = "id", right_on = "arv__").fillna(0)
        
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
    df_citibike = calHourlyBal(monthNet, starting_bal)
        
    return df_citibike
    
    
    
def calHourlyBal(df, starting_bal):
        
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

