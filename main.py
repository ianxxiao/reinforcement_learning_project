#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:54:01 2018

@author: Ian, Prince, Brenton, Alex

This is the main workflow of the RL program.

"""

from training import trainer

if __name__ == "__main__":
    
    #DUMMY UPDATE --- IAN#

    #training dqn model
    #episode_list = [eps for eps in range(100, 10001, 1000)]
    #episode_list = [eps for eps in range(100, 5001, 1000)]
    #episode_list = [10, 10, 10]
    
    #trainer = trainer()


    #episode_list = [eps for eps in range(1000, 2000, 200)]
    #episode_list = [10, 10, 10]
    
    #trainer = trainer()
    #trainer.start(episode_list, "random", logging  = 
    #             False, env_debug = False, rl_debug = False, brain = 'q')

    
    # Train an Agent

    episode_list = [eps for eps in range(100, 150, 100)]
    ID = int(input('Enter station ID (integer): '))
    brain = input("Enter agent type (q or dqn): ")
    if brain == 'q':
        modeled = input("Model-based? Y or N: ")
        if modeled == 'Y':
            model_based = True
        else:
            model_based = False
    if brain == 'dqn':
        model_based = False

    
    trainer = trainer()
    
    # ---- Bike Stock Parameters ----
    # linear: a linear increasing bike stock with 3 additional bikes per hour
    # random: a linear increasing bike stock with random fluctuation
    # actual_1: randomly pick traffic from one citibike stations
    # -------------------------------
    

    trainer.start(episode_list, "actual_1", logging  = 
                  True, env_debug = False, rl_debug = False,
                  brain=brain, ID = ID, model_based = model_based)

    # Run an Agent
    
    # TO BE DEVELOPED
    # Run an agent on some new environments with Q Table learned from above
    # Measure performance on operating in new environment

