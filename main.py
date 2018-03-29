#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:54:01 2018

@author: Ian, Prince, Brenton, Alex

This is the main workflow of the RL program.

"""

from training import trainer

if __name__ == "__main__":
    
    #episode_list = [eps for eps in range(1000, 2000, 200)]
    episode_list = [50, 50, 50]
    
    trainer = trainer()
    trainer.start(episode_list, "random", logging  = 
                  False, env_debug = False, rl_debug = False)
