#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:54:01 2018

@author: Ian, Prince, Brenton, Alex

This is the main workflow of the RL program.

"""

from training import trainer
import helper

if __name__ == "__main__":

    # Get Initial Parameters    
    episode_list, data, ID, brain, model_based, citi_df = helper.user_input()


    # Set Up a Training Environment

    if brain != 'all':
        trainer = trainer()
        trainer.start(episode_list, data, logging  = 
                      True, env_debug = False, rl_debug = False,
                      brain=brain, ID = ID, model_based = model_based)
    else:
        
        # TO DO: Brenton to add the citi_df as a parameter to trainer()
        # i.e. trainer(citi_df)
        # update the subsequent trainer init workflow
        # Run a end to end test
        
        trainer_QLN = trainer()
        trainer_FCT = trainer()
        trainer_DQN = trainer()
        
        trainer_QLN.start(episode_list, data, logging  = 
                      True, env_debug = False, rl_debug = False,
                      brain='q', ID = ID, model_based = False)
        
        trainer_FCT.start(episode_list, data, logging  = 
                      True, env_debug = False, rl_debug = False,
                      brain='q', ID = ID, model_based = True)
        
        trainer_DQN.start(episode_list, data, logging  = 
                      True, env_debug = False, rl_debug = False,
                      brain='dqn', ID = ID, model_based = False)