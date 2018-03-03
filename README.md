### Overview

We try to develop an intelligent agent to re-balance shared bikes using Reinforcement Learning and CitiBike data. The current version is still 
under development with basic features. 

### Key Features

1) **main.py**
This is the main workflow of initializing and training a RL agent and collecting analytics insights for debugging or reporting.

How to run: python main.py

2) **env.py**
This script is for creating an Environment class. Each environment represents
a bike stateion with the following methods:
    1) generate(): this initializes bike station with stock trend (e.g. linear, random, or based on real CitiBike history)
    2) ping(): this provides feedback to RL Agent with current stock info, reward, and episode termination status; it also simulates the internal clock of different hours of the day
    3) update: this updates the bike stock based on RL Agent Action
    4) reset: reset all environment properties for new episode training

3) **rl_brain.py**
This script is for creating a RL agent class object. This object has the 
following method:
    
    1) choose_action(): this choose an action based on Q(s,a) and greedy epsilon
    2) learn(): this updates the Q(s,a) table
    3) check_if_state_exist(): this check if a state exist based on environment feedback; create new state if it does not exist
    
### Known Issues Under Investigation
- success ratio (# of simulation with less than 50 bikes at hour 23 / total simulation) decreases as total simulation episode increases; ideally it should increase if the agent is learning properly
