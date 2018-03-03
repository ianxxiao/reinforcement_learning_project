### What are we building?

We try to develop an intelligent agent to re-balance shared bikes using Reinforcement Learning and CitiBike data. 

### How is bike balancing being done now?
Bike accumulates or depletes at certain popular locations. Companies, such as CitiBike, spend lots of effort and money to manage bike stock at each station to ensure availability to riders throughout the day. This operaiton is currently orchestrated by human operators. 

### How do we envision bike balancing to be done?
We aim to develop an computer agent that is able to decide how many bikes to move and to where without hard coding any rules. The only thing we specify is the reward, which the agent will receive if it manages to keep the number of bikes to be less than 50 at the end of each day. 

### What will be the benefit and impact of our work?
Hopefully the agent can be more precise, timely, and optimized than human agents. If we are successful, this is one step closer to creating a true smart city powered by Artificial Intelligence. 

The current version only has basic mechanics and is still under development with some known issues. 

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
