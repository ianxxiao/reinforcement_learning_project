### What are we building?

We try to develop an intelligent agent to re-balance shared bikes using Reinforcement Learning and CitiBike data. 

### How is bike balancing being done now?
Bike accumulates or depletes at certain popular locations. Companies, such as CitiBike, spend lots of effort and money to manage bike stock at each station to ensure availability to riders throughout the day. This operaiton is currently orchestrated by human operators. 

### How do we envision bike balancing to be done?
We aim to develop an computer agent that is able to decide how many bikes to move and to where without hard coding any rules. The only thing we specify is the reward, which the agent will receive if it manages to keep the number of bikes to be less than 50 at the end of each day. 

### How do we measure success?
We measure if the computer agent can achieve the following without deliberate programming:  
1) learn the limit of 50 bikes by end of day and 
2) maintain the allowable bike stock with minimum cost, which ties to the number of bikes moved in a day
3) develop new rebalancing strategy if external objectives changes (e.g. bike stock limit, flucturating daily traffic flow, allowed actions)

### What will be the benefit and impact of our work?
Hopefully the agent can be more precise, timely, and optimized than human agents. If we are successful, this is one step closer to creating a true smart city powered by Artificial Intelligence. 

The current version only has basic mechanics and is still under development with some known issues. Refer to Issue page for details.

### What are the key thesis to be done in near term?
- Can deep learning technique help to improve the cognitive capbility of the Reinforcement Learning agent?
- Can time series forecasting technique help to predict future rewards better and improve decisioning capability?

---
### Preliminary Results and Baseline

The agent was able to limit bikes under 50 more oftern after interacting with the environment more.
**Figure 1**: Comparison of success ratio as the number of interaction increases
![image](/result_snapshot/session_success_rate_2018-03-05101044410867.png)

The agent was being more strategic when moving bikes out of stations after 2000 interactions.
**Figure 2**: Bike moving comparison between training episode 0 and 2000
![image](/result_snapshot/action_history_220180305223236794486.png)

The agent choose its action based on an intuitive Q Table it developed **independently** without active human programming.
**Figure 3**: Heatmap of expected reward of an action when given certain bike stocks
![image](/result_snapshot/q_table_explaination.png)

---

### Key Features

1) **main.py**
This is the main workflow of initializing and training a RL agent and collecting analytics insights for debugging or reporting.

How to run: python main.py

2) **env.py**
This script is for creating an Environment class. Each environment represents
a bike stateion with the following methods:
    1) generate(): this initializes bike station with stock trend (e.g. linear, random, or based on real CitiBike history)
    2) ping(): this provides feedback to RL Agent with current stock info, reward, and episode termination status; it also simulates the internal clock of different hours of the day
    3) update_stock(): this updates the bike stock based on RL Agent Action
    4) update_hour(): this updates the simulated hour of the day
    5) reset(): reset all environment properties for new episode training

3) **rl_brain.py**
This script is for creating a RL agent class object. This object has the following methods:
    
    1) choose_action(): this choose an action based on Q(s,a) and greedy epsilon
    2) learn(): this updates the Q(s,a) table
    3) check_if_state_exist(): this check if a state exist based on environment feedback; create new state if it does not exist
    4) print_q_table(): this prints the Q Table for debugging purposes
    5) export_q_table(): this export the Q Table to a csv to a local folder
    
4) **training.py**
This script is for creating a trainer class. It has the following methods:
    1) start(): this initiate a training session with all necessary properties
    2) train_operator(): this runs a training session with environment and RL agent objects
    3) get_timestamp(): this returns a time stamp for logging purposes
    4) cal_performance(): this calculates detailed performance metrics after each training session
    5) save_session_results(): this saves all performance report assets
    6) reset_episode_action_history: this is a helper function for performance tracking
