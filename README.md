### What are we building?

We try to develop an intelligent agent to re-balance shared bikes using Reinforcement Learning and CitiBike data. 

### How is bike balancing being done now?
Bikes accumulate or deplete at certain popular locations. Companies, such as CitiBike, spend lots of effort and money to manage bike stock at each station to ensure availability to riders throughout the day. This operation is currently orchestrated by human operators. 

### How do we envision bike balancing to be done?
We aim to develop an computer agent that is able to decide how many bikes to move and to where without hard coding any rules. The only thing we specify is the reward, which the agent will receive if it manages to keep the number of bikes to be less than 50 at the end of each day (23:00). The agent will also receive penalty based on the number of bikes it moves and if the number of bikes exceed 50.

### How do we measure success (KPIs)?
We measure if the computer agent can achieve the following without deliberate human programming, but only with reward and penalty based on bike stocks:  
1) **High Success Ratio**: % of times it learns to limit bike stock equal or less than 50 by each day 23:00
2) **Low Cost**: Total cost the agent used to balance bike stocks; the agent will receive a penalty of -0.1 * (number of bikes moved)
3) **Short Learning Time**: the agent should be able to develop new rebalancing strategy quickly when external objectives change (e.g. bike stock limit, flucturating daily traffic flow, allowed actions)

### What will be the benefit and impact of our work?
Hopefully the agent can be more precise, timely, and optimized than human agents. If we are successful, this is one step closer to creating a true smart city powered by Artificial Intelligence. 

The current version only has basic mechanics and is still under development with some known issues. Refer to Issue page for details.

### What are the key thesis to investigate in near term?
- Can Deep Learning (e.g. RNN) and time series forecasting techniques improve the performance KPIs (e.g. increase success ratio, lower cost, and shorten learning time)?
- Can the agent balance multiple stations (e.g. 5 stations with inter-connected traffic flow)?
- Can the agent develop its own actions space, which is specified by human programmer at the moment?
---
### Preliminary Results and Baseline

**Setup**: The following results were generated using a simulation with linearly increasing bike stock. The initial stock at 00:00 is 20 and 3 additional bikes were added hourly. There would be 89 bikes at 23:00 if no bikes were removed during the day.

**Terminology**
- Agent: Reinforcement Learning object acting as a "bike re-balancing operator"
- Environment: a bike station object that will provide feedback such as the number of bikes and reward / penalty
- Training: interactions between the agent and environment for the agent to learn what the goal is and how to achieve it the best
- Episode: number of independent training session (the environment is reset, but agent keeps the learning from episode to another); each episode has 24 hour inter-dependent instances with bike stock info based on the environment setup and agent actions
- Session: each session has multiple episodes with both environment and agent reset; the goal is to benchmark agent performances based on the number of episodes (e.g. will more training episode leads to high success ratio?)
- Q-Table: a matrix the agent use to decide future action based on state-action-reward tuples; the agent develop this Q-Table from each training episode based on environment feedback

**The agent was able to limit bikes under 50 more oftern after interacting with the environment more.**

**Figure 1**: Comparison of success ratio as the number of episode increases. The success ratio is defined as % of times when the agent limited bikes to be equal or less than 50.

![image](/result_snapshot/session_success_rate_2018-03-05101044410867.png)

**The agent was being more strategic when moving bikes out of stations after learning from 1500 episodes.**

**Figure 2**: Bike moving comparison between the first and last training episode

![image](/result_snapshot/action_history_220180305223236794486.png)

**The agent chose actions based on a Q-Table it developed without explicit human programming. The Q-Table shows intuitive patterns.**

**Figure 3**: Heatmap of expected reward of an action when the agent was given certain bike stocks after 98000 training episodes.

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
