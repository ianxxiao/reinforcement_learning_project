This README includes the following high level information:
- Project Overview
- Solution Design
- Results
- Future Development Goals
- Contributions

For a short presentation of our project, please refer to [this document](/Reports/Presentation_RL_citiBike_20180514.pdf)

For a detailed technical report with extensive discussion and results, please refer to [this document](/Reports/Full_Report_RL_citiBike_20180514.pdf)

---

![image](/result_snapshot/Slide02.png)

Bikes accumulate or deplete quickly at certain popular locations. Companies, such as CitiBike, spend lots of effort and money to manage bike stock at each station to ensure availability to riders throughout the day. Re-balancing is currently monitored and orchestrated by human based on conversation with citiBike frontline operators.

### What should the intelligent solution be able to do?
We want the solution to be able to do the following: 
- learn without explicit human instruction and intervention
- continuous improvement over time
- capture and adjust to complex and changing system dynamics


### What is Reinforcement Learning and its applications?
Reinforcement Learning (RL) is designed to let computors learn through reward and punishment, which is similar to how human learn. Some considered RL is the closest to real Artificial Intelligence.

RL has been applied to many use cases, such as: 
- Google's Alpha Go and energy saving solution at its data centers
- Tesla's self-driving car
- Boston Dynamics' Robots
- algorithmic traders at hedge funds
- personalized marketing at Jing Dong (the largest e-Commerce platform in China)

There are various applications beyond the ones we mentioned. Large scale operations with complex constraints and changing conditions, such as Smart City operation and multi-channel digital marketing, are the ideal condidates. There are lots of untapped potentials to gain productivity and new human-manchine interaction in these domains.

### What will be the impact of our work?
Hopefully the agent can be more precise, timely, and optimized than human agents. If we are successful, doing this may be one step closer to creating an "operational brain" for smart cities. Also, doing this is an attempt to apply Reinforcement Learning in large scale operational and public sector context, which is beyond games and finance.

### How do we measure success (KPIs)?
We measure if the computer agent can achieve the following without deliberate human programming, but only with reward and penalty based on bike stocks:  
1) **High Success Ratio**: % of times it learns to limit bike stock equal or less than 50 by each day 23:00
2) **Low Cost**: Total cost the agent used to balance bike stocks; the agent will receive a penalty of -0.1 * (number of bikes moved)
3) **Short Learning Time**: the agent should be able to develop new rebalancing strategy quickly when external objectives change (e.g. bike stock limit, fluctuating  daily traffic flow, allowed actions)

### What are the key thesis to investigate in near term?
- Can Deep Learning (e.g. RNN) and time series forecasting or simulation techniques improve the performance KPIs (e.g. increase success ratio, lower cost, and shorten learning time)?
- Can the agent balance multiple stations (e.g. 5 stations with inter-connected traffic flow) at the same time?
- Can the agent develop its own action space, which is specified by human programmer at the moment?
- Can parallel learning (Asynchronous Advantaged Actor-Critic)  improve the speed and accuracy of operations in new dynamics?

---
### Design and Results

**Terminology**
- **Agent**: Reinforcement Learning object acting as a "bike re-balancing operator"
    - Policy: agent's behaviour function, which is a map from state to action
    - Value Function: a prediction of future rewards
    - Model: agent's representation of the environment
- **Environment**: a bike station object that will provide feedback such as the number of bikes and reward / penalty
- **State**: the number of bike stock at a given time
- **Training**: interactions between the agent and environment for the agent to learn what the goal is and how to achieve it the best
- **Episode**: number of independent training session (the environment is reset, but agent keeps the learning from one episode to another); each episode has 24 hour inter-dependent instances with bike stock info based on the environment setup and agent actions
- **Session**: each session has multiple episodes with both environment and agent reset; the goal is to benchmark agent performances based on the number of episodes (e.g. will more training episode leads to high success ratio? When should we stop the training?)
- **Q-Table**: a matrix the agent use to decide future action based on state-action-reward tuples; the agent develop this Q-Table from each training episode based on environment feedback

**Setup**

- **Environment**: The following results were generated using a simulation with linearly increasing bike stock. The initial stock at 00:00 is 20 and 3 additional bikes were added hourly. There would be 89 bikes at 23:00 if no bikes were removed during the day.

- **Agent**: The agent can only remove bikes from a station in a quantity of 0, -1, -3, or -10 in each hour. 

- **Reward / Penalty**: 
    - +10 if the bike stock is equal or less than 50 at hour 23:00
    - -10 if the bike stock is more than 50 at any given hour
    - -0.1 * number of bike removed at each hour
    - -20 if bike stock becomes negative at any given hour

### High Level Architecture

![image](/result_snapshot/Slide07.png)


![image](/result_snapshot/Slide08.png)

![image](/result_snapshot/Slide09.png)

![image](/result_snapshot/Slide10.png)

![image](/result_snapshot/Slide11.png)

![image](/result_snapshot/Slide12.png)

![image](/result_snapshot/Slide13.png)


![image](/result_snapshot/Slide14.png)

---

### Contributions
- **Ian Xiao**: Overall solution design, development of base code, and integration and QA; Supported presentation and report writing.

- **Alex Shannon**: Attempted to transfer training methods onto HPC to decrease training time; compiled literature review; reached out to Citibike for info regarding current distribution methods and the applicability of our approach; assisted with writing of report and presentation.

- **Prince Abunku**: Wrote the code for the Deep Q network, as well as the logging functions for this method. Assisted in writing the report and presentation.

- **Brenton Arnaboldi**: Built random forests model to predict a station’s hourly net flow; incorporated predictions into Q-Learning method (Q-Learning + Forecasting); assisted with writing of report and presentation.
