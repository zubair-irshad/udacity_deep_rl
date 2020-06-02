[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Problem Statement

For this project, we will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Instructions to reproduce results

- Clone this repo 

- Complete all dependencies here: [Enviornment dependencies](https://github.com/udacity/deep-reinforcement-learning/#dependencies)

- Download the respective UnityML Navigation Environment from the list below and extract it under ./2-Continuous_Control
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

### Train agent from scratch
```
python main.py
```

### Evaulate a trained model
```
python eval.py
```

### Code Structure (Helpful if you would like to make changes and try out different algorithms/architecture)
**model.py :** Actor-Critic model 
**ddpg_agent.py :** Implementation of Deep deterministic policy gradient agent. Helpful methods include 'act' for getting deterministic polocy with some added OU Noise, 'step' to store data in a replay buffer and learn every few timestep and Replay Buffer Class
**monitor.py :** Implementation includes running an agent through 'num_episodes' inside a simualtion 

### Algorithm Implementation Details

This work implements a deep deterministic policy gradient algorithm. This is an off-polocy algorithm which is based on the actor-critic method. 
![alt text](https://github.com/zubair-irshad/udacity_deep_rl/blob/master/4-Projects/2-Continuous_Control/logging/DDPG_algorithm.svg "Algorithm")

### Results

![alt text](https://github.com/zubair-irshad/udacity_deep_rl/blob/master/4-Projects/3-Collaborative_Reinforcement_Learning_Agent/logging/plot.jpg "Results")

### Future Work

- [Solve the crawler enviornment using DDPG](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#crawler)

