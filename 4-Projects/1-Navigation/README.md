[//]: # (Image References)
[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
# Project 1: Autonomous Navigation Agent

### Problem Statement
In this project, an agent has to start from scratch in a previously unknown UnityML Enviornment and learn to navigate the enviornment by collecting the maximum amount of reward(yellow bananas) and avoid bad reward(blue bananas). A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. 
The agent previously does not know the rules of this game and solely learns from interactions and reward feedback mechanism. 

![Trained Agent][image1]

The state space constitutes of 37 dimensions including agent's velocity and ray-based perception vector of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

### Instructions to reproduce results

- Clone this repo 

- Complete all dependencies here: [Enviornment dependencies](https://github.com/udacity/deep-reinforcement-learning/#dependencies)

- Download the respective UnityML Navigation Environment from the list below and extract it under ./navigation

 - Linux (Version1 with GUI): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - Linux (Version2 headless): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip)
 - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

Now you are ready to start training your agent

### Train agent from scratch
```
python main.py
```

### Evaulate a trained model
```
python eval.py
```

### Code Structure (Helpful if you would like to make changes and try out different algorithms/architecture)
**model.py :** Deep Neural Network model (Current implementation: Fully Connected Architecture)
**nav_agent.py :** Implementation of navigation agent object. Helpful methods include 'act' for getting epsilon-greedy policy, 'step' to store data in a replay buffer and learn every few timestep and Replay Buffer Class
**monitor.py :** Implementation includes running an agent through 'num_episodes' inside a simualtion 

### Algorithm Implementation Details

This work implements a Deep Q-Network from Deep Mind's paper [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf).

![alt text](https://github.com/zubair-irshad/udacity_deep_rl/blob/master/4-Projects/1-Navigation/logging/dqn_algorithm.png "Algorithm")


### Results

![alt text](https://github.com/zubair-irshad/udacity_deep_rl/blob/master/4-Projects/1-Navigation/logging/plot.jpg "Results")

### Future Work

- [Implement Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Implement Double DQN](https://arxiv.org/abs/1509.06461)

