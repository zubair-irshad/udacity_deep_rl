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

Deep Q Learning improves upon the classical Q-learning algorithm which builds upon value iteration and bellman equation to update the action-value of each visited state in the network. Using feedback signals from the enviornment, the agent updates its action-value pair and selects the action with the highest action-value in an uncertain scenario. Deep Q Leanring combines this classical methodology with the power of Deep Networks to achieve this task in continous enviornments. 


### Results



### Future Work

- [Prioritized Experienced Repaly](https://arxiv.org/abs/1511.05952)
- [Dueling Q Networks](https://arxiv.org/abs/1511.06581)
- [Double Q Learning](https://arxiv.org/abs/1509.06461)

