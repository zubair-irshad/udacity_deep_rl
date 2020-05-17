My solutions (with explanations) to the Udacity Deep Reinforcement Learning Nano Degree Program

Current Progress:

Introduction to Deep RL
-----------------------

**Monte Carlo Methods**

- Implementation of [Monteo Carlo Methods](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf#page=113) for the enviornment [BlackJack](https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py)

**Temporal Difference Methods**

- Implementation of [Sarsa](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf#page=154), [Q-learning](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf#page=157) and [Expected-Sarsa](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf#page=157)  for the enviornment [CliffWalking](https://github.com/openai/gym/blob/master/gym/envs/toy_text/cliffwalking.py)

Deep Value Iterations
-----------------------

**Deep Q Network - DQN**

- Implementation of [Deep Q Network](https://arxiv.org/abs/1312.5602) for the enviornment [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/)


Project1: Navigation
-----------------------
- Implemented [Deep Q Network](https://arxiv.org/abs/1312.5602) to navigate an agent inside an enviornment while avoiding obstacle(bad reward paths)


Policy Gradients
-----------------------

**REINFORCE**

- Implementation of [REINOFRCE Algorithm](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) to teach an agent to play [Pong from scratch](https://gym.openai.com/envs/Pong-v0/)

**Proximal Policy Optimizaiton**

- Implementation of [PPO Algorithm](https://openai.com/blog/openai-baselines-ppo/) to teach an agent to play [Pong from scratch](https://gym.openai.com/envs/Pong-v0/)


Project2: Continuous Control
-----------------------
- Implemented [Deep Deterministic Policy Gradients](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) to teach an 2-DOF robotic manipulator to reach a goal location. Enviornment used for training and testing: [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
