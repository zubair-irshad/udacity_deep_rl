from collections import deque
import torch
import numpy as np
import math
import time


def interact(env, brain_name, agent, num_agents, num_episodes=200, window=30, max_iter=1000):
    scores = []
    scores_window = deque(maxlen=window)
    # get the default brain of UnityML Agents
    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]
    for i_episode in range(1, num_episodes+1):
        # Reset env and get current state
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(num_agents)
        agent.reset()
        for t in range(max_iter):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if any(dones):
                break
        scores.append(np.mean(score))
        scores_window.append(np.mean(score))
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(), './logging/checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), './logging/checkpoint_critic.pth')
        if np.mean(scores_window)>30:
            scores_filename = "./logging/ddpg_agent_" +str(i_episode) + ".csv"
            np.savetxt(scores_filename, scores, delimiter=",")
        if i_episode % window == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return scores
