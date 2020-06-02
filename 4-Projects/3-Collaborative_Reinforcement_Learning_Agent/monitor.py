from collections import deque
import torch
import numpy as np
import math
import time


def interact(env,state_dim, brain_name, agent, num_agents,max_t=2000, num_episodes=10000, window=100):
    scores = []
    PRINT_EVERY=10
    rolling_avg=[]
    best_score=0
    scores_window = deque(maxlen=window)
    for i_episode in range(1, num_episodes+1):
        # Reset env and get current state
        env_info = env.reset(train_mode=True)[brain_name]
        states = np.reshape(env_info.vector_observations, (1,48)) # flatten states
        score = np.zeros(num_agents)
        agent.reset()
        while True:
            actions = agent.ma_act(states)
            env_info = env.step(actions)[brain_name]
            next_states = np.reshape(env_info.vector_observations, (1,48)) # flatten states
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.ma_step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            if any(dones):
                break
        scores.append(np.max(score))
        scores_window.append(np.max(score))
        rolling_avg.append(np.mean(scores_window))
        # print results
        if i_episode % PRINT_EVERY == 0:
            print('Episodes {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(i_episode, np.max(score), np.mean(scores_window)))
        # print('\rEpisode {}\tMax Reward: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, np.max(scores_all[-PRINT_EVERY:]),np.mean(scores_window)))
        if np.mean(scores_window)>=0.5:
            torch.save(agent.agents[0].actor_local.state_dict(), './logging/checkpoint_actor_0.pth')
            torch.save(agent.agents[0].critic_local.state_dict(), './logging/checkpoint_critic_0.pth')
            torch.save(agent.agents[1].actor_local.state_dict(), './logging/checkpoint_actor_1.pth')
            torch.save(agent.agents[1].critic_local.state_dict(), './logging/checkpoint_critic_1.pth')
            scores_filename = "./logging/ma_ddpg_agent_score_" +str(i_episode) + ".csv"
            rolling_avg_filename = "./logging/ma_ddpg_agent_rolling_avg_" +str(i_episode) + ".csv"
            np.savetxt(scores_filename, scores, delimiter=",")
            np.savetxt(rolling_avg_filename, rolling_avg, delimiter=",")
        if np.mean(scores_window)>=best_score:
            torch.save(agent.agents[0].actor_local.state_dict(), './logging/best_checkpoint_actor_0.pth')
            torch.save(agent.agents[0].critic_local.state_dict(), './logging/best_checkpoint_critic_0.pth')
            torch.save(agent.agents[1].actor_local.state_dict(), './logging/best_checkpoint_actor_1.pth')
            torch.save(agent.agents[1].critic_local.state_dict(), './logging/best_checkpoint_critic_1.pth')
            best_score = np.mean(scores_window)
        if i_episode % window == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
    return scores
