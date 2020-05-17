from collections import deque
import torch
import numpy as np
import math
import time


def interact(env, brain_name, agent, num_episodes=2000, window=100, max_iter=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    scores = []
    scores_window = deque(maxlen=window)
    # get the default brain of UnityML Agents
    # brain_name = env.brain_names[0]
    # brain = env.brains[brain_name]
    eps = eps_start
    for i_episode in range(1, num_episodes+1):
        # Reset env and get current state
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_iter):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        scores.append(score)
        scores_window.append(score)
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 15:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            timestr = time.strftime("%Y%m%d-%H%M%S")
            model_filename = "./logging/nav_dqn_model_" + timestr + ".pth"
            torch.save(agent.qNetwork_local.state_dict(), model_filename)

            # Save the recorded Scores data
            scores_filename = "./logging/nav_dqn_scores_" + timestr + ".csv"
            np.savetxt(scores_filename, scores, delimiter=",")
            break

    return scores
