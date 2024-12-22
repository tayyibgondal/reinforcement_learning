from action_select import *
from DQN import *
from optimized_hyperparameters import *
from replay_buffer import *
from visualizer import *
from helper_functions import *
import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import matplotlib.pyplot as plt  # For plotting graphs
import os
from tqdm import tqdm

def main():
    # Initialize environment
    env = gym.make(ENV_NAME)
    num_actions = env.action_space.n

    # Initialize networks
    input_channels = 4  # Stacked frames
    policy_net = DQN(input_channels, num_actions).to(DEVICE)
    target_net = DQN(input_channels, num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START
    epsilon_decay = (EPS_START - EPS_END) / EPS_DECAY

    # Lists to store metrics
    episode_rewards = []
    episode_epsilons = []
    episode_losses = []
    episodes = []

    # Directory to save plots and model
    save_directory = 'dqn_training_results'
    os.makedirs(save_directory, exist_ok=True)

    episode_bar = tqdm(range(1, NUM_EPISODES + 1), desc="Training Episodes", unit="episode")
    for episode in episode_bar:
        obs, info = env.reset()
        state = preprocess_frame(obs)
        state_stack = deque([state] * 4, maxlen=4)  # Initialize with 4 frames
        total_reward = 0
        done = False
        loss_per_episode = 0
        steps = 0

        for step in range(MAX_STEPS):
            stacked_state = np.stack(state_stack, axis=0)
            action = select_action(stacked_state, policy_net, epsilon, num_actions)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            next_state = preprocess_frame(next_obs)
            state_stack.append(next_state)
            stacked_next_state = np.stack(state_stack, axis=0)

            replay_buffer.push(stacked_state, action, reward, stacked_next_state, done)

            if len(replay_buffer) > MIN_MEMORY_SIZE:
                # Sample a batch
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                # Compute current Q values
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q values
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

                # Compute loss
                loss = criterion(q_values, target_q_values)
                loss_per_episode += loss.item()

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update epsilon
                if epsilon > EPS_END:
                    epsilon -= epsilon_decay

                # Update target network
                if step % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
            steps += 1

        # Record metrics
        episode_rewards.append(total_reward)
        episode_epsilons.append(epsilon)
        average_loss = loss_per_episode / steps if steps > 0 else 0
        episode_losses.append(average_loss)
        episodes.append(episode)

        # Update progress bar with latest metrics
        episode_bar.set_postfix({
            'Total Reward': total_reward,
            'Epsilon': f"{epsilon:.4f}",
            'Avg Loss': f"{average_loss:.4f}"
        })

        # Plot and save metrics after each episode
        plot_metrics(episodes, episode_rewards, episode_epsilons, episode_losses, save_directory, episode)

        # Print progress every 100 episodes
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}, Average Loss: {average_loss:.4f}")

    # Plot and save final metrics
    plot_metrics(episodes, episode_rewards, episode_epsilons, episode_losses, save_directory, 'final')

    # Save the trained model
    model_path = os.path.join(save_directory, 'dqn_breakout_model.pth')
    torch.save(policy_net.state_dict(), model_path)
    print(f"Training completed! Model saved at {model_path}")
    env.close()

if __name__ == "__main__":
    main()