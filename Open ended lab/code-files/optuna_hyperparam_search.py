import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2
import os
from tqdm import tqdm
import optuna

# Register Atari environments (optional, helps IDEs)
gym.register_envs(ale_py)

# Define device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame):
    """Preprocesses a single frame: grayscale, resize, normalize."""
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # Resize to 84x84
    frame = frame / 255.0  # Normalize pixel values
    return frame

class DQN(nn.Module):
    """Simplified Deep Q-Network for faster training."""
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """Experience Replay Buffer."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(actions, dtype=torch.long).to(device),
            torch.tensor(rewards, dtype=torch.float32).to(device),
            torch.tensor(next_states, dtype=torch.float32).to(device),
            torch.tensor(dones, dtype=torch.float32).to(device)
        )
    
    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, epsilon, num_actions):
    """Selects an action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state)
            return q_values.argmax().item()

def train_dqn(trial):
    # Hyperparameters to tune
    ENV_NAME = 'ALE/Breakout-v5'
    GAMMA = trial.suggest_float('gamma', 0.90, 0.999, step=0.001)
    LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    BATCH_SIZE = trial.suggest_categorical('batch_size', [32, 64, 128])
    MEMORY_SIZE = trial.suggest_categorical('memory_size', [5000, 10000, 20000])
    MIN_MEMORY_SIZE = trial.suggest_categorical('min_memory_size', [1000, 2000, 5000])
    EPS_START = trial.suggest_float('eps_start', 0.3, 1.0, step=0.1)
    EPS_END = trial.suggest_float('eps_end', 0.05, 0.2, step=0.05)
    EPS_DECAY = trial.suggest_int('eps_decay', 50000, 200000, step=50000)
    TARGET_UPDATE_FREQ = trial.suggest_int('target_update_freq', 50, 500, step=50)
    NUM_EPISODES = 100  # Reduced for faster trials
    MAX_STEPS = 1000
    
    # Initialize environment
    env = gym.make(ENV_NAME)
    num_actions = env.action_space.n

    # Initialize networks
    input_channels = 4  # Stacked frames
    policy_net = DQN(input_channels, num_actions).to(device)
    target_net = DQN(input_channels, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START
    epsilon_decay_value = (EPS_START - EPS_END) / EPS_DECAY

    episode_rewards = []

    for episode in range(1, NUM_EPISODES + 1):
        obs, info = env.reset()
        state = preprocess_frame(obs)
        state_stack = deque([state] * 4, maxlen=4)  # Initialize with 4 frames
        total_reward = 0
        done = False
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

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update epsilon
                if epsilon > EPS_END:
                    epsilon -= epsilon_decay_value

                # Update target network
                if step % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break
            steps += 1

        episode_rewards.append(total_reward)

    env.close()

    # Return the average reward as the objective to maximize
    avg_reward = np.mean(episode_rewards)
    return avg_reward

def main():
    # Create an Optuna study
    study = optuna.create_study(direction='maximize')
    
    # Optimize the study
    study.optimize(train_dqn, n_trials=50)

    # Output the best hyperparameters
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"Best average reward: {study.best_value}")

if __name__ == "__main__":
    main()
