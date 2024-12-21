import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import os

# For the progress bar
from tqdm import trange

# Set random seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# -----------------------------
# Define DQN Model
# -----------------------------
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()
        # Define network layers (smaller hidden size to run more quickly)
        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        # x should be a 1D or 2D tensor containing the state values
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# -----------------------------
# Define memory for Experience Replay
# -----------------------------
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# -----------------------------
# LunarLander DQL
# -----------------------------
class LunarLanderDQL():
    """
    DQN to solve the LunarLander-v3 environment with the following hyperparameters:

    gamma (discount_factor_g) = 0.85
    epsilon (exploration) = 0.2 (can decay further if desired)
    episodes = 2500
    batch size (mini_batch_size) = 32
    replay_memory_size = 100000
    """
    # Hyperparameters
    discount_factor_g   = 0.85          # discount rate (gamma)
    learning_rate_a     = 0.001         # learning rate (alpha)
    network_sync_rate   = 1000          # steps between syncing policy and target network
    replay_memory_size  = 100000        # replay memory size
    mini_batch_size     = 32            # batch size

    # We start epsilon at 0.2. If you want to decay further, you can adjust code below.
    initial_epsilon     = 0.2
    epsilon_min         = 0.01
    epsilon_decay       = 1e-4          # how quickly epsilon decays each step (example)

    # Loss + optimizer (assigned later)
    loss_fn   = nn.MSELoss()
    optimizer = None

    def __init__(self):
        # Make sure the result folders exist
        if not os.path.exists("results"):
            os.makedirs("results")

        # We'll store MSE loss every time we do a gradient update
        self.mse_history = []

    def train(self, episodes=2500, render=False):
        """
        Train a DQN on the LunarLander-v3 environment using a standard feed-forward network.
        Plots are saved to the 'results' folder, and the policy is saved as 'lunarlander_dql.pt'.
        """
        # Create environment
        env = gym.make('LunarLander-v2', render_mode='human' if render else None)

        # Extract observation and action sizes
        obs_dim = env.observation_space.shape[0]  # should be 8 for LunarLander
        act_dim = env.action_space.n              # should be 4

        # Create policy and target networks
        policy_dqn = DQN(in_states=obs_dim, h1_nodes=128, out_actions=act_dim)
        target_dqn = DQN(in_states=obs_dim, h1_nodes=128, out_actions=act_dim)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Create an optimizer for the policy network
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # Replay memory
        memory = ReplayMemory(self.replay_memory_size)

        # Tracking variables
        rewards_per_episode = np.zeros(episodes)
        epsilon = self.initial_epsilon
        epsilon_history = []
        mean_scores = []

        # For step-based decays
        step_count = 0

        # Use tqdm to display a progress bar over episodes
        for i in trange(episodes, desc="Training"):
            # Reset environment
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32)
            done = False
            total_reward = 0

            while not done:
                step_count += 1

                # Epsilon-greedy
                if random.random() < epsilon:
                    # Random action
                    action = random.randint(0, act_dim - 1)
                else:
                    # Exploit: pick best discrete action from Q-network
                    with torch.no_grad():
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # shape [1, obs_dim]
                        q_values = policy_dqn(obs_tensor)                # shape [1, act_dim]
                        action = q_values.argmax(dim=1).item()

                # Step the environment
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.array(next_obs, dtype=np.float32)
                done = terminated or truncated
                total_reward += reward

                # Store in replay memory
                memory.append((obs, action, next_obs, reward, done))

                # Move to next state
                obs = next_obs

                # Decay epsilon
                # If you do NOT want to decay further from 0.2, you can remove or comment out below:
                if epsilon > self.epsilon_min:
                    epsilon = max(epsilon - self.epsilon_decay, self.epsilon_min)

                # Once we have enough in replay memory, sample and optimize
                if len(memory) > self.mini_batch_size:
                    self.optimize(memory, policy_dqn, target_dqn)

                # Sync policy -> target after some fixed steps
                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode[i] = total_reward
            epsilon_history.append(epsilon)

            # Track average reward every 100 episodes
            if (i + 1) % 100 == 0:
                mean_score_100 = np.mean(rewards_per_episode[i-99:i+1])
                mean_scores.append(mean_score_100)

        env.close()

        # --------------------------------------------------
        # Save learned policy
        torch.save(policy_dqn.state_dict(), "results/lunarlander_dql.pt")

        # --------------------------------------------------
        # Plotting and saving separate figures
        # --------------------------------------------------

        # 1) Reward per Episode
        plt.figure()
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('LunarLander: Reward per Episode')
        plt.savefig("results/lunarlander_rewards_per_episode.png")
        plt.close()

        # 2) Mean Reward (Every 100 Episodes)
        plt.figure()
        plt.plot(mean_scores)
        plt.xlabel('Block (each = 100 episodes)')
        plt.ylabel('Avg Reward')
        plt.title('LunarLander: Mean Reward (every 100 eps)')
        plt.savefig("results/lunarlander_mean_scores.png")
        plt.close()

        # 3) Epsilon Decay
        plt.figure()
        plt.plot(epsilon_history)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay')
        plt.savefig("results/lunarlander_epsilon_decay.png")
        plt.close()

        # 4) MSE Loss (over training steps)
        plt.figure()
        plt.plot(self.mse_history)
        plt.xlabel('Optimization Steps')
        plt.ylabel('Loss')
        plt.title('LunarLander: MSE Loss')
        plt.savefig("results/lunarlander_mse_loss.png")
        plt.close()

        print("Training complete. Model saved to results/lunarlander_dql.pt")
        print("All plots saved individually in the 'results' folder.")

    def optimize(self, memory, policy_dqn, target_dqn):
        """
        Sample a mini-batch from the replay buffer, then perform one gradient-descent step.
        We'll record MSE for debugging.
        """
        mini_batch = memory.sample(self.mini_batch_size)

        # Prepare tensors
        states      = []
        actions     = []
        next_states = []
        rewards     = []
        dones       = []

        for (s, a, ns, r, d) in mini_batch:
            states.append(s)
            actions.append(a)
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)

        states      = torch.FloatTensor(states)       # shape [batch, obs_dim]
        actions     = torch.LongTensor(actions).view(-1, 1)
        next_states = torch.FloatTensor(next_states)  # shape [batch, obs_dim]
        rewards     = torch.FloatTensor(rewards).view(-1, 1)
        dones       = torch.BoolTensor(dones).view(-1, 1)

        # Current Q-values from policy network
        q_vals = policy_dqn(states)  # shape [batch, act_dim]
        current_q = q_vals.gather(1, actions)  # shape [batch, 1]

        # Next Q-values from target network
        with torch.no_grad():
            target_q_vals = target_dqn(next_states)  # shape [batch, act_dim]
            max_target_q = target_q_vals.max(dim=1, keepdim=True)[0]  # shape [batch, 1]

        # Bellman target
        # If done, target = reward
        # else, target = reward + gamma * max(Q(s', a'))
        target = rewards + self.discount_factor_g * max_target_q * (~dones)

        # Compute loss
        loss = self.loss_fn(current_q, target)

        # Store for plotting
        self.mse_history.append(loss.item())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def test(self, episodes=10):
        """
        Run the learned policy for a few episodes in LunarLander-v3 (rendered).
        """
        env = gym.make('LunarLander-v3', render_mode='human')

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        # Load policy
        policy_dqn = DQN(in_states=obs_dim, h1_nodes=128, out_actions=act_dim)
        policy_dqn.load_state_dict(torch.load("results/lunarlander_dql.pt"))
        policy_dqn.eval()

        avg_reward = 0.0
        for ep in range(episodes):
            obs, _ = env.reset()
            obs = np.array(obs, dtype=np.float32)

            done = False
            total_reward = 0

            while not done:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)  # shape [1, obs_dim]
                    q_values = policy_dqn(obs_tensor)                # shape [1, act_dim]
                    action = q_values.argmax(dim=1).item()

                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_obs = np.array(next_obs, dtype=np.float32)
                done = terminated or truncated
                total_reward += reward
                obs = next_obs

            print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}")
            avg_reward += total_reward

        env.close()
        avg_reward /= episodes
        print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")


# -----------------------------
# Run the code
# -----------------------------
if __name__ == '__main__':
    lander_agent = LunarLanderDQL()

    # Train with the specified hyperparameters
    # episodes=2500, gamma=0.85, epsilon=0.2, batch_size=32, memory=100,000
    lander_agent.train(episodes=2500, render=False)

    # Test the learned policy
    lander_agent.test(episodes=10)
