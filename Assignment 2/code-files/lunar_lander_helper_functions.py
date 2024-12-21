import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.2  # Exploration factor
num_episodes = 2000  # Number of episodes for LunarLander
lambda_val = 0.8  # SARSA(λ) decay parameter

# Initialize environment
env = gym.make("LunarLander-v3")
n_actions = env.action_space.n
n_states = tuple((10, 10, 10, 10, 10, 10, 2, 2))  # Discretized state space
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[-2] = (-0.5, 0.5)  # Angle correction
state_bounds[-1] = (-0.5, 0.5)  # Angular velocity correction

# Discretize state space
def discretize_state(state, state_bounds, n_states):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    discrete_state = [int(ratio * (n_states[i] - 1)) for i, ratio in enumerate(ratios)]
    discrete_state = np.clip(discrete_state, 0, np.array(n_states) - 1)
    return tuple(discrete_state)

# Epsilon-greedy policy
def epsilon_greedy(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)  # Explore
    return np.argmax(Q[state])  # Exploit

# Evaluate policy
def evaluate(Q, max_steps=1000):
    state = discretize_state(env.reset()[0], state_bounds, n_states)
    action = epsilon_greedy(state, Q, 0)  # Exploit-only policy
    done = False
    steps = 0
    total_reward = 0
    start_time = time.time()

    while not done and steps < max_steps:
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        next_state = discretize_state(next_state, state_bounds, n_states)
        action = epsilon_greedy(next_state, Q, 0)
        steps += 1

    elapsed_time = time.time() - start_time
    return steps, total_reward, elapsed_time