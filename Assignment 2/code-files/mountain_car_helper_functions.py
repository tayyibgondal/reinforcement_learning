import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
import time  # For timing execution

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.2  # Exploration factor
num_episodes = 500  # Number of episodes for MountainCar

# Initialize the environment
env = gym.make("MountainCar-v0")
n_actions = env.action_space.n
n_states = (10, 10)  # Discretized state space (position, velocity)
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

# Discretize state space
def discretize_state(state, state_bounds, n_states):
    ratios = [(state[i] - state_bounds[i][0]) / (state_bounds[i][1] - state_bounds[i][0]) for i in range(len(state))]
    discrete_state = [int(ratio * (n_states[i] - 1)) for i, ratio in enumerate(ratios)]
    discrete_state = np.clip(discrete_state, 0, np.array(n_states) - 1)  # Correct clipping
    return tuple(discrete_state)

# Epsilon-greedy policy
def epsilon_greedy(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(n_actions)  # Explore
    return np.argmax(Q[state])  # Exploit


def evaluate(Q, max_steps=20000):
    state = discretize_state(env.reset()[0], state_bounds, n_states)
    action = epsilon_greedy(state, Q, 0)  # Exploit-only policy
    done = False
    steps = 0
    start_time = time.time()  # Start timing
    
    while not done and steps < max_steps:
        next_state, _, done, _, _ = env.step(action)
        next_state = discretize_state(next_state, state_bounds, n_states)
        action = epsilon_greedy(next_state, Q, 0)
        steps += 1
    
    elapsed_time = time.time() - start_time  # Calculate elapsed time
    
    if not done:
        print("The goal was not reached within the step limit.")
    return steps, elapsed_time