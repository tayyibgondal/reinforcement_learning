from collections import deque
import torch
import random
from optimized_hyperparameters import DEVICE

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
            torch.tensor(states, dtype=torch.float32).to(DEVICE),
            torch.tensor(actions, dtype=torch.long).to(DEVICE),
            torch.tensor(rewards, dtype=torch.float32).to(DEVICE),
            torch.tensor(next_states, dtype=torch.float32).to(DEVICE),
            torch.tensor(dones, dtype=torch.float32).to(DEVICE)
        )
    
    def __len__(self):
        return len(self.buffer)