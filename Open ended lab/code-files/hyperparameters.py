import torch

# Hyperparameters
ENV_NAME = 'ALE/Breakout-v5'
GAMMA = 0.99
LEARNING_RATE = 1e-2
BATCH_SIZE = 32
MEMORY_SIZE = 1000
MIN_MEMORY_SIZE = 1000
EPS_START = 1
EPS_END = 0.3
EPS_DECAY = 1000
TARGET_UPDATE_FREQ = 1000
NUM_EPISODES = 500
MAX_STEPS = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")