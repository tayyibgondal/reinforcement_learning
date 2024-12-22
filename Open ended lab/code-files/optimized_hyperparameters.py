import torch

# Hyperparameters
ENV_NAME = 'ALE/Breakout-v5'
GAMMA = 0.85
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 1000
EPS_START = 0.5
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 500
MAX_STEPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")