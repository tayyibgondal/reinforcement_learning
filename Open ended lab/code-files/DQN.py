import torch
import torch.nn as nn

class DQN(nn.Module):
    """Simplified Deep Q-Network for faster training."""
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=8, stride=4),  # Reduced channels
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),  # Reduced channels
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),  # Reduced channels
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 256),  # Reduced linear layer size
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    
    def forward(self, x):
        return self.net(x)