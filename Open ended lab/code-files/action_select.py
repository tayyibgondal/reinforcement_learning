import random
import torch

def select_action(state, policy_net, epsilon, num_actions):
    """Selects an action using epsilon-greedy policy."""
    if random.random() < epsilon:
        return random.randrange(num_actions)
    else:
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            q_values = policy_net(state)
            return q_values.argmax().item()