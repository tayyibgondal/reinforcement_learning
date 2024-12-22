import matplotlib.pyplot as plt
import os

def plot_metrics(episodes, rewards, epsilons, losses, save_dir, episode):
    """Plots and saves the training metrics after each episode."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot Total Rewards per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards Over Episodes')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'total_rewards_episode.png'))
    plt.close()
    
    # Plot Epsilon Decay
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, epsilons, label='Epsilon Value')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay Over Episodes')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'epsilon_decay_episode.png'))
    plt.close()
    
    # Plot Loss Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, losses, label='Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Over Episodes')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'loss_episode.png'))
    plt.close()