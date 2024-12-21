from mountain_car_helper_functions import *
from mountain_car_Q_learning import *
from mountain_car_sarsa_0 import *

# Train both algorithms
Q_sarsa, returns_sarsa = train_sarsa()
Q_q_learning, returns_q_learning = train_q_learning()

# Evaluate the learned strategies
steps_sarsa, time_sarsa = evaluate(Q_sarsa)
steps_q_learning, time_q_learning = evaluate(Q_q_learning)

print(f"Steps to solve using SARSA: {steps_sarsa}, Time taken: {time_sarsa:.4f} seconds")
print(f"Steps to solve using Q-Learning: {steps_q_learning}, Time taken: {time_q_learning:.4f} seconds")

# Plot returns for Sarsa-0
plt.figure(figsize=(10, 6))
plt.plot(returns_sarsa, label="SARSA")
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("SARSA on MountainCar")
plt.legend()
plt.grid()
plt.show()

# Plot returns for Q learning
plt.figure(figsize=(10, 6))
plt.plot(returns_q_learning, label="Q-Learning")
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Q-Learning on MountainCar")
plt.legend()
plt.grid()
plt.show()

# Plot returns for both algorithms
plt.figure(figsize=(10, 6))
plt.plot(returns_sarsa, label="SARSA")
plt.plot(returns_q_learning, label="Q-Learning")
plt.xlabel("Episodes")
plt.ylabel("Returns")
plt.title("Comparison of SARSA and Q-Learning on MountainCar")
plt.legend()
plt.grid()
plt.show()