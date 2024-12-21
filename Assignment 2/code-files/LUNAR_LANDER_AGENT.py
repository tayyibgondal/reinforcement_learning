from lunar_lander_helper_functions import *
from lunar_lander_sarsa_0 import *
from lunar_lander_Q_learning import *
from lunar_lander_sarsa_lambda import *

# Train and compare algorithms
Q_sarsa, returns_sarsa = sarsa_0()
Q_q_learning, returns_q_learning = q_learning()
Q_sarsa_lambda, returns_sarsa_lambda = sarsa_lambda(lambda_val)

# Evaluate the learned policies
steps_sarsa, reward_sarsa, time_sarsa = evaluate(Q_sarsa)
steps_q_learning, reward_q_learning, time_q_learning = evaluate(Q_q_learning)
steps_sarsa_lambda, reward_sarsa_lambda, time_sarsa_lambda = evaluate(Q_sarsa_lambda)
print(f"SARSA(0): Steps={steps_sarsa}, Reward={reward_sarsa:.2f}, Time={time_sarsa:.2f} seconds")
print(f"Q-Learning: Steps={steps_q_learning}, Reward={reward_q_learning:.2f}, Time={time_q_learning:.2f} seconds")
print(f"SARSA(λ): Steps={steps_sarsa_lambda}, Reward={reward_sarsa_lambda:.2f}, Time={time_sarsa_lambda:.2f} seconds")

# Plot performance sarsa 0
plt.plot(returns_sarsa, label="SARSA(0)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("SARSA(0) on LunarLander")
plt.legend()
plt.show()

# Plot performance q learning
plt.plot(returns_q_learning, label="Q-Learning")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Q-Learning on LunarLander")
plt.legend()
plt.show()

# Plot performance sarsa lambda
plt.plot(returns_sarsa_lambda, label="SARSA(λ)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("SARSA(λ) on LunarLander")
plt.legend()
plt.show()

# Plot comparison
plt.plot(returns_sarsa, label="SARSA(0)")
plt.plot(returns_q_learning, label="Q-Learning")
plt.plot(returns_sarsa_lambda, label="SARSA(λ)")
plt.xlabel("Episodes")
plt.ylabel("Total Reward")
plt.title("Comparison of SARSA(0), Q-Learning, and SARSA(λ) on LunarLander")
plt.legend()
plt.show()