from lunar_lander_helper_functions import *

# SARSA(0) Algorithm
def sarsa_0():
    Q = np.zeros(n_states + (n_actions,))
    returns = []
    for episode in tqdm(range(num_episodes), desc="SARSA(0)"):
        state = discretize_state(env.reset()[0], state_bounds, n_states)
        action = epsilon_greedy(state, Q, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, state_bounds, n_states)
            next_action = epsilon_greedy(next_state, Q, epsilon)

            Q[state + (action,)] += alpha * (reward + gamma * Q[next_state + (next_action,)] - Q[state + (action,)])

            state, action = next_state, next_action
            total_reward += reward

        returns.append(total_reward)

    return Q, returns