from lunar_lander_helper_functions import *

# Q-Learning Algorithm
def q_learning():
    Q = np.zeros(n_states + (n_actions,))
    returns = []
    for episode in tqdm(range(num_episodes), desc="Q-Learning"):
        state = discretize_state(env.reset()[0], state_bounds, n_states)
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(state, Q, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, state_bounds, n_states)

            Q[state + (action,)] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state + (action,)])

            state = next_state
            total_reward += reward

        returns.append(total_reward)

    return Q, returns