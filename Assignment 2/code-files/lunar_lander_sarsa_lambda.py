from lunar_lander_helper_functions import *


# SARSA(λ) Algorithm
def sarsa_lambda(lambda_val=0.8):
    Q = np.zeros(n_states + (n_actions,))
    returns = []
    for episode in tqdm(range(num_episodes), desc="SARSA(λ)"):
        state = discretize_state(env.reset()[0], state_bounds, n_states)
        action = epsilon_greedy(state, Q, epsilon)
        total_reward = 0
        done = False
        E = np.zeros_like(Q)  # Eligibility trace

        cur_step = 0
        while not done:
            next_state, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_state, state_bounds, n_states)
            next_action = epsilon_greedy(next_state, Q, epsilon)

            delta = reward + gamma * Q[next_state + (next_action,)] - Q[state + (action,)]
            E[state + (action,)] += 1  # Update eligibility trace

            # Update Q-values and decay eligibility traces
            Q += alpha * delta * E
            E *= gamma * lambda_val

            state, action = next_state, next_action
            total_reward += reward

            cur_step += 1
            if cur_step > 20:
              break

        returns.append(total_reward)

    return Q, returns