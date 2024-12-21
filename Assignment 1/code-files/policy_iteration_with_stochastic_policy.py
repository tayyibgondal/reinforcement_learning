# -------------------------------------------------
# imports
# -------------------------------------------------
import numpy as np

# -------------------------------------------------
# parameters
# -------------------------------------------------
num_states = 3 # (0=top, 1=rolling_down, 2=bottom)
num_actions = 2 # (0=drive, 1=no_drive)
discount_factor = 0.9

state_names = {0:'top', 1:'rolling down', 2:'bottom'}
action_names = {0:'drive', 1:"don't drive"}

# -------------------------------------------------
# initializing transition probabilities and rewards
# -------------------------------------------------
transitions = np.zeros((num_states, num_actions, num_states))
rewards = np.zeros((num_states, num_actions, num_states))

# -------------------------------------------------
# MDP Definition
# -------------------------------------------------
# Transition dynamics/uncertainties
transitions[0, 0, 0] = 0.5  # (top, drive, top)
transitions[0, 0, 1] = 0.5  # (top, drive, rolling_down)
transitions[0, 0, 2] = 0.0  # (top, drive, bottom)
transitions[0, 1, 0] = 0.5  # (top, no_drive, top)
transitions[0, 1, 1] = 0.5  # (top, no_drive, rolling_down)
transitions[0, 1, 2] = 0.0  # (top, no_drive, bottom)
transitions[1, 0, 0] = 0.3  # (rolling_down, drive, top)
transitions[1, 0, 1] = 0.4  # (rolling_down, drive, rolling_down)
transitions[1, 0, 2] = 0.3  # (rolling_down, drive, bottom)
transitions[1, 1, 0] = 0.0  # (rolling_down, no_drive, top)
transitions[1, 1, 1] = 0.0  # (rolling_down, no_drive, rolling_down)
transitions[1, 1, 2] = 1.0  # (rolling_down, no_drive, bottom)
transitions[2, 0, 0] = 0.5  # (bottom, drive, top)
transitions[2, 0, 1] = 0.0  # (bottom, drive, rolling_down)
transitions[2, 0, 2] = 0.5  # (bottom, drive, bottom)
transitions[2, 1, 0] = 0.0  # (bottom, no_drive, top)
transitions[2, 1, 1] = 0.0  # (bottom, no_drive, rolling_down)
transitions[2, 1, 2] = 1.0  # (bottom, no_drive, bottom)

# Rewards
rewards[0, 0, 0] = 2.0  # (top, drive, top)
rewards[0, 0, 1] = 2.0  # (top, drive, rolling_down)
rewards[0, 0, 2] = 0.0  # (top, drive, bottom)
rewards[0, 1, 0] = 3.0  # (top, no_drive, top)
rewards[0, 1, 1] = 1.0  # (top, no_drive, rolling_down)
rewards[0, 1, 2] = 0.0  # (top, no_drive, bottom)
rewards[1, 0, 0] = 2.0  # (rolling_down, drive, top)
rewards[1, 0, 1] = 1.5  # (rolling_down, drive, rolling_down)
rewards[1, 0, 2] = 0.5  # (rolling_down, drive, bottom)
rewards[1, 1, 0] = 0.0  # (rolling_down, no_drive, top)
rewards[1, 1, 1] = 0.0  # (rolling_down, no_drive, rolling_down)
rewards[1, 1, 2] = 1.0  # (rolling_down, no_drive, bottom)
rewards[2, 0, 0] = 2.0  # (bottom, drive, top)
rewards[2, 0, 1] = 0.0  # (bottom, drive, rolling_down)
rewards[2, 0, 2] = 2.0  # (bottom, drive, bottom)
rewards[2, 1, 0] = 0.0  # (bottom, no_drive, top)
rewards[2, 1, 1] = 0.0  # (bottom, no_drive, rolling_down)
rewards[2, 1, 2] = 1.0  # (bottom, no_drive, bottom)

# -------------------------------------------------
# policy iteration
# -------------------------------------------------
def policy_evaluation(policy, transitions, rewards, discount_factor, tol=1e-6):
    values = np.zeros(num_states)
    # if all elements in policy are -1, then stochastic policy will be assumed
    stochastic = all(elem == -1 for elem in policy)

    while True:
        delta = 0
        for s in range(num_states):
            v = values[s]
            action = policy[s]

            # if policy is not stochastic
            if not stochastic:
                values[s] = sum(transitions[s, action, s_next] * (rewards[s, action, s_next] + discount_factor * values[s_next]) for s_next in range(num_states))
            # if policy is stochastic, then average over all actions (equal probability for each action)
            else:
                values[s] = sum(1/num_actions * sum(transitions[s, action, s_next] * (rewards[s, action, s_next] + discount_factor * values[s_next]) for s_next in range(num_states)) for action in range(num_actions))
            
            delta = max(delta, abs(v - values[s]))

        # if convergence achieved, then exit
        if delta < tol:
            break

    return values
    
def policy_iteration(policy, transitions, rewards, discount_factor, tol=1e-6):
    # Initialize q(s, a)
    q_values = np.zeros((num_states, num_actions))

    while True:
        stop = True  # Assume policy is stable initially (won't change, and we need to exit after one policy iteration)
        # Evaluate the policy to get state values
        values = policy_evaluation(policy, transitions, rewards, discount_factor, tol)

        # Update the policy for each state
        for s in range(num_states):
            # Find the best action and its Q-value for this state
            best_action = policy[s]
            best_q_value = q_values[s, best_action]

            for action in range(num_actions):
                new_q_value = sum(
                    transitions[s, action, s_next] * (rewards[s, action, s_next] + discount_factor * values[s_next])
                    for s_next in range(num_states)
                )
                
                # Update if this action is better than the best one found so far
                if new_q_value > best_q_value:
                    best_q_value = new_q_value
                    best_action = action
                    stop = False  # Mark that we made a policy change
            
            # Update policy and q_values with the best action for state s
            policy[s] = best_action
            q_values[s, best_action] = best_q_value
        
        # Stop if the policy is stable
        if stop:
            break

    return values, policy

# -------------------------------------------------
# Example usage - Stochastic policy
# -------------------------------------------------
policy = np.zeros(num_states, dtype=int)
# stochastic policy 
# (no policy is specified for any state, so policy iteration will use stochastic policy at the start)
policy[0] = -1
policy[1] = -1
policy[2] = -1

values, policy = policy_iteration(policy, transitions, rewards, discount_factor)

for i in range(num_states):
    print(f'Optimal value for state "{state_names[i]}":', np.round(values[i], 2))
    print(f'Policy for state "{state_names[i]}":', action_names[policy[i]])
    print('---------------------------------------')