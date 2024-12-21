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
# value iteration
# -------------------------------------------------
def get_optimal_policy(values, transitions, rewards, discount_factor):
    '''
    once the value iteration is converved, this function is used to get
    the optimal policy based on converged v(s) values
    '''
    q_values = np.zeros((num_states, num_actions))
    
    # compute q(s, a) for all states and actions
    for s in range(num_states):
        for action in range(num_actions):
            q_values[s, action] = sum(transitions[s, action, s_next] * (rewards[s, action, s_next] + discount_factor * values[s_next]) for s_next in range(num_states))

    # select the best actions as the policy for that state     
    policy = np.argmax(q_values, axis=1)
    return policy
    

# value iteration
def value_iteration(transitions, rewards, discount_factor, tolerance=1e-6):
    values = np.zeros(num_states)

    while(True):
        delta = 0
        for s in range(num_states):
            v = values[s]

            # update value of current state from optimal v values of next states (from last iteration)
            values[s] = max([sum(transitions[s, action, s_next] * (rewards[s, action, s_next]+discount_factor*values[s_next]) for s_next in range(num_states)) for action in range(num_actions)])

            delta = max(delta, abs(v-values[s]))
        
        # if values converge, break
        if delta < tolerance:
            break
    
    # once value iteration is converged, get the best policy
    policy = get_optimal_policy(values, transitions, rewards, discount_factor)

    return values, policy

# -------------------------------------------------
# Example usage - Value iteration
# -------------------------------------------------
values, policy = value_iteration(transitions, rewards, discount_factor)

for i in range(num_states):
    print(f'Optimal value for state "{state_names[i]}":', np.round(values[i], 2))
    print(f'Policy for state "{state_names[i]}":', action_names[policy[i]])
    print('---------------------------------------')