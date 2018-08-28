import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# np.random.seed(2)   # reproducible

N_STATUS = 6    # the length of the 1 dimensional world like '------'
ACTIONS = ['left', 'right']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9     # discount factor
MAX_EPISODES = 30   # maximum episodes
FRESH_TIME = 0.05    # fresh time for one move
FOUND_REST_TIME = 0.5

# Total steps counter
TOTAL_STEPS = []


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),     # q_table initial values
        columns=actions,      # actions' name
    )
    print(table)        # show table
    return table


def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):   # act non-greedy or state-action
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()
    return action_name


def get_env_feedback(state, action):
    # This is how agent will interact with the environment
    if action == 'right':    # move right
        if state == N_STATUS - 2:   # terminal
            next_state = 'terminal'
            reward = 1
        else:
            next_state = state + 1
            reward = 0
    else:   # move left
        reward = 0
        if state == 0:
            next_state = state  # reach the wall
        else:
            next_state = state - 1
    return next_state, reward


def update_env(state, episode, step_counter):
    # This si how environment be updated
    env_list = ['-'] * (N_STATUS - 1) + ['T']     # like '-----T'
    if state == 'terminal':
        interaction = 'Episode %s: total steps = %s' % (episode + 1, step_counter)
        TOTAL_STEPS.append(step_counter)    # for figure data
        print('\r{}'.format(interaction))
        time.sleep(FOUND_REST_TIME)
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATUS, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0   # initial state
        is_terminated = False
        update_env(state, episode, step_counter)    # update environment
        while not is_terminated:
            action = choose_action(state, q_table)
            # take action & get next state and reward
            next_state, reward = get_env_feedback(state, action)
            q_predict = q_table.loc[state, action]
            if next_state != 'terminal':
                q_target = reward + GAMMA * q_table.iloc[next_state, :].max()
            else:
                q_target = reward       # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)     # update
            state = next_state      # move to next state
            step_counter += 1
            update_env(state, episode, step_counter)

    return q_table


def run():
    q_table = rl()
    print('\r\nQ-table:')
    print(q_table)

    x = range(1, MAX_EPISODES + 1)
    plt.plot(x, TOTAL_STEPS, 'r')
    plt.show()


if __name__ == "__main__":
    run()
