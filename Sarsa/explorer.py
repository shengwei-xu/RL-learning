from maze_env import Maze
from RLbrain import SarsaTable

MAX_EPISODES = 10


def update():
    for episode in range(MAX_EPISODES):
        # initial observation
        observation = env.reset()   # return the coordinates of red rectangle

        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render()

            # RL take action and get next observation and reward
            next_observation, reward, done = env.step(action)

            # RL choose action based on observation
            next_action = RL.choose_action(str(next_observation))

            # RL learn from this transition (s, a, r, s, a) ==> Sarsa
            RL.learn(str(observation), action, reward, str(next_observation), next_action)

            # swap observation and action
            observation = next_observation
            action = next_action

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print(RL.q_table)
    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
