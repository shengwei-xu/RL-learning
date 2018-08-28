from maze_env import Maze
from RLbrain import QLearningTable

MAX_EPISODES = 10


def update():
    for episode in range(MAX_EPISODES):
        # initial observation
        observation = env.reset()   # return the coordinates of red rectangle

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            next_observation, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(next_observation))

            # swap observation
            observation = next_observation

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print(RL.q_table)
    print('Game Over')
    env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
