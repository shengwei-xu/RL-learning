from maze_env import Maze
from RLBrain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            next_observation, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, next_observation)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            if done:
                break
            step += 1

        # end of game
        print('Game Over')
        env.destroy()


if __name__ == '__main__':
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
