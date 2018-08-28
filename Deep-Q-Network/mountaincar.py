import gym
from RLBrain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print(env.action_space)     # how many actions could be used
print(env.observation_space) # how many observations of states could be used
print(env.observation_space.high)   # maximum
print(env.observation_space.low)    # minimum

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000,
                  e_greedy_increment=0.008)

total_steps = 0

for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        next_observation, reward, done, info = env.step(action)

        position, velocity = next_observation

        reward = abs(position - (-0.5))

        RL.store_transition(observation, action, reward, next_observation)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward

        if done:
            get = '| Get' if next_observation[0] >= env.unwrapped.goal_position else '|----'
            print('episode:', i_episode,
                  get,
                  'ep_r:', round(ep_r, 4),
                  'epsilon:', round(RL.epsilon, 2))
            break

        observation = next_observation
        total_steps += 1

# output cost
RL.plot_cost()
