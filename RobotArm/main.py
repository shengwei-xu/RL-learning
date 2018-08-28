from env import ArmEnv
from rl import DDPG

MAX_EPISODES = 500
MAX_EP_STEPS = 200
ON_TRAIN = True

# set env
env = ArmEnv()
state_dim = env.state_dim
action_dim = env.action_dim
action_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(state_dim, action_dim, action_bound)


def train():
    # start training
    for i in range(MAX_EPISODES):
        state = env.reset()
        ep_reward = 0.
        for j in range(MAX_EP_STEPS):
            env.render()
            action = rl.choose_action(state)
            state_, reward, done = env.step(action)

            # memory storage
            rl.store_transition(state, action, reward, state_)

            if rl.memory_full:
                # start to learn once has fulfilled the memory
                rl.learn()

            state = state_
            if done or j == MAX_EP_STEPS - 1:
                print('Ep: %i | %s | ep_reward: %.1f | steps: %i' %
                      (i, '---' if not done else 'done', ep_reward, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    while True:
        state = env.reset()
        for _ in range(200):
            env.render()
            action = rl.choose_action(state)
            state, reward, done = env.step(action)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()

