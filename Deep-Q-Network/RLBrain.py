import numpy as np
import tensorflow as tf
import time

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:

    def __init__(self, n_actions, n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None,
                 output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step (for judge whether to replace params of target net)
        self.learn_step_counter = 0

        # initialize zero memory [state, action, reward, next_state]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # replace params of target net
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir='./logs/'
            tf.summary.FileWriter('./logs/', self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []  # record changes of 'cost' for present loss by plt

    def _build_net(self):
        # ---------------- all inputs -------------------
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name='state')  # input state
        self.next_state = tf.placeholder(tf.float32, [None, self.n_features], name='next_state')  # input next state
        self.reward = tf.placeholder(tf.float32, [None, ], name='reward')   # input reward
        self.action = tf.placeholder(tf.int32, [None, ], name='action')     # input action

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # -------------------- build evaluate_net ----------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.state, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='q')

        # -------------------- build target_net ------------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.next_state, 20, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.reward + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None,)
            self.q_target = tf.stop_gradient(q_target)
        with tf.variable_scope('q_eval'):
            action_indices = tf.stack([tf.range(tf.shape(self.action)[0], dtype=tf.int32), self.action], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=action_indices)  # shape=(None, )
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        '''
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with  tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer,
                                     collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.next_state, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer,
                                     collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer,
                                     collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2
        '''

    def store_transition(self, state, action, reward, next_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # record a record
        transition = np.hstack((state, [action, reward], next_state))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # replace
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf.placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.state: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced')
            print(time.asctime(time.localtime(time.time())), '\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        '''
        q_next, q_eval = self.sess.run(
            [self.q_target, self.q_eval],
            feed_dict={
                self.next_state: batch_memory[:, -self.n_features],  # fixed params
                self.state: batch_memory[:, :self.n_features],  # newest params
            }
        )

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        '''

        # train eval network
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={
                                    self.state: batch_memory[:, :self.n_features],
                                    self.action: batch_memory[:, self.n_features],
                                    self.reward: batch_memory[:, self.n_features + 1],
                                    self.next_state: batch_memory[:, -self.n_features:],
                                })
        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    DQN = DeepQNetwork(3, 4, output_graph=True)
    DQN.plot_cost()
