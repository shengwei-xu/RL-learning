import tensorflow as tf
import numpy as np


# hyper parameters
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32


class DDPG(object):

    def __init__(self, state_dim, action_dim, action_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.action_dim, self.state_dim, self.action_bound = action_dim, state_dim, action_bound[1]
        self.S = tf.placeholder(tf.float32, [None, state_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, state_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.action = self._build_a(self.S, scope="eval", trainanle=True)
            action_ = self._build_a(self.S_, scope='eval', trainable=True)
        with tf.variable_scope('Critic'):
            # assign self.action = action in memory when calculating q for td_error
            # otherwise the self.action is from Actor when updating Actor
            q = self._build_c(self.S, self.action, scope='eval', trainnable=True)
            q_ = self._build_c(self.S_, action_, scope='target', trainable=False)

        # network parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1-TAU) * ta + TAU * ea), tf.assign(tc, (1-TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]
        # go on here
        pass

    def choose_action(self, state):
        return self.sess.run(self.action, {self.S: state[None, :]})[0]

    def learn(self):
        pass

    def store_transition(self, state, action, reward, state_):
        pass

    def _build_a(self, state, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(state, 100, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.action_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.action_bound, name="scaled_a")

    def _build_c(self, state, action, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 100
            w1_state = tf.get_variable("w1_state", [self.state_dim, n_l1], trainable=trainable)
            w1_action = tf.get_variable("w1_action", [self.action_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(state, w1_state) + tf.matmul(action, w1_action) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)     # Q(state, action)

    # 保存功能
    def save(self):
        saver = tf.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    # 提取功能
    def restore(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, './params')
