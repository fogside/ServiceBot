"""
The double DQN based on this paper: https://arxiv.org/abs/1509.06461
"""

import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import normalize

tf.set_random_seed(1)


class DoubleDQN:
    def __init__(self,**kwargs):
        required_parametrs = ['n_features', 'n_actions', 'memory_size', 'random_state']
        for p in required_parametrs:
            if kwargs.get(p) is None:
                raise Exception('No parametr {} given'.format(p))

        self.kwargs = kwargs

        self.n_actions = kwargs.get('n_actions')
        self.n_features = kwargs.get('n_features')
        self.lr = kwargs.get('learning_rate', 0.01)
        self.gamma = kwargs.get('reward_decay', 0.9)
        self.epsilon_max = kwargs.get('e_greedy', 0.9)
        self.batch_size = kwargs.get('batch_size', 128)
        self.memory_size = kwargs.get('memory_size')
        self.epsilon_increment = kwargs.get('e_greedy_increment', 1e-4)
        self.replace_target_iter = kwargs.get('replace_target_iter', 200)
        self.epsilon = 0.5
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.random_state = kwargs.get('random_state')
        
        self.double_q = kwargs.get('double_q', False)    # decide to use double q or not
        self.memory = np.zeros((self.memory_size, self.n_features*2+2))

        sess = kwargs.get('sess')
        self._build_net()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if kwargs.get('output_graph'):
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, w_initializer, b_initializer):
            dense = tf.layers.dense(inputs=s, units=200, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            dense = tf.layers.dense(inputs=dense, units=100, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            dense = tf.layers.dense(inputs=dense, units=self.n_actions, activation=tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            return dense

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            w_initializer = tf.random_normal_initializer(0., 0.3)
            b_initializer = tf.constant_initializer(0.1)
            self.q_eval = build_layers(self.s,w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            self.q_next = build_layers(self.s_, w_initializer, b_initializer)

        self._init_replace_ops()

    def choose_action(self, observation, act_indexes_to_ignore):
        if len(observation.shape)!=2:
            observation = observation[np.newaxis, :]

        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        for i in act_indexes_to_ignore:
            actions_value[0][i] = -10**3

        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        was_random = False
        if self.random_state.uniform() > self.epsilon:  # choosing action
            action_probs = [1/(self.n_actions-len(act_indexes_to_ignore)) if i not in act_indexes_to_ignore else 0 for i in range(self.n_actions)]
            action_probs = normalize(np.array(action_probs).reshape((1, -1)), norm='l1')[0]
            action = self.random_state.choice(list(range(self.n_actions)), p=action_probs)
            was_random = True

        return action, was_random
    
    def store_transition(self, transition):
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition.flatten()
        self.memory_counter += 1

    def _init_replace_ops(self):
        tau = 1
        tfVars = tf.trainable_variables()
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign((var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        self.replace_ops = op_holder
        
    def _replace_target_params(self):
        # t_params = tf.get_collection('target_net_params')
        # e_params = tf.get_collection('eval_net_params')
        # self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        for op in self.replace_ops:
            self.sess.run(op)

        print('\ntarget_params_replaced\n')

    def get_q_value(self, batch, action_indexes):
        if len(batch.shape) != 2:
            batch = batch[np.newaxis, :]

        q_eval = self.sess.run(self.q_eval, feed_dict={self.s: batch})
        return q_eval[0][action_indexes].mean()

    def learn(self):
        #if self.memory_counter<self.memory_size:
         #   return

        if self.learn_step_counter % self.replace_target_iter == 0 and self.learn_step_counter>0:
            self._replace_target_params()

        if self.memory_counter > self.memory_size:
            sample_index = self.random_state.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = self.random_state.choice(self.memory_counter, size=self.batch_size)

        sample_index = list(set(sample_index))
        batch_memory = self.memory[sample_index, :]
        
        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            selected_q_next = q_next[:, max_act4next]  # Double DQN, select q_next depending on above actions
        else:
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[np.arange(q_target.shape[0]), eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save(self, filepath):
        self.kwargs['sess'] = None
        pickle.dump(self.kwargs, open(filepath + '.params', 'wb'))

        saver = tf.train.Saver(max_to_keep=1)
        # Save model weights to disk
        save_path= saver.save(self.sess, filepath, global_step=0)
        print('Save path = {}'.format(save_path))

    @classmethod
    def load(self, filepath):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        constructor_params = pickle.load(open(filepath.split('.tf')[0]+'.tf.params', 'rb'))
        constructor_params['sess'] = sess
        result = DoubleDQN(**constructor_params)
        saver = tf.train.Saver()
        sess = saver.restore(sess, filepath)
        return result

