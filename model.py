from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Flatten
from tensorflow import keras
import tensorflow as tf
import numpy as np

class HASL():
    def __init__(self, comm, controller, rank, sess_config=None, state_shape=(42, 42), n_base_acts=12, n_act_seqs=12):
        if sess_config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=sess_config)

        self.comm = comm
        self.controller = controller
        self.rank = rank
        self.sess_config = sess_config
        self.state_shape = state_shape
        self.n_base_acts = n_base_acts
        self.n_act_seqs = n_act_seqs
        self.n_curr_acts = n_act_seqs # Equal to n_act_seqs until the set_act_seqs has been called to update it
        self.as_nets = []
        self.create_phs(state_shape=self.state_shape)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.create_encoder_ops(state_shape=state_shape, n_base_acts=self.n_base_acts)
        self.create_policy_ops(n_act_seqs=self.n_act_seqs)
        self.sess.run(tf.global_variables_initializer())
        self.sync_weights()

    def create_as_net(self, obs, acts, n_acts=3, batch_size=32, n_epochs=200, test_frac=0.15):
        scope_name = f'as_net_{self.n_act_seqs}'

        with tf.variable_scope(scope_name):
            act_seq_ph = tf.placeholder(dtype=tf.int32, shape=(None, n_acts))
            flat_act_seqs = tf.reshape(act_seq_ph, (-1,))

            dense = Dense(128, activation='relu')(self.obs_op)
            act_probs = []
            act_ohs = []
            losses = []
            for i in range(n_acts):
                act_indices = tf.range(i, tf.shape(flat_act_seqs)[0], n_acts)
                resp_acts = tf.gather(flat_act_seqs, act_indices)
                act_ohs.append(tf.one_hot(resp_acts, self.n_curr_acts, dtype=tf.float32))
                dense2 = Dense(64, activation='relu')(dense)
                act_probs.append(Dense(self.n_curr_acts, activation=None)(dense2))
                losses.append(tf.losses.softmax_cross_entropy(act_ohs[-1], act_probs[-1]))
                
            as_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            total_loss = tf.math.add_n(losses)
            as_update = as_optimizer.minimize(total_loss)

        init_new_vars_ops = [x.initializer for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)]
        self.sess.run(init_new_vars_ops)

        # train_obs, test_obs = obs[:]

        for epoch in range(n_epochs):
            correct_preds = 0
            total_preds = 0
            for idx in range(0, len(obs), batch_size):
                aps, _ = self.sess.run([act_probs, as_update], 
                    feed_dict={
                        self.obs_op: obs[idx:idx+batch_size],
                        act_seq_ph: acts[idx:idx+batch_size]
                    })

                for i, ap in enumerate(aps):
                    ids = np.argmax(ap, axis=1)
                    correct_preds += (ids == acts[idx:idx+batch_size,i]).sum()
                    total_preds += batch_size
                # tl += loss[0]
            # print('Loss:', tl)
            print(correct_preds/total_preds)

        # TODO: Add to list of act sequences
        self.n_act_seqs += 1

    def create_phs(self, state_shape=(42, 42)):
        # Placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=(None, *list(state_shape), 1))
        self.state_p_ph = tf.placeholder(dtype=tf.float32, shape=(None, *list(state_shape), 1))
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    def create_policy_ops(self, n_act_seqs=12):
        with tf.variable_scope('policy'):
            # Creating a conv net for the policy and value estimator
            self.obs_op = Input(shape=(self.enc_dim,))
            dense1 = Dense(128, activation='relu')(self.obs_op)
            act_dense = Dense(128, activation='relu')(dense1)
            val_dense = Dense(128, activation='relu')(dense1)

            # Output probability distribution over possible actions
            self.act_probs_op = Dense(n_act_seqs, activation='softmax', name='act_probs')(act_dense)
            self.act_out = tf.squeeze(tf.random.multinomial(tf.log(self.act_probs_op), 1))

            # Output value of observed state
            self.value_op = Dense(1)(val_dense)

            self.act_masks = tf.one_hot(self.act_ph, n_act_seqs, dtype=tf.float32)
            self.log_probs = tf.log(self.act_probs_op)

            self.advantages = self.rew_ph - tf.squeeze(self.value_op)

            self.resp_acts = tf.reduce_sum(self.act_masks *  self.log_probs, axis=1)
            self.policy_loss = -tf.reduce_mean(self.resp_acts * self.advantages)

            self.policy_update = self.optimizer.minimize(self.policy_loss)

            with tf.control_dependencies([self.policy_update]):
                self.value_loss = tf.reduce_mean(tf.square(self.rew_ph - tf.squeeze(self.value_op)))
                self.value_update = self.optimizer.minimize(self.value_loss)

    def set_act_seqs(self, n_act_seqs):
        self.n_act_seqs = n_act_seqs

        ### Save weights and reset graph and session ###
        saved_weights = self.sess.run(tf.trainable_variables())

        tf.reset_default_graph()
        self.sess.close()
        if self.sess_config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=self.sess_config)

        ### Recreate all ops ###
        self.create_phs(state_shape=self.state_shape)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.create_encoder_ops(state_shape=self.state_shape, n_base_acts=self.n_base_acts)
        self.create_policy_ops(n_act_seqs=self.n_act_seqs)

        ### Reload applicable weights ###
        self.sess.run(tf.global_variables_initializer())
        t_vars = tf.trainable_variables()
        for pair in zip(t_vars, saved_weights):
            if 'policy/act_probs' not in pair[0].name:
                self.sess.run(tf.assign(pair[0], pair[1]))

    def choose_action(self, obs, batch_size=1024):
        return self.sess.run(self.act_out, feed_dict={self.obs_op: obs})

    def train_policy(self, states, actions, rewards):
        # TODO: Change the way the actions are selected when training the policy
        actions = [a[0] for a in actions]
        states = np.vstack(states)

        self.sess.run([self.policy_update, self.value_update],
            feed_dict={self.obs_op: states,
                self.act_ph: actions,
                self.rew_ph: rewards})

    def create_encoder_ops(self, state_shape=(42, 42), n_base_acts=12):
        """
        Creates the encoder used for states
        """
        self.enc_dim = 128 # Encoded Feature Dimension

        # State encoder layer ops
        with tf.variable_scope('encoder'):
            self.enc_layers = [
                ZeroPadding2D(),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                ZeroPadding2D(),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                ZeroPadding2D(),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                Flatten()
            ]

        # State encoder output ops
        self.enc_state = self.enc_layers[0](self.state_ph)
        for i in range(1, len(self.enc_layers)):
            self.enc_state = self.enc_layers[i](self.enc_state)

        self.enc_state_p = self.enc_layers[0](self.state_p_ph)
        for i in range(1, len(self.enc_layers)):
            self.enc_state_p = self.enc_layers[i](self.enc_state_p)

        # Inverse Dynamics Model
        with tf.variable_scope('inverse_model'):
            self.state_state_pair = tf.concat([self.enc_state, self.enc_state_p], axis=1)
            self.im_dense = Dense(256, activation='elu')(self.state_state_pair)
            self.act_pred = Dense(n_base_acts, activation='softmax', use_bias=False)(self.im_dense)

        # State encoder train ops
        self.act_actual_oh = tf.one_hot(self.act_ph, n_base_acts)
        self.loss_i = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(self.act_actual_oh, self.act_pred))

        self.im_train_vars = tf.trainable_variables(scope='encoder') + tf.trainable_variables(scope='inverse_model')
        self.update_im = self.optimizer.minimize(self.loss_i, var_list=self.im_train_vars)

    def apply_encoder(self, state):
        if type(state) is not np.ndarray:
            state = np.asarray(state)

        if state.shape == 2:
            state = [state]
            return self.sess.run(self.enc_state, feed_dict={self.state_ph: state})[0]

        return self.sess.run(self.enc_state, feed_dict={self.state_ph: state})

    def train_encoder(self, states, state_ps, actions, batch_size=1024):
        formatted_states = np.expand_dims(np.stack(states), axis=-1)
        formatted_state_ps = np.expand_dims(np.stack(state_ps), axis=-1)

        correct = 0
        for batch_idx in range(int(np.ceil(len(actions) / batch_size))):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            act_preds, _ = self.sess.run([self.act_pred, self.update_im], 
                        feed_dict={ 
                            self.state_ph: formatted_states[start_idx:end_idx],
                            self.state_p_ph: formatted_state_ps[start_idx:end_idx],
                            self.act_ph: actions[start_idx:end_idx]
                        })
            act_preds = np.argmax(act_preds, axis=1)
            correct += sum([1 if x[0] == x[1] else 0 for x in zip(act_preds, actions[start_idx:end_idx])])

        accuracy = correct / len(actions)

        return accuracy

    def sync_weights(self):
        if self.rank == self.controller:
            self.comm.bcast(self.sess.run(tf.trainable_variables()), self.controller)
        else:
            sync_vars = self.comm.bcast(None, self.controller)
            t_vars = tf.trainable_variables()
            for pair in zip(t_vars, sync_vars):
                self.sess.run(tf.assign(pair[0], pair[1]))