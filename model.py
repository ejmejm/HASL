from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Flatten, Dropout, UpSampling2D
from tensorflow import keras
import tensorflow as tf
import numpy as np

class HASL():
    def __init__(self, comm, controller, rank, sess_config=None, 
                 state_shape=(42, 42), state_depth=1, n_base_acts=12,
                 n_act_seqs=12, ppo_clip_val=0.2, ppo_val_coef=1.,
                 ppo_entropy_coef=0.01, ppo_iters=80, ppo_kl_limit=1.5):
        if sess_config is None:
            self.sess = tf.Session()
        else:
            self.sess = tf.Session(config=sess_config)

        self.comm = comm
        self.controller = controller # Index of the master process
        self.rank = rank
        self.sess_config = sess_config
        self.state_shape = state_shape       # Base state shape, minus the depth (should be 2D) 
        self.state_depth = state_depth       # Depth / 3rd dimension of the input states (i.e. 3 for RGB images)
        self.n_base_acts = n_base_acts       # Number of micro actions available (constant throughout)
        self.n_act_seqs = n_act_seqs         # Number of micro + macro actions currently avaliable and being created
        self.clip_val = ppo_clip_val         # Clip value PPO parameter
        self.val_coef = ppo_val_coef         # How much to weigh value optimization in the PPO update op
        self.entropy_coef = ppo_entropy_coef # How much to weigh entropy in the PPO update op
        self.ppo_iters = ppo_iters           # Number of PPO loops per training batch
        self.target_kl = ppo_kl_limit        # Max KL-divergence before the PPO loop haults
        self.n_curr_acts = n_act_seqs        # Equal to n_act_seqs until the set_act_seqs has been called to update it
                                             # Actual number of micro + macro actions currently available (curr # output nodes)
        self.as_nets = []
        self.create_phs(state_shape=self.state_shape, state_depth=state_depth)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.enc_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.create_encoder_ops(state_shape=state_shape, n_base_acts=self.n_base_acts)
        self.create_policy_ops(n_act_seqs=self.n_act_seqs)
        self.sess.run(tf.global_variables_initializer())
        self.encoder_saver = None
        self.sync_weights()

    def create_as_net(self, obs, acts, n_acts=3, batch_size=32, n_epochs=50, test_frac=0.15, hidden_dims=(128,64,)):
        scope_name = f'as_net_{self.n_act_seqs}'

        with tf.variable_scope(scope_name):
            act_seq_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
            flat_act_seqs = tf.reshape(act_seq_ph, (-1,))

            dense = Dense(hidden_dims[0], activation='relu')(self.obs_op)
            
            ###############################

            dense2 = Dense(hidden_dims[1], activation='relu')(dense)
            act_probs = Dense(self.n_curr_acts, activation=None)(dense2)

            act_ohs = tf.one_hot(act_seq_ph, self.n_curr_acts, dtype=tf.float32)
            
            loss = tf.losses.softmax_cross_entropy(act_ohs, act_probs)
            asn_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            asn_update = asn_optimizer.minimize(loss)

            ###############################

            # act_probs = []
            # act_ohs = []
            # losses = []
            # for i in range(n_acts):
            #     act_indices = tf.range(i, tf.shape(flat_act_seqs)[0], n_acts)
            #     resp_acts = tf.gather(flat_act_seqs, act_indices)
            #     act_ohs.append(tf.one_hot(resp_acts, self.n_curr_acts, dtype=tf.float32))
            #     dense2 = Dense(hidden_dims[1], activation='relu')(dense)
            #     act_probs.append(Dense(self.n_curr_acts, activation=None)(dense2))
            #     losses.append(tf.losses.softmax_cross_entropy(act_ohs[i], act_probs[-1]))
                
            # as_optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            # total_loss = tf.math.add_n(losses)
            # as_update = as_optimizer.minimize(total_loss)

        init_new_vars_ops = [x.initializer for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)]
        self.sess.run(init_new_vars_ops)

        train_obs, test_obs = obs[:-int(len(obs)*test_frac)], obs[-int(len(obs)*test_frac):]
        train_acts, test_acts = acts[:-int(len(acts)*test_frac)], acts[-int(len(acts)*test_frac):]

        for epoch in range(n_epochs):
            correct_preds = 0
            for idx in range(0, len(train_obs), batch_size):
                # Run and train ASN
                aps, _ = self.sess.run([act_probs, asn_update], 
                    feed_dict={
                        self.obs_op: train_obs[idx:idx+batch_size],
                        act_seq_ph: train_acts[idx:idx+batch_size]
                    })

                pred_acts = np.argmax(aps, axis=1)
                correct_preds += (pred_acts == train_acts[idx:idx+batch_size]).sum()
            
            train_acc = correct_preds/len(train_obs)*100

            # Re-run ASN for testing accuracy
            aps = self.sess.run(act_probs, 
                feed_dict={
                    self.obs_op: test_obs,
                    act_seq_ph: test_acts
                })

            pred_acts = np.argmax(aps, axis=1)
            correct_preds = (pred_acts == test_acts).sum()
            test_acc = correct_preds/len(test_obs)*100
                
            print('ASN Epoch {0} | Train acc: {1:.2f}% | Test acc {2:.2f}'.format(
                epoch, train_acc, test_acc))

        # TODO: Add to list of act sequences
        self.n_act_seqs += 1

    def create_phs(self, state_shape=(42, 42), state_depth=1):
        # Placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=(None, *list(state_shape), state_depth))
        self.state_p_ph = tf.placeholder(dtype=tf.float32, shape=(None, *list(state_shape), state_depth))
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.rew_ph = tf.placeholder(dtype=tf.float32, shape=(None,))

    def create_policy_ops(self, n_act_seqs=12):
        with tf.variable_scope('policy'):
            # Creating a conv net for the policy and value estimator
            self.obs_op = Input(shape=(self.enc_dim,))
            dense1 = Dense(256, activation='relu')(self.obs_op)
            act_dense = Dense(256, activation='relu')(dense1)
            val_dense = Dense(256, activation='relu')(dense1)

            # Output probability distribution over possible actions
            self.act_probs_op = Dense(n_act_seqs, activation='softmax', name='act_probs')(act_dense)
            self.act_out = tf.squeeze(tf.random.multinomial(tf.log(self.act_probs_op), 1))

            # Output value of observed state
            self.value_op = Dense(1)(val_dense)

            ### Training Ops ###

            self.act_masks = tf.one_hot(self.act_ph, n_act_seqs, dtype=tf.float32)
            self.log_probs = tf.log(self.act_probs_op)
            self.resp_acts = tf.reduce_sum(self.act_masks * self.act_probs_op, axis=1)

            self.advantages = self.rew_ph - tf.squeeze(self.value_op)

            ### PPO Repeated Pass ###

            self.advatange_ph = tf.placeholder(dtype=tf.float32, shape=self.advantages.shape)
            self.old_probs_ph = tf.placeholder(dtype=tf.float32, shape=self.resp_acts.shape)
            
            self.policy_ratio = self.resp_acts / self.old_probs_ph
            self.clipped_ratio = tf.clip_by_value(self.policy_ratio, 1 - self.clip_val, 1 + self.clip_val)

            self.min_loss = tf.minimum(self.policy_ratio * self.advatange_ph, self.clipped_ratio * self.advatange_ph)
        
            self.optimizer = tf.train.AdamOptimizer()

            ### Policy Update ###

            self.kl_divergence = tf.reduce_mean(tf.log(self.old_probs_ph) - tf.log(self.resp_acts))
            self.actor_loss = -tf.reduce_mean(self.min_loss)
            self.actor_update = self.optimizer.minimize(self.actor_loss)

            ### Value Update ###
        
            self.value_loss = tf.reduce_mean(tf.square(self.rew_ph - tf.squeeze(self.value_op)))
            self.value_update = self.optimizer.minimize(self.value_loss)

            ### Combined Update ###
            
            self.entropy = -tf.reduce_mean(tf.reduce_sum(self.act_probs_op * \
                tf.log(1. / tf.clip_by_value(self.act_probs_op, 1e-8, 1.0)), axis=1))
            self.combined_loss = self.actor_loss + self.val_coef * self.value_loss + self.entropy_coef * self.entropy
            self.combined_update = self.optimizer.minimize(self.combined_loss)

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

    # TODO: This function looks weird
    def choose_action(self, obs, batch_size=1024, epsilon=0.1, possible_acts=None):
        if possible_acts is None:
            return self.sess.run(self.act_out, feed_dict={self.obs_op: obs})

        if np.random.rand() < epsilon:
            return np.random.choice(possible_acts)
        else:
            return self.sess.run(self.act_out, feed_dict={self.obs_op: obs})

    def train_policy(self, states, actions, rewards):
        """
        Trains the policy using PPO. Currently not functional.
        """
        # TODO: Change the way the actions are selected when training the policy
        actions = [a[0] for a in actions] # Currently like this because actions are singular and of the shape (1,)
        states = np.vstack(states)

        self.old_probs, self.old_advantages = self.sess.run([self.resp_acts, self.advantages], 
                                feed_dict={self.obs_op: states,
                                           self.act_ph: actions,
                                           self.rew_ph: rewards})

        for i in range(self.ppo_iters):
            kl_div, _ = self.sess.run([self.kl_divergence, self.combined_update], 
                            feed_dict={self.obs_op: states,
                                       self.act_ph: actions,
                                       self.rew_ph: rewards,
                                       self.old_probs_ph: self.old_probs,
                                       self.advatange_ph: self.old_advantages})
            if kl_div > 1.5 * self.target_kl:
                break
    
        print('# PPO updates: {}'.format(i+1))
    
    def train_vanilla_policy(self, states, actions, rewards):
        """
        Trains the policy using vanilla policy gradient.
        """
        # TODO: Change the way the actions are selected when training the policy
        actions = [a[0] for a in actions] # Currently like this because actions are singular and of the shape (1,)
        states = np.vstack(states)

        loss = tf.reduce_sum(self.act_masks * self.log_probs, axis=1) * self.advantages
        loss = -tf.reduce_mean(loss)

        actor_update = self.optimizer.minimize(loss)

        with tf.control_dependencies([actor_update]):
            value_loss = tf.reduce_mean(tf.square(self.rew_ph - tf.squeeze(self.value_op)))
            value_update = self.optimizer.minimize(value_loss)

        self.sess.run([actor_update, value_update], 
            feed_dict={self.obs_op: states,
                    self.act_ph: actions,
                    self.rew_ph: rewards})
        

    def create_encoder_ops(self, state_shape=(42, 42), n_base_acts=12):
        """
        Creates the encoder used for states
        """
        for size in state_shape:
            assert size % 2**4 == 0, 'state shape must be divisible by 2^4!' 

        with tf.variable_scope('auto_encoder'):
            # State encoder layer ops
            self.enc_layers = [
                Conv2D(16, 3, activation='relu', padding='same'),
                MaxPool2D(2),
                Dropout(rate=0.4),
                Conv2D(32, 3, activation='relu', padding='same'),
                MaxPool2D(2),
                Conv2D(64, 3, activation='relu', padding='same'),
                MaxPool2D(2),
                Dropout(rate=0.4),
                Conv2D(64, 3, activation='relu', padding='same'),
                MaxPool2D(2)
            ]

            self.dec_layers = [
                Conv2D(128, 3, activation='relu', padding='same'),
                UpSampling2D(2),
                Dropout(rate=0.4),
                Conv2D(64, 3, activation='relu', padding='same'),
                UpSampling2D(2),
                Conv2D(32, 3, activation='relu', padding='same'),
                UpSampling2D(2),
                Dropout(rate=0.4),
                Conv2D(32, 3, activation='relu', padding='same'),
                UpSampling2D(2),
                Conv2D(self.state_depth, 3, activation='relu', padding='same')
            ]

            # State encoder output ops
            self.enc_state = self.enc_layers[0](self.state_ph)
            for i in range(1, len(self.enc_layers)):
                self.enc_state = self.enc_layers[i](self.enc_state)
        
            self.enc_vector = Flatten()(self.enc_state) # Used encoded representation op

            self.dec_state = self.enc_state
            for i in range(1, len(self.dec_layers)):
                self.dec_state = self.dec_layers[i](self.dec_state)

            self.enc_dim = self.enc_vector.shape[-1] # Encoded Feature Dimension
            print('Encoder output dimensions:', self.enc_dim)

            # State encoder train ops
            self.auto_enc_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(self.state_ph, self.dec_state))

        self.auto_enc_train_vars = tf.trainable_variables(scope='auto_encoder')
        self.update_auto_enc = self.enc_optimizer.minimize(self.auto_enc_loss, var_list=self.auto_enc_train_vars)

    def apply_encoder(self, state):
        if type(state) is not np.ndarray:
            state = np.asarray(state)

        if state.shape == 2:
            state = [state]
            return self.sess.run(self.enc_vector, feed_dict={self.state_ph: state})

        return self.sess.run(self.enc_vector, feed_dict={self.state_ph: state})

    def train_encoder(self, states, batch_size=64, save_path=None):
        formatted_states = np.stack(states)

        correct = 0
        losses = 0
        for batch_idx in range(int(np.ceil(len(formatted_states) / batch_size))):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            loss, _ = self.sess.run([self.auto_enc_loss, self.update_auto_enc], 
                    feed_dict={ 
                        self.state_ph: formatted_states[start_idx:end_idx]
                    })
            
            losses += loss

        if save_path:
            self.save_encoder(save_path)

        return losses / (batch_idx + 1)

    def save_encoder(self, path):
        """
        Creates a saver for the encoder if one does not exist, and then uses it to save the encoder variables.
        """
        if self.encoder_saver is None:
            self.saver = tf.train.Saver(tf.trainable_variables(scope='auto_encoder'))

        self.saver.save(self.sess, path)
        print('Saved encoder model to {}'.format(path))

    def load_encoder(self, path):
        """
        Loads the encoder from the specified model file path.
        """
        if self.encoder_saver is None:
            self.saver = tf.train.Saver(tf.trainable_variables(scope='auto_encoder'))

        self.saver.restore(self.sess, path)

    def sync_weights(self):
        if self.rank == self.controller:
            self.comm.bcast(self.sess.run(tf.trainable_variables()), self.controller)
        else:
            sync_vars = self.comm.bcast(None, self.controller)
            t_vars = tf.trainable_variables()
            for pair in zip(t_vars, sync_vars):
                self.sess.run(tf.assign(pair[0], pair[1]))