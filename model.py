from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, ZeroPadding2D, Flatten
import tensorflow as tf

class HASL():
    def __init__(self, sess, comm, controller, rank, state_shape=(42, 42), n_actions=12):
        self.comm = comm
        self.controller = controller
        self.rank = rank
        self.sess = sess
        self.create_phs(state_shape=state_shape)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.create_encoder_ops(state_shape=state_shape, n_actions=n_actions)
        self.sess.run(tf.global_variables_initializer())
        self.sync_weights()

    def create_phs(self, state_shape=(42, 42)):
        # Placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=(None, *list(state_shape), 1))
        self.state_p_ph = tf.placeholder(dtype=tf.float32, shape=(None, *list(state_shape), 1))
        self.act_ph = tf.placeholder(dtype=tf.int32, shape=(None,))

    def create_encoder_ops(self, state_shape=(42, 42), n_actions=12):
        """
        Creates the encoder used for states
        """
        enc_dim = 288 # Encoded Feature Dimension

        # State encoder layer ops
        with tf.variable_scope('encoder'):
            self.enc_layers = [
                ZeroPadding2D(),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                ZeroPadding2D(),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                ZeroPadding2D(),
                Conv2D(32, 3, strides=(2, 2), activation='elu'),
                ZeroPadding2D(),
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
            self.act_pred = Dense(n_actions, activation='softmax', use_bias=False)(self.im_dense)

        # State encoder train ops
        self.act_actual_oh = tf.one_hot(self.act_ph, n_actions)
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

    def train_encoder(self, states, state_ps, actions):
        tf.run(self.update_im, feed_dict={ 
                                            self.state_ph: states,
                                            self.state_p_ph: state_ps,
                                            self.act_ph: actions
                                         })

    def sync_weights(self):
        if self.rank == self.controller:
            self.comm.bcast(self.sess.run(tf.trainable_variables()), self.controller)
        else:
            sync_vars = self.comm.bcast(None, self.controller)
            t_vars = tf.trainable_variables()
            for pair in zip(t_vars, sync_vars):
                self.sess.run(tf.assign(pair[0], pair[1]))