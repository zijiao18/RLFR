import tensorflow as tf
#source ~/Desktop/tensorflow/bin/activate


class ActorNetwork():
    def __init__(
        self,
        sess,
        name,
        time_step,
        obs_dim,
        vel_dim,
        dir_dim,
        batch_size,
        lstm_state_dim,
        n_fc1_unit,
        n_fc2_unit,
        learning_rate,
        tau,
        training,
        master_network=None
    ):
        self.sess = sess
        self.name = name
        self.act_dim = 1
        self.obs_dim = obs_dim
        self.vel_dim = vel_dim
        self.dir_dim = dir_dim
        self.input_time_step = time_step
        self.input_batch_size = batch_size
        self.lstm_state_size = lstm_state_dim
        self.fc1_size = n_fc1_unit
        self.fc2_size = n_fc2_unit
        self.tau = tau
        self.learning_rate = learning_rate
        self.training = training
        self.net_param_offset=len(tf.trainable_variables())
        print("%s has params offset %d"%(
            name, 
            self.net_param_offset
            )
        )
        (
            self.obs_in, 
            self.vel_in, 
            self.dir_in, 
            self.act_out
        ) = self.build_network(self.name)
        self.net_params = tf.trainable_variables()[
            self.net_param_offset:
        ]
        (
            self.target_obs_in, 
            self.target_vel_in, 
            self.target_dir_in, 
            self.target_act_out
        ) = self.build_network(self.name+'_target')
        self.target_net_params = tf.trainable_variables()[
            self.net_param_offset+len(self.net_params):
        ]
        self.update_target_net_params = [
            self.target_net_params[i].assign(
                tf.multiply(
                    self.net_params[i], 
                    self.tau
                ) 
                +
                tf.multiply(
                    self.target_net_params[i],
                    1.0-self.tau
                )
            )
            for i in range(len(self.target_net_params))
        ]
        self.num_trainable_vars = (len(self.net_params)
                                + len(self.target_net_params))

        self.action_gradients = tf.placeholder(
            dtype=tf.float32, 
            shape=[1,None, self.act_dim]
        )
        self.actor_params_gradients = tf.gradients(
            ys=self.act_out, 
            xs=self.net_params,
            grad_ys=-self.action_gradients[0,:,:]
        )
        self.actor_norm_gradients = [
            tf.div(g, self.input_batch_size) 
            for g in self.actor_params_gradients
        ]
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.apply_gradients(
            grads_and_vars=zip(
                self.actor_norm_gradients, 
                self.net_params
            )
        )
        self.master_network = master_network
        if not master_network==None:
            self.apply_grads = self.optimizer.apply_gradients(
                grads_and_vars=zip(
                    self.actor_norm_gradients,
                    self.master_network.net_params
                )
            )
            self.copy_master_params = [
                self.net_params[i].assign(
                    self.master_network.net_params[i]
                ) 
                for i in range(len(self.master_network.net_params))
            ]


    def build_network(self, rnn_scope):
        obs_in=tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.input_time_step,
                None,
                self.obs_dim
            ]
        )
        vel_in=tf.placeholder(
            dtype=tf.float32,
            shape=[1,None,self.vel_dim]
        )
        dir_in=tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.dir_dim
            ]
        )
        lstm_in = tf.reverse(obs_in, [0])
        lstm = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.lstm_state_size,
            name='lstm'
        )
        lstm = tf.nn.rnn_cell.DropoutWrapper(
            lstm, 
            state_keep_prob=0.6
        )
        hs, _ = tf.nn.static_rnn(
            cell=lstm,
            inputs=tf.unstack(lstm_in),
            dtype=tf.float32,
            scope=rnn_scope
        )
        fc1_in=tf.concat(
            [
                tf.concat(
                    [
                        hs[-1],
                        vel_in[0,:,:] 
                    ],
                    1
                ),
                dir_in[0,:,:]
            ],
            1
        )
        fc1=tf.layers.dense(
            inputs=fc1_in,
            units=self.fc1_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        fc1 = tf.layers.dropout(
            inputs=fc1,
            rate=0.2,
            training=self.training
        )
        fc2=tf.layers.dense(
            inputs=fc1,
            units=self.fc2_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        fc2 = tf.layers.dropout(
            inputs=fc2,
            rate=0.2,
            training=self.training
        )
        act_out=tf.layers.dense(
            inputs=fc2,
            units=self.act_dim,
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        return obs_in, vel_in, dir_in, act_out

    def train(
        self, 
        obs_batch,
        vel_batch,
        dir_batch,
        act_grads
    ):
        return self.sess.run(
            self.optimize, 
            feed_dict={
                self.obs_in: obs_batch,
                self.vel_in: vel_batch,
                self.dir_in: dir_batch,
                self.action_gradients: act_grads
            }
        )

    def update_master_network(
        self,
        obs_batch,
        vel_batch,
        dir_batch,
        act_grads
    ):
        self.sess.run(
            self.apply_grads, 
            feed_dict={
                self.obs_in: obs_batch,
                self.vel_in: vel_batch,
                self.dir_in: dir_batch,
                self.action_gradients: act_grads
            }
        )

    def predict(
        self, 
        obs_batch,
        vel_batch,
        dir_batch
    ):
        return self.sess.run(
            self.act_out, 
            feed_dict={
                self.obs_in: obs_batch,
                self.vel_in: vel_batch,
                self.dir_in: dir_batch
            }
        )

    def predict_target(
        self, 
        obs_batch,
        vel_batch,
        dir_batch
    ):
        return self.sess.run(
            self.target_act_out, 
            feed_dict={
                self.target_obs_in: obs_batch,
                self.target_vel_in: vel_batch,
                self.target_dir_in: dir_batch
            }
        )

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

    def copy_master_network(self):
        self.sess.run(self.copy_master_params)

    def print_net_params(self):
        for p in self.net_params:
            self.sess.run(
                tf.Print(
                    p,
                    [p.get_shape()],
                    self.name+' net_params'
                )
            )
        for tp in self.target_net_params:
            self.sess.run(
                tf.Print(
                    tp,
                    [tp.get_shape()],
                    self.name+' target_net_params'
                )
            )
        

class CriticNetwork():
    def __init__(
        self,
        sess,
        name,
        time_step,
        obs_dim,
        vel_dim,
        dir_dim,
        batch_size,
        lstm_state_dim,
        n_fc1_unit,
        n_fc2_unit,
        learning_rate,
        tau,
        training,
        master_network=None
    ):
        self.sess = sess
        self.name = name
        self.act_dim = 1 
        self.obs_dim = obs_dim
        self.vel_dim = vel_dim
        self.dir_dim = dir_dim
        self.input_time_step = time_step
        self.input_batch_size = batch_size
        self.lstm_state_size = lstm_state_dim
        self.fc1_size = n_fc1_unit
        self.fc2_size = n_fc2_unit
        self.learning_rate = learning_rate
        self.tau = tau
        self.training = training
        self.net_param_offset = len(tf.trainable_variables())
        print("%s has params offset %d"%(
                name, 
                self.net_param_offset
            )
        )
        (
            self.obs_in, 
            self.vel_in, 
            self.dir_in, 
            self.act_in, 
            self.q_out
        ) = self.build_network(self.name)
        self.net_params = tf.trainable_variables()[
            self.net_param_offset:
        ]
        (
            self.target_obs_in, 
            self.target_vel_in, 
            self.target_dir_in, 
            self.target_act_in,
            self.target_q_out
        ) = self.build_network(self.name+'_target')
        self.target_net_params = tf.trainable_variables()[
            (self.net_param_offset+len(self.net_params)):
        ]
        self.update_target_net_params = [
            self.target_net_params[i].assign(
                tf.multiply(
                    self.net_params[i], 
                    self.tau
                ) 
                +
                tf.multiply(
                    self.target_net_params[i], 
                    1.0 - self.tau
                )
            )
            for i in range(len(self.target_net_params))
        ]
        self.target_q_value = tf.placeholder(
            tf.float32, 
            [None, 1]
        )
        self.loss = tf.reduce_mean(
            tf.squared_difference(
                self.target_q_value, 
                self.q_out
            )
        )
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimizer.minimize(
            self.loss,
            var_list=self.net_params
        )
        self.loss_grads = tf.gradients(
            ys=self.loss,
            xs=self.net_params
        )
        self.loss_norm_grads = [
            tf.div(g, self.input_batch_size) 
            for g in self.loss_grads
        ]
        self.action_grads = tf.gradients(
            self.q_out, 
            self.act_in
        )
        self.master_network = master_network
        if not master_network==None:
            self.apply_grads = self.optimizer.apply_gradients(
                grads_and_vars=zip(
                    self.loss_norm_grads,
                    self.master_network.net_params
                )
            )
            self.copy_master_params = [
                self.net_params[i].assign(
                    self.master_network.net_params[i]
                ) 
                for i in range(len(self.net_params))
            ]


    def build_network(self,rnn_scope):
        obs_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.input_time_step,
                None,
                self.obs_dim
            ]
        )
        vel_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.vel_dim
            ]
        )
        dir_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.dir_dim
            ]
        )
        act_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.act_dim
            ]
        )
        lstm_in = tf.reverse(obs_in,[0])
        lstm = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.lstm_state_size
        )
        lstm = tf.nn.rnn_cell.DropoutWrapper(
            lstm, 
            state_keep_prob=0.6
        )
        hs, _ = tf.nn.static_rnn(
            cell=lstm,
            inputs=tf.unstack(lstm_in),
            dtype=tf.float32,
            scope=rnn_scope
        )
        state = tf.concat(
            [
                hs[-1],
                tf.concat(
                    [
                        vel_in[0,:,:],
                        dir_in[0,:,:]
                    ],
                    1
                )
            ],
            1
        )
        fc1_in = tf.concat(
            [
                state,
                act_in[0,:,:]
            ],
            1
        )
        fc1 = tf.layers.dense(
            inputs=fc1_in,
            units=self.fc1_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        fc1 = tf.layers.dropout(
            inputs=fc1,
            rate=0.2,
            training=self.training
        )
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=self.fc2_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        fc2 = tf.layers.dropout(
            inputs=fc2,
            rate=0.2,
            training=self.training
        )
        q_out = tf.layers.dense(
            inputs=fc2,
            units=1,
            activation=None,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        return obs_in, vel_in, dir_in, act_in, q_out

    def train(
        self, 
        obs_batch,
        vel_batch,
        dir_batch,
        act_batch,
        tar_q_batch
    ):
        return self.sess.run(
            [self.q_out, self.optimize], 
            feed_dict={
                self.obs_in: obs_batch,
                self.vel_in: vel_batch,
                self.dir_in: dir_batch,
                self.act_in: act_batch,
                self.target_q_value: tar_q_batch
            }
        )

    def predict(
        self, 
        obs_batch,
        vel_batch,
        dir_batch,
        act_batch
    ):
        return self.sess.run(
            self.q_out, 
            feed_dict={
                self.obs_in: obs_batch,
                self.vel_in: vel_batch,
                self.dir_in: dir_batch,
                self.act_in: act_batch
            }
        )

    def predict_target(
        self, 
        obs_batch,
        vel_batch,
        dir_batch,
        act_batch
    ):
        return self.sess.run(
            self.target_q_out, 
            feed_dict={
                self.target_obs_in: obs_batch,
                self.target_vel_in: vel_batch,
                self.target_dir_in: dir_batch,
                self.target_act_in: act_batch
            }
        )

    def action_gradients(
        self, 
        obs_batch, 
        vel_batch, 
        dir_batch, 
        act_batch
    ):
        return self.sess.run(
            self.action_grads, 
            feed_dict={
                self.obs_in: obs_batch, 
                self.vel_in: vel_batch,
                self.dir_in: dir_batch,
                self.act_in: act_batch
            }
        )

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def update_master_network(
        self,
        obs_batch,
        vel_batch,
        dir_batch,
        act_batch,
        tar_q_batch
    ):
        self.sess.run(
            self.apply_grads, 
            feed_dict={
                self.obs_in: obs_batch,
                self.vel_in: vel_batch,
                self.dir_in: dir_batch,
                self.act_in: act_batch,
                self.target_q_value: tar_q_batch
            }
        )

    def copy_master_network(self):
        self.sess.run(self.copy_master_params)

    def print_net_params(self):
        for p in self.net_params:
            self.sess.run(
                tf.Print(
                    p,
                    [p.get_shape()],
                    self.name+' net_params'
                )
            )
        for tp in self.target_net_params:
            self.sess.run(
                tf.Print(
                    tp,
                    [tp.get_shape()],
                    self.name+' target_net_params'
                )
            )

