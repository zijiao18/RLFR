#!/usr/bin/env python
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
        act_dim,
        batch_size,
        lstm_state_dim,
        n_fc1_unit,
        n_fc2_unit,
        n_fc3_unit,
        learning_rate,
        tau,
        device,
        master_network=None
    ):
        self.sess = sess
        self.name = name
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.vel_dim = vel_dim
        self.dir_dim = dir_dim
        self.time_step = time_step
        self.batch_size = batch_size
        self.lstm_state_size = lstm_state_dim
        self.fc1_size = n_fc1_unit
        self.fc2_size = n_fc2_unit
        self.fc3_size = n_fc3_unit
        self.tau = tau
        self.learning_rate = learning_rate
        self.net_param_offset = len(tf.trainable_variables())
        print("%s has params offset %d"%(
                name, 
                self.net_param_offset
            )
        )
        with tf.device(device):
            (
                self.obs_in,
                self.dir_in,
                self.ind_in,
                self.act_out
            ) = self.build_network(rnn_scope=self.name)
            self.net_params=tf.trainable_variables()[
                self.net_param_offset:
            ]
            (
                self.target_obs_in,
                self.target_dir_in,
                self.target_ind_in,
                self.target_act_out
            ) = self.build_network(rnn_scope=self.name+'_target')
            self.target_net_params=tf.trainable_variables()[
                self.net_param_offset+len(self.net_params):
            ]
            self.action_gradients = tf.placeholder(
                dtype=tf.float32, 
                shape=[
                    1,
                    None, 
                    self.act_dim
                ]
            )
            self.actor_params_gradients = tf.gradients(
                ys=self.act_out, 
                xs=self.net_params,
                grad_ys= -self.action_gradients[0,:,:]
            )
            self.actor_norm_gradients = [
                tf.div(g, self.batch_size) 
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
                self.init_target_net_params = [
                    self.target_net_params[i].assign(
                        self.master_network.net_params[i]) 
                    for i in range(len(self.master_network.net_params))
                ]
                self.update_target_net_params = [
                    self.target_net_params[i].assign(
                        tf.multiply(
                            self.master_network.net_params[i],
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

        self.tb_actor_norm_gradients = [
            tf.summary.histogram(
                self.name+'_pg_'+str(i), 
                self.actor_norm_gradients[i]
            ) 
            for i in xrange(len(self.actor_norm_gradients))
        ]

    def build_network(self,rnn_scope):
        obs_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.time_step,
                None,
                self.obs_dim
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
        ind_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                1
            ]
        )
        lstm_in = tf.reverse(obs_in,[0])
        lstm = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.lstm_state_size,
            name='lstm'
        )
        hs, _ = tf.nn.static_rnn(
            cell=lstm,
            inputs=tf.unstack(lstm_in),
            dtype=tf.float32,
            scope=rnn_scope
        )
        fc1_in = tf.concat(
            [
                hs[-1],
                dir_in[0,:,:],
                ind_in[0,:,:]
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
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=self.fc2_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        fc3 = tf.layers.dense(
            inputs=fc2,
            units=self.fc3_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        lvel_out = tf.layers.dense(
            inputs=fc3,
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )

        avel_out = tf.layers.dense(
            inputs=fc3,
            units=1,
            activation=tf.nn.tanh,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        act_out = tf.concat(
            [lvel_out, avel_out],
            1
        )
        return obs_in, dir_in, ind_in, act_out

    def train(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_grads
    ):
        return self.sess.run(
            self.optimize, 
            feed_dict={
                self.obs_in: obs_batch,
                self.dir_in: dir_batch,
                self.ind_in: ind_batch,
                self.action_gradients: act_grads
            }
        )

    def update_master_network(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_grads
    ):
        self.sess.run(
            self.apply_grads, 
            feed_dict={
                self.obs_in: obs_batch,
                self.dir_in: dir_batch,
                self.ind_in: ind_batch,
                self.action_gradients: act_grads
            }
        )

    def predict(
        self, 
        obs_batch,
        dir_batch,
        ind_batch
    ):
        return self.sess.run(
            self.act_out, 
            feed_dict={
                self.obs_in: obs_batch,
                self.dir_in: dir_batch,
                self.ind_in: ind_batch
            }
        )

    def predict_target(
        self,
        obs_batch,
        dir_batch,
        ind_batch
    ):
        return self.sess.run(
            self.target_act_out, 
            feed_dict={
                self.target_obs_in: obs_batch,
                self.target_dir_in: dir_batch,
                self.target_ind_in: ind_batch
            }
        )

    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)


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
    
    def summary(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_grads
    ):
        return self.sess.run(
            self.tb_actor_norm_gradients, 
            feed_dict={
                self.obs_in: obs_batch,
                self.dir_in: dir_batch,
                self.ind_in: ind_batch,
                self.action_gradients: act_grads
            }
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
        act_dim,
        batch_size,
        lstm_state_dim,
        n_fc1_unit,
        n_fc2_unit,
        n_fc3_unit,
        learning_rate,
        tau,
        device,
        master_network=None
    ):
        self.sess = sess
        self.name = name
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.vel_dim = vel_dim
        self.dir_dim = dir_dim
        self.time_step = time_step
        self.batch_size = batch_size
        self.lstm_state_size = lstm_state_dim
        self.fc1_size = n_fc1_unit
        self.fc2_size = n_fc2_unit
        self.fc3_size = n_fc3_unit
        self.learning_rate = learning_rate
        self.tau = tau
        self.net_param_offset = len(tf.trainable_variables())
        print("%s has params offset %d"%(
                name, 
                self.net_param_offset
            )
        )
        with tf.device(device):
            (
                self.obs0_in,
                self.obs1_in,
                self.obs2_in,
                self.dir0_in,
                self.dir1_in,
                self.dir2_in,
                self.act0_in,
                self.act1_in,
                self.act2_in,
                self.ind_in,
                self.q_out
            ) = self.build_network(rnn_scope=self.name)
            self.net_params = tf.trainable_variables()[
                self.net_param_offset:
            ]
            (
                self.target_obs0_in,
                self.target_obs1_in,
                self.target_obs2_in,
                self.target_dir0_in,
                self.target_dir1_in,
                self.target_dir2_in,
                self.target_act0_in,
                self.target_act1_in,
                self.target_act2_in,
                self.target_ind_in,
                self.target_q_out
            ) = self.build_network(rnn_scope=self.name+'_target')
            self.target_net_params = tf.trainable_variables()[
                (self.net_param_offset+len(self.net_params)):
            ]
            self.target_q_value = tf.placeholder(
                dtype=tf.float32, 
                shape=[None, 1]
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
                tf.div(g, self.batch_size) 
                for g in self.loss_grads
            ]
            self.action0_grads = tf.gradients(self.q_out, self.act0_in)
            self.action1_grads = tf.gradients(self.q_out, self.act1_in)
            self.action2_grads = tf.gradients(self.q_out, self.act2_in)
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
                self.init_target_net_params = [
                    self.target_net_params[i].assign(
                        self.master_network.net_params[i]
                    ) 
                    for i in range(len(self.master_network.net_params))
                ]
                self.update_target_net_params = [
                    self.target_net_params[i].assign(
                        tf.multiply(
                            self.master_network.net_params[i], 
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
        self.tb_loss = tf.summary.scalar(
            self.name+'_loss',
            self.loss
        )

    def build_network(self,rnn_scope):
        obs0_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.time_step,
                None,
                self.obs_dim
            ]
        )
        obs1_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.time_step,
                None,
                self.obs_dim
            ]
        )
        obs2_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                self.time_step,
                None,
                self.obs_dim
            ]
        )
        dir0_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.dir_dim
            ]
        )
        dir1_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.dir_dim
            ]
        )
        dir2_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.dir_dim
            ]
        )
        act0_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.act_dim
            ]
        )
        act1_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.act_dim
            ]
        )
        act2_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                self.act_dim
            ]
        )
        ind_in = tf.placeholder(
            dtype=tf.float32,
            shape=[
                1,
                None,
                1
            ]
        )
        joint_dir_in = tf.concat(
            [
                dir0_in,
                dir1_in,
                dir2_in
            ],
            2
        )
        joint_act_in = tf.concat(
            [
                act0_in,
                act1_in,
                act2_in
            ],
            2
        )
        lstm = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.lstm_state_size
        )
        lstm_in_0 = tf.reverse(obs0_in, [0])    
        h0, _ = tf.nn.static_rnn(
            cell=lstm,
            inputs=tf.unstack(lstm_in_0),
            dtype=tf.float32,
            scope=rnn_scope
        )
        lstm_in_1 = tf.reverse(obs1_in, [0])    
        h1, _ = tf.nn.static_rnn(
            cell=lstm,
            inputs=tf.unstack(lstm_in_1),
            dtype=tf.float32,
            scope=rnn_scope
        )
        lstm_in_2 = tf.reverse(obs2_in, [0])    
        h2, _ = tf.nn.static_rnn(
            cell=lstm,
            inputs=tf.unstack(lstm_in_2),
            dtype=tf.float32,
            scope=rnn_scope
        )
        fc1_in = tf.concat(
            [
                h0[-1], 
                h1[-1], 
                h2[-1], 
                joint_dir_in[0,:,:], 
                joint_act_in[0,:,:], 
                ind_in[0,:,:]
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
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=self.fc2_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        fc3 = tf.layers.dense(
            inputs=fc2,
            units=self.fc3_size,
            activation=tf.nn.relu,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        q_out = tf.layers.dense(
            inputs=fc3,
            units=1,
            activation=None,
            kernel_initializer=tf.keras.initializers.he_normal(),
            bias_initializer=tf.initializers.zeros()
        )
        return (obs0_in,obs1_in,obs2_in,
                dir0_in,dir1_in,dir2_in,
                act0_in,act1_in,act2_in,
                ind_in,q_out)

    def train(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_batch,
        tar_q_batch
    ):
        return self.sess.run(
            [self.q_out, self.optimize], 
            feed_dict={
                self.obs0_in: obs_batch[0],
                self.obs1_in: obs_batch[1],
                self.obs2_in: obs_batch[2],
                self.dir0_in: dir_batch[0],
                self.dir1_in: dir_batch[1],
                self.dir2_in: dir_batch[2],
                self.act0_in: act_batch[0],
                self.act1_in: act_batch[1],
                self.act2_in: act_batch[2],
                self.ind_in: ind_batch,
                self.target_q_value: tar_q_batch
            }
        )

    def predict(
        self, 
        obs_batch,
        dir_batch,
        ind_batch,
        act_batch
    ):
        return self.sess.run(
            self.q_out, 
            feed_dict={
                self.obs0_in: obs_batch[0],
                self.obs1_in: obs_batch[1],
                self.obs2_in: obs_batch[2],
                self.dir0_in: dir_batch[0],
                self.dir1_in: dir_batch[1],
                self.dir2_in: dir_batch[2],
                self.act0_in: act_batch[0],
                self.act1_in: act_batch[1],
                self.act2_in: act_batch[2],
                self.ind_in: ind_batch
            }
        )

    def predict_target(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_batch
    ):
        return self.sess.run(
            self.target_q_out, 
            feed_dict={
                self.target_obs0_in: obs_batch[0],
                self.target_obs1_in: obs_batch[1],
                self.target_obs2_in: obs_batch[2],
                self.target_dir0_in: dir_batch[0],
                self.target_dir1_in: dir_batch[1],
                self.target_dir2_in: dir_batch[2],
                self.target_act0_in: act_batch[0],
                self.target_act1_in: act_batch[1],
                self.target_act2_in: act_batch[2],
                self.target_ind_in: ind_batch
            }
        )

    def action_gradients(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_batch,
        ind_scalar
    ):
        if ind_scalar==0:
            return self.sess.run(
                self.action0_grads, 
                feed_dict={
                    self.obs0_in: obs_batch[0],
                    self.obs1_in: obs_batch[1],
                    self.obs2_in: obs_batch[2],
                    self.dir0_in: dir_batch[0],
                    self.dir1_in: dir_batch[1],
                    self.dir2_in: dir_batch[2],
                    self.act0_in: act_batch[0],
                    self.act1_in: act_batch[1],
                    self.act2_in: act_batch[2],
                    self.ind_in: ind_batch
                }
            )
        if ind_scalar==1:
            return self.sess.run(
                self.action1_grads, 
                feed_dict={
                    self.obs0_in: obs_batch[0],
                    self.obs1_in: obs_batch[1],
                    self.obs2_in: obs_batch[2],
                    self.dir0_in: dir_batch[0],
                    self.dir1_in: dir_batch[1],
                    self.dir2_in: dir_batch[2],
                    self.act0_in: act_batch[0],
                    self.act1_in: act_batch[1],
                    self.act2_in: act_batch[2],
                    self.ind_in: ind_batch
                }
            )
        if ind_scalar==2:
            return self.sess.run(
                self.action2_grads, 
                feed_dict={
                    self.obs0_in: obs_batch[0],
                    self.obs1_in: obs_batch[1],
                    self.obs2_in: obs_batch[2],
                    self.dir0_in: dir_batch[0],
                    self.dir1_in: dir_batch[1],
                    self.dir2_in: dir_batch[2],
                    self.act0_in: act_batch[0],
                    self.act1_in: act_batch[1],
                    self.act2_in: act_batch[2],
                    self.ind_in: ind_batch
                }
            )
    
    def init_target_network(self):
        self.sess.run(self.init_target_net_params)

    def update_target_network(self):
        self.sess.run(self.update_target_net_params)

    def update_master_network(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_batch,
        tar_q_batch
    ):
        self.sess.run(
            self.apply_grads, 
            feed_dict={
                self.obs0_in: obs_batch[0],
                self.obs1_in: obs_batch[1],
                self.obs2_in: obs_batch[2],
                self.dir0_in: dir_batch[0],
                self.dir1_in: dir_batch[1],
                self.dir2_in: dir_batch[2],
                self.act0_in: act_batch[0],
                self.act1_in: act_batch[1],
                self.act2_in: act_batch[2],
                self.ind_in: ind_batch,
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

    def summary(
        self,
        obs_batch,
        dir_batch,
        ind_batch,
        act_batch,
        tar_q_batch
    ):
        return self.sess.run(
            self.tb_loss, 
            feed_dict={
                self.obs0_in: obs_batch[0],
                self.obs1_in: obs_batch[1],
                self.obs2_in: obs_batch[2],
                self.dir0_in: dir_batch[0],
                self.dir1_in: dir_batch[1],
                self.dir2_in: dir_batch[2],
                self.act0_in: act_batch[0],
                self.act1_in: act_batch[1],
                self.act2_in: act_batch[2],
                self.ind_in: ind_batch,
                self.target_q_value: tar_q_batch
            }
        )

