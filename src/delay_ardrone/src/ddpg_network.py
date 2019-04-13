import tensorflow as tf
#source ~/Desktop/tensorflow/bin/activate
class ActorNetwork():
	def __init__(self,sess,name,time_step,obs_dim,vel_dim,dir_dim,batch_size,
				lstm_state_dim,n_fc1_unit,n_fc2_unit,learning_rate,tau,master_network=None):
		self.sess=sess
		self.name=name
		self.act_dim=1
		self.obs_dim=obs_dim
		self.vel_dim=vel_dim
		self.dir_dim=dir_dim
		self.input_time_step=time_step
		self.input_batch_size=batch_size
		self.lstm_state_size=lstm_state_dim
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.tau=tau
		self.learning_rate=learning_rate
		
		#with g.name_scope(self.name):
		self.net_param_offset=len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		# Actor network
		self.obs_in, self.vel_in, self.dir_in, self.act_out = self.build_network(self.name)
		self.net_params = tf.trainable_variables()[self.net_param_offset:]	#all the weights and bias in actor network
		# Target network
		self.target_obs_in, self.target_vel_in, self.target_dir_in, self.target_act_out = self.build_network(self.name+'_target')
		self.target_net_params = tf.trainable_variables()[self.net_param_offset+len(self.net_params):]	#all the weights and bias in target actor network
		# Op for periodically updating target network with online network weights
		self.update_target_net_params = [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
                                  									tf.multiply(self.target_net_params[i],1.0-self.tau))
										for i in range(len(self.target_net_params))]

		self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

		# Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
		# Temporary placeholder action gradient
		self.action_gradients = tf.placeholder(tf.float32, [1,None, self.act_dim])
		#self.action_gradients = tf.Print(self.action_gradients,[self.action_gradients])

		self.actor_params_gradients = tf.gradients(ys=self.act_out, xs=self.net_params,grad_ys= -self.action_gradients[0,:,:])
		#self.actor_params_gradients = [tf.Print(g,[g.get_shape()],"ag") for g in self.actor_params_gradients]

		self.actor_norm_gradients = [tf.div(g,self.input_batch_size) for g in self.actor_params_gradients]
		# Optimization Op
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		
		self.optimize = self.optimizer.apply_gradients(grads_and_vars=zip(self.actor_norm_gradients, self.net_params))
		
		self.master_network=master_network
		if not master_network==None:
			self.apply_grads=self.optimizer.apply_gradients(grads_and_vars=zip(self.actor_norm_gradients,self.master_network.net_params))
			self.copy_master_params = [self.net_params[i].assign(self.master_network.net_params[i]) 
								  		for i in range(len(self.master_network.net_params))]


	def build_network(self,rnn_scope):
		obs_in=tf.placeholder(dtype=tf.float32,shape=[self.input_time_step,None,self.obs_dim])
		vel_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.vel_dim])
		dir_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.dir_dim])
		lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size,name='lstm')	#lstm_state_size=8
		
		hs,c=tf.nn.static_rnn(cell=lstm,
							inputs=tf.unstack(tf.reverse(obs_in,[0])),
							dtype=tf.float32,
							scope=rnn_scope)
		#hs=tf.Print(hs,[hs[-1]],"act_h")
		fc1_in=tf.concat([tf.concat([hs[-1],vel_in[0,:,:] ],1),dir_in[0,:,:]],1)
		fc1=tf.layers.dense(inputs=fc1_in,
							units=self.fc1_size,
							activation=tf.nn.relu,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
							bias_initializer=tf.initializers.zeros())
		#fc1 = tf.layers.dropout(inputs=fc1,rate=0.2,training=True)
		#fc1=tf.Print(fc1,[fc1],"act_fc1")
		fc2=tf.layers.dense(inputs=fc1,
							units=self.fc2_size,
							activation=tf.nn.relu,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
							bias_initializer=tf.initializers.zeros())
		#fc2 = tf.layers.dropout(inputs=fc2,rate=0.2,training=True)
		#fc2=tf.Print(fc2,[fc2],"act_fc2")
		act_out=tf.layers.dense(inputs=fc2,
								units=self.act_dim,
								activation=tf.nn.tanh,
								kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
								bias_initializer=tf.initializers.zeros())
		#act_out: 1x4
		#act_out=tf.multiply(act_out,self.action_bound)
		#act_out=tf.Print(act_out,[act_out],"act_act_out")
		return obs_in, vel_in, dir_in, act_out

	def train(self, obs_batch,vel_batch,dir_batch,act_grads):
		# args [inputs, action_gradients, phase]
		return self.sess.run(self.optimize, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.action_gradients: act_grads
		})

	def update_master_network(self,obs_batch,vel_batch,dir_batch,act_grads):
		self.sess.run(self.apply_grads, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.action_gradients: act_grads
			})
		#print(self.name,"updated master network")

	def predict(self, obs_batch,vel_batch,dir_batch):
		return self.sess.run(self.act_out, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch
		})

	def predict_target(self, obs_batch,vel_batch,dir_batch):
		return self.sess.run(self.target_act_out, feed_dict={
			self.target_obs_in: obs_batch,
			self.target_vel_in: vel_batch,
			self.target_dir_in: dir_batch
		})

	def update_target_network(self):
		self.sess.run(self.update_target_net_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars

	def copy_master_network(self):
		self.sess.run(self.copy_master_params)

	def print_net_params(self):
		for p in self.net_params:
			self.sess.run(tf.Print(p,[p.get_shape()],self.name+' net_params'))
		for tp in self.target_net_params:
			self.sess.run(tf.Print(tp,[tp.get_shape()],self.name+' target_net_params'))
		


class CriticNetwork():

	def __init__(self,sess,name,time_step,obs_dim,vel_dim,dir_dim,batch_size,
				lstm_state_dim,n_fc1_unit,n_fc2_unit,learning_rate,tau,master_network=None):
		self.sess=sess
		self.name=name
		self.act_dim=1
		self.obs_dim=obs_dim
		self.vel_dim=vel_dim
		self.dir_dim=dir_dim
		self.input_time_step=time_step
		self.input_batch_size=batch_size
		self.lstm_state_size=lstm_state_dim
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.learning_rate=learning_rate
		self.tau=tau

		self.net_param_offset=len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		# Critic network
		self.obs_in, self.vel_in, self.dir_in, self.act_in, self.q_out = self.build_network(self.name)
		self.net_params = tf.trainable_variables()[self.net_param_offset:]
		# Target network
		self.target_obs_in, self.target_vel_in, self.target_dir_in, self.target_act_in,self.target_q_out = self.build_network(self.name+'_target')
		self.target_net_params = tf.trainable_variables()[(self.net_param_offset+len(self.net_params)):]
		# Op for periodically updating target network with online network weights
		self.update_target_net_params = [self.target_net_params[i].assign(tf.multiply(self.net_params[i], self.tau) +
																tf.multiply(self.target_net_params[i], 1. - self.tau))
										for i in range(len(self.target_net_params))]

		# Network target (y_i)
		# Obtained from the target networks
		self.target_q_value = tf.placeholder(tf.float32, [None, 1])

		# Define loss and optimization Op
		self.loss = tf.reduce_mean(tf.squared_difference(self.target_q_value, self.q_out))

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.optimize = self.optimizer.minimize(self.loss,var_list=self.net_params)

		self.loss_grads=tf.gradients(ys=self.loss,xs=self.net_params)
		self.loss_norm_grads=[tf.div(g,self.input_batch_size) for g in self.loss_grads]

		# Get the gradient of the critic w.r.t. the action, (critic the action)
		self.action_grads = tf.gradients(self.q_out, self.act_in)

		self.master_network=master_network
		if not master_network==None:
			#self.apply_grads=self.optimizer.apply_gradients(grads_and_vars=zip(self.grad_vars,self.master_network.net_params))
			self.apply_grads=self.optimizer.apply_gradients(grads_and_vars=zip(self.loss_norm_grads,self.master_network.net_params))
			self.copy_master_params = [self.net_params[i].assign(self.master_network.net_params[i]) 
								  		for i in range(len(self.net_params))]


	def build_network(self,rnn_scope):
		obs_in=tf.placeholder(dtype=tf.float32,shape=[self.input_time_step,None,self.obs_dim])
		vel_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.vel_dim])
		dir_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.dir_dim])
		act_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.act_dim])
		
		lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size)	#lstm_state_size=8
		hs,c=tf.nn.static_rnn(cell=lstm,
							inputs=tf.unstack(tf.reverse(obs_in,[0])),
							dtype=tf.float32,
							scope=rnn_scope)
		#h=tf.Print(hs,[hs[-1]],"cri_h")
		state=tf.concat([hs[-1],tf.concat([vel_in[0,:,:],dir_in[0,:,:]],1)],1)
		fc1_in=tf.concat([state,act_in[0,:,:]],1)
		#input_fc1=tf.Print(input_fc1,[input_fc1[0,:]],"input_fc1")
		fc1=tf.layers.dense(inputs=fc1_in,
								units=self.fc1_size,
								activation=tf.nn.relu,
								kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.,0.001),
								bias_initializer=tf.initializers.zeros())
		#fc1 = tf.layers.dropout(inputs=fc1,rate=0.2,training=True)
		#fc1=tf.Print(fc1,[fc1],"cri_fc1")
		fc2=tf.layers.dense(inputs=fc1,
								units=self.fc2_size,
								activation=tf.nn.relu,
								kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.,0.001),
								bias_initializer=tf.initializers.zeros())
		#fc2 = tf.layers.dropout(inputs=fc2,rate=0.2,training=True)
		#fc2=tf.Print(fc2,[fc2],"cri_fc2")
		q_out=tf.layers.dense(inputs=fc2,
								units=1,
								activation=None,
								kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.,0.001),
								bias_initializer=tf.initializers.zeros())
		#q_out=tf.Print(q_out,[q_out],"q_out")
		return obs_in, vel_in, dir_in, act_in, q_out

	def train(self, obs_batch,vel_batch,dir_batch,act_batch,tar_q_batch):
		return self.sess.run([self.q_out, self.optimize], feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.act_in: act_batch,
			self.target_q_value: tar_q_batch
		})

	def predict(self, obs_batch,vel_batch,dir_batch,act_batch):
		return self.sess.run(self.q_out, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.act_in: act_batch
		})

	def predict_target(self, obs_batch,vel_batch,dir_batch,act_batch):
		return self.sess.run(self.target_q_out, feed_dict={
			self.target_obs_in: obs_batch,
			self.target_vel_in: vel_batch,
			self.target_dir_in: dir_batch,
			self.target_act_in: act_batch
		})

	def action_gradients(self, obs_batch, vel_batch, dir_batch, act_batch):
		return self.sess.run(self.action_grads, feed_dict={
			self.obs_in: obs_batch, 
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.act_in: act_batch
		})

	def update_target_network(self):
		self.sess.run(self.update_target_net_params)

	def update_master_network(self,obs_batch,vel_batch,dir_batch,act_batch,tar_q_batch):
		self.sess.run(self.apply_grads, feed_dict={
				self.obs_in: obs_batch,
				self.vel_in: vel_batch,
				self.dir_in: dir_batch,
				self.act_in: act_batch,
				self.target_q_value: tar_q_batch
			})
		#print(self.name,"updated master network")

	def copy_master_network(self):
		self.sess.run(self.copy_master_params)

	def print_net_params(self):
		for p in self.net_params:
			self.sess.run(tf.Print(p,[p.get_shape()],self.name+' net_params'))
		for tp in self.target_net_params:
			self.sess.run(tf.Print(tp,[tp.get_shape()],self.name+' target_net_params'))

