#!usr/bin/env python
import tensorflow as tf
#source ~/Desktop/tensorflow/bin/activate


class ManagerActorNetwork():
	def __init__(
		self,
		sess,
		name,
		obs_dim,		
		obs_seq_len,
		obs_emb_size,
		traj_dim,
		traj_emb_size,
		temp_abs,
		skip_steps,
		n_fc1_unit,
		n_fc2_unit,
		batch_size,
		learning_rate,
		device='/device:GPU:0',
		master_network=None,
		tb_writer=None
	):
		self.sess = sess
		self.name = name
		self.obs_dim = obs_dim	
		self.obs_seq_len = obs_seq_len
		self.obs_emb_size = obs_emb_size
		self.traj_dim = traj_dim
		self.traj_emb_size = traj_emb_size
		self.temp_abs = temp_abs
		self.skip_steps = skip_steps
		self.traj_len = (
			self.temp_abs/(1+self.skip_steps) 
			+ min(self.temp_abs%(1+self.skip_steps), 1)
		)  # length of sampled temperal trajectory
		self.fc1_size = n_fc1_unit
		self.fc2_size = n_fc2_unit
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.pose_dim = 6
		self.net_param_offset = len(tf.trainable_variables())
		print("%s has params offset %d"%(
				name, 
				self.net_param_offset
			)
		)
		self.tb_writer = tb_writer
		with tf.device(device):
			(
				self.traj_in,
				self.obs_in,
				self.pose_in,  
				self.goal_out
			) = self.build_network(self.name+'_actor')
			self.net_params=tf.trainable_variables()[
				self.net_param_offset:
			]  # all the weights and bias in actor network
			(
				self.target_traj_in,
				self.target_obs_in,
				self.target_pose_in,
				self.target_goal_out
			) = self.build_network(self.name+'_target_actor')
			self.target_net_params=tf.trainable_variables()[
				self.net_param_offset
				+ len(self.net_params):
			]  # all the weights and bias in target actor network

			self.master_network = master_network
			if master_network:
				self.action_gradients = tf.placeholder(
					dtype=tf.float32,
					shape=[None, 2]
				)
				self.gradients = tf.gradients(
					ys=self.goal_out, 
					xs=self.net_params,
					grad_ys=-self.action_gradients
				)
				self.normalized_gradients=[
					# tf.clip_by_value(
					# 	tf.div(
					# 		g, 
					# 		self.batch_size
					# 	),
					# 	clip_value_min=-0.1,
					# 	clip_value_max=0.1,
					# )
					tf.div(
						g, 
						self.batch_size
					)
					for g in self.gradients
				]
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
				self.apply_grads=self.optimizer.apply_gradients(
					grads_and_vars=zip(
						self.normalized_gradients,
						self.master_network.net_params
					)
				)
				self.copy_master_params=[
					self.net_params[i].assign(
						self.master_network.net_params[i]
					) 
					for i in range(
						len(self.master_network.net_params)
					)
				]
				# init target_net with master_net
				self.init_target_net_params=[
					self.target_net_params[i].assign(
						self.master_network.net_params[i]
					) 
					for i in range(len(self.master_network.net_params))
				]
				# slowly update target_net
				self.update_target_net_params=[
					self.target_net_params[i].assign(
						tf.multiply(
							self.master_network.net_params[i],
							0.1
						)
						+ tf.multiply(
							self.target_net_params[i],
							0.9
						)
					)
					for i in range(len(self.target_net_params))
				]
		if self.master_network:
			with tf.device('/cpu:0'):
				self.tb_normalized_gradients = [
					tf.summary.histogram(
						self.name+'_normalized_gradients', 
						norm_grad
					)
					for norm_grad in self.normalized_gradients
				]

	def build_network(self, rnn_scope):
		traj_in = tf.placeholder(
			dtype=tf.float32,
			shape=[self.traj_len, None, self.traj_dim]
		)
		obs_in = tf.placeholder(
			dtype=tf.float32,
			shape=[self.obs_seq_len, None, self.obs_dim]
		)
		pose_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.pose_dim],
		)
		# traj_emb, _ = tf.nn.static_rnn(
		# 	cell=tf.nn.rnn_cell.DropoutWrapper(
		# 		tf.contrib.rnn.BasicLSTMCell(
		# 			num_units=self.traj_emb_size,
		# 			activation=tf.nn.leaky_relu
		# 		),
		# 		state_keep_prob=1.0
		# 	),
		# 	inputs=tf.unstack(tf.reverse(traj_in,[0])),
		# 	dtype=tf.float32,
		# 	scope=rnn_scope+"_traj_emb"
		# )
		# obs_emb, _ = tf.nn.static_rnn(
		# 	cell=tf.nn.rnn_cell.DropoutWrapper(
		# 		tf.contrib.rnn.BasicLSTMCell(
		# 			num_units=self.obs_emb_size,
		# 			activation=tf.nn.leaky_relu
		# 		),
		# 		state_keep_prob=1.0
		# 	),
		# 	inputs=tf.unstack(tf.reverse(obs_in,[0])),
		# 	dtype=tf.float32,
		# 	scope=rnn_scope+"_obs_emb"
		# )
		# fc1_in=tf.concat(
		# 	[
		# 		# traj_emb[-1],
		# 		# obs_emb[-1],
		# 		pose_in,
		# 	],
		# 	1
		# )
		fc1_in = pose_in
		fc1=tf.layers.dense(
			inputs=fc1_in,
			units=self.fc1_size,
			activation=tf.nn.leaky_relu,
		)
		fc1_dropout=tf.layers.dropout(
			inputs=fc1,
			rate=0.2,
			training=True
		)
		fc2=tf.layers.dense(
			inputs=fc1_dropout,
			units=self.fc2_size,
			activation=tf.nn.leaky_relu,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		fc2_dropout=tf.layers.dropout(
			inputs=fc2,
			rate=0.2,
			training=True
		)
		goal_out=tf.layers.dense(
			inputs=fc2_dropout,
			units=2,
			activation=tf.nn.tanh,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		return traj_in, obs_in, pose_in, goal_out

	def train(
		self, 
		traj_batch,
		obs_batch,
		pose_batch,  
		action_gradients
	):
		return self.sess.run(
			self.apply_grads, 
			feed_dict={
				self.traj_in:traj_batch,
				self.obs_in:obs_batch,
				self.pose_in:pose_batch,
				self.action_gradients:action_gradients
			}
		)

	def predict(
		self, 
		traj_batch,
		obs_batch,
		pose_batch,
	):
		return self.sess.run(
			self.goal_out, 
			feed_dict={
				self.traj_in:traj_batch,
				self.pose_in:pose_batch,
				self.obs_in:obs_batch,
			}
		)

	def predict_target(
		self,
		traj_batch,
		obs_batch,
		pose_batch,
	):
		return self.sess.run(
			self.target_goal_out, 
			feed_dict={
				self.target_traj_in: traj_batch,
				self.target_pose_in: pose_batch,
				self.target_obs_in: obs_batch
			}
		)

	def init_target_network(self):
		self.sess.run(self.init_target_net_params)

	def update_target_network(self):
		self.sess.run(self.update_target_net_params)

	def copy_master_network(self):
		self.sess.run(self.copy_master_params)

	def tensorboard_summary(
		self, 
		traj_batch,
		obs_batch,
		pose_batch, 
		action_gradients,
		timestamp
	):
		summary = self.sess.run(
			self.tb_normalized_gradients,
			feed_dict={
				self.traj_in:traj_batch,
				self.obs_in:obs_batch,
				self.pose_in:pose_batch,
				self.action_gradients: action_gradients
			}
		)
		for s in summary:
			self.tb_writer.add_summary(
				s,
				timestamp
			)


class ManagerCriticNetwork():
	def __init__(
		self,
		sess,
		name,
		obs_dim,		
		obs_seq_len,
		obs_emb_size,
		traj_dim,
		traj_emb_size,
		temp_abs,
		skip_steps,
		n_fc1_unit,
		n_fc2_unit,
		batch_size,
		learning_rate,
		device='/device:GPU:0',
		master_network=None,
		tb_writer=None
	):
		self.sess = sess
		self.name = name
		self.obs_dim = obs_dim	
		self.obs_seq_len = obs_seq_len
		self.obs_emb_size = obs_emb_size
		self.traj_dim = traj_dim
		self.traj_emb_size = traj_emb_size
		self.temp_abs = temp_abs
		self.skip_steps = skip_steps
		self.traj_len = (
			self.temp_abs/(1+self.skip_steps) 
			+ min(self.temp_abs%(1+self.skip_steps), 1)
		)  # length of sampled temperal trajectory
		self.batch_size = batch_size
		self.fc1_size = n_fc1_unit
		self.fc2_size = n_fc2_unit
		self.learning_rate = learning_rate
		self.traj_dim = traj_dim
		self.traj_emb_size = traj_emb_size
		self.goal_dim = 2
		self.pose_dim = 6
		self.obs_seq_len = obs_seq_len
		self.obs_dim = obs_dim
		self.obs_emb_size = obs_emb_size
		self.net_param_offset = len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		self.tb_writer = tb_writer
		with tf.device(device):
			(
				self.traj_in,
				self.obs_in,
				self.pose_in,
				self.goal_in,  
				self.q_out
			) = self.build_network(self.name+'_critic')
			self.net_params = tf.trainable_variables()[
				self.net_param_offset:
			]

			(
				self.target_traj_in,
				self.target_obs_in,
				self.target_pose_in, 
				self.target_goal_in,  
				self.target_q_out
			) = self.build_network(self.name+'_target_critic')
			self.target_net_params = tf.trainable_variables()[
				(self.net_param_offset+len(self.net_params)):
			]
			
			self.master_network=master_network
			if master_network!=None:
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
				self.target_q = tf.placeholder(tf.float32, [None, 1])
				self.loss = tf.reduce_mean(
					tf.squared_difference(
						self.target_q, 
						self.q_out
					)
				)
				self.gradients=tf.gradients(
					ys=self.loss,
					xs=self.net_params
				)
				self.action_gradients = tf.gradients(
					ys=self.q_out,
					xs=self.goal_in
				)
				self.normalized_gradiants=[
					# tf.clip_by_value(
					# 	tf.div(
					# 		g, 
					# 		self.batch_size
					# 	),
					# 	clip_value_min=-0.1,
					# 	clip_value_max=0.1
					# )
					tf.div(
						g, 
						self.batch_size
					)
					for g in self.gradients
				]
				self.apply_grads=self.optimizer.apply_gradients(
					grads_and_vars=zip(
						self.normalized_gradiants,
						self.master_network.net_params
					)
				)
				self.copy_master_params=[
					self.net_params[i].assign(
						self.master_network.net_params[i]
					) 
					for i in range(len(self.net_params))
				]
				self.init_target_net_params=[
					self.target_net_params[i].assign(
						self.master_network.net_params[i]
					) 
					for i in range(len(self.master_network.net_params))
				]
				self.update_target_net_params=[
					self.target_net_params[i].assign(
						tf.multiply(
							self.master_network.net_params[i], 
							0.1
						)
						+
						tf.multiply(
							self.target_net_params[i], 
							0.9
						)
					)
					for i in range(len(self.target_net_params))
				]
		if self.master_network:
			with tf.device('/cpu:0'):
				self.tb_normalized_gradients = [
					tf.summary.histogram(
						self.name+'_normalized_gradients',
						norm_grad
					)
					for norm_grad in self.normalized_gradiants
				]
				self.tb_loss = tf.summary.scalar(
					self.name+'_loss',
					self.loss
				)

	def build_network(self, rnn_scope):
		traj_in = tf.placeholder(
			dtype=tf.float32,
			shape=[self.traj_len, None, self.traj_dim]
		)
		obs_in=tf.placeholder(
			dtype=tf.float32,
			shape=[self.obs_seq_len, None, self.obs_dim]
		)
		pose_in = tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.pose_dim]
		)
		goal_in = tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.goal_dim]
		)
		# traj_emb, _ = tf.nn.static_rnn(
		# 	cell=tf.nn.rnn_cell.DropoutWrapper(
		# 		tf.contrib.rnn.BasicLSTMCell(
		# 			num_units=self.traj_emb_size,
		# 			activation=tf.nn.leaky_relu
		# 		),
		# 		state_keep_prob=1.0
		# 	),
		# 	inputs=tf.unstack(tf.reverse(traj_in,[0])),
		# 	dtype=tf.float32,
		# 	scope=rnn_scope+"_traj_emb"
		# )
		# obs_emb, _ = tf.nn.static_rnn(
		# 	cell=tf.nn.rnn_cell.DropoutWrapper(
		# 		tf.contrib.rnn.BasicLSTMCell(
		# 			num_units=self.obs_emb_size,
		# 			activation=tf.nn.leaky_relu
		# 		),
		# 		state_keep_prob=1.0
		# 	),
		# 	inputs=tf.unstack(tf.reverse(obs_in,[0])),
		# 	dtype=tf.float32,
		# 	scope=rnn_scope+"_obs_emb"
		# )
		fc1_in = tf.concat(
			[
				# traj_emb[-1],
				# obs_emb[-1],
				pose_in,
				goal_in,
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
		fc1_dropout = tf.layers.dropout(
			inputs=fc1,
			rate=0.2,
			training=True
		)
		fc2 = tf.layers.dense(
			inputs=fc1_dropout,
			units=self.fc2_size,
			activation=tf.nn.relu,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		fc2_dropout = tf.layers.dropout(
			inputs=fc2,
			rate=0.2,
			training=True
		)
		q_out = tf.layers.dense(
			inputs=fc2_dropout,
			units=1,
			activation=None,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		return traj_in, obs_in, pose_in, goal_in, q_out

	def train(
		self,
		traj_batch,
		obs_batch,
		pose_batch, 
		goal_batch, 
		target_q_batch
	):
		return self.sess.run(
			self.apply_grads, 
			feed_dict={
				self.traj_in:traj_batch,
				self.obs_in:obs_batch,
				self.pose_in:pose_batch,
				self.goal_in:goal_batch,
				self.target_q:target_q_batch
			}
		)

	def predict(
		self, 
		traj_batch,
		obs_batch,
		pose_batch, 
		goal_batch,
	):
		return self.sess.run(
			self.q_out, 
			feed_dict={
				self.traj_in:traj_batch,
				self.obs_in:obs_batch,
				self.pose_in:pose_batch,
				self.goal_in:goal_batch,
			}
		)

	def predict_target(
		self, 
		traj_batch,
		obs_batch,
		pose_batch, 
		goal_batch,
	):
		return self.sess.run(
			self.target_q_out, 
			feed_dict={
				self.target_traj_in:traj_batch,
				self.target_obs_in:obs_batch,
				self.target_pose_in:pose_batch,
				self.target_goal_in:goal_batch,
			}
		)
	
	def cal_action_gradients(
		self,
		traj_batch,
		obs_batch,
		pose_batch, 
		goal_batch, 
	):
		return self.sess.run(
			self.action_gradients,
			feed_dict={
				self.traj_in:traj_batch,
				self.obs_in:obs_batch,
				self.pose_in:pose_batch,
				self.goal_in:goal_batch,
			}
		)[0]

	def init_target_network(self):
		self.sess.run(
			self.init_target_net_params
		)

	def update_target_network(self):
		self.sess.run(
			self.update_target_net_params
		)

	def copy_master_network(self):
		self.sess.run(
			self.copy_master_params
		)

	def tensorboard_summary(
		self,
		traj_batch,
		obs_batch,
		pose_batch, 
		goal_batch,
		target_q_batch,
		timestamp
	):
		summary = self.sess.run(
			[self.tb_loss]+self.tb_normalized_gradients,
			feed_dict={
				self.traj_in:traj_batch,
				self.obs_in:obs_batch,
				self.pose_in:pose_batch,
				self.goal_in:goal_batch,
				self.target_q:target_q_batch
			}
		)
		for s in summary:
			self.tb_writer.add_summary(
				s,
				timestamp
			)
