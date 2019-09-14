#!usr/bin/env python
import tensorflow as tf
#source ~/Desktop/tensorflow/bin/activate


class ManagerActorNetwork():
	def __init__(
		self,
		sess,
		name,
		n_agent,
		n_fc1_unit,
		n_fc2_unit,
		batch_size,
		learning_rate,
		device='/device:GPU:0',
		master_network=None
	):
		self.sess=sess
		self.name=name
		self.n_agent=n_agent
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.batch_size=batch_size
		self.learning_rate=learning_rate
		self.joint_pos_dim=n_agent*2
		self.net_param_offset=len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		with tf.device(device):
			(
				self.joint_pos_in, 
				self.ind_in, 
				self.goal_out
			) = self.build_network()
			self.net_params=tf.trainable_variables()[
				self.net_param_offset:
			]  # all the weights and bias in actor network
			(
				self.target_joint_obs_in,
				self.target_ind_in,
				self.target_goal_out
			) = self.build_network()
			self.target_net_params=tf.trainable_variables()[
				self.net_param_offset
				+ len(self.net_params):
			]  # all the weights and bias in target actor network

			self.master_network = master_network
			if master_network:
				self.cpos_in=tf.placeholder(
					dtype=tf.float32,
					shape=[None, 2]
				)
				self.npos_in=tf.placeholder(
					dtype=tf.float32,
					shape=[None, 2]
				)
				self.value_in = tf.placeholder(
					dtype=tf.float32,
					shape=None
				)
				self.objective = -tf.multiply(
					tf.reduce_sum(
						tf.multiply(
							tf.nn.l2_normalize(
								self.npos_in-self.cpos_in, 
								1
							),
							tf.nn.l2_normalize(
								self.goal_out,
								1
							)
						),
						1
					),
					self.value_in
				)
				self.gradients = tf.gradients(
					ys=self.objective, 
					xs=self.net_params,
				)
				self.normalized_gradients=[
					tf.multiply(
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
							0.9
						)
						+ tf.multiply(
							self.target_net_params[i],
							1.0-0.9
						)
					)
					for i in range(len(self.target_net_params))
				]

	def build_network(self):
		joint_pos_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.joint_pos_dim],
			name=self.name+'_joint_pose_in',
		)
		ind_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, 1],
			name=self.name+'_ind_in'
		)
		fc1_in=tf.concat(
			[
				joint_pos_in, 
				ind_in
			],
			1
		)
		fc1=tf.layers.dense(
			inputs=fc1_in,
			units=self.fc1_size,
			activation=tf.nn.relu,
		)
		fc1_dropout=tf.layers.dropout(
			inputs=fc1,
			rate=0.4,
			training=True
		)
		fc2=tf.layers.dense(
			inputs=fc1_dropout,
			units=self.fc2_size,
			activation=tf.nn.relu,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		fc2_dropout=tf.layers.dropout(
			inputs=fc2,
			rate=0.4,
			training=True
		)
		goal_out=tf.layers.dense(
			inputs=fc2_dropout,
			units=2,
			activation=None,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		return joint_pos_in, ind_in, goal_out

	def train(
		self, 
		joint_pos_batch, 
		ind_batch, 
		cpos_batch, 
		npos_batch, 
		value_batch
	):
		return self.sess.run(
			self.apply_grads, 
			feed_dict={
				self.joint_pos_in: joint_pos_batch,
				self.ind_in: ind_batch,
				self.cpos_in: cpos_batch,
				self.npos_in: npos_batch,
				self.value_in: value_batch
			}
		)

	def predict(
		self, 
		joint_pos_batch, 
		ind_batch
	):
		return self.sess.run(
			self.goal_out, 
			feed_dict={
				self.joint_pos_in: joint_pos_batch,
				self.ind_in: ind_batch,
			}
		)

	def predict_target(
		self,
		joint_pos_batch, 
		joint_dir_batch, 
		ind_batch
	):
		return self.sess.run(
			self.target_goal_out, 
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

class ManagerCriticNetwork():
	def __init__(
		self,
		sess,
		name,
		n_agent,
		n_fc1_unit,
		n_fc2_unit,
		batch_size,
		learning_rate,
		device='/device:GPU:0',
		master_network=None
	):
		self.sess=sess
		self.name=name
		self.batch_size=batch_size
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.learning_rate=learning_rate
		self.joint_pos_dim=n_agent*2
		self.goal_dim=2
		self.net_param_offset=len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		with tf.device(device):
			(
				self.joint_pos_in, 
				self.goal_in, 
				self.ind_in, 
				self.q_out
			) = self.build_network()
			self.net_params = tf.trainable_variables()[
				self.net_param_offset:
			]

			(
				self.target_joint_pos_in, 
				self.target_goal_in, 
				self.target_ind_in, 
				self.target_q_out
			) = self.build_network()
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
				self.normalized_gradiants=[
					tf.div(g, self.batch_size) 
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
							0.9
						)
						+
						tf.multiply(
							self.target_net_params[i], 
							1.0-0.9
						)
					)
					for i in range(len(self.target_net_params))
				]

	def build_network(self):
		joint_pos_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.joint_pos_dim]
		)
		goal_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.goal_dim]
		)
		ind_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, 1]
		)
		fc1_in=tf.concat(
			[
				joint_pos_in,
				goal_in,
				ind_in
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
		fc1_dropout=tf.layers.dropout(
			inputs=fc1,
			rate=0.4,
			training=True
		)
		fc2=tf.layers.dense(
			inputs=fc1_dropout,
			units=self.fc2_size,
			activation=tf.nn.relu,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		fc2_dropout=tf.layers.dropout(
			inputs=fc2,
			rate=0.4,
			training=True
		)
		q_out=tf.layers.dense(
			inputs=fc2_dropout,
			units=1,
			activation=None,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		return joint_pos_in, goal_in, ind_in, q_out

	def train(
		self,
		joint_pos_batch, 
		goal_batch, 
		ind_batch,
		target_q_batch
	):
		return self.sess.run(
			self.apply_grads, 
			feed_dict={
				self.joint_pos_in:joint_pos_batch,
				self.goal_in:goal_batch,
				self.ind_in:ind_batch,
				self.target_q:target_q_batch
			}
		)

	def predict(
		self, 
		joint_pos_batch, 
		goal_batch, 
		ind_batch
	):
		return self.sess.run(
			self.q_out, 
			feed_dict={
				self.joint_pos_in:joint_pos_batch,
				self.goal_in:goal_batch,
				self.ind_in:ind_batch
			}
		)

	def predict_target(
		self, 
		joint_pos_batch, 
		goal_batch, 
		ind_batch
	):
		return self.sess.run(
			self.target_q_out, 
			feed_dict={
				self.target_joint_pos_in:joint_pos_batch,
				self.target_goal_in:goal_batch,
				self.target_ind_in:ind_batch
			}
		)
	
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

