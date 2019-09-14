#!usr/bin/env python
import tensorflow as tf
#source ~/Desktop/tensorflow/bin/activate

class WorkerActorNetwork():
	def __init__(
		self,
		sess,
		name,
		obs_seq_len,
		obs_dim,
		obs_emb_size,
		n_fc1_unit,
		n_fc2_unit,
		batch_size,
		learning_rate,
		device='/device:GPU:0',
		master_network=None
	):
		self.sess=sess
		self.name=name
		self.obs_dim=obs_dim
		self.pose_dim=6
		self.goal_dim=2
		self.act_dim=2
		self.obs_seq_len=obs_seq_len
		self.batch_size=batch_size
		self.obs_emb_size=obs_emb_size
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.learning_rate=learning_rate
		with tf.device(device):
			self.net_param_offset=len(tf.trainable_variables())
			print("%s has params offset %d"%(name, self.net_param_offset))
			(
				self.obs_in,
				self.pose_in,
				self.goal_in,
				self.act_out
			) = self.build_network(self.name+'_actor')
			self.net_params=tf.trainable_variables()[
				self.net_param_offset:
			]

			(
				self.target_obs_in,
				self.target_pose_in,
				self.target_goal_in,
				self.target_act_out
			) = self.build_network(self.name+'target_actor')
			self.target_net_params=tf.trainable_variables()[
				self.net_param_offset+len(self.net_params):
			]
			
			self.master_network=master_network
			if not master_network==None:
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
				self.action_gradients = tf.placeholder(
					dtype=tf.float32, 
					shape=[None, self.act_dim]
				)
				self.gradients = tf.gradients(
					ys=self.act_out, 
					xs=self.net_params,
					grad_ys=(-self.action_gradients)
				)
				self.normalized_gradients = [
					tf.div(g, self.batch_size) 
					for g in self.gradients
				]
				self.apply_grads = self.optimizer.apply_gradients(
					grads_and_vars=zip(
						self.normalized_gradients,
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
						self.master_network.net_params[i]
					) 
					for i in range(len(self.master_network.net_params))
				]
				self.update_target_net_params = [
					self.target_net_params[i].assign(
						tf.multiply(
							self.master_network.net_params[i],
							0.9)
						+
						tf.multiply(
							self.target_net_params[i],
							1.0-0.9
						)
					)
					for i in range(len(self.target_net_params))
				]

	def build_network(self, rnn_scope):
		obs_in=tf.placeholder(
			dtype=tf.float32,
			shape=[self.obs_seq_len, None, self.obs_dim]
		)
		pose_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.pose_dim]
		)
		goal_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.goal_dim]
		)
		obs_emb, _=tf.nn.static_rnn(
			cell=tf.nn.rnn_cell.DropoutWrapper(
				tf.contrib.rnn.BasicLSTMCell(
					num_units=self.obs_emb_size
				),
				state_keep_prob=0.6
			),
			inputs=tf.unstack(tf.reverse(obs_in,[0])),
			dtype=tf.float32,
			scope=rnn_scope
		)
		fc1_in=tf.concat(
			[
				obs_emb[-1], 
				pose_in, 
				goal_in
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
		lvel_out=tf.layers.dense(
			inputs=fc2_dropout,
			units=1,
			activation=tf.nn.sigmoid,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		avel_out=tf.layers.dense(
			inputs=fc2_dropout,
			units=1,
			activation=tf.nn.tanh,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		act_out=tf.concat(
			[lvel_out, avel_out],
			1
		)
		return obs_in, pose_in, goal_in, act_out

	def train(
		self,
		obs_batch,
		pose_batch,
		goal_batch,
		action_gradients
	):
		return self.sess.run(
			self.apply_grads, 
			feed_dict={
				self.obs_in: obs_batch,
				self.pose_in: pose_batch,
				self.goal_in: goal_batch,
				self.action_gradients: action_gradients
			}
		)
	def predict(
		self, 
		obs_batch,
		pose_batch,
		goal_batch
	):
		return self.sess.run(
			self.act_out, 
			feed_dict={
				self.obs_in: obs_batch,
				self.pose_in: pose_batch,
				self.goal_in: goal_batch,
			}
		)
	def predict_target(		
		self, 
		obs_batch,
		pose_batch,
		goal_batch
	):
		return self.sess.run(
			self.target_act_out, 
			feed_dict={
				self.target_obs_in: obs_batch,
				self.target_pose_in: pose_batch,
				self.target_goal_in: goal_batch,
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


class WorkerCriticNetwork():

	def __init__(
		self,
		sess,
		name,
		obs_dim,		
		obs_seq_len,
		obs_emb_size,
		n_fc1_unit,
		n_fc2_unit,
		batch_size,
		learning_rate,
		device='/device:GPU:0',
		master_network=None
	):
		self.sess=sess
		self.name=name
		self.obs_dim=obs_dim
		self.obs_seq_len=obs_seq_len
		self.obs_emb_size=obs_emb_size
		self.pose_dim=6
		self.goal_dim=2
		self.act_dim=2
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.batch_size=batch_size
		self.learning_rate=learning_rate
		with tf.device(device):
			self.net_param_offset=len(tf.trainable_variables())
			print("%s has params offset %d"%(name, self.net_param_offset))
			(
				self.obs_in,
				self.pose_in,
				self.goal_in,
				self.act_in,
				self.q_out
			) = self.build_network(self.name+'critic')
			self.net_params = tf.trainable_variables()[
				self.net_param_offset:
			]

			(
				self.target_obs_in,
				self.target_pose_in,
				self.target_goal_in,
				self.target_act_in,
				self.target_q_out	
			) = self.build_network(self.name+'target_critic')
			self.target_net_params = tf.trainable_variables()[
				(self.net_param_offset+len(self.net_params)):
			]

			self.master_network=master_network
			if not master_network==None:
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
				self.target_q = tf.placeholder(
					tf.float32, 
					[None, 1]
				)
				self.loss = tf.reduce_mean(
					tf.squared_difference(
						self.target_q, 
						self.q_out
					)
				)
				self.gradients = tf.gradients(
					ys=self.loss,
					xs=self.net_params
				)
				self.normalized_gradients=[
					tf.div(g, self.batch_size) 
					for g in self.gradients
				]
				self.action_gradients = tf.gradients(
					self.q_out, 
					self.act_in
				)
				self.apply_grads = self.optimizer.apply_gradients(
					grads_and_vars=zip(
						self.normalized_gradients,
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
							0.9
						) 
						+
						tf.multiply(
							self.target_net_params[i], 
							1.0-0.9))
					for i in range(len(self.target_net_params))
				]

	def build_network(self, rnn_scope):
		obs_in=tf.placeholder(
			dtype=tf.float32,
			shape=[self.obs_seq_len, None, self.obs_dim]
		)
		pose_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.pose_dim]
		)
		goal_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.goal_dim]
		)
		act_in=tf.placeholder(
			dtype=tf.float32,
			shape=[None, self.act_dim]
		)
		obs_emb, _=tf.nn.static_rnn(
			cell=tf.nn.rnn_cell.DropoutWrapper(
				tf.contrib.rnn.BasicLSTMCell(
					num_units=self.obs_emb_size
				),
				state_keep_prob=0.6
			),
			inputs=tf.unstack(tf.reverse(obs_in,[0])),
			dtype=tf.float32,
			scope=rnn_scope
		)
		fc1_in=tf.concat(
			[
				obs_emb[-1], 
				pose_in, 
				goal_in,
				act_in
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
			rate=0.4,
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
			rate=0.4,
			training=True
		)
		q_out = tf.layers.dense(
			inputs=fc2_dropout,
			units=1,
			activation=None,
			kernel_initializer=tf.keras.initializers.he_normal(),
			bias_initializer=tf.initializers.zeros()
		)
		return obs_in, pose_in, goal_in, act_in, q_out

	def train(
		self,
		obs_batch,
		pose_batch,
		goal_batch,
		act_batch,
		target_q_batch
	):
		return self.sess.run(
			self.apply_grads, 
			feed_dict={
				self.obs_in: obs_batch,
				self.pose_in: pose_batch,
				self.goal_in: goal_batch,
				self.act_in: act_batch,
				self.target_q: target_q_batch
			}
		)

	def predict(
		self,
		obs_batch,
		pose_batch,
		goal_batch,
		act_batch
	):
		return self.sess.run(
			self.q_out, 
			feed_dict={
				self.obs_in: obs_batch,
				self.pose_in: pose_batch,
				self.goal_in: goal_batch,
				self.act_in: act_batch
			}
		)
	def predict_target(
		self,
		obs_batch,
		pose_batch,
		goal_batch,
		act_batch
	):
		return self.sess.run(
			self.target_q_out, 
			feed_dict={
				self.obs_in: obs_batch,
				self.pose_in: pose_batch,
				self.goal_in: goal_batch,
				self.act_in: act_batch
			}
		)
	def cal_action_gradients(
		self,
		obs_batch,
		pose_batch,
		goal_batch,
		act_batch,
	):	
		return self.sess.run(
			self.action_gradients, 
			feed_dict={
				self.obs_in: obs_batch,
				self.pose_in: pose_batch,
				self.goal_in: goal_batch,
				self.act_in: act_batch
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

