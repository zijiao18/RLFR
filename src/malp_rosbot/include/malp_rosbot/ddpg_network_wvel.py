#!usr/bin/env python
import tensorflow as tf
#source ~/Desktop/tensorflow/bin/activate
class ActorNetwork():
	def __init__(self,sess,name,time_step,obs_dim,vel_dim,dir_dim,act_dim,batch_size,
				lstm_state_dim,n_fc1_unit,n_fc2_unit,learning_rate,tau,training,master_network=None):
		self.sess=sess
		self.name=name

		self.act_dim=act_dim
		self.obs_dim=obs_dim
		self.vel_dim=vel_dim
		self.dir_dim=dir_dim

		self.time_step=time_step
		self.batch_size=batch_size
		self.lstm_state_size=lstm_state_dim
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.tau=tau
		self.learning_rate=learning_rate
		self.training=training
		
		#with g.name_scope(self.name):
		self.net_param_offset=len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		# Actor network
		self.obs_in,self.vel_in,self.dir_in,self.ind_in,self.act_out = self.build_network(self.name)
		self.net_params=tf.trainable_variables()[self.net_param_offset:]	#all the weights and bias in actor network
		# Target network
		self.target_obs_in,self.target_vel_in,self.target_dir_in,self.target_ind_in,self.target_act_out = self.build_network(self.name+'_target')
		self.target_net_params=tf.trainable_variables()[self.net_param_offset+len(self.net_params):]	#all the weights and bias in target actor network

		# Op for periodically updating target network with online network weights
		self.update_target_net_params = [self.target_net_params[i].assign(tf.multiply(self.net_params[i],self.tau)
										 +tf.multiply(self.target_net_params[i],1.0-self.tau))
										for i in range(len(self.target_net_params))]

		self.num_trainable_vars=len(self.net_params)+len(self.target_net_params)

		# Combine dnetScaledOut/dnetParams with criticToActionGradient to get actorGradient
		# Temporary placeholder action gradient
		self.action_gradients = tf.placeholder(tf.float32, [1,None, self.act_dim])
		#self.action_gradients = tf.Print(self.action_gradients,[self.action_gradients])

		self.actor_params_gradients = tf.gradients(ys=self.act_out, xs=self.net_params,grad_ys= -self.action_gradients[0,:,:])
		#self.actor_params_gradients = [tf.Print(g,[g.get_shape()],"ag") for g in self.actor_params_gradients]

		self.actor_norm_gradients = [tf.div(g,self.batch_size) for g in self.actor_params_gradients]
		self.tb_actor_norm_gradients=[tf.summary.histogram(self.name+'_pg_'+str(i), self.actor_norm_gradients[i]) 
										for i in xrange(len(self.actor_norm_gradients))]
		
		# Optimization Op
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		
		self.optimize = self.optimizer.apply_gradients(grads_and_vars=zip(self.actor_norm_gradients, self.net_params))
		
		self.master_network=master_network
		if not master_network==None:
			self.apply_grads=self.optimizer.apply_gradients(grads_and_vars=zip(self.actor_norm_gradients,self.master_network.net_params))
			self.copy_master_params = [self.net_params[i].assign(self.master_network.net_params[i]) 
								  		for i in range(len(self.master_network.net_params))]


	def build_network(self,rnn_scope):
		
		obs_in=tf.placeholder(dtype=tf.float32,shape=[self.time_step,None,self.obs_dim])
		vel_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.vel_dim])
		dir_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.dir_dim])
		ind_in=tf.placeholder(dtype=tf.float32,shape=[1,None,1])

		lstm_in = tf.reverse(obs_in,[0])
		lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size,name='lstm')
		lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=1.0)#was 0.6
		hs,c=tf.nn.static_rnn(cell=lstm,
							inputs=tf.unstack(lstm_in),
							dtype=tf.float32,
							scope=rnn_scope)
		#hs=tf.Print(hs,[hs[-1]],"act_h")
		fc1_in=tf.concat([hs[-1],vel_in[0,:,:],dir_in[0,:,:],ind_in[0,:,:]],1)
		fc1=tf.layers.dense(inputs=fc1_in,
							units=self.fc1_size,
							activation=tf.nn.relu,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
							bias_initializer=tf.initializers.zeros())
		fc1 = tf.layers.dropout(inputs=fc1,rate=0.0,training=self.training)#rate was 0.2
		#fc1=tf.Print(fc1,[fc1],"act_fc1")
		fc2=tf.layers.dense(inputs=fc1,
							units=self.fc2_size,
							activation=tf.nn.relu,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
							bias_initializer=tf.initializers.zeros())
		fc2 = tf.layers.dropout(inputs=fc2,rate=0.0,training=self.training)#rate was 0.2
		#fc2=tf.Print(fc2,[fc2],"act_fc2")
		
		lvel_out=tf.layers.dense(inputs=fc2,
								units=1,
								activation=tf.nn.sigmoid,
								kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
								bias_initializer=tf.initializers.zeros())

		avel_out=tf.layers.dense(inputs=fc2,
								units=1,
								activation=tf.nn.tanh,
								kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.0,0.001),
								bias_initializer=tf.initializers.zeros())

		act_out=tf.concat([lvel_out,avel_out],1)
		#act_out=tf.multiply(act_out,self.action_bound)
		#act_out=tf.Print(act_out,[act_out],"act_act_out")
		return obs_in,vel_in,dir_in,ind_in,act_out

	def train(self, obs_batch,vel_batch,dir_batch,ind_batch,act_grads):
		# args [inputs, action_gradients, phase]
		return self.sess.run(self.optimize, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.ind_in: ind_batch,
			self.action_gradients: act_grads
		})

	def update_master_network(self,obs_batch,vel_batch,dir_batch,ind_batch,act_grads):
		self.sess.run(self.apply_grads, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.ind_in: ind_batch,
			self.action_gradients: act_grads
			})
		#print(self.name,"updated master network")

	def predict(self, obs_batch,vel_batch,dir_batch,ind_batch):
		return self.sess.run(self.act_out, feed_dict={
			self.obs_in: obs_batch,
			self.vel_in: vel_batch,
			self.dir_in: dir_batch,
			self.ind_in: ind_batch
		})

	def predict_target(self,obs_batch,vel_batch,dir_batch,ind_batch):
		return self.sess.run(self.target_act_out, feed_dict={
			self.target_obs_in: obs_batch,
			self.target_vel_in: vel_batch,
			self.target_dir_in: dir_batch,
			self.target_ind_in: ind_batch
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
	
	def summary(self,obs_batch,vel_batch,dir_batch,ind_batch,act_grads):
		return self.sess.run(self.tb_actor_norm_gradients, feed_dict={
						self.obs_in: obs_batch,
						self.vel_in: vel_batch,
						self.dir_in: dir_batch,
						self.ind_in: ind_batch,
						self.action_gradients: act_grads})

class CriticNetwork():

	def __init__(self,sess,name,time_step,obs_dim,vel_dim,dir_dim,act_dim,batch_size,
				lstm_state_dim,n_fc1_unit,n_fc2_unit,learning_rate,tau,training,master_network=None):
		self.sess=sess
		self.name=name
		self.act_dim=act_dim
		self.obs_dim=obs_dim
		self.vel_dim=vel_dim
		self.dir_dim=dir_dim
		self.time_step=time_step
		self.batch_size=batch_size
		self.lstm_state_size=lstm_state_dim
		self.fc1_size=n_fc1_unit
		self.fc2_size=n_fc2_unit
		self.learning_rate=learning_rate
		self.tau=tau
		self.training=training

		self.net_param_offset=len(tf.trainable_variables())
		print("%s has params offset %d"%(name, self.net_param_offset))
		# Critic network
		(self.obs0_in,self.obs1_in,self.obs2_in,
		self.vel0_in,self.vel1_in,self.vel2_in,
		self.dir0_in,self.dir1_in,self.dir2_in,
		self.act0_in,self.act1_in,self.act2_in,
		self.ind_in,self.q_out)=self.build_network(self.name)
		self.net_params = tf.trainable_variables()[self.net_param_offset:]
		# Target network
		(self.target_obs0_in,self.target_obs1_in,self.target_obs2_in,
		self.target_vel0_in,self.target_vel1_in,self.target_vel2_in,
		self.target_dir0_in,self.target_dir1_in,self.target_dir2_in,
		self.target_act0_in,self.target_act1_in,self.target_act2_in,
		self.target_ind_in,self.target_q_out)=self.build_network(self.name+'_target')
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
		self.tb_loss=tf.summary.scalar(self.name+'_loss',self.loss)

		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.optimize = self.optimizer.minimize(self.loss,var_list=self.net_params)

		self.loss_grads=tf.gradients(ys=self.loss,xs=self.net_params)
		self.loss_norm_grads=[tf.div(g,self.batch_size) for g in self.loss_grads]

		# Get the gradient of the critic w.r.t. the action, (critic the action)
		self.action0_grads = tf.gradients(self.q_out,self.act0_in)
		self.action1_grads = tf.gradients(self.q_out,self.act1_in)
		self.action2_grads = tf.gradients(self.q_out,self.act2_in)

		self.master_network=master_network
		if not master_network==None:
			#self.apply_grads=self.optimizer.apply_gradients(grads_and_vars=zip(self.grad_vars,self.master_network.net_params))
			self.apply_grads=self.optimizer.apply_gradients(grads_and_vars=zip(self.loss_norm_grads,self.master_network.net_params))
			self.copy_master_params = [self.net_params[i].assign(self.master_network.net_params[i]) 
								  		for i in range(len(self.net_params))]


	def build_network(self,rnn_scope):
		obs0_in=tf.placeholder(dtype=tf.float32,shape=[self.time_step,None,self.obs_dim])
		obs1_in=tf.placeholder(dtype=tf.float32,shape=[self.time_step,None,self.obs_dim])
		obs2_in=tf.placeholder(dtype=tf.float32,shape=[self.time_step,None,self.obs_dim])

		vel0_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.vel_dim])
		vel1_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.vel_dim])
		vel2_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.vel_dim])

		dir0_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.dir_dim])
		dir1_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.dir_dim])
		dir2_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.dir_dim])
		
		act0_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.act_dim])
		act1_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.act_dim])
		act2_in=tf.placeholder(dtype=tf.float32,shape=[1,None,self.act_dim])
		
		ind_in=tf.placeholder(dtype=tf.float32,shape=[1,None,1])

		joint_obs_in=tf.concat([obs0_in,obs1_in,obs2_in],2)
		joint_vel_in=tf.concat([vel0_in,vel1_in,vel2_in],2)
		joint_dir_in=tf.concat([dir0_in,dir1_in,dir2_in],2)
		joint_act_in=tf.concat([act0_in,act1_in,act2_in],2)

		lstm_in = tf.reverse(joint_obs_in,[0])
		lstm_in = tf.layers.dropout(inputs=lstm_in,rate=0.0,training=self.training)
		lstm = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size)	#lstm_state_size=8
		lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, state_keep_prob=1.0) #state_keep_prob was 0.6
		h,c=tf.nn.static_rnn(cell=lstm,
							inputs=tf.unstack(lstm_in),
							dtype=tf.float32,
							scope=rnn_scope)

		#lstm0_in = tf.reverse(obs0_in,[0])
		#lstm0_in = tf.layers.dropout(inputs=lstm0_in,rate=0.0,training=self.training)
		#lstm0 = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size)	#lstm_state_size=8
		#lstm0 = tf.nn.rnn_cell.DropoutWrapper(lstm0, state_keep_prob=1.0) #state_keep_prob was 0.6
		#h0,c0=tf.nn.static_rnn(cell=lstm0,
		#					inputs=tf.unstack(lstm0_in),
		#					dtype=tf.float32,
		#					scope=rnn_scope+'0')

		#lstm1_in = tf.reverse(obs1_in,[0])
		#lstm1_in = tf.layers.dropout(inputs=lstm1_in,rate=0.0,training=self.training)
		#lstm1 = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size)	#lstm_state_size=8
		#lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1, state_keep_prob=1.0) #state_keep_prob was 0.6
		#h1,c1=tf.nn.static_rnn(cell=lstm1,
		#					inputs=tf.unstack(lstm1_in),
		#					dtype=tf.float32,
		#					scope=rnn_scope+'1')

		#lstm2_in = tf.reverse(obs2_in,[0])
		#lstm2_in = tf.layers.dropout(inputs=lstm2_in,rate=0.0,training=self.training)
		#lstm2 = tf.contrib.rnn.BasicLSTMCell(num_units=self.lstm_state_size)	#lstm_state_size=8
		#lstm2 = tf.nn.rnn_cell.DropoutWrapper(lstm2, state_keep_prob=1.0) #state_keep_prob was 0.6
		#h2,c2=tf.nn.static_rnn(cell=lstm2,
		#					inputs=tf.unstack(lstm2_in),
		#					dtype=tf.float32,
		#					scope=rnn_scope+'2')

		#h=tf.Print(h,[h[-1]],"cri_h")
		fc1_in=tf.concat([h[-1],joint_vel_in[0,:,:],joint_dir_in[0,:,:],joint_act_in[0,:,:],ind_in[0,:,:]],1)
		#fc1_in=tf.concat([h0[-1],h1[-1],h2[-1],joint_vel_in[0,:,:],joint_dir_in[0,:,:],joint_act_in[0,:,:],ind_in[0,:,:]],1)
		#input_fc1=tf.Print(input_fc1,[input_fc1[0,:]],"input_fc1")
		fc1=tf.layers.dense(inputs=fc1_in,
							units=self.fc1_size,
							activation=tf.nn.relu,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.,0.001),
							bias_initializer=tf.initializers.zeros())
		fc1 = tf.layers.dropout(inputs=fc1,rate=0.0,training=self.training)#rate was 0.2
		#fc1=tf.Print(fc1,[fc1],"cri_fc1")
		fc2=tf.layers.dense(inputs=fc1,
							units=self.fc2_size,
							activation=tf.nn.relu,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.,0.001),
							bias_initializer=tf.initializers.zeros())
		fc2 = tf.layers.dropout(inputs=fc2,rate=0.0,training=self.training)#rate was 0.2
		#fc2=tf.Print(fc2,[fc2],"cri_fc2")
		q_out=tf.layers.dense(inputs=fc2,
							units=1,
							activation=None,
							kernel_initializer=tf.keras.initializers.he_normal(),#tf.initializers.random_normal(0.,0.001),
							bias_initializer=tf.initializers.zeros())
		#q_out=tf.Print(q_out,[q_out],"q_out")
		return (obs0_in,obs1_in,obs2_in,
				vel0_in,vel1_in,vel2_in,
				dir0_in,dir1_in,dir2_in,
				act0_in,act1_in,act2_in,
				ind_in,q_out)

	def train(self, obs_batch,vel_batch,dir_batch,ind_batch,act_batch,tar_q_batch):
		return self.sess.run([self.q_out, self.optimize], feed_dict={
			self.obs0_in: obs_batch[0],
			self.obs1_in: obs_batch[1],
			self.obs2_in: obs_batch[2],
			self.vel0_in: vel_batch[0],
			self.vel1_in: vel_batch[1],
			self.vel2_in: vel_batch[2],
			self.dir0_in: dir_batch[0],
			self.dir1_in: dir_batch[1],
			self.dir2_in: dir_batch[2],
			self.act0_in: act_batch[0],
			self.act1_in: act_batch[1],
			self.act2_in: act_batch[2],
			self.ind_in: ind_batch,
			self.target_q_value: tar_q_batch})

	def predict(self, obs_batch,vel_batch,dir_batch,ind_batch,act_batch):
		return self.sess.run(self.q_out, feed_dict={
				self.obs0_in: obs_batch[0],
				self.obs1_in: obs_batch[1],
				self.obs2_in: obs_batch[2],
				self.vel0_in: vel_batch[0],
				self.vel1_in: vel_batch[1],
				self.vel2_in: vel_batch[2],
				self.dir0_in: dir_batch[0],
				self.dir1_in: dir_batch[1],
				self.dir2_in: dir_batch[2],
				self.act0_in: act_batch[0],
				self.act1_in: act_batch[1],
				self.act2_in: act_batch[2],
				self.ind_in: ind_batch})

	def predict_target(self,obs_batch,vel_batch,dir_batch,ind_batch,act_batch):
		return self.sess.run(self.target_q_out, feed_dict={
			self.target_obs0_in: obs_batch[0],
			self.target_obs1_in: obs_batch[1],
			self.target_obs2_in: obs_batch[2],
			self.target_vel0_in: vel_batch[0],
			self.target_vel1_in: vel_batch[1],
			self.target_vel2_in: vel_batch[2],
			self.target_dir0_in: dir_batch[0],
			self.target_dir1_in: dir_batch[1],
			self.target_dir2_in: dir_batch[2],
			self.target_act0_in: act_batch[0],
			self.target_act1_in: act_batch[1],
			self.target_act2_in: act_batch[2],
			self.target_ind_in: ind_batch})

	def action_gradients(self,obs_batch,vel_batch,dir_batch,ind_batch,act_batch,ind_scalar):
		if ind_scalar==0:
			return self.sess.run(self.action0_grads, feed_dict={
					self.obs0_in: obs_batch[0],
					self.obs1_in: obs_batch[1],
					self.obs2_in: obs_batch[2],
					self.vel0_in: vel_batch[0],
					self.vel1_in: vel_batch[1],
					self.vel2_in: vel_batch[2],
					self.dir0_in: dir_batch[0],
					self.dir1_in: dir_batch[1],
					self.dir2_in: dir_batch[2],
					self.act0_in: act_batch[0],
					self.act1_in: act_batch[1],
					self.act2_in: act_batch[2],
					self.ind_in: ind_batch})
		if ind_scalar==1:
			return self.sess.run(self.action1_grads, feed_dict={
					self.obs0_in: obs_batch[0],
					self.obs1_in: obs_batch[1],
					self.obs2_in: obs_batch[2],
					self.vel0_in: vel_batch[0],
					self.vel1_in: vel_batch[1],
					self.vel2_in: vel_batch[2],
					self.dir0_in: dir_batch[0],
					self.dir1_in: dir_batch[1],
					self.dir2_in: dir_batch[2],
					self.act0_in: act_batch[0],
					self.act1_in: act_batch[1],
					self.act2_in: act_batch[2],
					self.ind_in: ind_batch})
		if ind_scalar==2:
			return self.sess.run(self.action2_grads, feed_dict={
					self.obs0_in: obs_batch[0],
					self.obs1_in: obs_batch[1],
					self.obs2_in: obs_batch[2],
					self.vel0_in: vel_batch[0],
					self.vel1_in: vel_batch[1],
					self.vel2_in: vel_batch[2],
					self.dir0_in: dir_batch[0],
					self.dir1_in: dir_batch[1],
					self.dir2_in: dir_batch[2],
					self.act0_in: act_batch[0],
					self.act1_in: act_batch[1],
					self.act2_in: act_batch[2],
					self.ind_in: ind_batch})

	def update_target_network(self):
		self.sess.run(self.update_target_net_params)

	def update_master_network(self,obs_batch,vel_batch,dir_batch,ind_batch,act_batch,tar_q_batch):
		self.sess.run(self.apply_grads, feed_dict={
				self.obs0_in: obs_batch[0],
				self.obs1_in: obs_batch[1],
				self.obs2_in: obs_batch[2],
				self.vel0_in: vel_batch[0],
				self.vel1_in: vel_batch[1],
				self.vel2_in: vel_batch[2],
				self.dir0_in: dir_batch[0],
				self.dir1_in: dir_batch[1],
				self.dir2_in: dir_batch[2],
				self.act0_in: act_batch[0],
				self.act1_in: act_batch[1],
				self.act2_in: act_batch[2],
				self.ind_in: ind_batch,
				self.target_q_value: tar_q_batch})
		#print(self.name,"updated master network")

	def copy_master_network(self):
		self.sess.run(self.copy_master_params)

	def print_net_params(self):
		for p in self.net_params:
			self.sess.run(tf.Print(p,[p.get_shape()],self.name+' net_params'))
		for tp in self.target_net_params:
			self.sess.run(tf.Print(tp,[tp.get_shape()],self.name+' target_net_params'))

	def summary(self,obs_batch,vel_batch,dir_batch,ind_batch,act_batch,tar_q_batch):
		return self.sess.run(self.tb_loss, feed_dict={
								self.obs0_in: obs_batch[0],
								self.obs1_in: obs_batch[1],
								self.obs2_in: obs_batch[2],
								self.vel0_in: vel_batch[0],
								self.vel1_in: vel_batch[1],
								self.vel2_in: vel_batch[2],
								self.dir0_in: dir_batch[0],
								self.dir1_in: dir_batch[1],
								self.dir2_in: dir_batch[2],
								self.act0_in: act_batch[0],
								self.act1_in: act_batch[1],
								self.act2_in: act_batch[2],
								self.ind_in: ind_batch,
								self.target_q_value: tar_q_batch})

