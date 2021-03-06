import rospy
import numpy as np
import tensorflow as tf
import random as rand
import Queue
from threading import Thread
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32MultiArray
from ddpg_network import ActorNetwork
from ddpg_network import CriticNetwork
from utilities import State
from utilities import Feedback
from utilities import ReplayBuffer


class Worker():
	def __init__(
		self,
		sess,
		name,
		init_pose,
		goal,
		master_actor,
		master_critic,
		tb_writer
	):
		self.sess = sess
		self.name = name
		self.state=State(
			goal=goal,
			pose=init_pose.copy(),
			obs_dim=360,
			obs_seq_len=16
		)
		self.pre_state = State(
			goal=goal,
			pose=init_pose.copy(),
			obs_dim=360,
			obs_seq_len=16
		)
		self.action = np.zeros(
			[1,1],
			dtype=float
		)  # yaw angular velocity
		self.goal = goal  # [1,3]
		self.init_pose = init_pose.copy()  # [[x,y,z,r,p,y]]
		self.episode = 0
		self.end_of_episode = False
		self.step = 0
		self.epoch = 0
		self.epoch_max = 10
		self.max_step = 500
		self.lvel = 10.0
		self.max_avel = 3.141592654
		self.training_batch_size = 256
		self.epslion = 1.0
		self.epslion_t = 0
		self.ou_noise = 0;
		self.lidar_res_h = 360
		self.lidar_res_v = 1
		self.lidar_view_h = 3.141592654
		self.lidar_view_v = 3.141592654/180.0
		self.lidar_max_range = 5.0
		#setting up topics takes about 1 second
		self.velcmd_pub = rospy.Publisher(
			"/"+self.name+"/cmd_vel", 
			Twist, 
			queue_size=1000
		)
		self.reset_pub = rospy.Publisher(
			"/"+self.name+"/reset", 
			Empty, 
			queue_size=10
		)
		self.feedback_sub = rospy.Subscriber(
			"/"+self.name+"/pycontroller", 
			Float32MultiArray,
			self.feedback_receiver
		)
		self.training_rate = rospy.Rate(100)
		self.replay_buffer = ReplayBuffer(100000)
		self.min_repbuf_size = 2000
		self.feedback_q = Queue.Queue(100)
		self.training_q = Queue.Queue(100)

		self.actor = ActorNetwork(
			sess=sess,
			name=self.name+"_actor",
			time_step=self.state.obs_seq_len,
			obs_dim=self.state.obs_dim,
			vel_dim=self.state.vel_dim,
			dir_dim=self.state.dir_dim,
			batch_size=self.training_batch_size, 
			lstm_state_dim=128, 
			n_fc1_unit=1024, 
			n_fc2_unit=1024, 
			learning_rate=0.00001, 
			tau=0.01,
			master_network=master_actor
		)
		self.critic = CriticNetwork(
			sess=sess,
			name=self.name+"_critic",
			time_step=self.state.obs_seq_len,
			obs_dim=self.state.obs_dim,
			vel_dim=self.state.vel_dim,
			dir_dim=self.state.dir_dim,
			batch_size=self.training_batch_size,
			lstm_state_dim=128, 
			n_fc1_unit=1024, 
			n_fc2_unit=1024, 
			learning_rate=0.001, 
			tau=0.01,
			master_network=master_critic
		)
		self.master_actor = master_actor
		self.master_critic = master_critic
		self.env_worker = Thread(
			target=self.interact_with_environment
		)
		self.train_worker = Thread(
			target=self.train
		)

		self.tb_writer = tb_writer

	def train(self):
		while not rospy.is_shutdown():
			try:
				if self.training_q.qsize()>0:
					sras = self.training_q.get(timeout=1)
					self.replay_buffer.add(sras)
				if self.replay_buffer.size()<self.min_repbuf_size:
					print(
						"train_worker: replay_buffer "
						+ "size: %d/%d, epslion: %f"%(
							self.replay_buffer.size(),
							self.min_repbuf_size,self.epslion
						)
					)
					self.training_rate.sleep()
					continue
			except Queue.Empty as e:
				print("train_worker: no transaction is received...")
				break	
			batch = self.replay_buffer.sample(
				batch_size=self.training_batch_size
			)
			next_act = self.actor.predict_target(
				obs_batch=batch['next_obs'],
				vel_batch=batch['next_vel'],
				dir_batch=batch['next_dir']
			)
			next_q = self.critic.predict_target(
				obs_batch=batch['next_obs'],
				vel_batch=batch['next_vel'],
				dir_batch=batch['next_dir'],
				act_batch=[next_act]
			)
			tar_q = batch['reward']+0.99*(next_q*(1-batch['terminal']))
			self.critic.update_master_network(
				obs_batch=batch['cur_obs'],
				vel_batch=batch['cur_vel'],
				dir_batch=batch['cur_dir'],
				act_batch=batch['action'].reshape(
					(1,self.training_batch_size,1)
				),
				tar_q_batch=tar_q[0]
			)
			act_out = self.actor.predict(
				obs_batch=batch['cur_obs'],
				vel_batch=batch['cur_vel'],
				dir_batch=batch['cur_dir']
			)
			act_grads = self.critic.action_gradients(
				obs_batch=batch['cur_obs'],
				vel_batch=batch['cur_vel'],
				dir_batch=batch['cur_dir'],
				act_batch=[act_out]
			)
			self.actor.update_master_network(
				obs_batch=batch['cur_obs'],
				vel_batch=batch['cur_vel'],
				dir_batch=batch['cur_dir'],
				act_grads=act_grads[0]
			)
			self.actor.copy_master_network()
			self.critic.copy_master_network()
			self.actor.update_target_network()
			self.critic.update_target_network()
			self.training_rate.sleep()
		print("%s train_worker: exits..."%(self.name))

	def interact_with_environment(self):
		while not rospy.is_shutdown():
			self.actuate(self.action)
			try:
				self.state.update(
					feedback=self.feedback_q.get(
						timeout=1.0
					),
					goal=self.goal)
			except Queue.Empty:
				print("no feedback is received...")
				break
			rwd,terminal=self.reward_model(self.state)
			sars={
					'cur_state':self.pre_state.clone(),
					'action':self.action.copy(),
				  	'reward':rwd, 
				  	'terminal':terminal,
				  	'next_state':self.state.clone()
				}
			self.training_q.put(sars)
			self.pre_state.copy(self.state)#deepcopy
			self.end_of_episode=terminal or self.step>self.max_step
			if not self.end_of_episode:
				if self.replay_buffer.size()<self.min_repbuf_size:
					self.action[0][0] = rand.uniform(-1,1)
				else:
					self.action = self.actor.predict(
						self.state.obs_in(),
						self.state.vel_in(),
						self.state.dir_in()
					)  # 1x1
					self.action[0][0] += self.ornstein_uhlenbeck_noise()
					if self.action[0][0]>1:
						self.action[0][0] = 1
					if self.action[0][0]<-1:
						self.action[0][0] = -1
				self.step += 1
			else:
				self.reset_pose()
				self.state.reset(
					goal=self.goal.copy(),
					pose=self.init_pose.copy()
				)
				self.pre_state.reset(
					goal=self.goal.copy(),
					pose=self.init_pose.copy()
				)
				self.action = np.zeros(
					shape=[1,1],
					dtype=float
				)
				self.episode += 1
				self.step = 0
			self.training_rate.sleep()
		print("%s env_worker: exits..."%(self.name))

	def evaluate(self):
		'''
		evaluate the performance of the 
		policy leared by the robot so far
		'''
		succ=0  # 0: not success, 1: success
		ep_rwd=0
		loss=[]
		actions=[]
		traj=[self.init_pose.copy()[0]]
		self.end_of_episode=False
		self.step=0
		self.action=np.zeros(
			shape=[1,1],
			dtype=float
		)
		self.state.reset(
			goal=self.goal,
			pose=self.init_pose.copy()
		)
		self.pre_state.reset(
			goal=self.goal,
			pose=self.init_pose.copy()
		)
		while not rospy.is_shutdown() and not self.end_of_episode:
			self.actuate(self.action)
			actions.append(self.action[0][0])
			try:
				self.state.update(
					feedback=self.feedback_q.get(
						timeout=1.0
					),
					goal=self.goal
				)
			except Queue.Empty:
				print("no feedback is received...")
				break
			rwd,terminal = self.reward_model(self.state)
			sars = {
					'cur_state':self.pre_state.clone(),
					'action':self.action.copy(),
					'reward':rwd, 
					'terminal':terminal,
					'next_state':self.state.clone()
				}
			self.pre_state.copy(self.state)
			traj.append(self.state.pose[0].copy())
			cur_q = self.critic.predict(
				obs_batch=sars['cur_state'].obs_in(),
				vel_batch=sars['cur_state'].vel_in(),
				dir_batch=sars['cur_state'].dir_in(),
				act_batch=sars['action'].reshape((1,1,1))
			)
			ep_rwd += rwd
			self.end_of_episode = (terminal or self.step>self.max_step)
			if not self.end_of_episode:
				self.action = self.actor.predict(
					self.state.obs_in(),
					self.state.vel_in(),
					self.state.dir_in()
				)  # 1x2
				next_q = self.critic.predict_target(
					obs_batch=sars['next_state'].obs_in(),
					vel_batch=sars['next_state'].vel_in(),
					dir_batch=sars['next_state'].dir_in(),
					act_batch=[self.action]
				)
				tar_q = sars['reward']+0.99*(next_q[0][0]*(1-sars['terminal']))
				loss.append((tar_q-cur_q[0][0])**2)
				self.step += 1
			else:
				self.reset_pose()
				if rwd == 1:
					succ = 1
				loss.append((rwd-cur_q[0][0])**2)
			self.training_rate.sleep()
		ttl_loss = 0.0
		for l in loss:
			ttl_loss += l
		avg_loss = ttl_loss if len(loss)==0 else ttl_loss/float(len(loss))
		return ep_rwd, self.step, avg_loss, actions, traj, succ

	def actuate(self,action):
		velcmd=Twist()
		velcmd.linear.x = self.lvel
		velcmd.linear.y = 0.0
		velcmd.linear.z = 0.0
		velcmd.angular.x = 0.0
		velcmd.angular.y = 0.0
		velcmd.angular.z = action[0][0]*self.max_avel
		self.velcmd_pub.publish(velcmd)

	def reward_model(self,state):
		if self.reached_goal(state):#reached goal
			return 1.0,True
		elif state.event==0:#safe
			pre_goal_dist = np.linalg.norm(
				self.pre_state.pose[0,0:3]-self.goal
			)
			cur_goal_dist = np.linalg.norm(
				self.state.pose[0,0:3]-self.goal
			)
			goal_rwd = max(
				0.0, 
				pre_goal_dist-cur_goal_dist
			)
			ossc_rwd = -0.1*np.linalg.norm(
				state.vel[0,3:6]
			)
			rwd = goal_rwd+ossc_rwd
			return rwd, False
		elif state.event==1:
			return -1.0, True
		else:
			raise NameError(
				self.name
				+ " reward_model: invalide event-"
				+ str(feedback.reward)
			)

	def reached_goal(self, state):
		dist=np.linalg.norm(
			state.pose[0,0:3]-self.goal[0,:]
		)
		return dist<=1

	def circulating(self,state):
		dot_veldir = np.dot(
			state.vel[0,0:3],
			state.dir[0,:]
		)
		angle = np.arccos(
			dot_veldir/(np.linalg.norm(state.vel[0,0:3])*np.linalg.norm(state.dir))
		)
		if angle<3.05 and angle>0.0001:
			return False
		else:
			return True
	def closer_to_goal(self, pre_state, state):
		pre_goal_dist = np.linalg.norm(
			pre_state.pose[0,0:3]-self.goal
		)
		cur_goal_dist = np.linalg.norm(
			state.pose[0,0:3]-self.goal
		)
		return pre_goal_dist-cur_goal_dist

	def feedback_receiver(self,msg):
		feedback = Feedback()
		feedback.load(msg.data)
		feedback.set_sender(self.name)
		self.feedback_q.put(feedback)

	def reset_pose(self):
		reset_sig=Empty()
		self.reset_pub.publish(reset_sig)

	def terminate(self):
		if(not rospy.is_shutdown()):
			rospy.signal_shutdown("training is completed...")
		self.env_worker.join()
		self.train_worker.join()

	def start(self):
		self.env_worker.start()
		self.train_worker.start()
		return self.env_worker, self.train_worker

	def epslion_greedy(self):
		self.epslion_t += 1
		if self.replay_buffer.size()>self.min_repbuf_size:
			self.epslion = max(self.epslion*0.95,0.05)
		return rand.uniform(0,1)>self.epslion

	def ornstein_uhlenbeck_noise(self):
		sigma = 0.2  # Standard deviation.
		mu = 0.  # Mean.
		tau = .05  # Time constant.
		dt = .001  # Time step.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise = (self.ou_noise
						+ dt*(-(self.ou_noise-mu)/tau)
						+ sigma_bis*sqrtdt*np.random.randn())
		return self.ou_noise




