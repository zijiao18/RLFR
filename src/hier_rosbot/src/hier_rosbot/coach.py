#!usr/bin/env python
import rospy
import numpy as np
import tensorflow as tf
import tf as rostf
import random as rand
import Queue
import time
from threading import Thread
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Empty, Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from manager_net import ManagerActorNetwork, ManagerCriticNetwork
from worker_net import WorkerActorNetwork, WorkerCriticNetwork
from utilities import State, ManagerReplayBuffer, WorkerReplayBuffer


class Coach():
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
		temp_abs_skip_steps,
		collision_thresh,
		manager_batch_size,
		worker_batch_size,
		manager_actor_fc1_size,
		manager_actor_fc2_size,
		manager_critic_fc1_size,
		manager_critic_fc2_size,
		manager_actor_lr,
		manager_critic_lr,
		manager_replay_buffer_size,
		worker_actor_fc1_size,
		worker_actor_fc2_size,
		worker_critic_fc1_size,
		worker_critic_fc2_size,
		worker_actor_lr,
		worker_critic_lr,
		worker_replay_buffer_size,
		master_manager_actor,
		master_manager_critic,
		master_worker_actor,
		master_worker_critic,
		init_world_pose,
		final_goal=None,
		device='/device:GPU:0',
		tb_writer=None
	):
		self.sess = sess
		self.name = name
		self.manager_epoch = 0
		self.worker_epoch = 0
		self.episode = 0
		self.episode_goal = 0  # num of asigned goals during an episode
		self.episode_reached_goal = 0
		self.episode_extrinsic_reward = 0
		self.episode_intrinsic_reward = 0
		self.time_step = 0.0
		self.temp_abs = temp_abs
		self.skip_steps = temp_abs_skip_steps
		self.episode_step_max = 200.0
		self.max_lvel = 1.0
		self.max_avel = 3.14
		self.traj_dim = traj_dim
		self.traj_emb_size = traj_emb_size
		self.obs_dim = obs_dim
		self.obs_seq_len = obs_seq_len
		self.obs_emb_size = obs_emb_size
		self.obs_max_range = 2.0
		self.pos_dim = 2
		self.vel_dim = 3
		self.goal_dim = 2
		self.act_dim = 2
		self.collision_dist = 0.22
		self.ou_noise_lv = 0.0
		self.ou_noise_av = 0.0
		self.epsilon = 0.05
		self.manager_batch_size = manager_batch_size
		self.worker_batch_size = worker_batch_size
		self.collision_thresh = collision_thresh
		self.final_goal = final_goal
		self.init_world_pose = init_world_pose
		self.terminal = False
		self.collided = False
		self.device = device
		self.manager_actor_fc1_size = manager_actor_fc1_size
		self.manager_actor_fc2_size = manager_actor_fc2_size
		self.manager_critic_fc1_size = manager_critic_fc1_size
		self.manager_critic_fc2_size = manager_critic_fc2_size
		self.manager_actor_lr = manager_actor_lr
		self.manager_critic_lr = manager_critic_lr
		self.manager_replay_buffer_size = manager_replay_buffer_size
		self.worker_actor_fc1_size = worker_actor_fc1_size
		self.worker_actor_fc2_size = worker_actor_fc2_size
		self.worker_critic_fc1_size = worker_critic_fc1_size
		self.worker_critic_fc2_size = worker_critic_fc2_size
		self.worker_actor_lr = worker_actor_lr
		self.worker_critic_lr = worker_critic_lr
		self.worker_replay_buffer_size = worker_replay_buffer_size
		self.master_manager_actor = master_manager_actor
		self.master_manager_critic = master_manager_critic
		self.master_worker_actor = master_worker_actor
		self.master_worker_critic = master_worker_critic
		self.tb_writer = tb_writer
		self.tb_episode_intrinsic_reward_in = tf.placeholder(
			shape=None,
			dtype=tf.float32
		)
		self.tb_episode_goal_in = tf.placeholder(
			shape=None,
			dtype=tf.float32
		)
		self.tb_episode_intrinsic_reward = tf.summary.scalar(
			'avg_intrinsic_rewards',
			tf.div(
				self.tb_episode_intrinsic_reward_in,
				self.tb_episode_goal_in
			)
		)
		self.tb_episode_extrinsic_reward_in = tf.placeholder(
			shape=None,
			dtype=tf.float32
		)
		self.tb_episode_extrinsic_reward = tf.summary.scalar(
			'ttl_extrinsic_rewards',
			self.tb_episode_extrinsic_reward_in
		)
		self.manager_actor = ManagerActorNetwork(
			sess=self.sess,
			name=self.name+'_manager_actor',
			obs_dim=self.obs_dim,		
			obs_seq_len=self.obs_seq_len,
			obs_emb_size=self.obs_emb_size,
			traj_dim=self.traj_dim,
			traj_emb_size=self.traj_emb_size,
			temp_abs=self.temp_abs,
			skip_steps=self.skip_steps,
			n_fc1_unit=self.manager_actor_fc1_size,
			n_fc2_unit=self.manager_actor_fc2_size,
			batch_size=self.manager_batch_size,
			learning_rate=self.manager_actor_lr,
			device=self.device,
			master_network=self.master_manager_actor,
			tb_writer=self.tb_writer
		)
		self.manager_critic = ManagerCriticNetwork(
			sess=self.sess,
			name=self.name+'_manager_critic',
			obs_dim=self.obs_dim,		
			obs_seq_len=self.obs_seq_len,
			obs_emb_size=self.obs_emb_size,
			traj_dim=self.traj_dim,
			traj_emb_size=self.traj_emb_size,
			temp_abs=self.temp_abs,
			skip_steps=self.skip_steps,
			n_fc1_unit=self.manager_critic_fc1_size, 
			n_fc2_unit=self.manager_critic_fc2_size,
			batch_size=self.manager_batch_size,
			learning_rate=self.manager_critic_lr, 
			device=self.device,
			master_network=self.master_manager_critic,
			tb_writer=self.tb_writer
		)
		self.worker_actor = WorkerActorNetwork(
			sess=self.sess,
			name=self.name+'_worker_actor',
			obs_seq_len=self.obs_seq_len,
			obs_dim=self.obs_dim,
			obs_emb_size=self.obs_emb_size,
			n_fc1_unit=self.worker_actor_fc1_size,
			n_fc2_unit=self.worker_actor_fc2_size,
			batch_size=self.worker_batch_size,
			learning_rate=self.worker_actor_lr,
			device=self.device,
			master_network=self.master_worker_actor,
			tb_writer=self.tb_writer
		)
		self.worker_critic = WorkerCriticNetwork(
			sess=self.sess,
			name=self.name+'_worker_critic',
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_emb_size=self.obs_emb_size,
			n_fc1_unit=self.worker_critic_fc1_size,
			n_fc2_unit=self.worker_critic_fc2_size,
			batch_size=self.worker_batch_size,
			learning_rate=self.worker_critic_lr,
			device=self.device,
			master_network=self.master_worker_critic,
			tb_writer=self.tb_writer
		)
		self.manager_replay_buffer = ManagerReplayBuffer(
			max_size=self.manager_replay_buffer_size,
			batch_size=self.manager_batch_size,
			final_goal=self.final_goal,
			temp_abs=self.temp_abs,
			skip_steps=self.skip_steps
		)
		self.worker_replay_buffer = WorkerReplayBuffer(
			max_size=self.worker_replay_buffer_size,
			batch_size=self.worker_batch_size,
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
		)
		self.cur_traj = [
			State(
				world_pose=self.init_world_pose.copy(),
				obs_dim=self.obs_dim,
				obs_seq_len=self.obs_seq_len,
				obs_max_range=self.obs_max_range,
				collision_thresh=self.collision_thresh
			)
			for _ in xrange(self.temp_abs)
		]
		self.next_traj = []
		self.cur_state = State(
			world_pose=self.init_world_pose.copy(),
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_max_range=self.obs_max_range,
			collision_thresh=self.collision_thresh
		)
		self.next_state = State(
			world_pose=init_world_pose.copy(),
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_max_range=self.obs_max_range,
			collision_thresh=self.collision_thresh
		)
		self.action = np.zeros(
			[1, self.act_dim],
			dtype=float
		)
		self.intrinsic_reward = 0.0
		self.extrinsic_reward = 0.0
		self.velcmd_pub = rospy.Publisher(
			"/rosbot0/cmd_vel", 
			Twist, 
			queue_size=1000
		)
		self.scan_sub = rospy.Subscriber(
			"/rosbot0/scan",
			LaserScan,
			self.lidar_receiver,
		)
		self.odom_sub = rospy.Subscriber(
			"/rosbot0/odom",
			Odometry,
			self.odom_receiver
		)
		self.reset_client = rospy.ServiceProxy(
			"/gazebo/set_model_state",
			SetModelState
		)
		self.worker_training_rate = rospy.Rate(40)
		self.manager_training_rate = rospy.Rate(40)
		self.interact_rate = rospy.Rate(5)
		self.load_lidar = False
		self.load_odom = False
		self.t_behavior = Thread(target=self.behave)
		self.t_manager_train = Thread(target=self.train_manager)
		self.t_worker_train = Thread(target=self.train_worker)

	def behave(self):
		self.initMDP()
		while not rospy.is_shutdown():
			self.observe_state()
			self.time_step += 1.0
			manager_terminal = self.update_extrinsic_reward()
			worker_terminal = self.update_intrinsic_reward()
			terminal = manager_terminal or worker_terminal
			self.episode_intrinsic_reward = (
				self.episode_intrinsic_reward
				+ self.intrinsic_reward
			)
			self.worker_replay_buffer.add(
				cur_state=self.cur_state,
				action=self.action,
				reward=self.intrinsic_reward,
				next_state=self.next_state,
				terminal=terminal
			)
			if self.time_step%self.temp_abs!=0:
				self.next_traj.append(self.next_state.clone())
			else:
				# self.manager_replay_buffer.add(
				# 	cur_traj=self.cur_traj,
				# 	reward=self.extrinsic_reward,
				# 	next_traj=self.next_traj,
				# 	terminal=terminal,
				# )
				self.episode_extrinsic_reward += self.extrinsic_reward
				# if self.next_state.reached_goal():
				# 	self.episode_reached_goal += 1
				# setup the next goal
				self.extrinsic_reward = 0.0
				# next_traj_tensor = np.concatenate(
				# 	[
				# 		[
				# 			self.next_traj[t].get_goal_world_pos()
				# 			- self.next_traj[t].get_world_pos()
				# 		]
				# 		for t in xrange(
				# 			0, 
				# 			len(self.next_traj), 
				# 			1+self.skip_steps
				# 		)
				# 	],
				# 	0
				# )
				# next_goal = self.manager_actor.predict(
				# 	traj_batch=next_traj_tensor,
				# 	obs_batch=self.next_state.get_obs(),
				# 	pose_batch=self.next_state.get_world_pose(),
				# )
				# the end state of current goal travel   
				# is the start state of the next goal
				# self.next_state.set_goal(next_goal)
				# self.next_state.set_time(self.time_step/self.episode_step_max)
				self.episode_goal += 1   
				print(
					'ep:%d'%(self.episode),  
					# 'next goal: ', next_goal,
					'goal world pos', self.next_state.get_goal_world_pos()
				)
				# transite state trajectory
				# for t, s in enumerate(self.next_traj):
				# 	self.cur_traj[t].copy(s)
				# self.next_traj=[self.next_state.clone()]  # store the start state of the next goal
				
			self.transite_state()
			self.action = self.worker_actor.predict(
				obs_batch=self.cur_state.get_obs(),
				pose_batch=self.cur_state.get_world_pose(),
				goal_batch=self.cur_state.get_goal_dir(),
				time_batch=self.cur_state.get_time(),
			)
			self.exploration_noise(self.action)
			self.actuate(self.action)
			# terminate episode after goal execution
			if  terminal or self.time_step>=self.episode_step_max:
				print('episode: ', self.episode)
				print('time_step: ', self.time_step)
				print('episode_goal: ', self.episode_goal)
				print(
					'episode_reached_goal: ', 
					self.episode_reached_goal
				)
				print(
					'ttl_episode_extrinsic_reward: ', 
					self.episode_extrinsic_reward
				)
				print(
					'ttl_episode_intrinsic_reward: ', 
					self.episode_intrinsic_reward
				)
				print(
					'manager replay buffer size: ', 
					self.manager_replay_buffer.size()
				)
				print(
					'worker replay buffer size: ', 
					self.worker_replay_buffer.size()
				)
				extrinsic_rewards_summary = self.sess.run(
					self.tb_episode_extrinsic_reward,
					feed_dict={
						self.tb_episode_extrinsic_reward_in: 
						self.episode_extrinsic_reward,
					}
				)
				intrinsic_rewards_summary = self.sess.run(
					self.tb_episode_intrinsic_reward,
					feed_dict={
						self.tb_episode_intrinsic_reward_in:
						self.episode_intrinsic_reward,

						self.tb_episode_goal_in:
						1,
					}
				)
				self.tb_writer.add_summary(
					extrinsic_rewards_summary,
					self.episode
				)
				self.tb_writer.add_summary(
					intrinsic_rewards_summary,
					self.episode
				)
				self.episode += 1
				self.reset()
				self.initMDP()
			self.interact_rate.sleep()

	def train_manager(self):
		while not rospy.is_shutdown():
			batch=self.manager_replay_buffer.sample()
			if batch:
				next_goal = self.manager_actor.predict_target(
					traj_batch=batch['nexttraj'],
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose']
				)
				next_q = self.manager_critic.predict_target(
					traj_batch=batch['nexttraj'],
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose'],
					goal_batch=next_goal,
				)
				target_q = (
					batch['reward']
					+ 0.99*next_q*(1-batch['terminal'])
				)
				self.manager_critic.train(
					traj_batch=batch['nexttraj'],
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose'],
					goal_batch=batch['goal'], 
					target_q_batch=target_q
				)
				self.manager_critic.copy_master_network()
				action_gradients = self.manager_critic.cal_action_gradients(
					traj_batch=batch['nexttraj'],
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose'],
					goal_batch=batch['goal'],
				)
				self.manager_actor.copy_master_network()
				self.manager_actor.train(
					traj_batch=batch['nexttraj'],
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose'],
					action_gradients=action_gradients
				)		
				if self.manager_epoch%100==0:
					self.manager_actor.tensorboard_summary(
						traj_batch=batch['nexttraj'],
						obs_batch=batch['nextobs'],
						pose_batch=batch['nextpose'],
						action_gradients=action_gradients,
						timestamp=self.manager_epoch
					)
					self.manager_critic.tensorboard_summary(
						traj_batch=batch['nexttraj'],
						obs_batch=batch['nextobs'],
						pose_batch=batch['nextpose'],
						goal_batch=batch['goal'], 
						target_q_batch=target_q,
						timestamp=self.manager_epoch
					)
				self.manager_actor.update_target_network()
				self.manager_critic.update_target_network()
				self.manager_epoch+=1
			self.manager_training_rate.sleep()
	
	def train_worker(self):
		while not rospy.is_shutdown():
			batch = self.worker_replay_buffer.sample()
			if batch:
				next_action = self.worker_actor.predict_target(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					time_batch=batch['curtime'],
				)
				next_q = self.worker_critic.predict_target(
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose'],
					goal_batch=batch['nextgoal'],
					time_batch=batch['nexttime'],
					act_batch=next_action,
				)
				target_q = (
					batch['reward']
					+ 0.99*next_q*(1-batch['terminal'])
				)
				self.worker_critic.train(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					time_batch=batch['curtime'],
					act_batch=batch['action'],
					target_q_batch=target_q
				)
				self.worker_critic.copy_master_network()
				action_gradients = self.worker_critic.cal_action_gradients(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					time_batch=batch['curtime'],
					act_batch=batch['action'],
				)
				# print(action_gradients)
				self.worker_actor.copy_master_network()
				self.worker_actor.train(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					time_batch=batch['curtime'],
					action_gradients=action_gradients
				)
				if self.worker_epoch%100==0:
					self.worker_actor.tensorboard_summary(
						obs_batch=batch['curobs'],
						pose_batch=batch['curpose'],
						goal_batch=batch['curgoal'],
						time_batch=batch['curtime'],
						action_gradients=action_gradients,
						timestamp=self.worker_epoch
					)
					self.worker_critic.tensorboard_summary(
						obs_batch=batch['curobs'],
						pose_batch=batch['curpose'],
						goal_batch=batch['curgoal'],
						time_batch=batch['curtime'],
						act_batch=batch['action'],
						target_q_batch=target_q,
						timestamp=self.worker_epoch
					)
				self.worker_actor.update_target_network()
				self.worker_critic.update_target_network()
				self.worker_epoch += 1
			self.worker_training_rate.sleep()

	def initMDP(self):
		self.observe_state()
		# next_goal = self.final_goal-self.next_state.get_world_pos()
		# self.next_state.set_goal(next_goal)
		# assume that the first goal of an agent is <0,0>
		self.next_traj.append(self.next_state.clone())
		self.transite_state()
		self.action = self.worker_actor.predict(
			obs_batch=self.cur_state.get_obs(),
			pose_batch=self.cur_state.get_world_pose(),
			goal_batch=self.cur_state.get_goal_dir(),
			time_batch=self.cur_state.get_time(),
		)
		self.exploration_noise(self.action)
		self.time_step += 1.0
		self.actuate(self.action)
		rospy.sleep(0.2)  # wait state transition for 0.2s

	def actuate(self,action):
		velcmd=Twist()
		velcmd.linear.x = (
			self.action[0][0]
			* self.max_lvel
		)
		velcmd.linear.y = 0.0
		velcmd.linear.z = 0.0
		velcmd.angular.x = 0.0
		velcmd.angular.y = 0.0
		velcmd.angular.z = (
			self.action[0][1]
			* self.max_avel
		)
		self.velcmd_pub.publish(velcmd)

	def update_intrinsic_reward(self):
		ndist_to_final_goal = np.linalg.norm(
			self.next_state.get_world_pos()
			- self.final_goal
		)
		cdist_to_final_goal = np.linalg.norm(
			self.cur_state.get_world_pos()
			- self.final_goal
		)
		dist_to_sub_goal = np.linalg.norm(
			self.next_state.get_goal_pos()
			- self.next_state.get_pos()
		)
		self.intrinsic_reward = 20.0*(cdist_to_final_goal-ndist_to_final_goal)
		terminal = False
		if self.next_state.collided():
			self.intrinsic_reward += -10.0
			terminal=True
		if ndist_to_final_goal<0.5:
			self.intrinsic_reward += 10.0
			terminal=True
		return terminal

	def update_extrinsic_reward(self):
		dist_to_final_goal = np.linalg.norm(
			self.next_state.get_world_pos()
			- self.final_goal
		)
		dist_to_sub_goal = np.linalg.norm(
			self.next_state.get_goal_pos()
			- self.next_state.get_pos()
		)
		self.extrinsic_reward += -0.05*(
			dist_to_final_goal
			+ dist_to_sub_goal
		)
		if self.next_state.collided():
			self.extrinsic_reward += -1.0
		if dist_to_final_goal<0.5:
			return True
		return False

	def observe_state(self):
		self.load_lidar=True
		self.load_odom=True
		transiting=True
		while self.load_lidar or self.load_odom:
			pass
		self.next_state.set_time(self.time_step/self.episode_step_max)

	def transite_state(self):
		self.cur_state.copy(
			self.next_state
		)  # deepcopy

	#lidar callback for subscriber ith worker
	def lidar_receiver(self,msg):
		if self.load_lidar:
			ranges=np.asarray(msg.ranges)
			for i in xrange(len(ranges)):
				if ranges[i]==float('inf'):
					ranges[i]=1.0
				else:
					ranges[i]/=self.obs_max_range
			self.next_state.update_obsseq(ranges)
			self.next_state.update_event(ranges)
			self.load_lidar=False
			#print(self.name,"updated lidar obsseq")

	#odometry callback for subscriber i
	def odom_receiver(self, msg):
		if self.load_odom:
			vel=np.zeros(shape=6, dtype=float)
			vel[0]=msg.twist.twist.linear.x
			vel[1]=msg.twist.twist.linear.y
			vel[2]=msg.twist.twist.linear.z
			vel[3]=msg.twist.twist.angular.x
			vel[4]=msg.twist.twist.angular.y
			vel[5]=msg.twist.twist.angular.z
			pose=np.zeros(shape=6, dtype=float)
			r, p, y=rostf.transformations.euler_from_quaternion(
				[
					msg.pose.pose.orientation.x,
					msg.pose.pose.orientation.y,
					msg.pose.pose.orientation.z,
					msg.pose.pose.orientation.w
				]
			)
			pose[0]=msg.pose.pose.position.x
			pose[1]=msg.pose.pose.position.y
			pose[2]=msg.pose.pose.position.z
			pose[3]=r
			pose[4]=p
			pose[5]=y
			self.next_state.set_vel(vel=vel)
			self.next_state.set_world_pose(world_pose=pose)
			self.load_odom=False

	def reset(self):
		rate=rospy.Rate(10)	
		stop_time = rospy.get_time()
		while rospy.get_time()-stop_time<2:
			self.velcmd_pub.publish(Twist())
			rate.sleep()
		modelstate = ModelState()
		modelstate.model_name = "rosbot0"
		modelstate.reference_frame = "world"
		modelstate.pose.position.x = self.init_world_pose[0][0]
		modelstate.pose.position.y = self.init_world_pose[0][1]
		x,y,z,w = rostf.transformations.quaternion_from_euler(
			self.init_world_pose[0][3], 
			self.init_world_pose[0][4],
			self.init_world_pose[0][5]
		)
		modelstate.pose.orientation.x = x
		modelstate.pose.orientation.y = y
		modelstate.pose.orientation.z = z
		modelstate.pose.orientation.w = w
		reps = self.reset_client(modelstate)
		self.cur_state = State(
			world_pose=self.init_world_pose.copy(),
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_max_range=self.obs_max_range,
			collision_thresh=self.collision_thresh
		) 
		self.next_state = State(
			world_pose=self.init_world_pose.copy(),
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_max_range=self.obs_max_range,
			collision_thresh=self.collision_thresh
		)
		self.cur_traj = []
		self.next_traj = []
		self.episode_extrinsic_reward = 0
		self.episode_intrinsic_reward = 0
		self.extrinsic_reward = 0.0
		self.intrinsic_reward = 0.0
		self.time_step = 0.0
		self.episode_goal = 0.0
		self.episode_reached_goal = 0.0

	def start_training(self):
		self.manager_actor.init_target_network()
		self.manager_actor.copy_master_network()
		self.worker_actor.init_target_network()
		self.worker_actor.copy_master_network()
		self.manager_critic.init_target_network()
		self.manager_critic.copy_master_network()
		self.worker_critic.init_target_network()
		self.worker_critic.copy_master_network()
		self.t_behavior.start()
		# self.t_manager_train.start()
		self.t_worker_train.start()

	def exploration_noise(self,action):
		#lvel
		action[0][0]+=self.ornstein_uhlenbeck_noise_lv();
		action[0][0]=min(max(action[0][0],0),1)
		#avel
		action[0][1]+=self.ornstein_uhlenbeck_noise_av();
		action[0][1]=min(max(action[0][1],-1),1)

	def epsilon_greedy(self, goal):
		r = rand.uniform(0, 1)
		if r<self.epsilon or self.episode<30:
			goal = 2.0*(np.random.rand(1, self.goal_dim)-0.5)

	def ornstein_uhlenbeck_noise_lv(self):
		sigma = 0.1  # Standard deviation.
		mu = 0.  # Mean.
		tau = .05  # Time constant.
		dt = .001  # Time step.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise_lv += (dt*(-(self.ou_noise_lv-mu)/tau)
								+ sigma_bis*sqrtdt*np.random.randn())
		return self.ou_noise_lv

	def ornstein_uhlenbeck_noise_av(self):
		sigma = 0.1  # Standard deviation.
		mu = 0.  # Mean.
		tau = .05  # Time constant.
		dt = .001  # Time step.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise_av += (dt*(-(self.ou_noise_av-mu)/tau)
								+ sigma_bis*sqrtdt*np.random.randn())
		return self.ou_noise_av




