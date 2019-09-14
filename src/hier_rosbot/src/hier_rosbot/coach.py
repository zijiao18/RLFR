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
		n_agent,
		obs_dim,
		obs_seq_len,
		obs_emb_size,
		collision_thresh,
		batch_size,
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
		init_poses,
		final_goals=None,
		device='/device:GPU:0',
		tb_writer=None
	):
		self.sess=sess
		self.name=name
		self.n_agent=n_agent
		self.epoch=0
		self.episode=0
		self.eprwd=0
		self.step=0
		self.max_step=200
		self.max_lvel=1.0
		self.max_avel=3.14
		self.obs_dim=obs_dim
		self.obs_seq_len=obs_seq_len
		self.obs_max_range=2.0
		self.pos_dim=2
		self.vel_dim=3
		self.goal_dim=2
		self.act_dim=2
		self.collision_dist=0.22
		self.ou_noise_lv=[0.0 for _ in xrange(self.n_agent)]
		self.ou_noise_av=[0.0 for _ in xrange(self.n_agent)]
		self.batch_size=batch_size
		self.collision_thresh=collision_thresh
		self.final_goals=final_goals
		self.init_poses=init_poses
		self.terminal=False
		self.collided=False
		self.device=device
		self.manager_actor_fc1_size=manager_actor_fc1_size
		self.manager_actor_fc2_size=manager_actor_fc2_size
		self.manager_critic_fc1_size=manager_critic_fc1_size
		self.manager_critic_fc2_size=manager_critic_fc2_size
		self.manager_actor_lr=manager_actor_lr
		self.manager_critic_lr=manager_critic_lr
		self.manager_replay_buffer_size=manager_replay_buffer_size
		self.worker_actor_fc1_size=worker_actor_fc1_size
		self.worker_actor_fc2_size=worker_actor_fc2_size
		self.worker_critic_fc1_size=worker_critic_fc1_size
		self.worker_critic_fc2_size=worker_critic_fc2_size
		self.worker_actor_lr=worker_actor_lr
		self.worker_critic_lr=worker_critic_lr
		self.worker_replay_buffer_size=worker_replay_buffer_size
		self.master_manager_actor=master_manager_actor
		self.master_manager_critic=master_manager_critic
		self.master_worker_actor=master_worker_actor
		self.master_worker_critic=master_worker_critic
		self.manager_actor=ManagerActorNetwork(
			sess=self.sess,
			name=self.name+'_manager_actor',
			n_agent=self.n_agent,
			n_fc1_unit=self.manager_actor_fc1_size,
			n_fc2_unit=self.manager_actor_fc2_size,
			batch_size=self.batch_size,
			learning_rate=self.manager_actor_lr,
			device=self.device,
			master_network=self.master_manager_actor
		)
		self.manager_critic=ManagerCriticNetwork(
			sess=sess,
			name=name+'_manager_critic',
			n_agent=self.n_agent,
			n_fc1_unit=self.manager_critic_fc1_size, 
			n_fc2_unit=self.manager_critic_fc2_size,
			batch_size=batch_size,
			learning_rate=self.manager_critic_lr, 
			device=device,
			master_network=master_manager_critic
		)
		self.worker_actor=WorkerActorNetwork(
			sess=sess,
			name=name+'_worker_actor',
			obs_seq_len=obs_seq_len,
			obs_dim=obs_dim,
			obs_emb_size=obs_emb_size,
			n_fc1_unit=self.worker_actor_fc1_size,
			n_fc2_unit=self.worker_actor_fc2_size,
			batch_size=batch_size,
			learning_rate=worker_actor_lr,
			device=device,
			master_network=master_worker_actor
		)
		self.worker_critic=WorkerCriticNetwork(
			sess=sess,
			name=name+'_worker_critic',
			obs_dim=obs_dim,
			obs_seq_len=obs_seq_len,
			obs_emb_size=obs_emb_size,
			n_fc1_unit=self.worker_critic_fc1_size,
			n_fc2_unit=self.worker_critic_fc2_size,
			batch_size=self.batch_size,
			learning_rate=worker_critic_lr,
			device=device,
			master_network=master_worker_critic
		)
		self.manager_replay_buffer=ManagerReplayBuffer(
			max_size=self.manager_replay_buffer_size,
			batch_size=self.batch_size,
			n_agent=self.n_agent,
		)
		self.worker_replay_buffer=WorkerReplayBuffer(
			max_size=self.worker_replay_buffer_size,
			batch_size=self.batch_size,
			obs_dim=obs_dim,
			obs_seq_len=self.obs_seq_len,
		)
		self.cur_states=[
			State(
				pose=self.init_poses[i].copy(),
				index=i,
				obs_dim=self.obs_dim,
				obs_seq_len=self.obs_seq_len,
				obs_max_range=self.obs_max_range,
				collision_thresh=self.collision_thresh
			) 
			for i in xrange(self.n_agent)
		]
		self.next_states=[
			State(
				pose=init_poses[i].copy(),
				index=i,
				obs_dim=self.obs_dim,
				obs_seq_len=self.obs_seq_len,
				obs_max_range=self.obs_max_range,
				collision_thresh=self.collision_thresh
			)
			for i in xrange(self.n_agent)
		]
		self.actions=[
			np.zeros(
				[1, self.act_dim],
				dtype=float
			)
			for _ in xrange(self.n_agent)
		]
		self.intrinsic_rewards=np.zeros(
			shape=self.n_agent, 
			dtype=float
		)
		self.extrinsic_rewards=np.zeros(
			shape=self.n_agent, 
			dtype=float
		)
		self.velcmd_pubs=[
			rospy.Publisher(
				"/rosbot"+str(i)+"/cmd_vel", 
				Twist, 
				queue_size=1000
			)
			for i in xrange(self.n_agent)
		]
		self.scan_subs=[
			rospy.Subscriber(
				"/rosbot"+str(i)+"/scan",
				LaserScan,
				self.lidar_receiver,
				i
			)
			for i in xrange(self.n_agent)
		]
		self.odom_subs=[
			rospy.Subscriber(
				"/rosbot"+str(i)+"/odom",
				Odometry,
				self.odom_receiver,
				i
			)
			for i in xrange(self.n_agent)
		]
		self.reset_client=rospy.ServiceProxy(
			"/gazebo/set_model_state",
			SetModelState
		)
		self.training_rate=rospy.Rate(30)
		self.interact_rate=rospy.Rate(10)
		self.load_lidar=[False for _ in xrange(self.n_agent)]
		self.load_odom=[False for _ in xrange(self.n_agent)]
		self.t_behavior=Thread(target=self.behave)
		self.t_manager_train=Thread(target=self.train_manager)
		self.t_worker_train=Thread(target=self.train_worker)

	def behave(self):
		self.initMDP()
		while not rospy.is_shutdown():
			self.observe_state()
			ext_rewards, terminal=self.recieve_extrinsic_reward()
			int_rewards=self.receive_intrinsic_reward()
			joint_nextpos = np.concatenate(
				[
					s.get_pos() 
					for s in self.next_states
				],
				1
			)
			for i in xrange(self.n_agent):
				if self.next_states[i].reached_goal():
					goal = self.manager_actor.predict(
						joint_pos_batch=joint_nextpos,
						ind_batch=np.array([[i]], dtype=float)
					)
					self.next_states[i].set_goal(goal)
					self.manager_replay_buffer.add(
						cur_states=self.cur_states,
						reward=ext_rewards[i],
						next_states=self.next_states,
						terminal=terminal,
						index=i
					)
				self.worker_replay_buffer.add(
					cur_state=self.cur_states[i],
					action=self.actions[i],
					reward=int_rewards[i],
					next_state=self.next_states[i],
					terminal=self.next_states[i].reached_goal()
				)
			self.transite_state()
			self.actions=[]
			for i in xrange(self.n_agent):
				action=self.worker_actor.predict(
					obs_batch=self.cur_states[i].get_obs(),
					pose_batch=self.cur_states[i].get_pose(),
					goal_batch=self.cur_states[i].get_goal_dir(),
				)
				self.exploration_noise(action,i)
				self.actions.append(action)
			self.actuate(self.actions)
			self.step+=1
			if terminal:
				print('-----------------------'+self.name+'--------------------------')
				print('episode:',self.episode)
				print('step:',self.step)
				print('manager replay buffer size: ', self.manager_replay_buffer.size())
				print('worker replay buffer size: ', self.worker_replay_buffer.size())
				self.step=0
				self.eprwd=0
				self.episode+=1#read by train()
				self.reset()
				self.initMDP()
			self.interact_rate.sleep()

	def train_manager(self):
		while not rospy.is_shutdown():
			batch=self.manager_replay_buffer.sample()
			if batch:
				self.manager_actor.copy_master_network()
				next_goal = self.manager_actor.predict(
					joint_pos_batch=batch['joint_nextpos'],
					ind_batch=batch['index']
				)
				self.manager_critic.copy_master_network()
				next_q = self.manager_critic.predict(
					joint_pos_batch=batch['joint_nextpos'],
					goal_batch=next_goal,
					ind_batch=batch['index'],
				)
				target_q = batch['reward']+0.99*next_q*(1-batch['terminal'])
				self.manager_critic.train(
					joint_pos_batch=batch['joint_curpos'], 
					goal_batch=batch['goal'], 
					ind_batch=batch['index'],
					target_q_batch=target_q
				)
				q_values = self.manager_critic.predict(
					joint_pos_batch=batch['joint_curpos'],
					goal_batch=batch['goal'],
					ind_batch=batch['index'],
				)
				self.manager_actor.train(
					joint_pos_batch=batch['joint_curpos'], 
					ind_batch=batch['index'], 
					cpos_batch=batch['curpos'],
					npos_batch=batch['nextpos'],
					value_batch=q_values
				)		
				self.manager_actor.update_target_network()
				self.manager_critic.update_target_network()
				self.epoch+=1
			self.training_rate.sleep()
	
	def train_worker(self):
		while not rospy.is_shutdown():
			batch = self.worker_replay_buffer.sample()
			if batch:
				self.worker_actor.copy_master_network()
				next_action = self.worker_actor.predict(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
				)
				self.worker_critic.copy_master_network()
				next_q = self.worker_critic.predict(
					obs_batch=batch['nextobs'],
					pose_batch=batch['nextpose'],
					goal_batch=batch['nextgoal'],
					act_batch=next_action
				)
				target_q = batch['reward']+0.99*next_q*(1-batch['terminal'])
				self.worker_critic.train(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					act_batch=batch['action'],
					target_q_batch=target_q
				)
				self.worker_critic.copy_master_network()
				action_gradients = self.worker_critic.cal_action_gradients(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					act_batch=batch['action'],
				)
				self.worker_actor.train(
					obs_batch=batch['curobs'],
					pose_batch=batch['curpose'],
					goal_batch=batch['curgoal'],
					action_gradients=action_gradients
				)
			self.training_rate.sleep()

	def initMDP(self):
		self.observe_state()
		self.transite_state()
		joint_pos = np.concatenate(
			[s.get_pos() for s in self.cur_states],
			1
		)
		self.actions=[]
		for i in xrange(self.n_agent):
			goal=self.manager_actor.predict(
				joint_pos_batch=joint_pos,
				ind_batch=np.array([[i]])
			)
			self.next_states[i].set_goal(goal)
			action=self.worker_actor.predict(
				obs_batch=(self.cur_states[i].get_obs()),
				pose_batch=self.cur_states[i].get_pose(),
				goal_batch=self.cur_states[i].get_goal_dir()
			)
			self.exploration_noise(action, i)
			self.actions.append(action)
		self.actuate(self.actions)
		self.interact_rate.sleep() #wait for state transition

	def actuate(self,actions):
		for i in xrange(self.n_agent):
			velcmd=Twist()
			velcmd.linear.x=self.actions[i][0][0]*self.max_lvel
			velcmd.linear.y=0.0
			velcmd.linear.z=0.0
			velcmd.angular.x=0.0
			velcmd.angular.y=0.0
			velcmd.angular.z=self.actions[i][0][1]*self.max_avel
			self.velcmd_pubs[i].publish(velcmd)

	#determine rewards received by each robot
	def receive_intrinsic_reward(self):
		int_rewards=np.array([0 for _ in xrange(self.n_agent)])
		return int_rewards

	def recieve_extrinsic_reward(self):
		ext_rewards=np.array([0 for _ in xrange(self.n_agent)])
		return ext_rewards, 0

	#observe the next joint state
	def observe_state(self):
		for i in xrange(self.n_agent):
			self.load_lidar[i]=True
			self.load_odom[i]=True
		transiting=True
		while transiting:
			transiting=False
			for i in xrange(self.n_agent):
				transiting |= (self.load_lidar[i]|self.load_odom[i])

	def transite_state(self):
		for i in xrange(self.n_agent):
			self.cur_states[i].copy(
				self.next_states[i]
			)  # deepcopy

	#lidar callback for subscriber ith worker
	def lidar_receiver(self,msg,wi):
		if self.load_lidar[wi]:
			ranges=np.asarray(msg.ranges)
			for i in xrange(len(ranges)):
				if ranges[i]==float('inf'):
					ranges[i]=1.0
				else:
					ranges[i]/=self.obs_max_range
			self.next_states[wi].update_obsseq(ranges)
			self.next_states[wi].update_event(ranges)
			self.load_lidar[wi]=False
			#print(self.name,"updated lidar obsseq")

	#odometry callback for subscriber i
	def odom_receiver(self, msg, wi):
		if self.load_odom[wi]:
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
			self.next_states[wi].set_vel(vel)
			self.next_states[wi].set_pose(pose)
			self.load_odom[wi]=False

	def reset(self):
		rate=rospy.Rate(10)		
		while rospy.get_time()-rospy.get_time()<2:
			for i in xrange(self.n_agent):
				self.velcmd_pubs[i].publish(Twist())
			rate.sleep()
		for i in xrange(self.n_agent):
			modelstate = ModelState()
			modelstate.model_name = "rosbot"+str(i+self.rtoffset)
			modelstate.reference_frame = "world"
			modelstate.pose.position.x=self.init_poses[i][0]
			modelstate.pose.position.y=self.init_poses[i][1]
			x,y,z,w=rostf.transformations.quaternion_from_euler(
				self.init_poses[i][3], 
				self.init_poses[i][4],
				self.init_poses[i][5]
			)
			modelstate.pose.orientation.x=x
			modelstate.pose.orientation.y=y
			modelstate.pose.orientation.z=z
			modelstate.pose.orientation.w=w
			reps=self.reset_client(modelstate)
		self.cur_states=[
			State(
				pose=init_poses[i].copy(),
				index=i,
				obs_dim=self.obs_dim,
				obs_seq_len=self.obs_seq_len,
				obs_max_range=self.obs_max_range,
				collision_thresh=self.collision_thresh
			) 
			for i in xrange(self.n_agent)
		]
		self.next_states=[
			State(
				pose=self.init_poses[i].copy(),
				index=i,
				obs_dim=self.obs_dim,
				obs_seq_len=self.obs_seq_len,
				obs_max_range=self.obs_max_range,
				collision_thresh=self.collision_thresh
			)
			for i in xrange(self.n_agent)
		]
		self.actions=[
			np.zeros(
				[1, self.act_dim],
				dtype=float
			)
			for _ in xrange(self.n_agent)
		]

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
		self.t_manager_train.start()
		self.t_worker_train.start()

	def start_testing(self):
		self.actor.copy_master_network()
		self.critic.copy_master_network()
		self.t_testing.start()

	def exploration_noise(self,action,wi):
		#lvel
		action[0][0]+=self.ornstein_uhlenbeck_noise_lv(wi);
		action[0][0]=min(max(action[0][0],0),1)
		#avel
		action[0][1]+=self.ornstein_uhlenbeck_noise_av(wi);
		action[0][1]=min(max(action[0][1],-1),1)

	def ornstein_uhlenbeck_noise_lv(self,i):
		sigma = 0.1  # Standard deviation.
		mu = 0.  # Mean.
		tau = .05  # Time constant.
		dt = .001  # Time step.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise_lv[i] += (dt*(-(self.ou_noise_lv[i]-mu)/tau)
								+ sigma_bis*sqrtdt*np.random.randn())
		return self.ou_noise_lv[i]

	def ornstein_uhlenbeck_noise_av(self,i):
		sigma = 0.1  # Standard deviation.
		mu = 0.  # Mean.
		tau = .05  # Time constant.
		dt = .001  # Time step.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise_av[i] += (dt*(-(self.ou_noise_av[i]-mu)/tau)
								+ sigma_bis*sqrtdt*np.random.randn())
		return self.ou_noise_av[i]

if __name__ == '__main__':
	init_pose=np.array([[[-4.0,14.0,0.0,0,0,-1.57]],
						[[-4.0,4.0,0.0,0,0,0]],
						[[-4.0,-6.0,0.0,0,0,0]],
						#[[11.0,14.0,3.0,0,0,0]],
						#[[19.0,4.0,3.0,0,0,0]],
						[[-39.0,9.5,0.0,0,0,-1.57]]],dtype=float)

	goals=np.array([[[8,10.0,0.0]],
					[[4.0,-4.0,0.0]],
					[[4.0,-13.0,0.0]],
					#[[17.0,10.0,0.0]],
					#[[13.5,0.0,0.0]],
					[[-11.0,-9.5,0.0]]],dtype=float)
	rospy.init_node("coach_test")
	c = Coach(sess=None,initpose=init_pose,goals=goals,
			master_actor=None,master_critic=None,rtoffset=0)



