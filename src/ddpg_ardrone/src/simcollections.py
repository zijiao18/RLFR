#!/usr/bin/env python
# coding: utf8
import numpy as np
import random as rand
from threading import Lock


class SimulatorFeedback():
	'''
	lidar setup:
		based on the "RPLIDAR A2M8 360° Laser Scanner"
		n_horizon_ray = 400
		n_vertical_ray = 1
	'''
	def __init__(self):
		self.sender = ""
		self.reward = 0; 
		self.velocity = np.zeros([6,1])  # lx,ly,lz,rx,ry,rz
		self.lidar_ranges = np.zeros([360,1])
		self.pose = np.zeros([6,1])  # x,y,z,r,p,y

	def load(self, data):
		'''
		data from simulator:
		[
			isterminal,reward,
			linear_velocity,
			angular_velocity,
			position_xyz,l
			idar_observation
		]
		'''
		self.reward=data[0]
		i=1;
		for i in range(1,7):
			self.velocity[i-1][0]=data[i]
		for i in range(7,13):
			self.pose[i-7][0]=data[i]
		for i in range(13,len(data)):
			self.lidar_ranges[i-13][0]=data[i]

	def set_sender(self,name):
		self.sender=name;


class ReplayBuffer:
	'''
	Implemented as round array. 
	The oldest experience will 
	be overwritten when the buf
	is full filled
	'''
	def __init__(self,size,thresh):
		self.max_size=size#
		self.min_size=thresh
		self.ttl_size=0
		self.iterator=0
		self.dim_eoe=int(1)
		self.dim_rwd=int(1)
		self.dim_act=int(1)  # yaw rotational velocity
		self.dim_vel=int(6)
		self.dim_dir=int(3)
		self.dim_obs=int(360)
		self.dim_exp=int(
			self.dim_eoe
			+ self.dim_rwd
			+ self.dim_act
			+ self.dim_vel
			+ self.dim_dir
			+ self.dim_obs
		)
		self.max_lidar_range=float(5.0)  # max sensor range
		self.gdir_normalizer=float(100.0)  # max range between the robot and the goal
		self.buf=np.zeros(
			shape=[
				self.dim_exp,
				self.max_size
			],
			dtype=float
		)  # [r,a,s], dimission: [1,2,6+360]
		self.buf_lock=Lock()

	def sample_batch(self,time_step,batch_size):
		try:
			if self.ttl_size<self.min_size:
				print(
					"ReplayBuffer.sample(): "
					+ "ReplayBuffer is not initialized,"
					+ " %d/%d"%(self.ttl_size,self.min_size)
				)
				return None
			cur_obs_batch = np.zeros(
				[
					time_step,
					batch_size,
					self.dim_obs
				],
				dtype=float
			)
			cur_vel_batch = np.zeros(
				[
					1,
					batch_size,
					self.dim_vel
				],
				dtype=float
			)
			cur_dir_batch = np.zeros(
				[
					1,
					batch_size,
					self.dim_dir
				],
				dtype=float
			)
			cur_act_batch = np.zeros(
				[
					1,
					batch_size,
					self.dim_act
				],
				dtype=float
			)
			next_obs_batch = np.zeros(
				[
					time_step,
					batch_size,
					self.dim_obs
				],
				dtype=float
			)
			next_vel_batch=np.zeros(
				[
					1,
					batch_size,
					self.dim_vel
				],
				dtype=float
			)
			next_dir_batch=np.zeros(
				[
					1,
					batch_size,
					self.dim_dir
				],
				dtype=float
			)
			next_act_batch=np.zeros(
				[
					1,
					batch_size,
					self.dim_act
				],
				dtype=float
			)
			rwd_batch=np.zeros(
				[
					1,
					batch_size,
					self.dim_rwd
				],
				dtype=float
			)
			isterm_batch=np.zeros(
				[
					1,
					batch_size,
					1
				],
				dtype=float
			)
			rwd_i = self.dim_eoe
			act_i = (self.dim_eoe
					+ self.dim_rwd)
			vel_i = (self.dim_eoe
					+ self.dim_rwd
					+ self.dim_act)
			dir_i = (self.dim_eoe
					+ self.dim_rwd
					+ self.dim_act
					+ self.dim_vel)
			obs_i = (self.dim_eoe
					+ self.dim_rwd
					+ self.dim_act
					+ self.dim_vel
					+ self.dim_dir)
			for b in xrange(0,batch_size):
				bot_offset=rand.randint(
					1,
					self.ttl_size-1-time_step
				)  # index range: [time_step, self.ttl_size-2]
				cur_i = (self.iterator
						- 1
						- bot_offset
						+ self.max_size
						) % self.max_size
				if self.buf[0][cur_i]==2:   # The current state 
											# can not be at the 
											# cutoff of the episode
					cur_i = (cur_i
							- 1
							+ self.max_size
							) % self.max_size

				next_i = (cur_i+1)%self.max_size    # The next state can 
													# be at the cutoff of 
													# the episode

				isterm_batch[0][b][0] = 1.0 if self.buf[0][cur_i]==1 else 0.0
				rwd_batch[0][b][0]=self.buf[rwd_i][cur_i]

				cur_act_batch[0, b, :] = self.buf[
					act_i:act_i+self.dim_act,cur_i
				]
				cur_vel_batch[0, b, :] = self.buf[
					vel_i:vel_i+self.dim_vel,cur_i
				]
				cur_dir_batch[0, b, :] = self.buf[
					dir_i:dir_i+self.dim_dir,cur_i
				]

				t=0
				while t<time_step and t<(self.ttl_size-bot_offset):  # Avoid the jump 
																	 # from the first 
																	 # experience to 
																	 # the last experience
					cur_obs_batch[t,b,:]=self.buf[
						obs_i
						:
						obs_i+self.dim_obs,cur_i
					]
					cur_i=(cur_i-1+self.max_size)%self.max_size
					if self.buf[0][cur_i]>0:  # the cur_i has 
											  # exceeded the episode
						break
					t+=1
				#the current state is not the terminal state
				if isterm_batch[0][b][0]==0.0:
					#load the next state and action
					next_act_batch[0,b,:] = self.buf[
						act_i:act_i+self.dim_act,next_i
					]
					next_vel_batch[0,b,:] = self.buf[
						vel_i:vel_i+self.dim_vel,next_i
					]
					next_dir_batch[0,b,:] = self.buf[
						dir_i:dir_i+self.dim_dir,next_i
					]
					'''
					The observation is loaded backward, 
					starting from the next_i. Should avoid 
					jumpping from the first experience 
					to the last one.
					'''
					t=0
					while t<time_step and t<self.ttl_size-bot_offset:
						next_obs_batch[t,b,:]=self.buf[
							obs_i:obs_i+self.dim_obs,next_i
						]
						next_i=(next_i-1+self.max_size)%self.max_size
						if self.buf[0][next_i]>0:   # the next_i exceeds 
													# the head of the episode
							break
						t+=1

			return {
						'cur_obs':cur_obs_batch,
						'cur_vel':cur_vel_batch,
						'cur_dir':cur_dir_batch,
						'cur_act':cur_act_batch,
						'next_obs':next_obs_batch,
						'next_vel':next_vel_batch,
						'next_dir':next_dir_batch,
						'next_act':next_act_batch,
						'rwd':rwd_batch,
						'isterm':isterm_batch
					}
		except Exception as e:
			print("ReplayBuffer.sample_batch(): ",e)
			raise

	def add(self,feedback,action,goal,isterm,exhusted_episode): 
		try:
			bot_itr = self.iterator
			gdir = (goal-feedback.pose[0:3])
			gdir_norm = np.linalg.norm(gdir)
			if gdir_norm>0:
				gdir /= gdir_norm
			#[t,r,a,s] dimission: [1,1,2,6+3+360]
			i = 0
			# the order shall not be changed, 
			# in order to garrentee the terminal
			# state being marked as 1
			self.buf[i][bot_itr] = 0.0
			if exhusted_episode:
				self.buf[i][bot_itr] = 2.0
			if isterm:
				self.buf[i][bot_itr] = 1.0
			i += self.dim_eoe
			self.buf[i][bot_itr] = float(feedback.reward)
			i += self.dim_rwd
			self.buf[
				i:i+self.dim_act,bot_itr
			] = action[:, 0]
			i += self.dim_act
			self.buf[
				i:i+self.dim_vel,bot_itr
			] = feedback.velocity[:, 0]
			i += self.dim_vel
			self.buf[
				i:i+self.dim_dir,bot_itr
			] = gdir[:, 0]
			i += self.dim_dir
			self.buf[
				i:i+self.dim_obs,bot_itr
			] = (feedback.lidar_ranges[:, 0]/self.max_lidar_range)
			self.iterator = (self.iterator+1)%self.max_size
			if self.ttl_size<self.max_size:
				self.ttl_size += 1
		except Exception as e:
			print("ReplayBuffer.add(): ", e)

	def advance(self,exp):
		self.buf[:,self.iterator] = exp[:]
		self.iterator = (self.iterator+1)%self.max_size
		if self.ttl_size<self.max_size:
			self.ttl_size += 1

	def get(self, offset_from_back):
		exp_i = (self.iterator
				- 1
				- offset_from_back
				+ self.max_size
				) % self.max_size
		return np.array(self.buf[:,exp_i])

	def size(self):
		return self.ttl_size
	
	def next_state(self,time_step,goal,feedback):
		'''
		exp_i is initialized to be the most 
		recent experience of the robot i
		self.iterators are pointing to the 
		head of robot buffers
		'''
		exp_i = (self.iterator
				- 1 
				+ self.max_size
				) % self.max_size
		obs = np.zeros(
			[
				time_step,
				1,
				self.dim_obs
			],
			dtype=float
		)
		vel = np.zeros(
			[
				1,
				1,
				self.dim_vel
			],
			dtype=float
		)
		gdir = np.zeros(
			[
				1,
				1,
				self.dim_dir
			],
			dtype=float
		)
		obs[0,0,:] = feedback.lidar_ranges[:,0]
		vel[0,0,:] = feedback.velocity[:,0]
		gvec = (goal-feedback.pose[0:3])
		gvec_norm = np.linalg.norm(gvec)
		if gvec_norm>0:
			gvec /= gvec_norm
		gdir[0,0,:]=gvec[:,0]
		if self.buf[0][exp_i]==0:
			'''
			If the previous feedback if not 
			end of the episode, continue to 
			load previous observations until 
			reached the end of a episode or 
			exhusted the existing experience 
			in the robot's buffer
			'''
			t = 1
			obs_i = (self.dim_eoe
					+ self.dim_rwd
					+ self.dim_act
					+ self.dim_vel
					+ self.dim_dir)
			while t < time_step and t-1<self.ttl_size:
				obs[t,0,:] = self.buf[
					obs_i:obs_i+self.dim_obs,
					exp_i
				]
				exp_i = (exp_i
						- 1
						+ self.max_size
						) % self.max_size
				if self.buf[0][exp_i]>0:  # If reached the 
										  # the terminal of 
										  # the previous episode
					break
				t += 1
		return obs, vel, gdir


