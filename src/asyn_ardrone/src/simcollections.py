#!/usr/bin/env python
# coding: utf8
import numpy as np
import random as rand
from threading import Lock
'''
Consult the DDPG_ardrone/simcollections 
for the implementation.
'''
class SimulatorFeedback():
	def __init__(self):
		self.sender  = ""
		self.reward = 0; 
		self.velocity = np.zeros([6,1])
		self.lidar_ranges = np.zeros([360,1])
		self.pose = np.zeros([6,1])

	def load(self, data):
		'''
		data from simulator:
		[
			isterminal,
			reward,
			linear_velocity,
			angular_velocity,
			position_xyz,
			lidar_observation
		]
		'''
		self.reward=data[0]
		i=1;
		for i in range(1,7):
			self.velocity[i-1][0] = data[i]
		for i in range(7,13):
			self.pose[i-7][0] = data[i]
		for i in range(13,len(data)):
			self.lidar_ranges[i-13][0] = data[i]

	def set_sender(self,name):
		self.sender=name;


class ReplayBuffer:
	def __init__(self,size,thresh):
		self.max_size = size
		self.min_size = thresh
		self.ttl_size = 0
		self.iterator = 0
		self.dim_eoe = int(1)
		self.dim_rwd = int(1)
		self.dim_act = int(1)
		self.dim_vel = int(6)
		self.dim_dir = int(3)
		self.dim_obs = int(360)
		self.dim_exp = int(
			self.dim_eoe
			+ self.dim_rwd
			+ self.dim_act
			+ self.dim_vel
			+ self.dim_dir
			+ self.dim_obs
		)
		self.max_lidar_range = float(5.0)
		self.gdir_normalizer = float(100.0)
		self.buf = np.zeros(
			shape=[
				self.dim_exp,
				self.max_size
			],
			dtype=float
		)

	def sample_batch(self,time_step,batch_size):
		try:
			if self.ttl_size<self.min_size:
				print("ReplayBuffer.sample(): "
					+ "ReplayBuffer is not "
					+ "initialized, %d/%d"%(
						self.ttl_size,
						self.min_size
					)
				)
				return None
			cur_obs_batch = np.zeros([time_step,batch_size,self.dim_obs])
			cur_vel_batch = np.zeros([1,batch_size,self.dim_vel])
			cur_dir_batch = np.zeros([1,batch_size,self.dim_dir])
			cur_act_batch = np.zeros([1,batch_size,self.dim_act])
			next_obs_batch = np.zeros([time_step,batch_size,self.dim_obs])
			next_vel_batch = np.zeros([1,batch_size,self.dim_vel])
			next_dir_batch = np.zeros([1,batch_size,self.dim_dir])
			next_act_batch = np.zeros([1,batch_size,self.dim_act])
			rwd_batch = np.zeros([1,batch_size,self.dim_rwd])
			isterm_batch = np.zeros([1,batch_size,1])
			rwd_i = self.dim_eoe
			act_i = self.dim_eoe+self.dim_rwd
			vel_i = self.dim_eoe+self.dim_rwd+self.dim_act
			dir_i = self.dim_eoe+self.dim_rwd+self.dim_act+self.dim_vel
			obs_i = self.dim_eoe+self.dim_rwd+self.dim_act+self.dim_vel+self.dim_dir
			
			for b in xrange(0,batch_size):
				bot_offset = rand.randint(
					1,
					self.ttl_size-1-time_step
				)
				cur_i = (self.iterator
						- 1
						- bot_offset
						+ self.max_size
						) % self.max_size
				if self.buf[0][cur_i]==2:
					cur_i = (cur_i
							- 1
							+ self.max_size
							) % self.max_size
				next_i = (cur_i+1)%self.max_size
				isterm_batch[0][b][0] = 1.0 if self.buf[0][cur_i]==1 else 0.0
				rwd_batch[0][b][0] = self.buf[rwd_i][cur_i]
				cur_act_batch[0,b,:] = self.buf[act_i:act_i+self.dim_act,cur_i]
				cur_vel_batch[0,b,:] = self.buf[vel_i:vel_i+self.dim_vel,cur_i]
				cur_dir_batch[0,b,:] = self.buf[dir_i:dir_i+self.dim_dir,cur_i]
				t = 0
				while t<time_step and t<self.ttl_size-bot_offset:
					cur_obs_batch[t,b,:] = self.buf[obs_i:obs_i+self.dim_obs,cur_i]
					cur_i = (cur_i
							- 1
							+ self.max_size
							) % self.max_size
					if self.buf[0][cur_i]>0:
						break
					t+=1
				if isterm_batch[0][b][0]==0.0:
					next_act_batch[0,b,:] = self.buf[
						act_i:act_i+self.dim_act,next_i
					]
					next_vel_batch[0,b,:] = self.buf[
						vel_i:vel_i+self.dim_vel,next_i
					]
					next_dir_batch[0,b,:] = self.buf[
						dir_i:dir_i+self.dim_dir,next_i
					]
					t = 0
					while t<time_step and t<self.ttl_size-bot_offset:
						next_obs_batch[t,b,:] = self.buf[
							obs_i:obs_i+self.dim_obs,next_i
						]
						next_i = (next_i
								 - 1
								 + self.max_size
								 ) % self.max_size
						if self.buf[0][next_i]>0:
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

	def add(
		self,
		feedback,
		action,
		goal,
		isterm,
		exhusted_episode
	): 
		try:
			bot_itr = self.iterator
			gdir = (goal-feedback.pose[0:3])
			gdir_norm = np.linalg.norm(gdir)
			if gdir_norm>0:
				gdir /= gdir_norm
			i = 0
			self.buf[i][bot_itr] = 0.0
			if exhusted_episode:
				self.buf[i][bot_itr] = 2.0
			if isterm:
				self.buf[i][bot_itr] = 1.0
			i += self.dim_eoe
			self.buf[i][bot_itr] = float(feedback.reward)
			i += self.dim_rwd
			self.buf[i:i+self.dim_act,bot_itr] = action[:,0]
			i += self.dim_act
			self.buf[i:i+self.dim_vel,bot_itr] = feedback.velocity[:,0]
			i += self.dim_vel
			self.buf[i:i+self.dim_dir,bot_itr] = gdir[:,0]
			i += self.dim_dir
			self.buf[i:i+self.dim_obs,bot_itr] = (feedback.lidar_ranges[:,0]
												/ self.max_lidar_range)
			self.iterator = (self.iterator+1)%self.max_size
			if self.ttl_size<self.max_size:
				self.ttl_size+=1

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
		return np.array(self.buf[:,exp_i])#1d array

	def size(self):
		return self.ttl_size
	
	def next_state(
		self,
		time_step,
		goal,
		feedback
	):
		exp_i = (self.iterator
				- 1
				+ self.max_size
				) % self.max_size
		obs = np.zeros(
			shape=[
				time_step,
				1,
				self.dim_obs
			].
			dtype=float
		)
		vel = np.zeros(
			shape=[
				1,
				1,
				self.dim_vel
			],
			dtype=float
		)
		gdir = np.zeros(
			shape=[
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
		gdir[0,0,:] = gvec[:,0]
		if self.buf[0][exp_i]==0:
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
				if self.buf[0][exp_i]>0:
					break
				t += 1
		return obs, vel, gdir


