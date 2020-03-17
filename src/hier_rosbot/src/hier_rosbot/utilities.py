#!usr/bin/env python
import numpy as np
import random as rand; rand.seed(0)
from collections import deque
import threading

class State():
	def __init__(
		self,
		world_pose,
		obs_dim,
		obs_seq_len,
		obs_max_range,
		collision_thresh
	):
		self.event = 0
		self.pose_dim = 6
		self.vel_dim = 3
		self.pos_dim = 2
		self.goal_dim = 2
		self.obs_dim = obs_dim
		self.obs_itr = 0
		self.obs_seq_len = obs_seq_len
		self.obs_max_range = obs_max_range
		self.collision_thresh = collision_thresh
		self.goal_range = 1.0  # max distance for reaching a goal
		# initial world pose
		self.init_world_pose = np.array(
			world_pose,
			dtype=float
		)
		# reference world pose
		self.ref_world_pose = np.array(
			world_pose,
			dtype=float
		)
		# pose relative to init_pose
		self.pose = np.zeros(
			shape=[1, self.pose_dim],
			dtype=float
		)
		self.goal = np.zeros(
			shape=[1, self.goal_dim],
			dtype=float
		)
		self.goal_world_pos = np.zeros(
			shape=[1, self.goal_dim],
			dtype=float
		)
		self.time = np.array(
			[[0.0]],
			dtype=float
		)
		self.vel = np.zeros(
			shape=[1, self.vel_dim],
			dtype=float
		)
		self.obs = np.zeros(
			shape=[
				self.obs_seq_len, 
				1, 
				self.obs_dim
			],
			dtype=float
		)

	def reached_goal(self):
		dist = np.linalg.norm(
			(
				(
					self.pose
					+ self.ref_world_pose
				)[0, 0:self.pos_dim]
				- self.goal_world_pos
			)
		)
		return dist < 0.5

	def collided(self):
		return self.event==1

	def update_obsseq(self, lidar_ranges):
		if self.obs_itr==self.obs_seq_len:		
			for i in xrange(self.obs_seq_len-1):
				self.obs[i,0,:]=self.obs[i+1,0,:]
			self.obs[self.obs_seq_len-1,0,:]=lidar_ranges
		else:
			self.obs[self.obs_itr,0,:]=lidar_ranges
			self.obs_itr+=1

	def update_event(self, lidar_ranges):
		# 0, 1: no collision, collided
		min_r=self.obs_max_range
		for r in lidar_ranges:
			min_r = min(min_r, r)
		if min_r<=self.collision_thresh:
			self.event = 1  

	def set_time(self, time):
		self.time[0, :] = time 

	def set_pose(self, pose):
		self.pose[0, :] = pose

	def reset_pose(self):
		self.ref_world_pose = self.pose+self.ref_world_pose
		self.pose.fill(0.0)

	def set_world_pose(self, world_pose):
		self.pose[0,:] = world_pose-self.ref_world_pose

	def set_goal(self, goal):
		self.goal[0, :] = goal
		self.goal_world_pos[0, :] = (
			(
				self.pose
				+ self.ref_world_pose
			)[0,0:2]
			+ self.goal_range*goal
		)
		

	def set_vel(self, vel):
		self.vel[0][0] = vel[0]  # x
		self.vel[0][1] = vel[1]  # y
		self.vel[0][2] = vel[5]  # yaw
	
	def get_time(self):
		return self.time

	def get_obs(self):
		return self.obs

	def get_vel(self):
		return self.vel

	def get_pose(self):
		return self.pose

	def get_pos(self):
		return self.pose[:, 0:self.pos_dim]

	def get_world_pose(self):
		# final_pos = np.array(
		# 	[[-1.5,-1.7]],
		# 	dtype=float
		# )
		# world_pose = self.pose+self.ref_world_pose
		# goal_dir = final_pos-world_pose[:,0:2]
		# world_pose[:,0:2]=goal_dir
		# return world_pose
		return self.pose+self.ref_world_pose

	def get_world_pos(self):
		return (
			self.pose[
				:, 
				0:self.pos_dim
			]
			+ 
			self.ref_world_pose[
				:, 
				0:self.pos_dim
			]
		)

	def get_init_world_pose(self):
		return self.init_world_pose

	def get_ref_world_pose(self):
		return self.ref_world_pose

	def get_ori(self):
		return self.pose[:, self.pos_dim:]

	def get_goal(self):
		return self.goal

	def get_goal_dir(self):
		return (
			self.goal_world_pos
			- self.ref_world_pose[:, 0:self.pos_dim]
			- self.pose[:, 0:self.pos_dim]
		)
		
	def get_goal_pos(self):
		return (
			self.goal_world_pos
			- self.ref_world_pose[:,0:self.pos_dim]
		)

	def get_goal_world_pos(self):
		return self.goal_world_pos

	def copy(self,state):
		self.event = state.event
		self.vel_dim = state.vel_dim
		self.pos_dim = state.pos_dim
		self.goal_dim = state.goal_dim
		self.obs_dim = state.obs_dim
		self.obs_itr = state.obs_itr
		self.obs_seq_len = state.obs_seq_len
		self.obs_max_range = state.obs_max_range
		self.collision_thresh = state.collision_thresh
		self.goal_range = state.goal_range
		self.init_world_pose = state.init_world_pose.copy()
		self.ref_world_pose = state.ref_world_pose.copy()
		self.pose = state.pose.copy()
		self.goal = state.goal.copy()
		self.goal_world_pos = state.goal_world_pos.copy()
		self.vel = state.vel.copy()
		self.obs = state.obs.copy()
		self.time = state.time.copy()
		
	def clone(self):
		c = State(
			world_pose=self.pose,
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_max_range=self.obs_max_range,
			collision_thresh=self.collision_thresh
		)
		c.copy(self)
		return c


class ManagerReplayBuffer():
	def __init__(
		self,
		max_size,
		batch_size,
		final_goal,
		temp_abs,
		skip_steps,
	):
		self.iter = 0
		self.cur_size = 0
		self.max_size = max_size
		self.batch_size = batch_size
		self.final_goal = final_goal
		self.temp_abs = temp_abs
		self.skip_steps = skip_steps
		self.traj_len = int(
			self.temp_abs/(1+self.skip_steps) 
			+ min(self.temp_abs%(1+self.skip_steps), 1)
		)  # length of sampled temperal trajectory
		self.curtraj_buf = np.zeros(
			shape=[
				self.traj_len,
				self.max_size,
				2
			],
			dtype=float
		)
		self.nexttraj_buf = np.zeros(
			shape=[
				self.traj_len,
				self.max_size,
				2
			],
			dtype=float
		)
		self.curpose_buf = np.zeros(
			shape=[
				self.max_size, 
				6
			], 
			dtype=float
		)
		self.nextpose_buf = np.zeros(
			shape=[
				self.max_size, 
				6
			], 
			dtype=float
		)
		self.curobs_buf = np.zeros(
			shape=[
				4,
				self.max_size,
				180
			],
			dtype=float
		)
		self.nextobs_buf = np.zeros(
			shape=[
				4,
				self.max_size,
				180
			],
			dtype=float
		)
		self.goal_buf = np.zeros(
			shape=[
				self.max_size, 
				2
			],
			dtype=float
		)
		self.reward_buf = np.zeros(
			shape=[
				self.max_size,
				1
			],
			dtype=float
		)
		self.terminal_buf = np.zeros(
			shape=[
				self.max_size,
				1
			],
			dtype=float
		)
		self.curtraj_batch = np.zeros(
			shape=[
				self.traj_len,
				self.batch_size,
				2
			],
			dtype=float
		)
		self.nexttraj_batch = np.zeros(
			shape=[
				self.traj_len,
				self.batch_size,
				2
			],
			dtype=float
		)
		self.curpose_batch = np.zeros(
			shape=[
				self.batch_size, 
				6
			],
			dtype=float
		)
		self.nextpose_batch = np.zeros(
			shape=[
				self.batch_size, 
				6
			],
			dtype=float
		)
		self.curobs_batch = np.zeros(
			shape=[
				4,
				self.batch_size,
				180
			]
		)
		self.nextobs_batch = np.zeros(
			shape=[
				4,
				self.batch_size,
				180
			]
		)
		self.goal_batch = np.zeros(
			shape=[
				self.batch_size, 
				2
			],
			dtype=float
		)
		self.reward_batch = np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.terminal_batch = np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.lock = threading.Lock()

	def size(self):
		return self.cur_size
		
	def add(
		self,
		cur_traj,
		reward,
		next_traj,
		terminal,
	):
		try:
			self.lock.acquire()
			self.curtraj_buf[:,self.iter:self.iter+1,:] = np.concatenate(
				[
					[
						cur_traj[i].get_goal_world_pos()
						- cur_traj[i].get_world_pos()
					]
					for i in xrange(
						0, 
						len(cur_traj), 
						1+self.skip_steps
					)
				],
				0
			)
			self.nexttraj_buf[:,self.iter:self.iter+1,:] = np.concatenate(
				[
					[
						next_traj[i].get_goal_world_pos()
						- next_traj[i].get_world_pos()
					]
					for i in xrange(
						0, 
						len(next_traj), 
						1+self.skip_steps
					)
				],
				0
			)
			self.curpose_buf[self.iter:self.iter+1,:] = cur_traj[-1].get_world_pose()
			self.nextpose_buf[self.iter:self.iter+1,:] = next_traj[-1].get_world_pose()
			self.curobs_buf[:,self.iter:self.iter+1,:] = cur_traj[-1].get_obs()
			self.nextobs_buf[:,self.iter:self.iter+1,:] = next_traj[-1].get_obs()
			self.goal_buf[self.iter:self.iter+1,:] = next_traj[0].get_goal()
			self.reward_buf[self.iter,0] = reward
			self.terminal_buf[self.iter,0] = terminal
			self.iter = (self.iter+1)%self.max_size
			self.cur_size = min(self.max_size, self.cur_size+1)
		finally:
			self.lock.release()

	def sample(self):
		try:
			self.lock.acquire()
			if self.cur_size>100:
				for i in xrange(self.batch_size):
					r = rand.randint(0,self.cur_size-1)
					self.curtraj_batch[:,i,:] = self.curtraj_buf[:,r,:]
					self.nexttraj_batch[:,i,:] = self.nexttraj_buf[:,r,:]
					self.curpose_batch[i,:] = self.curpose_buf[r,:]
					self.nextpose_batch[i,:] = self.nextpose_buf[r,:]
					self.curobs_batch[:,i,:] = self.curobs_buf[:,r,:]
					self.nextobs_batch[:,i,:] = self.nextobs_buf[:,r,:]
					self.goal_batch[i,:] = self.goal_buf[r,:]
					self.reward_batch[i,:] = self.reward_buf[r,:]
					self.terminal_batch[i,:] = self.terminal_buf[r,:]
				return {
						'curtraj': self.curtraj_batch,
						'nexttraj': self.nexttraj_batch,
						'curpose': self.curpose_batch,
						'nextpose': self.nextpose_batch,
						'curobs': self.curobs_batch,
						'nextobs': self.nextobs_batch,
						'reward': self.reward_batch,
						'goal': self.goal_batch,
						'terminal': self.terminal_batch,
						}
			else:
				return {}
		finally:
			self.lock.release()


class WorkerReplayBuffer():
	def __init__(
		self,
		max_size,
		batch_size,
		obs_dim,
		obs_seq_len,
	):
		self.buf = deque([])
		self.iter = 0
		self.lock = threading.Lock()
		self.max_size = max_size
		self.obs_seq_len = obs_seq_len
		self.obs_dim = obs_dim
		self.vel_dim = 3
		self.pose_dim = 6
		self.goal_dim = 2
		self.act_dim = 2
		self.batch_size = batch_size
		self.curtime_batch = np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.curobs_batch = np.zeros(
			shape=[
				self.obs_seq_len,
				self.batch_size,
				self.obs_dim
			],
			dtype=float
		)
		self.curpose_batch = np.zeros(
			shape=[
				self.batch_size,
				self.pose_dim
			],
			dtype=float
		)
		self.curgoal_batch = np.zeros(
			shape=[
				self.batch_size,
				self.goal_dim
			],
		)
		self.nexttime_batch = np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.nextobs_batch = np.zeros(
			shape=[
				self.obs_seq_len,
				self.batch_size,
				self.obs_dim
			],
			dtype=float
		)
		self.nextpose_batch = np.zeros(
			shape=[
				self.batch_size,
				self.pose_dim
			],
			dtype=float
		)
		self.nextgoal_batch = np.zeros(
			shape=[
				self.batch_size,
				self.goal_dim
			],
			dtype=float
		)
		self.action_batch = np.zeros(
			shape=[
				self.batch_size,
				self.act_dim
			],
			dtype=float
		)
		self.reward_batch = np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.terminal_batch = np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)

	def size(self):
		return len(self.buf)

	def clear(self):
		self.buf = []
		
	def add(
		self,
		cur_state,
		action,
		reward,
		next_state,
		terminal
	):
		try:
			self.lock.acquire()
			sars = {
				'cur_state':cur_state.clone(),
				'action':action.copy(),
				'reward':reward, 
				'next_state':next_state.clone(),
				'terminal':terminal,
			}
			if len(self.buf)==self.max_size:
				self.buf.popleft()	
			self.buf.append(sars)
		finally:
			self.lock.release()

	def sample(self):
		try:
			self.lock.acquire()
			if len(self.buf) > 500:
				for i in range(self.batch_size):
					r = rand.randint(0,len(self.buf)-1)
					sars=self.buf[r]
					self.curtime_batch[i,0] = sars['cur_state'].get_time()
					self.curobs_batch[:,i:i+1,:] = sars['cur_state'].get_obs()
					self.curpose_batch[i:i+1,:] = sars['cur_state'].get_world_pose()
					self.curgoal_batch[i:i+1,:] = sars['cur_state'].get_goal_dir()
					self.nexttime_batch[i,0] = sars['next_state'].get_time()
					self.nextobs_batch[:,i:i+1,:] = sars['next_state'].get_obs()
					self.nextpose_batch[i:i+1,:] = sars['next_state'].get_world_pose()
					self.nextgoal_batch[i:i+1,:] = sars['next_state'].get_goal_dir()
					self.action_batch[i:i+1,:] = sars['action']
					self.reward_batch[i,0] = sars['reward']
					self.terminal_batch[i,0] = sars['terminal']
				return {
						'curtime': self.curtime_batch,
						'curobs': self.curobs_batch,
						'curgoal': self.curgoal_batch,
						'curpose': self.curpose_batch,
						'nexttime': self.nexttime_batch,
						'nextobs': self.nextobs_batch,
						'nextgoal': self.nextgoal_batch,
						'nextpose': self.nextpose_batch,
						'action': self.action_batch,
						'reward': self.reward_batch,
						'terminal': self.terminal_batch,
						}
			else:
				return {}
		finally:
			self.lock.release()

	
