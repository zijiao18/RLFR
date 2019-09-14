#!usr/bin/env python
import numpy as np
import random as rand; rand.seed(0)
from collections import deque
import threading

class State():
	def __init__(
		self,
		pose,
		index,
		obs_dim,
		obs_seq_len,
		obs_max_range,
		collision_thresh
	):
		self.event=0
		self.vel_dim=3
		self.pos_dim=2
		self.goal_dim=2
		self.obs_dim=obs_dim
		self.obs_itr=0
		self.obs_seq_len=obs_seq_len
		self.obs_max_range=obs_max_range
		self.collision_thresh=collision_thresh
		self.init_pose=np.array(
			pose,
			dtype=float
		)
		self.pose=np.array(
			pose,
			dtype=float
		)
		self.goal_pos=np.zeros(
			shape=[1, self.goal_dim],
			dtype=float
		)
		self.goal_dir=np.zeros(
			shape=[1, self.goal_dim],
			dtype=float
		)
		self.index=np.array(
			[[index]],
			dtype=float
		)
		self.vel=np.zeros(
			shape=[1, self.vel_dim],
			dtype=float
		)
		self.obs=np.zeros(
			shape=[
				self.obs_seq_len, 
				1, 
				self.obs_dim
			],
			dtype=float
		)

	def reached_goal(self):
		dist = np.linalg.norm(
			(self.pose[0, 0:self.pos_dim]
			- self.goal_pos[0, :])
		)
		return dist < 0.1

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

	def set_pose(self, pose):
		self.pose[0,:]=pose

	def set_goal(self, goal):
		self.goal_pos[0, :] = self.pose[0,0:2]+goal
		self.goal_dir[0, :] = goal

	def set_vel(self, vel):
		self.vel[0][0]=vel[0] #x
		self.vel[0][1]=vel[1] #y
		self.vel[0][2]=vel[5] #yaw

	def get_obs(self):
		return self.obs

	def get_vel(self):
		return self.vel

	def get_index(self):
		return self.index

	def get_pose(self):
		return self.pose

	def get_pos(self):
		return self.pose[:, 0:self.pos_dim]

	def get_ori(self):
		return self.pose[:, self.pos_dim:]

	def get_goal_dir(self):
		self.goal_dir[:, :] = (self.goal_pos
							- self.pose[:,0:self.pos_dim])
		return self.goal_dir

	def get_goal_pos(self):
		return self.goal_pos

	def copy(self,state):
		self.event=state.event
		self.vel_dim=state.vel_dim
		self.pos_dim=state.pos_dim
		self.goal_dim=state.goal_dim
		self.obs_dim=state.obs_dim
		self.obs_itr=state.obs_itr
		self.obs_seq_len=state.obs_seq_len
		self.obs_max_range=state.obs_max_range
		self.collision_thresh=state.collision_thresh
		self.init_pose=state.init_pose.copy()
		self.pose=state.pose.copy()
		self.goal_pos=state.goal_pos.copy()
		self.goal_dir=state.goal_dir.copy()
		self.index=state.index.copy()
		self.vel=state.vel.copy()
		self.obs=state.obs.copy()
		
	def clone(self):
		c=State(
			pose=self.pose,
			index=self.index,
			obs_dim=self.obs_dim,
			obs_seq_len=self.obs_seq_len,
			obs_max_range=self.obs_max_range,
			collision_thresh=self.collision_thresh
		)
		c.copy(self)
		return c

	def reset(self):
		self.event=0;
		self.obs_itr=0
		self.obs.fill(0.0)
		self.pose[:,:]=self.init_pose
		self.goal_pos.fill(0.0)
		self.goal_dir.fill(0.0)
		self.vel.fill(0.0)


class ManagerReplayBuffer():
	def __init__(
		self,
		max_size,
		batch_size,
		n_agent
	):
		self.iter=0
		self.size=0
		self.max_size=max_size
		self.batch_size=batch_size
		self.n_agent=n_agent
		self.joint_curpos_buf=np.zeros(
			shape=[
				self.max_size, 
				self.n_agent*2
			], 
			dtype=float
		)
		
		self.joint_nextpos_buf=np.zeros(
			shape=[
				self.max_size, 
				self.n_agent*2
			], 
			dtype=float
		)
		self.curpos_buf=np.zeros(
			shape=[
				self.max_size,
				2
			],
			dtype=float
		)
		self.nextpos_buf=np.zeros(
			shape=[
				self.max_size,
				2
			],
			dtype=float
		)
		self.goal_buf=np.zeros(
			shape=[
				self.max_size, 
				2
			],
			dtype=float
		)
		self.reward_buf=np.zeros(
			shape=[
				self.max_size,
				1
			],
			dtype=float
		)
		self.index_buf=np.zeros(
			shape=[
				self.max_size,
				1
			],
			dtype=float
		)
		self.terminal_buf=np.zeros(
			shape=[
				self.max_size,
				1
			],
			dtype=float
		)
		self.joint_curpos_batch=np.zeros(
			shape=[
				self.batch_size, 
				self.n_agent*2
			],
			dtype=float
		)
		self.joint_nextpos_batch=np.zeros(
			shape=[
				self.batch_size, 
				self.n_agent*2
			],
			dtype=float
		)
		self.curpos_batch=np.zeros(
			shape=[
				self.batch_size,
				2
			],
			dtype=float
		)
		self.nextpos_batch=np.zeros(
			shape=[
				self.batch_size,
				2
			],
			dtype=float
		)
		self.goal_batch=np.zeros(
			shape=[
				self.batch_size, 
				2
			],
			dtype=float
		)
		self.reward_batch=np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.index_batch=np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.terminal_batch=np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.lock=threading.Lock()

	def size(self):
		return self.size
		
	def add(
		self,
		index,
		cur_states,
		reward,
		next_states,
		terminal,
	):
		try:
			self.lock.acquire()
			self.joint_curpos_buf[self.iter:self.iter+1,:]=np.concatenate(
				[	s.get_pos() 
					for s in cur_states
				],
				1
			)
			self.joint_nextpos_buf[self.iter:self.iter+1,:]=np.concatenate(
				[
					s.get_pos() 
					for s in next_states
				],
				1
			)
			self.curpos_buf[self.iter:self.iter+1,:]=cur_states[index].get_pos()
			self.nextpos_buf[self.iter:self.iter+1,:]=next_states[index].get_pos()
			self.goal_buf[self.iter:self.iter+1,:]=cur_states[index].get_goal_dir()
			self.index_buf[self.iter,0]=index
			self.reward_buf[self.iter,0]=reward
			self.terminal_buf[self.iter,0]=terminal
			self.iter=(self.iter+1)%self.max_size
			self.size=min(self.max_size, self.size+1)
		finally:
			self.lock.release()

	def sample(self):
		try:
			self.lock.acquire()
			if self.size>0:
				for i in xrange(self.batch_size):
					r=rand.randint(0,self.size-1)
					self.joint_curpos_batch[i,:]=self.joint_curpos_buf[r,:]
					self.joint_nextpos_batch[i,:]=self.joint_nextpos_buf[r,:]
					self.curpos_batch[i,:]=self.curpos_buf[r,:]
					self.nextpos_batch[i,:]=self.nextpos_buf[r,:]
					self.goal_batch[i,:]=self.goal_buf[r,:]
					self.reward_batch[i,:]=self.reward_buf[r,:]
					self.terminal_batch[i,:]=self.terminal_buf[r,:]
					self.index_batch[i,:]=self.index_buf[r,:]
				return {
							'joint_curpos': self.joint_curpos_batch,
							'joint_nextpos': self.joint_nextpos_batch,
							'curpos': self.curpos_batch,
							'nextpos': self.nextpos_batch,
							'reward': self.reward_batch,
							'goal': self.goal_batch,
							'terminal': self.terminal_batch,
							'index': self.index_batch
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
		self.buf=deque([])
		self.iter=0
		self.lock=threading.Lock()
		self.max_size=max_size
		self.obs_seq_len=obs_seq_len
		self.obs_dim=obs_dim
		self.vel_dim=3
		self.pose_dim=6
		self.goal_dim=2
		self.act_dim=2
		self.batch_size=batch_size
		self.curobs_batch=np.zeros(
			shape=[
				self.obs_seq_len,
				self.batch_size,
				self.obs_dim
			],
			dtype=float
		)
		self.curpose_batch=np.zeros(
			shape=[
				self.batch_size,
				self.pose_dim
			],
			dtype=float
		)
		self.curgoal_batch=np.zeros(
			shape=[
				self.batch_size,
				self.goal_dim
			],
		)
		self.nextobs_batch=np.zeros(
			shape=[
				self.obs_seq_len,
				self.batch_size,
				self.obs_dim
			],
			dtype=float
		)
		self.nextpose_batch=np.zeros(
			shape=[
				self.batch_size,
				self.pose_dim
			],
			dtype=float
		)
		self.nextgoal_batch=np.zeros(
			shape=[
				self.batch_size,
				self.goal_dim
			],
			dtype=float
		)
		self.action_batch=np.zeros(
			shape=[
				self.batch_size,
				self.act_dim
			],
			dtype=float
		)
		self.reward_batch=np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)
		self.terminal_batch=np.zeros(
			shape=[
				self.batch_size,
				1
			],
			dtype=float
		)

	def size(self):
		return len(self.buf)

	def clear(self):
		self.buf=[]
		
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
			sars={
				'cur_state':cur_state.clone(),
				'action':action.copy(),
				'reward':reward, 
				'next_state':next_state.clone(),
				'terminal':terminal
			}
			if(len(self.buf)==self.max_size):
				self.buf.popleft()	
			self.buf.append(sars)
		finally:
			self.lock.release()

	def sample(self):
		try:
			self.lock.acquire()
			if self.buf:
				for i in range(self.batch_size):
					r=rand.randint(0,len(self.buf)-1)
					sars=self.buf[r]
					self.curobs_batch[:,i:i+1,:]=sars['cur_state'].get_obs()
					self.curgoal_batch[i:i+1,:]=sars['cur_state'].get_goal_dir()
					self.curpose_batch[i:i+1,:]=sars['cur_state'].get_pose()
					self.nextobs_batch[:,i:i+1,:]=sars['next_state'].get_obs()
					self.nextgoal_batch[i:i+1,:]=sars['next_state'].get_goal_dir()
					self.nextpose_batch[i:i+1,:]=sars['next_state'].get_pose()
					self.action_batch[i:i+1,:]=sars['action']
					self.reward_batch[i,0]=sars['reward']
					self.terminal_batch[i,0]=sars['terminal']
				return {
							'curobs': self.curobs_batch,
							'curgoal': self.curgoal_batch,
							'curpose': self.curpose_batch,
							'nextobs': self.nextobs_batch,
							'nextgoal': self.nextgoal_batch,
							'nextpose': self.nextpose_batch,
							'action': self.action_batch,
							'reward': self.reward_batch,
							'terminal': self.terminal_batch
						}
			else:
				return {}
		finally:
			self.lock.release()

	
