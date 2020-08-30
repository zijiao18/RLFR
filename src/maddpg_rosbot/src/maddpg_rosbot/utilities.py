#!/usr/bin/env python
import numpy as np
import random as rand
from collections import deque
import threading


class State():
    def __init__(
        self,
        goal,
        pose,
        index,
        pos_dim,
        vel_dim,
        obs_dim,
        obs_seqlen,
        obs_rmax,
        coll_dist
    ):
        self.laser_range_max = obs_rmax
        self.coll_dist = coll_dist
        self.vel_dim = vel_dim
        self.dir_dim = pos_dim
        self.obs_dim = obs_dim
        self.event = 0
        self.pose = pose
        self.goal = goal
        self.ind = np.array(
            [index],
            dtype=float
        ).reshape([1,1,1])
        self.vel = np.zeros(
            shape=[1,self.vel_dim],
            dtype=float
        )
        self.dir=np.zeros(
            shape=[1,self.dir_dim],
            dtype=float
        )
        self.dir[0,:] = ((goal[0,:]-pose[0,0:self.dir_dim])
                        / np.linalg.norm(
                            goal[0,:]-pose[0,0:self.dir_dim]
                          )
                        )
        self.obs_itr = 0
        self.obs_seq_len = obs_seqlen
        self.obs = np.zeros(
            shape=[self.obs_seq_len,self.obs_dim],
            dtype=float
        )
        self.min_nr = 1 # normlaized minimum lidar range at most recent time step

    def update_obsseq_event(self,normalized_lidar_ranges):
        self.min_nr = 1
        for r in normalized_lidar_ranges:
            if r<self.min_nr:
                self.min_nr = r
        if self.min_nr<=(self.coll_dist/self.laser_range_max):
            self.event = 1
        if self.obs_itr==self.obs_seq_len:      
            for i in xrange(self.obs_seq_len-1):
                self.obs[i,:] = self.obs[i+1,:]
            self.obs[self.obs_itr-1,:] = normalized_lidar_ranges
        else:
            self.obs[self.obs_itr,:] = normalized_lidar_ranges
            self.obs_itr += 1

    def update_vel_dir_pose(
        self,
        velocity,
        pose,goal
    ):
        self.pose[0,:] = pose
        self.vel[0][0] = velocity[0][0] #x
        self.vel[0][1] = velocity[0][1] #y
        self.vel[0][2] = velocity[0][5] #yaw
        self.dir[0,:] = ((goal[0,:]-pose[0,0:self.dir_dim])
                        / np.linalg.norm(
                            goal-pose[0,0:self.dir_dim]
                          )
                        )

    def copy(self, state):
        self.event = state.event
        self.goal = state.goal.copy()
        self.pose = state.pose.copy()
        self.ind = state.ind.copy()
        self.vel = state.vel.copy()
        self.dir = state.dir.copy()
        self.obs = state.obs.copy()
        self.obs_itr = state.obs_itr
        self.obs_dim = state.obs_dim
        self.vel_dim = state.vel_dim
        self.dir_dim = state.dir_dim
        self.obs_seq_len = state.obs_seq_len
        self.laser_range_max = state.laser_range_max
        self.coll_dist = state.coll_dist

    def clone(self):
        c = State(
            self.goal,
            self.pose,
            self.ind,
            self.dir_dim,
            self.vel_dim,
            self.obs_dim,
            self.obs_seq_len,
            self.laser_range_max,
            self.coll_dist
        )
        c.copy(self)
        return c

    def reset(self, goal, pose):
        self.event = 0 
        self.goal = goal
        self.pose = pose
        self.vel = np.zeros(shape=[1,self.vel_dim],dtype=float)
        self.dir = np.zeros(shape=[1,self.dir_dim],dtype=float)
        self.dir[0,:] = ((goal[0,:]-pose[0,0:self.pos_dim])
                        / np.linalg.norm(
                            goal[0,:]-pose[0,0:self.pos_dim]
                          )
                        )
        self.obs_itr = 0
        self.obs = np.zeros(
            shape=[self.obs_seq_len, self.obs_dim],
            dtype=float
        )

    def obs_in(self):
        return self.obs.reshape(
            self.obs_seq_len,
            1,
            self.obs_dim
        )

    def vel_in(self):
        return self.vel.reshape(
            (1,1,self.vel_dim)
        )

    def dir_in(self):
        return self.dir.reshape(
            (1,1,self.dir_dim)
        )

    def ind_in(self):
        return self.ind

    def get_collision_distance(self):
        return self.min_nr*self.laser_range_max

#the buffer store SARS transactions
class ReplayBuffer():
    def __init__(
        self,
        max_size,
        pos_dim,
        vel_dim,
        obs_dim,
        act_dim,
        obs_seqlen,
        batch_size,
        n_worker
    ):
        self.iter = 0
        self.lock = threading.Lock()
        self.traj_head = deque([0])
        self.buf = []
        self.max_size = max_size
        self.obs_seq_len = obs_seqlen
        self.obs_dim = obs_dim
        self.vel_dim = vel_dim
        self.dir_dim = pos_dim
        self.act_dim = act_dim
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.cur_obs_batch = [
            np.zeros(
                shape=[
                    self.obs_seq_len,
                    batch_size,
                    self.obs_dim
                ],
                dtype=float
            ) 
            for _ in xrange(self.n_worker)
        ]
        self.cur_vel_batch = [
            np.zeros(
                shape=[
                    1,
                    batch_size,
                    self.vel_dim
                ],
                dtype=float
            ) 
            for _ in xrange(self.n_worker)
        ]
        self.cur_dir_batch = [
            np.zeros(
                shape=[
                    1,
                    batch_size,
                    self.dir_dim
                ],
                dtype=float
            ) 
            for _ in xrange(self.n_worker)
        ]
        self.next_obs_batch = [
            np.zeros(
                shape=[
                    self.obs_seq_len,
                    batch_size,
                    self.obs_dim
                ],
                dtype=float
            )
            for _ in xrange(self.n_worker)
        ]
        self.next_vel_batch = [
            np.zeros(
                shape=[
                    1,
                    batch_size,
                    self.vel_dim
                ],
                dtype=float
            )
            for _ in xrange(self.n_worker)
        ]
        self.next_dir_batch = [
            np.zeros(
                shape=[
                    1,
                    batch_size,
                    self.dir_dim
                ],
                dtype=float
            )
            for _ in xrange(self.n_worker)
        ]
        self.act_batch = [
            np.zeros(
                shape=[
                    1,
                    batch_size,
                    self.act_dim
                ],
                dtype=float
            ) 
            for _ in xrange(self.n_worker)
        ]
        self.rwd_batch = [
            np.zeros(
                shape=[
                    1,
                    batch_size,
                    1
                ],
                dtype=float
            ) 
            for i in xrange(self.n_worker)
        ]
        self.index_batch = [
            np.array(
                [i for _ in xrange(batch_size)],
                dtype=float
            ).reshape((1,batch_size,1)) 
            for i in xrange(self.n_worker)
        ]
        self.terminal_batch = np.zeros(
            shape=[1,batch_size,1],
            dtype=int
        )

    def size(self):
        return len(self.buf)

    def clear(self):
        self.buf=[]
        
    def add(
        self,
        joint_cstate,
        joint_action,
        joint_reward,
        joint_nstate,
        terminal,
        eoe
    ):
        try:
            self.lock.acquire()
            jcs=[cs.clone() for cs in joint_cstate]
            jns=[ns.clone() for ns in joint_nstate]
            ja=[act.copy() for act in joint_action]
            jrwd=[r for r in joint_reward]
            sars = {
                    'joint_cstate':jcs,
                    'joint_action':ja,
                    'joint_reward':jrwd, 
                    'joint_nstate':jns,
                    'terminal':terminal
            }
            if len(self.buf)<self.max_size:
                self.buf.append(sars)
            else:
                self.buf[self.iter]=sars
            if (
                    len(self.traj_head)>0 
                    and self.traj_head[0]==self.iter
            ):
                self.traj_head.popleft() #remove the overided trajectory
            self.iter = (self.iter+1)%self.max_size
            if eoe:  # end of an episode
                self.traj_head.append(self.iter) # update iter to the next head
        finally:
            self.lock.release()

    def sample(self):
        try:
            self.lock.acquire()
            for i in range(self.batch_size):
                r=rand.randint(0,len(self.buf)-1)
                sars=self.buf[r]
                for j in xrange(self.n_worker):
                    self.cur_obs_batch[j][:,i,:] = sars['joint_cstate'][j].obs[:,:]
                    self.cur_vel_batch[j][0,i,:] = sars['joint_cstate'][j].vel[0,:]
                    self.cur_dir_batch[j][0,i,:] = sars['joint_cstate'][j].dir[0,:]
                    self.next_obs_batch[j][:,i,:] = sars['joint_nstate'][j].obs[:,:]
                    self.next_vel_batch[j][0,i,:] = sars['joint_nstate'][j].vel[0,:]
                    self.next_dir_batch[j][0,i,:] = sars['joint_nstate'][j].dir[0,:]    
                    self.act_batch[j][0,i,:] = sars['joint_action'][j][0,:]
                    self.rwd_batch[j][0][i][0] = sars['joint_reward'][j]        
                self.terminal_batch[0][i][0] = sars['terminal']
            return {
                    'joint_cobs':self.cur_obs_batch, 
                    'joint_cvel':self.cur_vel_batch,
                    'joint_cdir':self.cur_dir_batch,
                    'joint_nobs':self.next_obs_batch,
                    'joint_nvel':self.next_vel_batch,
                    'joint_ndir':self.next_dir_batch,
                    'joint_action':self.act_batch,
                    'joint_reward':self.rwd_batch, 
                    'terminal':self.terminal_batch,
                    'index':self.index_batch
            }
        finally:
            self.lock.release()


    
