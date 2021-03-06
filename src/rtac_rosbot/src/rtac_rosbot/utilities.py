#!usr/bin/env python
import numpy as np
import random as rand


class Feedback():
    def __init__(self):
        self.sender  = ""
        self.event = 0;  # 0: no collision, 1: collided 
        self.velocity = np.zeros([1,6])  # lx,ly,lz,rx,ry,rz
        self.lidar_ranges = np.zeros([1,360])
        self.pose = np.zeros([1,6])  # [[x,y,z,r,p,y]]

    def load(self, data):
        '''
        data from simulator:
        [
            event,
            linear_velocity,
            angular_velocity,
            position_xyzrpy,
            lidar_observation
        ]
        '''
        self.event = data[0]
        self.velocity[0, :] = data[1:7]
        self.pose[0, :] = data[7:13]
        self.lidar_ranges[0, :] = data[13:len(data)]

    def set_sender(self,name):
        self.sender=name;


class State():
    def __init__(
        self,
        goal,
        pose,
        obs_dim,
        obs_seq_len
    ):
        self.event = 0
        self.pose = pose
        self.goal = goal
        self.vel_dim = 6
        self.vel = np.zeros(shape=[1,self.vel_dim],dtype=float)
        self.dir_dim = 3
        self.dir = np.zeros(shape=[1,self.dir_dim],dtype=float)
        self.dir[0,:] = (goal[0,:]-pose[0,0:3])/np.linalg.norm(goal[0,:]-pose[0,0:3])
        self.obs_dim = obs_dim
        self.obs_itr = 0
        self.obs_seq_len = obs_seq_len
        self.obs = np.zeros(
            shape=[obs_seq_len, obs_dim],
            dtype=float
        )
        self.laser_range_max = 2.0
        self.coll_dist = 0.22

    def update(self,feedback,goal):
        self.event = feedback.event
        self.pose[0,:] = feedback.pose
        self.vel[0,:] = feedback.velocity
        self.dir[0,:] = ((goal[0,:]-feedback.pose[0,0:3])
                        / np.linalg.norm(
                            goal-feedback.pose[0,0:3]
                          )
                        )
        if self.obs_itr==self.obs_seq_len:      
            for i in xrange(self.obs_seq_len-1):
                self.obs[i, :] = self.obs[i+1, :]
            self.obs[self.obs_itr-1, :] = feedback.lidar_ranges
        else:
            self.obs[self.obs_itr, :] = feedback.lidar_ranges
            self.obs_itr += 1

    def update_obsseq_event(self,lidar_ranges):
        min_r = 1.0
        for r in lidar_ranges:
            if r<min_r:
                min_r = r
        if min_r <= (self.coll_dist/self.laser_range_max):
            self.event = 1
        if self.obs_itr==self.obs_seq_len:      
            for i in xrange(self.obs_seq_len-1):
                self.obs[i,:] = self.obs[i+1,:]
            self.obs[self.obs_itr-1,:] = lidar_ranges
        else:
            self.obs[self.obs_itr,:] = lidar_ranges
            self.obs_itr += 1

    def update_vel_dir_pose(self,velocity,pose,goal):
        self.pose[0,:] = pose
        self.vel[0,:] = velocity
        self.dir[0,:] = ((goal[0,:]-pose[0,0:3])
                        / np.linalg.norm(
                            goal-pose[0,0:3]
                            )
                        )

    def copy(self,state):
        self.event = state.event
        self.pose = state.pose.copy()
        self.vel = state.vel.copy()
        self.dir = state.dir.copy()
        self.obs = state.obs.copy()
        self.obs_itr = state.obs_itr
        self.obs_dim = state.obs_dim
        self.obs_seq_len = state.obs_seq_len

    def clone(self):
        c = State(
            self.goal,
            self.pose,
            self.obs_dim,
            self.obs_seq_len
        )
        c.copy(self)
        return c

    def reset(self,goal,pose):
        self.event = 0;
        self.goal = goal
        self.pose = pose
        self.vel = np.zeros(
            shape=[1,self.vel_dim],
            dtype=float
        )
        self.dir = np.zeros(
            shape=[1,self.dir_dim],
            dtype=float
        )
        self.dir[0,:] = ((goal[0,:]-pose[0,0:3])
                        / np.linalg.norm(
                            goal[0,:]-pose[0,0:3]
                            )
                        )
        self.obs_itr = 0
        self.obs = np.zeros(
            shape=[
                self.obs_seq_len,
                self.obs_dim
            ],
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
            (1,1,6)
        )

    def dir_in(self):
        return self.dir.reshape(
            (1,1,3)
        )


class ReplayBuffer():
    def __init__(self, max_size):
        self.buf = []
        self.max_size = max_size
        self.iter = 0
        self.obs_seq_len = 16
        self.obs_dim = 360
        self.vel_dim = 6
        self.dir_dim = 3
        self.act_dim = 1

    def add(self,sars):
        if len(self.buf)==0:
            self.obs_seq_len = sars['cur_state'].obs_seq_len
            self.obs_dim = sars['cur_state'].obs_dim
            self.vel_dim = sars['cur_state'].vel_dim
            self.dir_dim = sars['cur_state'].dir_dim
            self.act_dim = len(sars['action'][0])
        if len(self.buf)<self.max_size:
            self.buf.append(sars)
        else:
            self.buf[self.iter] = sars
        self.iter = (self.iter+1)%self.max_size

    def sample(self,batch_size):
        cur_obs_batch = np.zeros(
            shape=[
                self.obs_seq_len,
                batch_size,
                self.obs_dim
            ],
            dtype=float
        )
        cur_vel_batch = np.zeros(
            shape=[
                1,
                batch_size,
                self.vel_dim
            ],
            dtype=float
        )
        cur_dir_batch = np.zeros(
            shape=[
                1,
                batch_size,
                self.dir_dim
            ],
            dtype=float
        )
        act_batch = np.zeros(
            shape=[
                1,
                batch_size,
                self.act_dim
            ],
            dtype=float)
        rwd_batch = np.zeros(
            shape=[
                1,
                batch_size,
                1
            ],
            dtype=float
        )
        terminal_batch = np.zeros(
            shape=[
                1,
                batch_size,
                1
            ],
            dtype=float
        )
        next_obs_batch = np.zeros(
            shape=[
                self.obs_seq_len,
                batch_size,
                self.obs_dim
            ],
            dtype=float
        )
        next_vel_batch = np.zeros(
            shape=[
                1,
                batch_size,
                self.vel_dim
            ],
            dtype=float
        )
        next_dir_batch = np.zeros(
            shape=[
                1,
                batch_size,
                self.dir_dim
            ],
            dtype=float
        )
        for i in range(batch_size):
            r = rand.randint(
                0,
                len(self.buf)-1
            )
            sars = self.buf[r]
            cur_obs_batch[:,i,:] = sars['cur_state'].obs[:,:]
            cur_vel_batch[0,i,:] = sars['cur_state'].vel[0,:]
            cur_dir_batch[0,i,:] = sars['cur_state'].dir[0,:]
            act_batch[0][i][0] = sars['action'][0][0]
            rwd_batch[0][i][0] = sars['reward']
            terminal_batch[0][i][0] = sars['terminal']
            next_obs_batch[:,i,:] = sars['next_state'].obs[:,:]
            next_vel_batch[0,i,:] = sars['next_state'].vel[0,:]
            next_dir_batch[0,i,:] = sars['next_state'].dir[0,:]
        return {
                    'cur_obs':cur_obs_batch, 
                    'cur_vel':cur_vel_batch,
                    'cur_dir':cur_dir_batch,
                    'action':act_batch,
                    'reward':rwd_batch, 
                    'terminal':terminal_batch,
                    'next_obs':next_obs_batch,
                    'next_vel':next_vel_batch,
                    'next_dir':next_dir_batch
        }

    def size(self):
        return len(self.buf)

    def clear(self):
        self.buf=[]