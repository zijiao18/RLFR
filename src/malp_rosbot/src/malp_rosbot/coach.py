#!usr/bin/env python
import rospy
import numpy as np
import tensorflow as tf
import tf as rostf
import random as rand
import Queue
from threading import Thread
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Empty, Float32MultiArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rec_maddpg_net_ie import ActorNetwork, CriticNetwork
#from rec_maddpg_net_je import ActorNetwork, CriticNetwork
#from maddpg_net import ActorNetwork, CriticNetwork
from utilities import State
from utilities import ReplayBuffer
import time

class Coach():
	def __init__(self,sess,name,initpose,goals,
				act_dim,pos_dim,vel_dim,lidar_dim,
				lidar_seqlen,batch_size,
				actor_lstm_state_dim,critic_lstm_state_dim,
				actor_fc1_unit,actor_fc2_unit,actor_fc3_unit,
				critic_fc1_unit,critic_fc2_unit,critic_fc3_unit,
				actor_lr,critic_lr,actor_tau,critic_tau,
				master_actor,master_critic,
				rtoffset,device,tb_writer):
		self.sess=sess
		self.name=name
		self.rtoffset=rtoffset
		self.n_worker=3 #fixed, should not be changed
		self.device=device

		self.epoch=0
		self.episode=0
		self.eprwd=0
		self.step=0
		self.max_step=200
		
		self.max_lvel=1.0
		self.max_avel=3.14
		self.act_dim=act_dim
		self.pos_dim=pos_dim
		self.vel_dim=vel_dim
		self.lidar_dim=lidar_dim
		self.lidar_seqlen=lidar_seqlen
		self.lidar_rmax=2.0
		self.lidar_colldist=0.25
		
		self.batch_size=batch_size

		self.epslion=1.0
		self.epslion_t=0
		self.ou_noise_lv=[0.0,0.0,0.0]
		self.ou_noise_av=[0.0,0.0,0.0]
		
		self.joint_goal=goals
		self.joint_initpose=initpose
		
		self.joint_cstate=[State(goal=self.joint_goal[i].copy(),
								pose=self.joint_initpose[i].copy(),
								index=i,
								pos_dim=self.pos_dim,
								vel_dim=self.vel_dim,
								obs_dim=self.lidar_dim,
								obs_seqlen=self.lidar_seqlen,
								obs_rmax=self.lidar_rmax,
								coll_dist=self.lidar_colldist) 
							for i in xrange(self.n_worker)]
		
		self.joint_nstate=[State(goal=self.joint_goal[i].copy(),
								pose=self.joint_initpose[i].copy(),
								index=i,
								pos_dim=self.pos_dim,
								vel_dim=self.vel_dim,
								obs_dim=self.lidar_dim,
								obs_seqlen=self.lidar_seqlen,
								obs_rmax=self.lidar_rmax,
								coll_dist=self.lidar_colldist)
							for i in xrange(self.n_worker)]
		
		self.joint_action=[np.zeros([1,self.act_dim],dtype=float)
							for _ in xrange(self.n_worker)]

		self.joint_reward=[0.0 for _ in xrange(self.n_worker)]

		self.terminal=False

		self.collided=False

		#topics for all robots in an environment
		self.velcmd_pubs=[rospy.Publisher("/rosbot"+str(i+rtoffset)+"/cmd_vel", 
										Twist, 
										queue_size=1000)
						for i in xrange(self.n_worker)]
		
		self.scan_subs=[rospy.Subscriber("/rosbot"+str(i+rtoffset)+"/scan",
										LaserScan,
										self.lidar_receiver,
										i)
						for i in xrange(self.n_worker)]
		
		self.odom_subs=[rospy.Subscriber("/rosbot"+str(i+rtoffset)+"/odom",
										Odometry,
										self.odom_receiver,
										i)
						for i in xrange(self.n_worker)]

		self.reset_client=rospy.ServiceProxy("/gazebo/set_model_state",SetModelState) #controls all models

		
		self.load_lidar=[False for _ in xrange(self.n_worker)]
		self.load_odom=[False for _ in xrange(self.n_worker)]

		self.training_rate=rospy.Rate(30)
		self.interact_rate=rospy.Rate(10)
		self.stop_rate=rospy.Rate(10)
		self.replay_buffer=ReplayBuffer(max_size=400000,
										pos_dim=self.pos_dim,
										vel_dim=self.vel_dim,
										obs_dim=self.lidar_dim,
										act_dim=self.act_dim,
										obs_seqlen=self.lidar_seqlen,
										batch_size=self.batch_size,
										n_worker=self.n_worker)
		self.min_repbuf_size=10

		self.actor = ActorNetwork(sess=sess,
								name=self.name+"_actor",
								time_step=self.lidar_seqlen,
								obs_dim=self.lidar_dim,
								vel_dim=self.vel_dim,
								dir_dim=self.pos_dim,
								act_dim=self.act_dim,
								batch_size=self.batch_size,
								lstm_state_dim=actor_lstm_state_dim,
								n_fc1_unit=actor_fc1_unit,
								n_fc2_unit=actor_fc2_unit,
								n_fc3_unit=actor_fc3_unit,
								learning_rate=actor_lr,
								tau=actor_tau,
								device=device,
								master_network=master_actor)

		self.critic = CriticNetwork(sess=sess,
								name=self.name+"_critic",
								time_step=self.lidar_seqlen,
								obs_dim=self.lidar_dim,
								vel_dim=self.vel_dim,
								dir_dim=self.pos_dim,
								act_dim=self.act_dim,
								batch_size=self.batch_size,
								lstm_state_dim=critic_lstm_state_dim, 
								n_fc1_unit=critic_fc1_unit, 
								n_fc2_unit=critic_fc2_unit,
								n_fc3_unit=critic_fc3_unit,
								learning_rate=critic_lr, 
								tau=critic_tau,
								device=device,
								master_network=master_critic)
		self.master_actor=master_actor
		self.master_critic=master_critic

		self.tb_writer=tb_writer
		self.tb_eprwd_in=tf.placeholder(dtype=tf.float32,shape=())
		self.tb_eprwd=tf.summary.scalar(self.name+'_episodic_rewards',self.tb_eprwd_in)

		self.batch_joint_nact=[np.zeros([1,self.batch_size,2]) 
								for _ in xrange(self.n_worker)]
		self.batch_joint_cact=[np.zeros([1,self.batch_size,2]) 
								for _ in xrange(self.n_worker)]
		
		self.t_behavior=Thread(target=self.interact_with_environment)
		self.t_training=Thread(target=self.train)

		self.log = open('/media/zilong/Backup/RLFR/save/malp_rosbot/log/'+self.name,'w+')


	def train(self):
		wi=0
		while not rospy.is_shutdown():
			try:
				if self.replay_buffer.size()<self.min_repbuf_size:
					print("%s train_worker: replay_buffer size: %d/%d"%(self.name,
																		self.replay_buffer.size(),
						 												self.min_repbuf_size))
					self.training_rate.sleep()
					continue
			except Queue.Empty as e:
				print("train_worker: no transaction is received...")
				break
			#train both global networks
			batch=self.replay_buffer.sample() #blocking method
			#print("sampled batch",self.replay_buffer.size(),len(batch['joint_cobs']))
			#print("batch0","joint_cobs: ", batch['joint_cobs'][0][:,:,0:5],"joint_nobs: ",batch['joint_nobs'][0][:,:,0:5])
			#print("batch1","joint_cobs: ", batch['joint_cobs'][1][:,:,0:5],"joint_nobs: ",batch['joint_nobs'][1][:,:,0:5])
			#print("batch2","joint_cobs: ", batch['joint_cobs'][2][:,:,0:5],"joint_nobs: ",batch['joint_nobs'][2][:,:,0:5])
			#print("batch0","joint_cvel: ", batch['joint_cvel'][0],"joint_nvel: ",batch['joint_nvel'][0])
			#print("batch1","joint_cvel: ", batch['joint_cvel'][1],"joint_nvel: ",batch['joint_nvel'][1])
			#print("batch2","joint_cvel: ", batch['joint_cvel'][2],"joint_nvel: ",batch['joint_nvel'][2])
			#print("batch0","joint_cdir: ", batch['joint_cdir'][0],"joint_ndir: ",batch['joint_ndir'][0])
			#print("batch1","joint_cdir: ", batch['joint_cdir'][1],"joint_ndir: ",batch['joint_ndir'][1])
			#print("batch2","joint_cdir: ", batch['joint_cdir'][2],"joint_ndir: ",batch['joint_ndir'][2])
			#print("batch","reward:", batch['reward'])
			#print("batch","terminal: ",batch['terminal'])
			#print("batch0","joint_action: ", batch['joint_action'][0])
			#print("batch1","joint_action: ", batch['joint_action'][1])
			#print("batch2","joint_action: ", batch['joint_action'][2])
			#print("batch","index",batch['index'])
			#print("-------------------------------------------------")

			#predict the agent's next action based on its next observation and velocity (Q-learning)
			for i in xrange(self.n_worker):
				self.batch_joint_nact[i][0,:,:]=self.actor.predict_target(obs_batch=batch['joint_nobs'][i],
																		dir_batch=batch['joint_ndir'][i],
																		ind_batch=batch['index'][i]) #[batch,2]
			
			#the q-value of the next state-action batch shape=[batch,1]
			next_q=self.critic.predict_target(obs_batch=batch['joint_nobs'],
											dir_batch=batch['joint_ndir'],
											ind_batch=batch['index'][wi],
											act_batch=self.batch_joint_nact)
			
			#the target q-value for TD(0). Gamma=0.95 
			tar_q=batch['joint_reward'][wi]+0.99*(next_q*(1-batch['terminal']))
			#print("tar_q",tar_q.shape,tar_q[0])

			self.critic.copy_master_network()
			self.critic.update_master_network(obs_batch=batch['joint_cobs'],
											dir_batch=batch['joint_cdir'],
											ind_batch=batch['index'][wi],
											act_batch=batch['joint_action'],
											tar_q_batch=tar_q[0])

			self.actor.copy_master_network()
			for i in xrange(self.n_worker):
				self.batch_joint_cact[i][0,:,:]=self.actor.predict(obs_batch=batch['joint_cobs'][i],
																dir_batch=batch['joint_cdir'][i],
																ind_batch=batch['index'][i]) #[batch,2]
			self.critic.copy_master_network()
			#critic a_out. the act_grads is a IndexSlicesValue(), [[act_grads_array,dtype,shape]]
			act_grads=self.critic.action_gradients(obs_batch=batch['joint_cobs'],
												dir_batch=batch['joint_cdir'],
												ind_batch=batch['index'][wi],
												act_batch=self.batch_joint_cact,
												ind_scalar=wi)
			#print("act_grads", act_grads)
			#based on the action gradient, train the actor network
			self.actor.update_master_network(obs_batch=batch['joint_cobs'][wi],
											dir_batch=batch['joint_cdir'][wi],
											ind_batch=batch['index'][wi],
											act_grads=act_grads[0])			

			self.actor.update_target_network()
			self.critic.update_target_network()

			if self.name=="coach0" and self.epoch%10==0:
				#the q-value of the next state-action batch shape=[batch,1]
				next_q=self.critic.predict(obs_batch=batch['joint_nobs'],
										dir_batch=batch['joint_ndir'],
										ind_batch=batch['index'][wi],
										act_batch=self.batch_joint_nact)
				#the target q-value for TD(0). Gamma=0.95 
				tar_q=batch['joint_reward'][0]+0.99*(next_q*(1-batch['terminal']))
				#store loss and policy gradients to tensorboard
				tb_loss=self.critic.summary(obs_batch=batch['joint_cobs'],
											dir_batch=batch['joint_cdir'],
											ind_batch=batch['index'][wi],
											act_batch=batch['joint_action'],
											tar_q_batch=tar_q[0])
				tb_pg=self.actor.summary(obs_batch=batch['joint_cobs'][wi],
										dir_batch=batch['joint_cdir'][wi],
										ind_batch=batch['index'][wi],
										act_grads=act_grads[0])
				self.tb_writer.add_summary(tb_loss,self.epoch)
				for g in tb_pg:
					self.tb_writer.add_summary(g,self.epoch)

			self.epoch+=1
			wi=(wi+1)%self.n_worker
			self.training_rate.sleep()
			#print("train timestamp:",rospy.get_time())
			

	#controls all the robots in an environment
	def interact_with_environment(self):
		self.initMDP() #determine o_0,a_0
		while not rospy.is_shutdown():

			#observe the transited state
			self.observe_state() 

			#determine joint action in the transited state
			joint_next_action=[]
			for i in xrange(self.n_worker):
				next_action=self.actor.predict(self.joint_nstate[i].obs_in(),
										self.joint_nstate[i].dir_in(),
										self.joint_nstate[i].ind_in())
				self.exploration_noise(next_action,i)
				joint_next_action.append(next_action)
			self.actuate(joint_next_action)

			self.joint_reward,self.terminal,self.collided=self.receive_reward()

			#self.training_q.put(self.sars())
			self.replay_buffer.add(joint_cstate=self.joint_cstate,
								joint_action=self.joint_action,
								joint_reward=self.joint_reward,
								joint_nstate=self.joint_nstate,
								terminal=self.terminal,
								eoe=(self.terminal|self.collided|(self.step>=self.max_step)))

			#transite state and action, maybe into terminal state
			for i in xrange(self.n_worker):
				self.joint_cstate[i].copy(self.joint_nstate[i])#deepcopy
			self.joint_action=joint_next_action
			self.step+=1
			self.eprwd+=sum(self.joint_reward)
			
			if self.terminal or self.collided or self.step>=self.max_step:
				tb_eprwd_sum=self.sess.run(self.tb_eprwd,feed_dict={self.tb_eprwd_in:self.eprwd})
				self.tb_writer.add_summary(tb_eprwd_sum,self.episode)
				print('-----------------------'+self.name+'--------------------------')
				print('episode:',self.episode)
				print('eprwd:',self.eprwd)
				print('step:',self.step)
				print('collided: ',self.collided)
				print('replay buffer size: ', self.replay_buffer.size())
				self.log.write(str(self.episode)+","+str(self.eprwd)+"\n")

				self.step=0
				self.eprwd=0
				self.episode+=1#read by train()
				self.reset_pose()
				self.reset_state_action()
				self.initMDP()
			else:
				#print("interact_with_environment timestamp:",rospy.get_time())
				#print("wall time: ", time.clock())
				self.interact_rate.sleep()
			

	def initMDP(self):
		self.interact_rate.sleep() #reset loop timer to now
		self.actor.copy_master_network()
		self.observe_state()
		joint_next_action=[]
		for i in xrange(self.n_worker):
			next_action=self.actor.predict(self.joint_nstate[i].obs_in(),
									self.joint_nstate[i].dir_in(),
									self.joint_nstate[i].ind_in())
			self.exploration_noise(next_action,i)
			joint_next_action.append(next_action)
		self.actuate(joint_next_action)

		for i in xrange(self.n_worker):
			self.joint_cstate[i].copy(self.joint_nstate[i])#deepcopy
		self.joint_action=joint_next_action
		
		self.interact_rate.sleep() #wait for state transition

	def actuate(self,actions):
		for i in xrange(self.n_worker):
			velcmd=Twist()
			velcmd.linear.x=self.joint_action[i][0][0]*self.max_lvel
			velcmd.linear.y=0.0
			velcmd.linear.z=0.0
			velcmd.angular.x=0.0
			velcmd.angular.y=0.0
			velcmd.angular.z=self.joint_action[i][0][1]*self.max_avel
			self.velcmd_pubs[i].publish(velcmd)

	#determine rewards received by each robot
	def receive_reward(self):
		'''
		Args:
			None
		Return:
			rewards:a list rewards for all robots.
			terminal: robots reached terminal state
		'''
		rewards=[]
		all_reached_goal=True
		one_has_collided=False
		for i in xrange(self.n_worker):
			cgoal_dist=np.linalg.norm(self.joint_cstate[i].pose[0,0:self.pos_dim]-self.joint_goal[i])
			ngoal_dist=np.linalg.norm(self.joint_nstate[i].pose[0,0:self.pos_dim]-self.joint_goal[i])
			goal_rwd=20.0*(cgoal_dist-ngoal_dist)#was 3.0
			#ossc_rwd=-0.1*np.linalg.norm(self.joint_nstate[i].pose[0,3:6])
			#print ossc_rwd
			#print(self.joint_nstate[i].pose[0][3:6])
			#print(goal_rwd)
			rwd=goal_rwd#+ossc_rwd
			if self.joint_nstate[i].event==1:#collided
				rwd-=10.0
				one_has_collided=True

			if self.reached_goal(self.joint_nstate[i],self.joint_goal[i]):
				all_reached_goal&=True
			else:
				all_reached_goal&=False
			rewards.append(rwd)

		if all_reached_goal:
			for i in xrange(self.n_worker):
				rewards[i]+=5.0

		terminal=int(all_reached_goal)
		return rewards,terminal,one_has_collided

	#state stansition based on a joint action
	def observe_state(self):
		for i in xrange(self.n_worker):
			self.load_lidar[i]=True
			self.load_odom[i]=True
		transiting=True
		while transiting:
			transiting=False
			for i in xrange(self.n_worker):
				transiting|=(self.load_lidar[i]|self.load_odom[i])


	#lidar callback for subscriber ith worker
	def lidar_receiver(self,msg,wi):
		if self.load_lidar[wi]:
			ranges=np.asarray(msg.ranges)
			for i in xrange(len(ranges)):
				if ranges[i]==float('inf'):
					ranges[i]=1.0
				else:
					ranges[i]/=self.lidar_rmax
			self.joint_nstate[wi].update_obsseq_event(ranges)
			self.load_lidar[wi]=False
			#print(self.name,"updated lidar obsseq")

	#odometry callback for subscriber i
	def odom_receiver(self,msg,wi):
		if self.load_odom[wi]:
			vel=np.zeros([1,6],dtype=float)
			vel[0][0]=msg.twist.twist.linear.x
			vel[0][1]=msg.twist.twist.linear.y
			vel[0][2]=msg.twist.twist.linear.z
			vel[0][3]=msg.twist.twist.angular.x
			vel[0][4]=msg.twist.twist.angular.y
			vel[0][5]=msg.twist.twist.angular.z
			pose=np.zeros([1,6],dtype=float)
			r,p,y=rostf.transformations.euler_from_quaternion([msg.pose.pose.orientation.x,
															msg.pose.pose.orientation.y,
															msg.pose.pose.orientation.z,
															msg.pose.pose.orientation.w])
			pose[0][0]=msg.pose.pose.position.x
			pose[0][1]=msg.pose.pose.position.y
			pose[0][2]=msg.pose.pose.position.z
			pose[0][3]=r
			pose[0][4]=p
			pose[0][5]=y
			self.joint_nstate[wi].update_vel_dir_pose(vel,pose,self.joint_goal[wi])
			self.load_odom[wi]=False
			#print(self.name,"updated vel dir pose")

	#reset robot poses in the world frame to their inital poses
	def reset_pose(self):
		#stop the robot before reset its pose, since it's
		#difficult to reset the velocities of all the links
		stop=Twist()
		time=rospy.get_time()
		while rospy.get_time()-time<2:
			for i in xrange(self.n_worker):
				self.velcmd_pubs[i].publish(stop)
			self.stop_rate.sleep()
		for i in xrange(self.n_worker):
			modelstate = ModelState()
			modelstate.model_name = "rosbot"+str(i+self.rtoffset)
			modelstate.reference_frame = "world"
			x,y,z,w=rostf.transformations.quaternion_from_euler(self.joint_initpose[i][0][3], 
																self.joint_initpose[i][0][4],
																self.joint_initpose[i][0][5])
			modelstate.pose.position.x=self.joint_initpose[i][0][0]
			modelstate.pose.position.y=self.joint_initpose[i][0][1]
			#modelstate.pose.position.z=self.joint_initpose[i][0][2]
			modelstate.pose.orientation.x=x
			modelstate.pose.orientation.y=y
			modelstate.pose.orientation.z=z
			modelstate.pose.orientation.w=w
			reps=self.reset_client(modelstate)

	def reset_state_action(self):
		self.joint_cstate=[State(goal=self.joint_goal[i].copy(),
								pose=self.joint_initpose[i].copy(),
								index=i,
								pos_dim=self.pos_dim,
								vel_dim=self.vel_dim,
								obs_dim=self.lidar_dim,
								obs_seqlen=self.lidar_seqlen,
								obs_rmax=self.lidar_rmax,
								coll_dist=self.lidar_colldist) 
							for i in xrange(self.n_worker)]
		
		self.joint_nstate=[State(goal=self.joint_goal[i].copy(),
								pose=self.joint_initpose[i].copy(),
								index=i,
								pos_dim=self.pos_dim,
								vel_dim=self.vel_dim,
								obs_dim=self.lidar_dim,
								obs_seqlen=self.lidar_seqlen,
								obs_rmax=self.lidar_rmax,
								coll_dist=self.lidar_colldist)
							for i in xrange(self.n_worker)]

		self.joint_action=[np.zeros([1,self.act_dim],dtype=float)
							for _ in xrange(self.n_worker)]

	def terminate(self):
		#if(not rospy.is_shutdown()):
		#	rospy.signal_shutdown("training is completed...")
		self.t_behavior.join()
		print("behavior thread is terminated...")
		self.t_training.join()
		print("training thread is terminated...")
		
	def start(self):
		self.actor.init_target_network()
		self.actor.copy_master_network()
		self.critic.init_target_network()
		self.critic.copy_master_network()
		self.t_behavior.start()
		self.t_training.start()
		#return self.t_behavior,self.t_training

	def epslion_greedy(self):
		self.epslion_t+=1
		if self.replay_buffer.size()>self.min_repbuf_size:
			self.epslion=max(self.epslion*0.95,0.05)
		return rand.uniform(0,1)>self.epslion

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
		#T = 1.  # Total time.
		#n = int(T / dt)  # Number of time steps.
		#t = np.linspace(0., T, n)  # Vector of times.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise_lv[i] = self.ou_noise_lv[i]+dt*(-(self.ou_noise_lv[i]-mu)/tau)+sigma_bis*sqrtdt*np.random.randn()
		#x = np.zeros(n)
		#for i in range(n - 1):
		#    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()
		#print(self.ou_noise_lv)
		return self.ou_noise_lv[i]

	def ornstein_uhlenbeck_noise_av(self,i):
		sigma = 0.1  # Standard deviation.
		mu = 0.  # Mean.
		tau = .05  # Time constant.
		dt = .001  # Time step.
		#T = 1.  # Total time.
		#n = int(T / dt)  # Number of time steps.
		#t = np.linspace(0., T, n)  # Vector of times.
		sigma_bis = sigma * np.sqrt(2. / tau)
		sqrtdt = np.sqrt(dt)
		self.ou_noise_av[i] = self.ou_noise_av[i]+dt*(-(self.ou_noise_av[i]-mu)/tau)+sigma_bis*sqrtdt*np.random.randn()
		#x = np.zeros(n)
		#for i in range(n - 1):
		#    x[i + 1] = x[i] + dt * (-(x[i] - mu) / tau) + sigma_bis * sqrtdt * np.random.randn()
		#print(self.ou_noise_av)
		return self.ou_noise_av[i]

	def reached_goal(self,state,goal):
		dist=np.linalg.norm(state.pose[0,0:self.pos_dim]-goal[0,:])
		return dist<=0.5





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



