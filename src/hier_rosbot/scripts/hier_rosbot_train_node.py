#!/usr/bin/env python
import tensorflow as tf; tf.set_random_seed(1)
import rospy
import numpy as np
import datetime
from hier_rosbot.manager_net import ManagerActorNetwork, ManagerCriticNetwork
from hier_rosbot.worker_net import WorkerActorNetwork, WorkerCriticNetwork
from hier_rosbot.coach import Coach

model_path = ''
log_path = ''
tb_path = '/media/zilong/Backup/RLFR/save/hier_rosbot'

traj_dim = 2
traj_emb_size = 32
obs_dim = 180
obs_seq_len = 4
obs_emb_size = 128
temp_abs = 20
temp_abs_skip_steps = 0
collision_thresh = 0.2

manager_actor_fc1_size = 64
manager_actor_fc2_size = 64
manager_critic_fc1_size = 64
manager_critic_fc2_size = 64
manager_actor_lr = 0.000001
manager_critic_lr = 0.00001
manager_replay_buffer_size = 2000
manager_batch_size = 64

worker_actor_fc1_size = 512
worker_actor_fc2_size = 512
worker_critic_fc1_size = 512
worker_critic_fc2_size = 512
worker_actor_lr = 0.000001
worker_critic_lr = 0.00001
worker_replay_buffer_size = 200000
worker_batch_size = 256

tb_writer = tf.summary.FileWriter(tb_path)

init_world_pose=np.array(
	[[-1.5,1.7,0.0,0,0,0.0]],
	dtype=float
)

final_goal=np.array(
	[[-1.5,-1.7]],
	dtype=float
)

if __name__ == '__main__':
	rospy.init_node("hier_rosbot_train_node")
	exp_timestamp = datetime.datetime.now().strftime(
		"%d-%m-%Y %H:%M:%S"
	)
	sess = tf.Session()
	master_manager_actor = ManagerActorNetwork(
		sess=sess,
		name='master_manager_actor',
		obs_dim=obs_dim,		
		obs_seq_len=obs_seq_len,
		obs_emb_size=obs_emb_size,
		traj_dim=traj_dim,
		traj_emb_size=traj_emb_size,
		temp_abs=temp_abs,
		skip_steps=temp_abs_skip_steps,
		n_fc1_unit=manager_actor_fc1_size,
		n_fc2_unit=manager_actor_fc2_size,
		batch_size=manager_batch_size,
		learning_rate=manager_actor_lr,
		device='/device:GPU:0',
	)
	master_manager_critic = ManagerCriticNetwork(
		sess=sess,
		name='master_manager_critic',
		obs_dim=obs_dim,		
		obs_seq_len=obs_seq_len,
		obs_emb_size=obs_emb_size,
		traj_dim=traj_dim,
		traj_emb_size=traj_emb_size,
		temp_abs=temp_abs,
		skip_steps=temp_abs_skip_steps,
		n_fc1_unit=manager_critic_fc1_size,
		n_fc2_unit=manager_critic_fc2_size,
		batch_size=manager_batch_size,
		learning_rate=manager_critic_lr,
		device='/device:GPU:0',
	)
	master_worker_actor = WorkerActorNetwork(
		sess=sess,
		name='master_worker_actor',
		obs_seq_len=obs_seq_len,
		obs_dim=obs_dim,
		obs_emb_size=obs_emb_size,
		n_fc1_unit=worker_actor_fc1_size,
		n_fc2_unit=worker_actor_fc2_size,
		batch_size=worker_batch_size,
		learning_rate=worker_actor_lr,
		device='/device:GPU:0',
	)
	master_worker_critic = WorkerCriticNetwork(
		sess=sess,
		name='master_worker_critic',
		obs_dim=obs_dim,		
		obs_seq_len=obs_seq_len,
		obs_emb_size=obs_emb_size,
		n_fc1_unit=worker_critic_fc1_size,
		n_fc2_unit=worker_critic_fc2_size,
		batch_size=worker_batch_size,
		learning_rate=worker_critic_lr,
		device='/device:GPU:0',
	)
	coach = Coach(
		sess=sess,
		name='coach',
		obs_dim=obs_dim,
		obs_seq_len=obs_seq_len,
		obs_emb_size=obs_emb_size,
		traj_dim=traj_dim,
		traj_emb_size=traj_emb_size,
		temp_abs=temp_abs,
		temp_abs_skip_steps=temp_abs_skip_steps,
		collision_thresh=collision_thresh,
		manager_batch_size=manager_batch_size,
		worker_batch_size=worker_batch_size,
		manager_actor_fc1_size=manager_actor_fc1_size,
		manager_actor_fc2_size=manager_actor_fc2_size,
		manager_critic_fc1_size=manager_critic_fc1_size,
		manager_critic_fc2_size=manager_critic_fc2_size,
		manager_actor_lr=manager_actor_lr,
		manager_critic_lr=manager_critic_lr,
		manager_replay_buffer_size=manager_replay_buffer_size,
		worker_actor_fc1_size=worker_actor_fc1_size,
		worker_actor_fc2_size=worker_actor_fc2_size,
		worker_critic_fc1_size=worker_critic_fc1_size,
		worker_critic_fc2_size=worker_critic_fc2_size,
		worker_actor_lr=worker_actor_lr,
		worker_critic_lr=worker_critic_lr,
		worker_replay_buffer_size=worker_replay_buffer_size,
		master_manager_actor=master_manager_actor,
		master_manager_critic=master_manager_critic,
		master_worker_actor=master_worker_actor,
		master_worker_critic=master_worker_critic,
		init_world_pose=init_world_pose,
		final_goal=final_goal,
		device='/device:GPU:0',
		tb_writer=tb_writer
	)
	try:
		sess.run(tf.global_variables_initializer())
		coach.start_training()
		rospy.spin()
	finally:
		print("hier_rosbot_node terminated...")
		
		