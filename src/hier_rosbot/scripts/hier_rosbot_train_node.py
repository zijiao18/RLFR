#!/usr/bin/env python
import tensorflow as tf; tf.set_random_seed(1)
import rospy
import numpy as np
import datetime
from hier_rosbot.manager_net import ManagerActorNetwork, ManagerCriticNetwork
from hier_rosbot.worker_net import WorkerActorNetwork, WorkerCriticNetwork
from hier_rosbot.coach import Coach

n_agent=3
obs_dim=180
obs_seq_len=4
obs_emb_size=64
collision_thresh=0.2
batch_size=1

manager_actor_fc1_size=128
manager_actor_fc2_size=128
manager_critic_fc1_size=256
manager_critic_fc2_size=256
manager_actor_lr=0.001
manager_critic_lr=0.001
manager_replay_buffer_size=10000

worker_actor_fc1_size=512
worker_actor_fc2_size=512
worker_critic_fc1_size=512
worker_critic_fc2_size=512
worker_actor_lr=0.001
worker_critic_lr=0.001
worker_replay_buffer_size=400000

init_poses=np.array(
	[
		[[0.0,1.5,0.0,0,0,-1.57]],
		[[-1.5,0.0,0.0,0,0,0.0]],
		[[1.5,0.0,0.0,0,0,3.14]]
	],
	dtype=float
)

final_goals=np.array(
	[
		[[0,-1.5]],
		[[1.5,0.0]],
		[[-1.5,0.0]]
	],
	dtype=float
)


# TODO: revise the file and start testing the implementation
if __name__ == '__main__':
	rospy.init_node("hier_rosbot_node")
	exp_timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
	sess=tf.Session()
	master_manager_actor=ManagerActorNetwork(
		sess=sess,
		name='master_manager_actor',
		n_agent=n_agent,
		n_fc1_unit=manager_actor_fc1_size,
		n_fc2_unit=manager_actor_fc2_size,
		batch_size=batch_size,
		learning_rate=manager_actor_lr,
		device='/device:GPU:0',
	)
	master_manager_critic=ManagerCriticNetwork(
		sess=sess,
		name='master_manager_critic',
		n_agent=n_agent,
		n_fc1_unit=manager_critic_fc1_size,
		n_fc2_unit=manager_critic_fc2_size,
		batch_size=batch_size,
		learning_rate=manager_critic_lr,
		device='/device:GPU:0',
	)
	master_worker_actor=WorkerActorNetwork(
		sess=sess,
		name='master_worker_actor',
		obs_seq_len=obs_seq_len,
		obs_dim=obs_dim,
		obs_emb_size=obs_emb_size,
		n_fc1_unit=worker_actor_fc1_size,
		n_fc2_unit=worker_actor_fc2_size,
		batch_size=batch_size,
		learning_rate=worker_actor_lr,
		device='/device:GPU:0',
	)
	master_worker_critic=WorkerCriticNetwork(
		sess=sess,
		name='master_worker_critic',
		obs_dim=obs_dim,		
		obs_seq_len=obs_seq_len,
		obs_emb_size=obs_emb_size,
		n_fc1_unit=worker_critic_fc1_size,
		n_fc2_unit=worker_critic_fc2_size,
		batch_size=worker_critic_lr,
		learning_rate=worker_critic_lr,
		device='/device:GPU:0',
	)
	coach=Coach(
		sess=sess,
		name='coach',
		n_agent=3,
		obs_dim=obs_dim,
		obs_seq_len=obs_seq_len,
		obs_emb_size=obs_emb_size,
		collision_thresh=collision_thresh,
		batch_size=batch_size,
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
		init_poses=init_poses,
		final_goals=final_goals,
		device='/device:GPU:0',
		tb_writer=None
	)
	try:
		sess.run(tf.global_variables_initializer())
		coach.start_training()
		rospy.spin()
	finally:
		print("malp_rosbot_node terminated...")
		
		