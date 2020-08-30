#!/usr/bin/env python
import rospy
import numpy as np
import csv
import tensorflow as tf
from asyn_ardrone.ddpg_network import (
    ActorNetwork,
    CriticNetwork
)
from asyn_ardrone.worker import Worker

rospy.init_node("asyn_ardrone_test_node")
model_restore_path = ''
model_save_path = ''
log_path = ''
tb_writer_path = ''

# the config below must match the config in the training script 
vel_dim = 6  
pos_dim = 3  
act_dim = 2  
obs_dim = 180 
obs_seqlen = 4
batch_size = 256

actor_lstm_state_dim = 128
actor_fc1_unit = 512
actor_fc2_unit = 512
critic_lstm_state_dim = 512  
critic_fc1_unit = 1024
critic_fc2_unit = 1024

actor_lr = 0.0001
critic_lr = 0.001
actor_tau = 0.01
critic_tau = 0.01
sess = tf.Session()
tb_writer = tf.summary.FileWriter(tb_writer_path)
evaluator = Worker(
    sess=sess,
    name="ardrone"+str(n_worker),
    init_pose=np.array([[-39.0,9.5,0.0,0,0,-1.57]],dtype=float),
    goal=np.array([[-11.0,-9.5,0.0]]),,
    act_dim=act_dim,
    pos_dim=pos_dim,
    vel_dim=vel_dim,
    obs_dim=obs_dim,
    obs_seqlen=obs_seqlen,
    batch_size=batch_size,
    actor_lstm_state_dim=actor_lstm_state_dim,
    critic_lstm_state_dim=critic_lstm_state_dim,
    actor_fc1_unit=actor_fc1_unit,
    actor_fc2_unit=actor_fc2_unit,
    critic_fc1_unit=critic_fc1_unit,
    critic_fc2_unit=critic_fc2_unit,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    actor_tau=actor_tau,
    critic_tau=critic_tau,
    master_actor=master_actor,
    master_critic=master_critic,
    training=True,
    model_saver = saver,
    tb_writer=tb_writer,
    model_save_path = model_save_path,
    log_path=log_path
)
saver = tf.train.Saver(evaluator.actor.net_params + evaluator.critic.net_params)

def main():
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_restore_path)
    print("restored model...")
    n_ep = 1
    for _ in range(n_ep):
        evaluator.evaluate()

if __name__ == '__main__':
    main()
