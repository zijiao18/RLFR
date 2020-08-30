#!/usr/bin/env python
import rospy
import numpy as np
import tensorflow as tf
from threading import Thread
import datetime
from rtac_rosbot.ddpg_network import (
    ActorNetwork,
    CriticNetwork
)
from rtac_rosbot.worker import Worker

rospy.init_node("rtac_rosbot_test_node")
model_save_path = ''
model_restore_path = ''
log_path = ''
tb_writer_path = ''
n_worker = 1
workers = []


vel_dim = 6  # (x,y,yaw), do not change
pos_dim = 3  # (x,y), do not change
act_dim = 2  # (lv,av), do not change
obs_dim = 180
obs_seqlen = 4
batch_size = 256

actor_lstm_state_dim = 128
actor_fc1_unit = 1024
actor_fc2_unit = 1024
critic_lstm_state_dim = 128  
critic_fc1_unit = 1024
critic_fc2_unit = 1024

actor_lr = 0.0001
critic_lr = 0.001

actor_tau = 0.01
critic_tau = 0.01

sess = tf.Session()
tb_writer = tf.summary.FileWriter(tb_writer_path)

master_actor = ActorNetwork(
    sess=sess,
    name='master_actor',
    time_step=obs_seqlen,
    obs_dim=obs_dim,
    vel_dim=vel_dim,
    dir_dim=pos_dim,
    act_dim=act_dim,
    batch_size=batch_size,
    lstm_state_dim=actor_lstm_state_dim,
    n_fc1_unit=actor_fc1_unit,
    n_fc2_unit=actor_fc2_unit,
    learning_rate=actor_lr,
    tau=actor_tau,
    training=False,
    master_network=None
)

master_critic = CriticNetwork(
    sess=sess,
    name="master_critic",
    time_step=obs_seqlen,
    obs_dim=obs_dim,
    vel_dim=vel_dim,
    dir_dim=pos_dim,
    act_dim=act_dim,
    batch_size=batch_size,
    lstm_state_dim=critic_lstm_state_dim, 
    n_fc1_unit=critic_fc1_unit, 
    n_fc2_unit=critic_fc2_unit, 
    learning_rate=critic_lr, 
    tau=critic_tau,
    training=False,
    master_network=None
)
saver = tf.train.Saver(master_actor.net_params + master_critic.net_params)
evaluator=Worker(
    sess=sess,
    name="ardrone0",
    init_pose=np.array([[-39.0,3.5,0.0,0,0,-1.57]],dtype=float),
    goal=np.array([[-22.6, -1.0, 0.0]]),
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
    training=False,
    model_saver = saver,
    tb_writer=tb_writer,
    model_save_path = model_save_path,
    log_path=log_path
)

def evaluate_master_network(itr):
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    evaluator.evaluate()
    

def main():
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_restore_path+"model")
    n_ep = 1
    for _ in range(n_ep):
        evaluator.evaluate()

if __name__ == '__main__':
    main()
