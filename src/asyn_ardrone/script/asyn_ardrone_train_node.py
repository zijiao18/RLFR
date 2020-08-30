#!/usr/bin/env python
import rospy
import csv
import numpy as np
import tensorflow as tf
from threading import Thread
from asyn_ardrone.ddpg_network import (
    ActorNetwork, 
    CriticNetwork
)
from asyn_ardrone.worker import Worker
import datetime

rospy.init_node("asyn_ardrone_train_node")
model_restore_path = ''
model_save_path = ''
log_path = ''
tb_writer_path = ''

n_worker = 3
vel_dim = 6  # (x,y,z,roll,pitch,yaw), do not change
pos_dim = 3  # (x,y,z), do not change
act_dim = 1
obs_dim = 180 # lidar ranges, do not change
obs_seqlen = 4
batch_size = 256

actor_lstm_state_dim = 64
actor_fc1_unit = 256
actor_fc2_unit = 256
critic_lstm_state_dim = 64  
critic_fc1_unit = 256
critic_fc2_unit = 256

actor_lr = 0.0001
critic_lr = 0.001
actor_tau = 0.01
critic_tau = 0.01

init_pose = [
    [[-4.0,14.0,3.0,0,0,-1.57]],
    [[-4.0,4.0,3.0,0,0,0]],
    [[-4.0,-6.0,3.0,0,0,0]],
    [[-39.0,9.5,3.0,0,0,-1.57]]
]
goals = [
    [[8.0,10.0,3.0]],
    [[4.0,-4.0,3.0]],
    [[4.0,-13.0,3.0]],
    [[-12.0,-10.0,3.0]]
]
sess = tf.Session()

master_actor = ActorNetwork(
    sess=sess,
    name="master_actor",
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
    training=True
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
    training=True
) 
tb_writer = tf.summary.FileWriter(tb_writer_path 
    + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S"))



evaluator = Worker(
    sess=sess,
    name="ardrone"+str(n_worker),
    init_pose=np.array(init_pose[-1]),
    goal=np.array(goals[-1]),
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
    model_saver = None,
    tb_writer=tb_writer,
    model_save_path = model_save_path,
    log_path=log_path
)

saver = tf.train.Saver(
    evaluator.actor.net_params
    + evaluator.critic.net_params
)

evaluator.model_saver = saver

workers = [Worker(sess=sess,
                name="ardrone"+str(i),
                init_pose=np.array(init_pose[i]),
                goal=np.array(goals[i]),
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
            ) for i in range(n_worker)
        ]

def evaluate_master_network():
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    evaluator.evaluate()

def main():
    try:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, model_restore_path)
        for worker in workers:
            worker.start()
        while not rospy.is_shutdown():
            evaluate_master_network()
            rospy.sleep(5)
    finally:
        for worker in workers:
            worker.terminate()
        saver.save(sess, model_save_path + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "_final/" + "model")

if __name__ == '__main__':
    main()
