#!/usr/bin/env python
import tensorflow as tf
import rospy
import numpy as np
import datetime
from maddpg_rosbot.coach import Coach
from maddpg_rosbot.rec_maddpg_net_ie import (
    ActorNetwork, 
    CriticNetwork
)
# from maddpg_rosbot.rec_maddpg_net_je import (
#     ActorNetwork, 
#     CriticNetwork
# )
# from maddpg_rosbot.maddpg_net import (
#     ActorNetwork, 
#     CriticNetwork
# )

model_restore_path = '' 
model_save_path = ''
log_path = ''
vel_dim = 3  # (x,y,yaw), do not change
pos_dim = 2  # (x,y), do not change
act_dim = 2  # (lv,av), do not change
obs_dim = 180
obs_seqlen = 4
batch_size = 256
actor_lstm_state_dim = 64
actor_fc1_unit  = 512
actor_fc2_unit = 512
critic_lstm_state_dim = 64 #ie: 128x3-1024-1024, je: 512-1024-1024, maddpg: n/a
critic_fc1_unit = 1024
critic_fc2_unit = 1024
actor_lr = 0.0
critic_lr = 0.0
actor_tau = 0.0
critic_tau = 0.0
init_pose_c0 = np.array(
    [
        [[0.0,1.5,0.0,0,0,-1.57]],
        [[-1.5,0.0,0.0,0,0,0.0]],
        [[1.5,0.0,0.0,0,0,3.14]]
    ],
    dtype=float
)
init_pose_c1 = np.array(
    [
        [[0.0,-4.5,0.0,0,0,-1.57]],
        [[-1.5,-6.0,0.0,0,0,0.0]],
        [[1.5,-6.0,0.0,0,0,3.14]]
    ],
    dtype=float
)
init_pose_c2 = np.array(
    [
        [[0.0,7.5,0.0,0,0,-1.57]],
        [[-1.5,6.0,0.0,0,0,0.0]],
        [[1.5,6.0,0.0,0,0,3.14]]
    ],
    dtype=float
)
goals_c0 = np.array(
    [
        [[0,-1.5]],
        [[1.5,0.0]],
        [[-1.5,0.0]]
    ],
    dtype=float
)
goals_c1 = np.array(
    [
        [[0,-7.5]],
        [[1.5,-6.0]],
        [[-1.5,-6.0]]
    ],
    dtype=float
)
goals_c2 = np.array(
    [
        [[0,4.5]],
        [[1.5,6.0]],
        [[-1.5,6.0]]
    ],
    dtype=float
)
if __name__ == '__main__':
    rospy.init_node("maddpg_rosbot_test_node")
    sess = tf.Session()
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
        device='/device:GPU:0',
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
        device='/device:GPU:0',
        master_network=None
    )
    saver=tf.train.Saver(
        master_actor.net_params
        + master_critic.net_params
    )
    c0 = Coach(
        sess=sess,
        name="coach0",
        initpose=init_pose_c0,
        goals=goals_c0,
        act_dim=act_dim,
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        lidar_dim=obs_dim,
        lidar_seqlen=obs_seqlen,
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
        rtoffset=0,
        log_path=log_path,
        model_save_path=model_save_path,
        device='/device:GPU:0',
        tb_writer=None,
        model_saver=None
    )
    c1 = Coach(
        sess=sess,
        name="coach1",
        initpose=init_pose_c1,
        goals=goals_c1,
        act_dim=act_dim,
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        lidar_dim=obs_dim,
        lidar_seqlen=obs_seqlen,
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
        rtoffset=3,
        log_path=log_path,
        model_save_path=model_save_path,
        device='/device:GPU:0',
        tb_writer=None,
        model_saver=None
    )
    c2 = Coach(
        sess=sess,
        name="coach2",
        initpose=init_pose_c2,
        goals=goals_c2,
        act_dim=act_dim,
        pos_dim=pos_dim,
        vel_dim=vel_dim,
        lidar_dim=obs_dim,
        lidar_seqlen=obs_seqlen,
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
        rtoffset=6,
        log_path=log_path,
        model_save_path=model_save_path,
        device='/device:GPU:0',
        tb_writer=None,
        model_saver=None
    )
    try:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_restore_path+"model")
        c0.start_testing()
        c1.start_testing()
        c2.start_testing()
        rospy.spin()
    finally:
        c0.terminate()
        c1.terminate()
        c2.terminate()
        print("maddpg_rosbot_test_node terminated...")
        
        