#!/usr/bin/env python
import tensorflow as tf
import rospy
import numpy as np
import datetime
from maddpg_rosbot.exp_5bot.coach import Coach
from maddpg_rosbot.exp_5bot.rec_maddpg_net_ie import (
    ActorNetwork, 
    CriticNetwork
)
# from maddpg_rosbot.rec_maddpg_net_je import (
#   ActorNetwork, CriticNetwork
# )
# from maddpg_rosbot.maddpg_net import (
#   ActorNetwork, 
#   CriticNetwork
# )

# path must end with '/'
model_restore_path = ''
model_save_path = ''
log_path = ''
tb_writer_path = ''
vel_dim = 3  # (x,y,yaw), do not change
pos_dim = 2  # (x,y), do not change
act_dim = 2  # (lv,av), do not change
obs_dim = 180
obs_seqlen = 4
batch_size = 256
actor_lstm_state_dim = 64
critic_lstm_state_dim = 64
actor_fc1_unit = 512
actor_fc2_unit = 512
critic_fc1_unit = 1024
critic_fc2_unit = 1024
actor_lr = 0.0001
critic_lr = 0.001
actor_tau = 0.01
critic_tau = 0.01
init_pose_c0=np.array(
    [
        [[1.5,0.0,0.0,0,0,-3.14]],
        [[0.46352549156242118,1.4265847744427302,0.0,0,0,-1.9]],
        [[-1.213525491562421,0.88167787843870982,0.0,0,0,-0.864785]],
        [[-1.2135254915624214,-0.88167787843870959,0.0,0,0,0.785838]],
        [[0.46352549156242084,-1.4265847744427305,0.0,0,0,2.010268]]
    ],
    dtype=float
)
goals_c0=np.array(
    [
        [[-1.5,0.0]],
        [[-0.46352549156242134,-1.4265847744427302]],
        [[1.213525491562421, -0.88167787843871004]],
        [[1.2135254915624214, 0.88167787843870937]],
        [[-0.46352549156242068, 1.4265847744427305]]
    ],
    dtype=float
)
if __name__ == '__main__':
    rospy.init_node("maddpg_rosbot_train_node")
    exp_timestamp = datetime.datetime.now().strftime(
        "%d-%m-%Y_%H:%M:%S"
    )
    sess = tf.Session()
    tb_writer=tf.summary.FileWriter(
        tb_writer_path
        + exp_timestamp
    )
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
        training=True,
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
        training=True,
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
        training=True,
        rtoffset=0,
        log_path=log_path,
        model_save_path=model_save_path,
        device='/device:GPU:0',
        tb_writer=tb_writer,
        model_saver=saver
    )
    
    try:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, model_restore_path+"model")
        c0.start_training()
        rospy.spin()
    finally:
        saver.save(sess, model_save_path+datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")+"_final/"+"model")
        print("malp_rosbot_node terminated...")
        
        