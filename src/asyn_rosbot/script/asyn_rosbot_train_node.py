#!/usr/bin/env python
import rospy
import numpy as np
import tensorflow as tf
from threading import Thread
from asyn_rosbot.ddpg_network import (
    ActorNetwork,
    CriticNetwork
)
from asyn_rosbot.worker import Worker

rospy.init_node("asyn_rosbot_train_node")
model_path = ''
log_path = ''
tb_writer_path = ''
ros_rate = rospy.Rate(0.2)
n_worker = 3
workers = []
init_pose = [
    [[-4.0,14.0,0.0,0,0,-1.57]],
    [[-4.0,4.0,0.0,0,0,0]],
    [[-4.0,-6.0,0.0,0,0,0]],
    [[-39.0,9.5,0.0,0,0,-1.57]]
]

goals = [
    [[8,10.0,0.0]],
    [[4.0,-4.0,0.0]],
    [[4.0,-13.0,0.0]],
    [[-11.0,-9.5,0.0]]
]

sess = tf.Session()
master_actor = ActorNetwork(
    sess=sess,
    name="master_actor",
    time_step=4,
    obs_dim=180,
    vel_dim=6,
    dir_dim=3,
    batch_size=256, 
    lstm_state_dim=128, 
    n_fc1_unit=1024, 
    n_fc2_unit=1024, 
    learning_rate=0.0001, 
    tau=0.01,
    training=True
)
master_critic = CriticNetwork(
    sess=sess,
    name="master_critic",
    time_step=4,
    obs_dim=180,
    vel_dim=6,
    dir_dim=3,
    batch_size=256,
    lstm_state_dim=128, 
    n_fc1_unit=1024, 
    n_fc2_unit=1024, 
    learning_rate=0.01, 
    tau=0.01, 
    training=True
)
'''
Save only master networks, in
order to avoid recunstructing 
entire session while restoring
the model for test
'''
saver = tf.train.Saver(
    master_actor.net_params
    + master_critic.net_params
)
model_save_cnt = 0
saver = tf.train.Saver(
    master_actor.net_params
    + master_critic.net_params
)
tb_writer = tf.summary.FileWriter(tb_writer_path)
ep_actions_in = tf.placeholder(
    dtype=tf.float32,
    shape=[None]
)
ep_loss_in = tf.placeholder(
    dtype=tf.float32,
    shape=()
)
ep_actions_tb = tf.summary.histogram(
    'ep_actions',
    ep_actions_in
)
ep_loss_tb = tf.summary.scalar(
    'ep_avg_loss',
    ep_loss_in
)
ep_rwd_in = tf.placeholder(
    dtype=tf.float32,
    shape=()
)
ep_step_in = tf.placeholder(
    dtype=tf.float32,
    shape=()
)
ep_rwd_tb = tf.summary.scalar(
    'episodic_reward',
    ep_rwd_in
)
ep_step_tb = tf.summary.scalar(
    'episodic_step',
    ep_step_in
)
merged = tf.summary.merge_all()

evaluator=Worker(
    sess=sess,
    name="rosbot3",
    init_pose=np.array(init_pose[-1]),
    goal=np.array(goals[-1]),
    master_actor=master_actor,
    master_critic=master_critic,
    training=False,
    tb_writer=tb_writer,
    log_path=log_path
)

def evaluate_master_network(itr):
    global model_save_cnt
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    (
        ep_rwd,
        ep_step,
        avg_loss,
        actions,
        succ
    ) = evaluator.evaluate()
    print("Eval master networks: ")
    print("ep_rwd: ", ep_rwd)
    print("ep_step: ", ep_step)
    print("avg_loss: ", avg_loss)
    print("actions: ",actions)
    print("----------------------")
    summary, cl, ao, er, es = sess.run(
        [
            merged,
            ep_loss_tb,
            ep_actions_tb,
            ep_rwd_tb,
            ep_step_tb
        ],
        feed_dict={
            ep_actions_in:actions,
            ep_loss_in:avg_loss,
            ep_rwd_in:ep_rwd,
            ep_step_in:ep_step
        }
    )   
    tb_writer.add_summary(summary, itr)
    if succ==1:  # save only good master networks
        saver.save(
            sess,
            model_path+str(model_save_cnt)
        )
        model_save_cnt+=1

def main():
    global master_actor, master_critic
    for i in xrange(n_worker):
        worker = Worker(
            sess=sess,
            name="rosbot"+str(i),
            init_pose=np.array(init_pose[i]),
            goal=np.array(goals[i]),
            master_actor=master_actor,
            master_critic=master_critic,
            training=True,
            tb_writer=tb_writer,
            log_path=log_path
        )
        workers.append(worker)
    sess.run(tf.global_variables_initializer())
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    for worker in workers:
        worker.actor.copy_master_network()
        worker.critic.copy_master_network()
    for worker in workers:
        worker.start()
    try:
        itr=0
        while not rospy.is_shutdown():
            evaluate_master_network(itr)
            itr += 1
            rospy.sleep(5)#sleep 10 seconds
        for worker in workers:
            worker.terminate()
    except Exception as e:
        print(e)
    finally:
        saver.save(sess, model_path)

if __name__ == '__main__':
    main()
