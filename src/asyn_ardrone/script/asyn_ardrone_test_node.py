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

model_path = ''
log_path = ''

def log_episode_info(
    log,
    ep_rwd,
    ep_step,
    avg_loss,
    actions,
    traj,
    succ
):
    act_str = ""
    for a in actions:
        act_str += (str(a)+",")
    traj_str=""
    for t in traj:
        traj_str += (str(t[0])
                    + ","
                    + str(t[1])
                    + ","
                    + str(t[2])
                    + ",")
    log.writerow(
        [
            str(ep_rwd),
            str(ep_step),
            str(avg_loss),
            str(succ),
            act_str[:-1],
            traj_str[:-1]
        ]
    )

def main():
    rospy.init_node("asyn_ardrone_test_node")
    ros_rate = rospy.Rate(0.2)
    sess = tf.Session()
    evaluator = Worker(
        sess=sess,
        name="master",
        init_pose=np.array(
            [[-39.0,9.5,0.0,0,0,-1.57]],
            dtype=float
        ),
        goal=np.array([[-11.0,-9.5,0.0]]),
        master_actor=None,
        master_critic=None,
        training=False,
        tb_writer=None
    )
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(
        evaluator.actor.net_params+evaluator.critic.net_params
    )
    saver.restore(
        sess,
        model_path
    )
    csv_file = open(log_path,'wb')
    test_log = csv.writer(
        csv_file,
        delimiter=',', 
        quotechar='"', 
        quoting=csv.QUOTE_MINIMAL
    )
    itr = 0
    while itr<1:
        (
            ep_rwd,
            ep_step,
            avg_loss,
            actions,
            traj,
            succ
        ) = evaluator.evaluate()
        print("eval the master networks: ")
        print("ep_rwd: ",ep_rwd)
        print("ep_step: ", ep_step)
        print("avg_loss: ", avg_loss)
        print("actions: ",actions)
        print("--------------------------")
        log_episode_info(
            test_log,
            ep_rwd,
            ep_step,
            avg_loss,
            actions,
            traj,
            succ
        )
        itr += 1
        rospy.sleep(2)

if __name__ == '__main__':
    main()
