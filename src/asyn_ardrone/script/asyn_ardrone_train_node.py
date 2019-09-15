#!usr/bin/env python
import rospy
import csv
import numpy as np
import tensorflow as tf
from threading import Thread
from ddpg_network import ActorNetwork
from ddpg_network import CriticNetwork
from worker import Worker

rospy.init_node("asyn_ardrone_train_node")
model_path = ''
log_path = ''
ros_rate = rospy.Rate(0.1)
model_save_cnt = 0
n_worker = 3
workers = []
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
    [[-7.5,-11.5,3.0]]
]
sess = tf.Session()
master_actor = ActorNetwork(
    sess=sess,
    name="master_actor",
    time_step=4,
    obs_dim=360,
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
    obs_dim=360,
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
tb_writer = tf.summary.FileWriter('./save/tbsum')
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
ep_loss_tb=tf.summary.scalar(
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
evaluator = Worker(
    sess=sess,
    name="ardrone3",
    init_pose=np.array(init_pose[-1]),
    goal=np.array(goals[-1]),
    master_actor=master_actor,
    master_critic=master_critic,
    training=False,
    tb_writer=tb_writer
)
saver = tf.train.Saver(
    evaluator.actor.net_params
    + evaluator.critic.net_params
)  # model of evaluator is saved
restored_for_test = False

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
    traj_str = ""
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

#keeps track of training progress
def evaluate_master_network(itr,log_csv_writer):
    global model_save_cnt
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    ep_rwd,ep_step,avg_loss,actions,traj,succ = evaluator.evaluate()
    print("evak master networks: ")
    print("ep_rwd: ", ep_rwd)
    print("ep_step: ", ep_step)
    print("avg_loss: ", avg_loss)
    print("actions: ", actions)
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
        })  
    tb_writer.add_summary(summary, itr)
    log_episode_info(
        log_csv_writer,
        ep_rwd,
        ep_step,
        avg_loss,
        actions,
        traj,
        succ
    )
    if succ==1:  # save each good model copied 
                 # from master networks
        saver.save(
            sess,
            model_path
        )
        model_save_cnt += 1

#Test the model with parameters copied from master networks
def test_for_evaluator_save():
    sess.run(tf.global_variables_initializer())
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
    itr=0
    while itr<50:
        (
            ep_rwd,
            ep_step,
            avg_loss,
            actions,
            traj,
            succ
        ) = evaluator.evaluate()
        print("eval master networks: ")
        print("ep_rwd: ",ep_rwd)
        print("ep_step: ", ep_step)
        print("avg_loss: ", avg_loss)
        print("actions: ",actions)
        print("----------------------")
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

#Test the master network model copied by evaluator 
'''
def test_for_master_save():
    sess.run(tf.global_variables_initializer())
    tf.train.Saver(
        master_actor.net_params+master_critic.net_params
    ).restore(
        sess,
        "./save/model/good/master_networks"
    )
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    csv_file = open('./save/log/test.csv','wb')
    test_log = csv.writer(
        csv_file,delimiter=',', 
        quotechar='"', 
        quoting=csv.QUOTE_MINIMAL
    )
    itr = 0
    while itr<5:
        ep_rwd,ep_step,avg_loss,actions,traj,succ = evaluator.evaluate()
        print(itr," the evaluation of the master networks: ")
        print("ep_rwd: ",ep_rwd)
        print("ep_step: ", ep_step)
        print("avg_loss: ", avg_loss)
        print("actions: ",actions)
        print("------------------------------------------------")
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
        rospy.sleep(1)
'''

def train_in_test_environment():
    worker=Worker(
        sess=sess,
        name="ardrone0",
        init_pose=np.array([[-4.0,14.0,3.0,0,0,-1.57]]),
        goal=np.array([[24.0,-4.5,3.0]]),
        master_actor=master_actor,
        master_critic=master_critic,
        training=True,
        tb_writer=tb_writer
    )
    sess.run(tf.global_variables_initializer())
    worker.actor.copy_master_network()
    worker.critic.copy_master_network()
    worker.start()
    evaluator.actor.copy_master_network()
    evaluator.critic.copy_master_network()
    csv_file=open(log_path,'wb')
    train_log =csv.writer(
        csv_file,
        delimiter=',', 
        quotechar='"', 
        quoting=csv.QUOTE_MINIMAL
    )
    try:
        itr=0
        while not rospy.is_shutdown():
            evaluate_master_network(itr,train_log)
            itr+=1
            rospy.sleep(5)
    except Exception as e:
        print(e)
    finally:
        worker.terminate()
        print("joined",worker.name)
        saver.save(
            sess,
            model_path
        )
        csv_file.close()

def main():
    global master_actor,master_critic
    for i in xrange(n_worker):
        worker = Worker(
            sess=sess,
            name="ardrone"+str(i),
            init_pose=np.array(init_pose[i]),
            goal=np.array(goals[i]),
            master_actor=master_actor,
            master_critic=master_critic,
            training=True,
            tb_writer=tb_writer
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
    csv_file=open(log_path,'wb')
    train_log =csv.writer(
        csv_file,
        delimiter=',', 
        quotechar='"', 
        quoting=csv.QUOTE_MINIMAL
    )
    try:
        itr=0
        while not rospy.is_shutdown():
            evaluate_master_network(itr, train_log)
            itr+=1
            rospy.sleep(5)
        
    except Exception as e:
        print(e)
    finally:
        for worker in workers:
            worker.terminate()
            print("joined",worker.name)
        saver.save(
            sess,
            model_path
        )
        csv_file.close()

if __name__ == '__main__':
    main()  # for training
    #test_for_master_save()
    #test_for_evaluator_save()
    #train_in_test_environment()
