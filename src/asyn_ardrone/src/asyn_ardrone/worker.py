import rospy
import numpy as np
import tensorflow as tf
import random as rand
import Queue
import datetime
from threading import Thread
from geometry_msgs.msg import Twist
from std_msgs.msg import (
    Empty, 
    Float32MultiArray
)
from asyn_ardrone.ddpg_network import (
    ActorNetwork, 
    CriticNetwork
)
from asyn_ardrone.utilities import (
    State, 
    Feedback, 
    ReplayBuffer
)

class Worker():
    def __init__(
        self,
        sess,
        name,
        init_pose,
        goal,
        act_dim,
        pos_dim,
        vel_dim,
        obs_dim,
        obs_seqlen,
        batch_size,
        actor_lstm_state_dim,
        critic_lstm_state_dim,
        actor_fc1_unit,
        actor_fc2_unit,
        critic_fc1_unit,
        critic_fc2_unit,
        actor_lr,
        critic_lr,
        actor_tau,
        critic_tau,
        master_actor,
        master_critic,
        training,
        tb_writer,
        model_saver,
        model_save_path,
        log_path
    ):
        self.sess = sess
        self.name = name
        self.goal = goal.copy()  # [1,3]
        self.init_pose = init_pose.copy()  # [[x,y,z,r,p,y]]
        self.episode = 0
        self.epoch = 0
        self.eprwd = 0
        self.collided = False
        self.end_of_episode = False
        self.step = 0
        self.epoch = 0
        self.obs_dim = obs_dim
        self.obs_seqlen = obs_seqlen
        self.act_dim = act_dim
        self.dir_dim = pos_dim
        self.vel_dim = vel_dim
        self.max_step = 500
        self.max_lvel = 2.0
        self.max_avel = 3.141592654
        self.training_batch_size = batch_size
        self.epslion = 1.0
        self.epslion_t = 0
        self.ou_noise_av = 0
        self.lidar_max_range = 5.0
        self.lidar_coll_dist = 0.25
        
        self.replay_buffer = ReplayBuffer(
            max_size=100000,
            dir_dim=self.dir_dim,
            vel_dim=self.vel_dim,
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            obs_seqlen=self.obs_seqlen
        )
        self.min_repbuf_size = 500
        
        self.training = training
        self.state = State(
            goal=self.goal,
            pose=self.init_pose,
            obs_dim=self.obs_dim,
            obs_seq_len=self.obs_seqlen,
            vel_dim=self.vel_dim,
            dir_dim=self.dir_dim,
            obs_rmax=self.lidar_max_range,
            coll_threash=self.lidar_coll_dist
        )
        self.pre_state = State(
            goal=self.goal,
            pose=self.init_pose,
            obs_dim=self.obs_dim,
            obs_seq_len=self.obs_seqlen,
            vel_dim=self.vel_dim,
            dir_dim=self.dir_dim,
            obs_rmax=self.lidar_max_range,
            coll_threash=self.lidar_coll_dist
        )
        self.action = np.zeros(
            [1, self.act_dim],
            dtype=float
        )  
        self.master_actor = master_actor
        self.master_critic = master_critic
        self.actor = ActorNetwork(
            sess=sess,
            name=self.name+"_actor",
            time_step=self.obs_seqlen,
            obs_dim=self.obs_dim,
            vel_dim=self.vel_dim,
            dir_dim=self.dir_dim,
            act_dim=self.act_dim,
            batch_size=self.training_batch_size, 
            lstm_state_dim=actor_lstm_state_dim, 
            n_fc1_unit=actor_fc1_unit, 
            n_fc2_unit=actor_fc2_unit, 
            learning_rate=actor_lr, 
            tau=actor_tau,
            training=self.training,
            master_network=self.master_actor
        )
        self.critic = CriticNetwork(
            sess=sess,
            name=self.name+"_critic",
            time_step=self.obs_seqlen,
            obs_dim=self.obs_dim,
            vel_dim=self.vel_dim,
            dir_dim=self.dir_dim,
            act_dim=self.act_dim,
            batch_size=self.training_batch_size,
            lstm_state_dim=critic_lstm_state_dim, 
            n_fc1_unit=critic_fc1_unit, 
            n_fc2_unit=critic_fc2_unit, 
            learning_rate=critic_lr, 
            tau=critic_tau,
            training=self.training,
            master_network=self.master_critic
        )
        self.env_worker = Thread(target=self.interact_with_environment)
        self.train_worker = Thread(target=self.train)   
        self.model_saver = model_saver
        self.tb_writer = tb_writer
        self.tb_eprwd_in = tf.placeholder(
            dtype=tf.float32,
            shape=()
        )
        self.tb_eprwd = tf.summary.scalar(
            self.name+'_eprwd',
            self.tb_eprwd_in
        )
        self.model_save_path = model_save_path
        self.log_path = log_path
        
        self.velcmd_pub = rospy.Publisher(
            "/"+self.name+"/cmd_vel", 
            Twist, 
            queue_size=10
        )
        self.reset_pub = rospy.Publisher(
            "/"+self.name+"/reset", 
            Empty, 
            queue_size=10
        )
        self.feedback_sub = rospy.Subscriber(
            "/"+self.name+"/pycontroller", 
            Float32MultiArray,
            self.feedback_receiver
        )
        self.feedback_q = Queue.Queue(10)
        self.training_q = Queue.Queue(10)
        self.training_rate = rospy.Rate(10)
        self.log = open(self.log_path+self.name+'_'+datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")+'.txt','w+')

    def train(self):
        while not rospy.is_shutdown():
            try:
                if self.training_q.qsize()>0:
                    sras=self.training_q.get(timeout=1)
                    self.replay_buffer.add(sras)
            except Queue.Empty as e:
                raise NameError("train_worker: no transaction is received...")

            if self.replay_buffer.size() < self.min_repbuf_size:
                print("train_worker: replay_buffer size: "
                    + "%d/%d"% (self.replay_buffer.size(),self.min_repbuf_size))
            else:
                batch=self.replay_buffer.sample(
                    batch_size=self.training_batch_size
                )
                next_act=self.actor.predict_target(
                    obs_batch=batch['next_obs'],
                    dir_batch=batch['next_dir']
                )
                next_q=self.critic.predict_target(
                    obs_batch=batch['next_obs'],
                    dir_batch=batch['next_dir'],
                    act_batch=[next_act]
                )
                tar_q=batch['reward']+0.99*(next_q*(1-batch['terminal']))
                self.critic.update_master_network(
                    obs_batch=batch['cur_obs'],
                    dir_batch=batch['cur_dir'],
                    act_batch=batch['action'],
                    tar_q_batch=tar_q[0]
                )
                act_out=self.actor.predict(
                    obs_batch=batch['cur_obs'],
                    dir_batch=batch['cur_dir']
                )
                act_grads=self.critic.action_gradients(
                    obs_batch=batch['cur_obs'],
                    dir_batch=batch['cur_dir'],
                    act_batch=[act_out]
                )
                self.actor.update_master_network(
                    obs_batch=batch['cur_obs'],
                    dir_batch=batch['cur_dir'],
                    act_grads=act_grads[0]
                )
                
                if (self.tb_writer and (self.name == "ardrone0") and (self.epoch % 10==0)):
                    next_q=self.critic.predict_target(
                        obs_batch=batch['next_obs'],
                        dir_batch=batch['next_dir'],
                        act_batch=[next_act]
                    )
                    tar_q=batch['reward']+0.99*(next_q*(1-batch['terminal']))
                    tb_loss=self.critic.summary(
                        obs_batch=batch['cur_obs'],
                        dir_batch=batch['cur_dir'],
                        act_batch=batch['action'],
                        tar_q_batch=tar_q[0]
                    )
                    tb_pg=self.actor.summary(
                        obs_batch=batch['cur_obs'],
                        dir_batch=batch['cur_dir'],
                        act_grads=act_grads[0]
                    )
                    self.tb_writer.add_summary(tb_loss[0], self.epoch)
                    for g in tb_pg:
                        self.tb_writer.add_summary(g, self.epoch)

                self.epoch += 1
                self.actor.copy_master_network()
                self.critic.copy_master_network()
                self.actor.update_target_network()
                self.critic.update_target_network()
            self.training_rate.sleep()

    def interact_with_environment(self):
        traj = ""
        vel = ""
        colldist = ""
        while not rospy.is_shutdown():
            self.actuate(self.action)
            try:
                self.state.update(
                    feedback=self.feedback_q.get(timeout=1.0),
                    goal=self.goal
                )
            except Queue.Empty:
                print("no feedback is received...")
                break
            rwd, terminal = self.reward_model(self.state)
            
            self.eprwd += rwd 
            traj += "%f;%f;"%(
                self.state.pose[0][0], 
                self.state.pose[0][1]
            )
            vel += "%f;%f;%f;%f;%f;%f;"%(
                self.state.vel[0][0],
                self.state.vel[0][1],
                self.state.vel[0][2],
                self.state.vel[0][3],
                self.state.vel[0][4],
                self.state.vel[0][5]
            )
            colldist += (str(self.state.get_collision_distance())+";")

            sars = {'cur_state':self.pre_state.clone(),
                    'action':self.action.copy(),
                    'reward':rwd, 
                    'terminal':terminal,
                    'next_state':self.state.clone()}

            self.training_q.put(sars)
            self.pre_state.copy(self.state)  # deepcopy
            self.end_of_episode = (terminal or (self.step > self.max_step))
            if not self.end_of_episode:
                self.action = self.actor.predict(
                    self.state.obs_in(),
                    self.state.dir_in()
                )
                self.exploration_noise(self.action)
                self.step += 1
            else:
                self.reset_pose()
                self.state.reset(
                    goal=self.goal.copy(),
                    pose=self.init_pose.copy()
                )
                self.pre_state.reset(
                    goal=self.goal.copy(),
                    pose=self.init_pose.copy()
                )
                self.action = np.zeros(
                    shape=[1,self.act_dim],
                    dtype=float
                )
                print(self.name + '--------------------------')
                print('episode:', self.episode)
                print('eprwd:', self.eprwd)
                print('step:', self.step)
                print('collided: ', self.collided)
                print('replay buffer size: ', self.replay_buffer.size())
                self.log.write(str(self.episode) + ","
                            + str(self.eprwd) + "," 
                            + str(self.step) + ","
                            + str(self.collided) + ","
                            + traj + ","
                            + vel + ","
                            + colldist
                            + "\n")
                self.step = 0
                self.eprwd = 0
                self.episode += 1
                traj = ""
                vel = ""
                colldist = ""
            self.training_rate.sleep()
        print("%s env_worker: exits..."%(self.name))

    def evaluate(self):
        if self.training:
            raise NameError("worker.evaluate: the worker is in training mode...")
            return
        traj = ""
        vel = ""
        colldist = ""
        self.step = 0
        self.action = np.zeros(
            shape=[1,self.act_dim],
            dtype=float
        )
        self.state.reset(
            goal=self.goal,
            pose=self.init_pose.copy()
        )
        self.pre_state.reset(
            goal=self.goal,
            pose=self.init_pose.copy()
        )
        while not rospy.is_shutdown():
            self.actuate(self.action)
            try:
                self.state.update(
                    feedback=self.feedback_q.get(timeout=1.0),
                    goal=self.goal
                )
            except Queue.Empty:
                raise NameError("worker.evaluate: no feedback is received...")
                
            rwd, terminal = self.reward_model(self.state)

            self.collided = (self.state.event == 1)
            self.eprwd += rwd 
            traj += "%f;%f;"%(
                self.state.pose[0][0], 
                self.state.pose[0][1]
            )
            vel += "%f;%f;%f;%f;%f;%f;"%(
                self.state.vel[0][0],
                self.state.vel[0][1],
                self.state.vel[0][2],
                self.state.vel[0][3],
                self.state.vel[0][4],
                self.state.vel[0][5]
            )
            colldist += (str(self.state.get_collision_distance())+";")

            self.pre_state.copy(self.state)

            if (not terminal) and (self.step < self.max_step):
                self.action = self.actor.predict(
                    self.state.obs_in(),
                    self.state.dir_in()
                )
                self.step += 1
            else:
                self.reset_pose()
                tb_eprwd_sum = self.sess.run(
                    self.tb_eprwd,
                    feed_dict={self.tb_eprwd_in: self.eprwd}
                )
                self.tb_writer.add_summary(
                    tb_eprwd_sum,
                    self.episode
                )
                print(self.name + '--------------------------')
                print('episode:', self.episode)
                print('eprwd:', self.eprwd)
                print('step:', self.step)
                print('collided: ', self.collided)
                print('replay buffer size: ', self.replay_buffer.size())
                self.log.write(str(self.episode) + ","
                            + str(self.eprwd) + "," 
                            + str(self.step) + ","
                            + str(self.collided) + ","
                            + traj + ","
                            + vel + ","
                            + colldist
                            + "\n")
                self.step = 0
                self.eprwd = 0
                self.episode += 1
                traj = ""
                vel = ""
                colldist = ""
                if self.model_saver and (not self.collided) and terminal:
                    self.model_saver.save(self.sess, self.model_save_path + "/"
                        + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "/"
                        + "model")
                break

    def actuate(self, action):
        velcmd = Twist()
        velcmd.linear.x = self.max_lvel
        velcmd.linear.y = 0.0
        velcmd.linear.z = 0.0
        velcmd.angular.x = 0.0
        velcmd.angular.y = 0.0
        velcmd.angular.z = action[0][0] * self.max_avel
        self.velcmd_pub.publish(velcmd)

    def reward_model(self,state):
        if self.reached_goal(state):
            return 1.0,True
        elif state.event==0:
            pre_goal_dist = np.linalg.norm(
                self.pre_state.pose[0,0:3]-self.goal
            )
            cur_goal_dist = np.linalg.norm(
                self.state.pose[0,0:3]-self.goal
            )
            # trvl_dist = np.linalg.norm(
            #     self.state.pose[
            #         0,
            #         0:self.dir_dim
            #     ]
            #     - self.pre_state.pose[
            #         0,
            #         0:self.dir_dim
            #     ]
            # )
            # coll_dist = self.state.get_collision_distance()

            goal_rwd = (pre_goal_dist-cur_goal_dist)
            # trvl_rwd = 0.01*trvl_dist
            # coll_rwd = 0.0001/coll_dist

            rwd = goal_rwd#-trvl_rwd-coll_rwd
            return rwd,False
        elif state.event==1:
            return -1.0,True
        else:
            raise NameError(
                self.name
                + " reward_model: invalide event-"
                + str(feedback.reward)
            )

    def reached_goal(self, state):
        dist = np.linalg.norm(
            state.pose[0,0:3]-self.goal[0,:]
        )
        return dist<=2

    def circulating(self,state):
        dot_veldir=np.dot(
            state.vel[0,0:3],
            state.dir[0,:]
        )
        angle=np.arccos(
            dot_veldir
            / (np.linalg.norm(state.vel[0,0:3])*np.linalg.norm(state.dir))
        )
        if angle<3.05 and angle>0.0001:
            return False
        else:
            return True

    def closer_to_goal(self, pre_state, state):
        pre_goal_dist = np.linalg.norm(
            pre_state.pose[0,0:3]-self.goal
        )
        cur_goal_dist = np.linalg.norm(
            state.pose[0,0:3]-self.goal
        )
        return pre_goal_dist-cur_goal_dist

    def feedback_receiver(self, msg):
        feedback = Feedback()
        feedback.load(msg.data)
        feedback.set_sender(self.name)
        self.feedback_q.put(feedback)

    def reset_pose(self):
        reset_sig=Empty()
        self.reset_pub.publish(reset_sig)

    def terminate(self):
        if not rospy.is_shutdown():
            rospy.signal_shutdown(
                "training is completed..."
            )
        self.log.close()
        self.env_worker.join()
        self.train_worker.join()

    def start(self):
        self.actor.init_target_network() 
        self.actor.copy_master_network()
        self.critic.init_target_network() 
        self.critic.copy_master_network()
        self.env_worker.start()
        self.train_worker.start()
        return self.env_worker, self.train_worker

    def epslion_greedy(self):
        self.epslion_t += 1
        if self.replay_buffer.size()>self.min_repbuf_size:
            self.epslion = max(
                self.epslion*0.95,
                0.05
            )
        return rand.uniform(0,1)>self.epslion

    def exploration_noise(self, action):
        #angular velocity
        action[0][0] += self.ornstein_uhlenbeck_noise_av();
        action[0][0] = min(max(action[0][0],-1),1)

    def ornstein_uhlenbeck_noise_av(self):
        sigma = 0.2  # Standard deviation.
        mu = 0.  # Mean.
        tau = .05  # Time constant.
        dt = .001  # Time step.
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)
        self.ou_noise_av = (self.ou_noise_av
                        + dt*(-(self.ou_noise_av-mu)/tau)
                        + sigma_bis*sqrtdt*np.random.randn())
        return self.ou_noise_av




