import rospy
import numpy as np
import tensorflow as tf
import tf as rostf
import random as rand
import time
import datetime
import Queue
from threading import Thread
from gazebo_msgs.srv import (
    SetModelState, 
    SetLinkState
)
from gazebo_msgs.msg import (
    ModelState, 
    LinkState
)
from geometry_msgs.msg import (
    Pose, 
    Twist
)
from std_msgs.msg import (
    Empty, 
    Float32MultiArray
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rtac_rosbot.ddpg_network import (
    ActorNetwork,
    CriticNetwork
)
from rtac_rosbot.utilities import (
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
        self.state = State(
            goal=goal,
            pose=init_pose,
            obs_dim=180,
            obs_seq_len=4
        )
        self.pre_state = State(
            goal=goal,
            pose=init_pose,
            obs_dim=180,
            obs_seq_len=4
        )
        self.action = np.zeros(
            shape=[1,1],
            dtype=float
        )  # yaw angular velocity
        self.goal = goal.copy()#[1,3]
        self.init_pose = init_pose.copy()  # [[x,y,z,r,p,y]]    
        self.episode = 0
        self.end_of_episode = False
        self.step = 0
        self.epoch = 0
        self.epoch_max = 10
        self.max_step = 1000
        self.lvel = 1.0
        self.max_avel = 3.141592654
        self.epslion = 1.0
        self.epslion_t = 0 
        self.ou_noise = 0.0
        self.act_dim = act_dim
        self.pos_dim = pos_dim
        self.vel_dim = vel_dim
        self.obs_dim = obs_dim
        self.obs_seqlen = obs_seqlen
        self.batch_size = batch_size
        self.lidar_max_range = 2.0
        self.velcmd_pub = rospy.Publisher(
            "/"+self.name+"/cmd_vel", 
            Twist, 
            queue_size=1000
        )
        self.scan_sub = rospy.Subscriber(
            "/"+self.name+"/scan", 
            LaserScan,
            self.lidar_receiver
        )
        self.odom_sub = rospy.Subscriber(
            "/"+self.name+"/odom", 
            Odometry, 
            self.odom_receiver
        )
        self.reset_client = rospy.ServiceProxy(
            "/gazebo/set_model_state",
            SetModelState
        )
        self.load_lidar_scan = False
        self.load_odometry = False
        self.state_trans_delay = 0.1
        self.delta_t = 0.05
        self.interact_rate = rospy.Rate(5)
        self.training_rate = rospy.Rate(100)
        self.stop_rate = rospy.Rate(10)
        self.replay_buffer = ReplayBuffer(100000)
        self.min_repbuf_size = 200
        self.feedback_q = Queue.Queue(100)
        self.training_q = Queue.Queue(100)
        self.training = training    
        self.actor = ActorNetwork(
            sess=sess,
            name=self.name+"_actor",
            time_step=self.obs_seqlen,
            obs_dim=self.obs_dim,
            vel_dim=self.vel_dim,
            dir_dim=self.pos_dim,
            act_dim=self.act_dim,
            batch_size=self.batch_size,
            lstm_state_dim=actor_lstm_state_dim,
            n_fc1_unit=actor_fc1_unit,
            n_fc2_unit=actor_fc2_unit,
            learning_rate=actor_lr,
            tau=actor_tau,
            training=training,
            master_network=master_actor
        )
        self.critic = CriticNetwork(
            sess=sess,
            name=self.name+"_critic",
            time_step=self.obs_seqlen,
            obs_dim=self.obs_dim,
            vel_dim=self.vel_dim,
            dir_dim=self.pos_dim,
            act_dim=self.act_dim,
            batch_size=self.batch_size,
            lstm_state_dim=critic_lstm_state_dim, 
            n_fc1_unit=critic_fc1_unit, 
            n_fc2_unit=critic_fc2_unit,
            learning_rate=critic_lr, 
            tau=critic_tau,
            training=training,
            master_network=master_critic
        )
        self.master_actor = master_actor
        self.master_critic = master_critic
        self.env_worker = Thread(target=self.interact_with_environment)
        self.train_worker = Thread(target=self.train_dec)
        self.train_seq_worker = Thread(target=self.train_seq)
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
        self.dec_control_delay_log = None 
        self.seq_control_delay_log = None 
        self.log = open(self.log_path+self.name+'_'+datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")+'.txt','w+')

    def train_dec(self):
        while not rospy.is_shutdown():
            try:
                if self.training_q.qsize()>0:
                    sras = self.training_q.get(timeout=1)
                    self.replay_buffer.add(sras)
            except Queue.Empty as e:
                print("train_worker: no transaction is received...")
                break 
            if self.replay_buffer.size()<self.min_repbuf_size:
                print("%s replay_buffer size: %d/%d" %(
                        self.name,
                        self.replay_buffer.size(),
                        self.min_repbuf_size))
                self.training_rate.sleep()
            else:
                batch = self.replay_buffer.sample(
                    batch_size=self.batch_size
                )
                next_act = self.actor.predict_target(
                    obs_batch=batch['next_obs'],
                    vel_batch=batch['next_vel'],
                    dir_batch=batch['next_dir']
                )
                next_q = self.critic.predict_target(
                    obs_batch=batch['next_obs'],
                    vel_batch=batch['next_vel'],
                    dir_batch=batch['next_dir'],
                    act_batch=[next_act]
                )
                tar_q = batch['reward']+0.99*(next_q*(1-batch['terminal']))
                self.critic.update_master_network(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir'],
                    act_batch=batch['action'].reshape(
                        (1,self.training_batch_size,1)
                    ),
                    tar_q_batch=tar_q[0]
                )
                act_out = self.actor.predict(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir']
                )
                act_grads = self.critic.action_gradients(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir'],
                    act_batch=[act_out]
                )
                self.actor.update_master_network(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir'],
                    act_grads=act_grads[0]
                )
                if (self.tb_writer and (self.name == "rosbot0") and (self.epoch % 10==0)):
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
        obs_time = 0  # timestamp of observing state
        delay_time = ''
        traj = ""
        vel = ""
        colldist = ""
        self.dec_control_delay_log = open('/media/zilong/Backup/RLFR/save/rtac_rosbot/log/dec_control_delay_'+self.name,'w+')
        while not rospy.is_shutdown():
            while(rospy.get_time()-obs_time < self.delta_t):
                continue
            delay_time += (str(rospy.get_time()-obs_time)+',')
            self.actuate(self.action)
            self.transit_state()
            obs_time = rospy.get_time()  # starting timestamp of control delay
            rwd,terminal = self.reward_model(self.state)

            self.collided = (self.state.event==1)
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

            sars = {
                    'cur_state':self.pre_state.clone(),
                    'action':self.action.copy(),
                    'reward':rwd, 
                    'terminal':terminal,
                    'next_state':self.state.clone()
            }
            self.training_q.put(sars)
            self.pre_state.copy(self.state)  # deepcopy
            self.end_of_episode = (terminal or self.step>self.max_step)
            if not self.end_of_episode:
                if self.replay_buffer.size() < self.min_repbuf_size:
                    self.action[0][0] = rand.uniform(-1,1)
                else:
                    self.action = self.actor.predict(
                        self.state.obs_in(),
                        self.state.vel_in(),
                        self.state.dir_in()
                    ) 
                    self.action[0][0] += self.ornstein_uhlenbeck_noise()
                    self.action[0][0] = max(-1, min(1, self.action[0][0]))
                self.step += 1
                # self.interact_rate.sleep()
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
                self.action=np.zeros(
                    shape=[1,1],
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
                            + colldist + ','
                            + delay_time
                            + "\n")
                if self.model_saver and (not self.collided) and terminal:
                    self.model_saver.save(
                        self.sess, self.model_save_path + "/"
                        + datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + "/"
                        + "model"
                    )
                self.step = 0
                self.eprwd = 0
                self.episode += 1
                traj = ""
                vel = ""
                colldist = ""
                delay_time = ""

    def train_seq(self):
        obs_time = time.time()
        delay_time = ''
        while not rospy.is_shutdown():
            if self.replay_buffer.size() > self.min_repbuf_size:
                batch = self.replay_buffer.sample(
                    batch_size=self.training_batch_size
                )
                next_act = self.actor.predict_target(
                    obs_batch=batch['next_obs'],
                    vel_batch=batch['next_vel'],
                    dir_batch=batch['next_dir']
                )
                next_q = self.critic.predict_target(
                    obs_batch=batch['next_obs'],
                    vel_batch=batch['next_vel'],
                    dir_batch=batch['next_dir'],
                    act_batch=[next_act]
                )
                tar_q = batch['reward']+0.99*(next_q*(1-batch['terminal']))
                self.critic.update_master_network(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir'],
                    act_batch=batch['action'].reshape(
                        (1,self.training_batch_size,1)
                    ),
                    tar_q_batch=tar_q[0]
                )
                act_out = self.actor.predict(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir']
                )
                act_grads = self.critic.action_gradients(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir'],
                    act_batch=[act_out]
                )
                self.actor.update_master_network(
                    obs_batch=batch['cur_obs'],
                    vel_batch=batch['cur_vel'],
                    dir_batch=batch['cur_dir'],
                    act_grads=act_grads[0]
                )
                self.actor.copy_master_network()
                self.critic.copy_master_network()
                self.actor.update_target_network()
                self.critic.update_target_network()
                rospy.sleep(rand.uniform(0, 0.2))  # simulate training time
            else:
                print("%s replay_buffer size: %d/%d, epslion: %f" %(
                        self.name,
                        self.replay_buffer.size(),
                        self.min_repbuf_size,
                        self.epslion
                    )
                )
            # explore environment
            self.actuate(self.action)
            delay_time += (str(rospy.get_time()-obs_time)+',')
            self.transit_state()
            obs_time = rospy.get_time()
            rwd,terminal = self.reward_model(self.state)
            sars = {
                    'cur_state':self.pre_state.clone(),
                    'action':self.action.copy(),
                    'reward':rwd, 
                    'terminal':terminal,
                    'next_state':self.state.clone()
            }
            self.replay_buffer.add(sars)
            self.pre_state.copy(self.state)  # deepcopy
            self.end_of_episode = (terminal or self.step>self.max_step)
            if not self.end_of_episode:
                if self.replay_buffer.size() < self.min_repbuf_size:
                    self.action[0][0] = rand.uniform(-1,1)
                else:
                    self.action = self.actor.predict(
                        self.state.obs_in(),
                        self.state.vel_in(),
                        self.state.dir_in()
                    )  # 1x1
                    self.action[0][0] += self.ornstein_uhlenbeck_noise()
                    self.action[0][0] = max(-1, min(1, self.action[0][0]))
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
                self.action=np.zeros(
                    shape=[1,1],
                    dtype=float
                )
                self.log.write(delay_time+'\n')
                delay_time = ''
                self.episode += 1
                self.step = 0

    def evaluate(self):
        if self.training:
            raise NameError("worker.evaluate: the worker is in training mode...")
            return
        traj = ""
        vel = ""
        colldist = ""
        delay_time = ""
        succ = 0
        ep_rwd = 0
        loss = []
        actions = []
        self.end_of_episode = False
        self.step = 0
        self.action = np.zeros(
            shape=[1,1],
            dtype=float
        )
        self.state.reset(
            goal=self.goal.copy(),
            pose=self.init_pose.copy()
        )
        self.pre_state.reset(
            goal=self.goal.copy(),
            pose=self.init_pose.copy()
        )
        obs_time = 0.0
        while not rospy.is_shutdown() and not self.end_of_episode:
            while rospy.get_time()-obs_time<self.delta_t:
                continue
            delay_time += (str(rospy.get_time()-obs_time)+',')
            self.actuate(self.action)
            actions.append(self.action[0][0])
            self.transit_state()
            obs_time = rospy.get_time()
            rwd, terminal = self.reward_model(self.state)

            self.collided = (self.state.event==1)
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

            sars = {
                    'cur_state':self.pre_state.clone(),
                    'action':self.action.copy(),
                    'reward':rwd, 
                    'terminal':terminal,
                    'next_state':self.state.clone()
            }
            self.pre_state.copy(self.state)
            cur_q=self.critic.predict(
                obs_batch=sars['cur_state'].obs_in(),
                vel_batch=sars['cur_state'].vel_in(),
                dir_batch=sars['cur_state'].dir_in(),
                act_batch=sars['action'].reshape(
                    (1,1,1)
                )
            )

            self.end_of_episode=terminal or self.step>self.max_step
            if not self.end_of_episode:
                self.action = self.actor.predict(
                    self.state.obs_in(),
                    self.state.vel_in(),
                    self.state.dir_in()
                )  # 1x1
                next_q = self.critic.predict_target(
                    obs_batch=sars['next_state'].obs_in(),
                    vel_batch=sars['next_state'].vel_in(),
                    dir_batch=sars['next_state'].dir_in(),
                    act_batch=[self.action]
                )
                tar_q = sars['reward']+0.99*(next_q[0][0]*(1-sars['terminal']))
                loss.append((tar_q-cur_q[0][0])**2)
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
                            + colldist + ","
                            + delay_time
                            + "\n")
                self.step = 0
                self.eprwd = 0
                self.episode += 1
                traj = ""
                vel = ""
                colldist = ""
                delay_time = ""
                

    def actuate(self,action):
        velcmd=Twist()
        velcmd.linear.x = self.lvel
        velcmd.linear.y = 0.0
        velcmd.linear.z = 0.0
        velcmd.angular.x = 0.0
        velcmd.angular.y = 0.0
        velcmd.angular.z = action[0][0]*self.max_avel
        self.velcmd_pub.publish(velcmd)

    def reward_model(self,state):
        if self.reached_goal(state):
            return 1.0, True
        elif state.event==0:
            pre_goal_dist = np.linalg.norm(
                self.pre_state.pose[0,0:3]-self.goal
            )
            cur_goal_dist = np.linalg.norm(
                self.state.pose[0,0:3]-self.goal
            )
            goal_rwd= max(
                0.0, 
                (pre_goal_dist-cur_goal_dist)
            )
            #ossc_rwd= -0.05*np.linalg.norm(state.vel[0,3:6])
            rwd = goal_rwd#+ossc_rwd
            return rwd, False
        elif state.event==1:
            return -1.0, True
        else:
            raise NameError(
                self.name
                + " reward_model: invalide event-"
                + str(feedback.reward)
            )

    def reached_goal(self, state):
        dist=np.linalg.norm(
            state.pose[0,0:3]-self.goal[0,:]
        )
        return dist <= 1.0


    def closer_to_goal(
        self, 
        pre_state, 
        state
    ):
        pre_goal_dist = np.linalg.norm(
            pre_state.pose[0,0:3]-self.goal
        )
        cur_goal_dist = np.linalg.norm(
            state.pose[0,0:3]-self.goal
        )
        return pre_goal_dist-cur_goal_dist

    def feedback_receiver(self,msg):
        feedback=Feedback()
        feedback.load(msg.data)
        feedback.set_sender(self.name)
        self.feedback_q.put(feedback)

    def transit_state(self):
        rospy.sleep(self.state_trans_delay)
        self.load_lidar_scan=True
        self.load_odometry=True
        rate=rospy.Rate(100)
        while self.load_odometry or self.load_lidar_scan:
            rate.sleep()

    def lidar_receiver(self,msg):
        if self.load_lidar_scan:
            ranges = np.asarray(msg.ranges)
            for i in xrange(len(ranges)):
                if ranges[i]==float('inf'):
                    ranges[i] = 1.0
                else:
                    ranges[i] /= self.lidar_max_range
            self.state.update_obsseq_event(ranges)
            self.load_lidar_scan = False

    def odom_receiver(self,msg):
        if self.load_odometry:
            vel = np.zeros([1,6], dtype=float)
            vel[0][0] = msg.twist.twist.linear.x
            vel[0][1] = msg.twist.twist.linear.y
            vel[0][2] = msg.twist.twist.linear.z
            vel[0][3] = msg.twist.twist.angular.x
            vel[0][4] = msg.twist.twist.angular.y
            vel[0][5] = msg.twist.twist.angular.z
            pose = np.zeros([1,6],dtype=float)
            r, p, y = rostf.transformations.euler_from_quaternion(
                [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w
                ]
            )
            pose[0][0] = msg.pose.pose.position.x
            pose[0][1] = msg.pose.pose.position.y
            pose[0][2] = msg.pose.pose.position.z
            pose[0][3] = r
            pose[0][3] = p
            pose[0][3] = y
            self.state.update_vel_dir_pose(
                vel,
                pose,
                self.goal
            )
            self.load_odometry=False

    def reset_pose(self):
        stop=Twist()
        time=rospy.get_time()
        while rospy.get_time()-time<2:
            self.velcmd_pub.publish(stop)
            self.stop_rate.sleep()
        modelstate = ModelState()
        modelstate.model_name = self.name
        modelstate.reference_frame = "world"
        x,y,z,w = rostf.transformations.quaternion_from_euler(
            self.init_pose[0][3], 
            self.init_pose[0][4], 
            self.init_pose[0][5]
        )
        modelstate.pose.position.x = self.init_pose[0][0]
        modelstate.pose.position.y = self.init_pose[0][1]
        modelstate.pose.position.z = self.init_pose[0][2]
        modelstate.pose.orientation.x = x
        modelstate.pose.orientation.y = y
        modelstate.pose.orientation.z = z
        modelstate.pose.orientation.w = w
        reps=self.reset_client(modelstate)

    def terminate(self):
        if not rospy.is_shutdown():
            rospy.signal_shutdown(
                "training is completed..."
            )
        self.env_worker.join()
        self.train_worker.join()

    def start_train_dec(self):
        self.env_worker.start()
        self.train_worker.start()

    def start_train_seq(self):
        self.train_seq_worker.start()

    def epslion_greedy(self):
        self.epslion_t += 1
        if self.replay_buffer.size()>self.min_repbuf_size:
            self.epslion = max(
                self.epslion*0.95,
                0.05
            )
        return rand.uniform(0,1)>self.epslion

    def ornstein_uhlenbeck_noise(self):
        sigma = 0.1  # Standard deviation.
        mu = 0.  # Mean.
        tau = .05  # Time constant.
        dt = .01  # Time step.
        sigma_bis = sigma * np.sqrt(2. / tau)
        sqrtdt = np.sqrt(dt)
        self.ou_noise = (self.ou_noise
                        + dt*(-(self.ou_noise-mu)/tau)
                        + sigma_bis*sqrtdt*np.random.randn())
        return self.ou_noise




