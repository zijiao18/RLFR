## Introduction
This repository contains a collection of ROS nodes that implement various reinforcment learning methods for robotic contol. Reinforcment learning is a general framework which enables agents to autonomously solve control tasks through trials and errors. It works best in stationary environments, such as games. However, applying reinforcment learing to robotic control requires consideration of practical factors, such as control delay, motion and sensing uncertainties, and resource limits. Addressing those issues, the reinforcement learning methods implemented here learn control policies in a cononical robotic simulator (i.e., Gazebo) which realistically simulates the motion and sensing of robots. 

## Installation
The packages hosted in the repository requires [ROS Kinetic](http://wiki.ros.org/kinetic/Installation), [Gazebo 7](http://gazebosim.org/tutorials?tut=ros_installing#InstallGazebo) and [TensorFlow 1.8.0](https://www.tensorflow.org/install/pip?lang=python2#nav-buttons-1). Clicking on their names leads to their installation instructions. To be compatible with the ROS Kinetic distribution, the reinforcement learning methods implemented here are based on Python 2.7, and will support ROS2 and Python3 later. Please install all the dependencies before proceeding to the rest of the installation. 

1. Clone the repository to your local machine
```
cd <mydir>

git https://github.com/zijiao18/RLFR.git
```
2. Compile and install the packages
```
sudo apt-get install ros-kinetic-ardrone-autonomy

source /opt/ros/kinetic/setup.bash

cd <mydir>/RLFR/src

catkin_init_workspace

cd <mydir>/RLFR

catkin_make

source <mydir>/RLFR/devel/setup.bash
```
3. Run simulation through .launch files.
```
roslaunch <mydir>/RLFR/src/<package_name>/launch/<train_or_test>.launch
``` 
Note: before launching simulation, please be sure to setup 
- model_path: the location where a trained model (policy) is stored or loaded by TensorFlow; 
- log_path: the localtion of a .csv file which logs agent(s) during training or testing;
- tb_writer_path: the location of tf.summary.FileWriter logs read by tensorboard.
Those variables defined in the 
```
<mydir>/RLFR/src/<package_name>/scripts/<train_or_test>.py
```

## Package Description
The following packages implement reinforcement learning methods for mobile robot control. Those reinforcment learning methods are off-policy, and are exetuated in parrallel with robot simulation. Their implementations aware the time constrains for real-time control during training and testing.

- **sim_drone**: the simulator for AirDrone 2.0 with front and down facing cameras. This package is customized from the [original implementation](http://wiki.ros.org/tum_simulator), in order to support multi-robot simulation.
- **sim_rosbot**: the [official ROSbot simulator](https://github.com/husarion/rosbot_description). 
- **ddpg_drone**: the application of the DDPG algorithm to AirDrone 2.0 collision avoidance in cluttered environments. 
- **asyn_drone, asyn_rosbot**: the implementation of **_"Asynchronous Multitask Reinforcement Learning with Dropout for Continuous Control" (accepted to ICMLA2019)_** using the AirDrone 2.0 and ROSbot simulators.
- **maddpg_rosbot**: the implementation of **_"End-to-End Reinforcement Learning for Multi-Agent Continuous Control" (accepted to ICMLA2019)_** using the ROSbot simulator.

*To report issues, please contact zijiao18@gmail.com*
