tum_simulator ported to Kinetic and Gazebo 7 (originally for Indigo)
=============

These packages are used to simulate the flying robot Ardrone in ROS environment using gazebo simulator. Totally they are 4 packages. Their functions are descript as below:

1. cvg_sim_gazebo: contains object models, sensor models, quadrocopter models, flying environment information and individual launch files for each objects and pure environment without any other objects.

2. cvg_sim_gazebo_plugins: contains gazebo plugins for the quadrocopter model. quadrotor_simple_controller is used to control the robot motion and deliver navigation information, such as: /ardrone/navdata. Others are plugins for sensors in the quadrocopter, such as: IMU sensor, sonar sensor, GPS sensor.

3. message_to_tf: is a package used to create a ros node, which transfers the ros topic /ground_truth/state to a /tf topic.

4. cvg_sim_msgs: contains message forms for the simulator.

Some packages are based on the tu-darmstadt-ros-pkg by Stefan Kohlbrecher, TU Darmstadt.

This package depends on ardrone_autonomy package and gazebo7 so install these first.

How to install the simulator:

1. Install gazebo7 and ardrone_autonomy package

2. Create a workspace for the simulator

    ```
    mkdir -p ~/ardrone_simulator/src
    cd  ~/ardrone_simulator/src
    catkin_init_workspace
    ```
2. Download package

    ```
    git clone https://github.com/iolyp/ardrone_simulator_gazebo7
    ```
3. Build the simulator

    ```
    cd ..
    catkin_make
    ```
4. Source the environment

    ```
    source devel/setup.bash
    ```
How to run a simulation:

1. Run a simulation by executing a launch file in cvg_sim_gazebo package:

    ```
    roslaunch cvg_sim_gazebo ardrone_testworld.launch
    ```

How to run a simulation using ar_track_alvar tags:

1. Move the contents of  ~/ardrone_simulator/src/cvg_sim_gazebo/meshes/ar_track_alvar_tags/ to  ~/.gazebo/models

2. Run simulation

    ```
    roslaunch cvg_sim_gazebo ar_tag.launch
    ```

Notes for spawning multiple ardrone in the envrionment.

The original tum_simulator is designed for signle UAV control. If multiple UAV is inserted into the envrionment, the ros topics are wired togehter. In the revised version, each UAV is spawned in its own namespace. In this case, the topic advertised in the UAV simulator will be isollated in the namespace, including the topics adversed in sensor plugins and the urdf files. However, the subscribed topics are not autmatically isolated. Therefore, they need to be manually prefixed with the robot_namespace. The original code is the tum_simulater ported to Kinetics and it is modified based on the revised Indigo code hosted on github. In addition to the changes in the Indigo tum_simulator, I have added the prefix for all the inpropperiate subcripted topics (particularly among the state_controller, simple_controller and the sensors plugins). The camera sensors (front and bottom) are removed from the crrent implimentation, since the they will not be used for mapping and collision avoidance. Otherwise, the computer vision part can be time-consumming. 

tum_simulator ported to Kinetics: https://github.com/angelsantamaria/tum_simulator
Indigo tum_simulator modified for multi-uav: https://github.com/basti35/tum_simulator/tree/5f758305d42a9f9e6fdd7447b3f494e5085486d7

LiDar sensors need to be added to each UAV for both mapping and collision avoidance



























