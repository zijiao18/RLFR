/*
* quadrotor_state_controller:
*
* This software is a state control gazebo plugin for the Ardrone simulator
*
* It receives the joystick command and the current simulator information to generate corresponding information in rostopic /ardrone/navdata
*
* Created on: Oct 22, 2012
* Author: Hongrong huang
* 
* Edited on: Apr 21, 2019, by Zilong Jiao
* 
* The simulator is revised as an emulator for training reinforcement learning methods. 
* Modifications: 
*   - removed subscriber, publisher, and services related to camera
*   - access camera data through the ros-topics published by camera plugins
*   - restrict a robot to have only FLYING_MODEL for collision avoidance research
*/


#include <hector_quadrotor_controller/quadrotor_state_controller.h>
#include "gazebo/common/Events.hh"
#include "gazebo/physics/physics.hh"

#include <cmath>

namespace gazebo {

GazeboQuadrotorStateController::GazeboQuadrotorStateController()
{
  robot_current_state = INITIALIZE_MODEL;
  m_isFlying          = false;
  m_takeoff           = false;
  m_drainBattery      = false;//was true
  m_batteryPercentage = 100;
  m_maxFlightTime     = 1200;
  m_timeAfterTakeOff  = 0;
  //m_selected_cam_num  = 0;
  state_reset         = false;
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
GazeboQuadrotorStateController::~GazeboQuadrotorStateController()
{
  event::Events::DisconnectWorldUpdateBegin(updateConnection);

  node_handle_->shutdown();
  delete node_handle_;
}

////////////////////////////////////////////////////////////////////////////////
// Load the controller
void GazeboQuadrotorStateController::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  world = _model->GetWorld();
  model = _model;

  // load parameters
  if (!_sdf->HasElement("robotNamespace"))
    robot_namespace_.clear();
  else
    robot_namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>() + "/";

  if (!_sdf->HasElement("topicName"))
    velocity_topic_ = "cmd_vel";
  else
    velocity_topic_ = _sdf->GetElement("topicName")->Get<std::string>();

  if (!_sdf->HasElement("takeoffTopic"))
    takeoff_topic_ = "takeoff";
  else
    takeoff_topic_ = _sdf->GetElement("takeoffTopic")->Get<std::string>();

  if (!_sdf->HasElement("landTopic"))
    land_topic_ = "land";
  else
    land_topic_ = _sdf->GetElement("landTopic")->Get<std::string>();

  if (!_sdf->HasElement("resetTopic"))
    reset_topic_ = "reset";
  else
    reset_topic_ = _sdf->GetElement("resetTopic")->Get<std::string>();

  if (!_sdf->HasElement("navdataTopic"))
    navdata_topic_ = "navdata";
  else
    navdata_topic_ = _sdf->GetElement("navdataTopic")->Get<std::string>();

  if (!_sdf->HasElement("navdatarawTopic"))
    navdataraw_topic_ = "navdata_raw_measures";
  else
    navdataraw_topic_ = _sdf->GetElement("navdatarawTopic")->Get<std::string>();

  if (!_sdf->HasElement("imuTopic"))
    imu_topic_.clear();
  else
    imu_topic_ = _sdf->GetElement("imuTopic")->Get<std::string>();

  if (!_sdf->HasElement("sonarTopic"))
    sonar_topic_.clear();
  else
    sonar_topic_ = _sdf->GetElement("sonarTopic")->Get<std::string>();

  if (!_sdf->HasElement("lidarTopic"))
    lidar_topic_.clear();
  else
    lidar_topic_ = _sdf->GetElement("lidarTopic")->Get<std::string>();

  if (!_sdf->HasElement("stateTopic"))
    state_topic_.clear();
  else
    state_topic_ = _sdf->GetElement("stateTopic")->Get<std::string>();

  link =  _model->GetChildLink("base_link");

  if (!link)
  {
    ROS_FATAL("gazebo_ros_baro plugin error: bodyName: %s does not exist\n", link_name_.c_str());
    return;
  }

  node_handle_ = new ros::NodeHandle(namespace_);

  // subscribe command: velocity control command
  if (!velocity_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<geometry_msgs::Twist>(
      robot_namespace_+velocity_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::VelocityCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    velocity_subscriber_ = node_handle_->subscribe(ops);
  }

  // subscribe command: take off command
  if (!takeoff_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<std_msgs::Empty>(
      robot_namespace_+takeoff_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::TakeoffCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    takeoff_subscriber_ = node_handle_->subscribe(ops);
  }

  // subscribe command: take off command
  if (!land_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<std_msgs::Empty>(
      robot_namespace_+land_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::LandCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    land_subscriber_ = node_handle_->subscribe(ops);
  }

  // subscribe command: take off command
  if (!reset_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<std_msgs::Empty>(
      robot_namespace_+reset_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::ResetCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    reset_subscriber_ = node_handle_->subscribe(ops);
  }

  	// publish navdata and navdataraw
    m_navdataPub = node_handle_->advertise< ardrone_autonomy::Navdata >(robot_namespace_+navdata_topic_ , 25 );
    m_navdatarawPub = node_handle_->advertise< ardrone_autonomy::navdata_raw_measures >(robot_namespace_+navdataraw_topic_ , 25 );

  // subscribe imu
  if (!imu_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<sensor_msgs::Imu>(
      robot_namespace_+imu_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::ImuCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    imu_subscriber_ = node_handle_->subscribe(ops);
    ROS_INFO_NAMED("quadrotor_state_controller", "Using imu information on topic %s as source of orientation and angular velocity.", imu_topic_.c_str());
  }

  // subscribe sonar senor info
  if (!sonar_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<sensor_msgs::Range>(
      robot_namespace_+sonar_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::SonarCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    sonar_subscriber_ = node_handle_->subscribe(ops);

    ROS_INFO_NAMED("quadrotor_state_controller", "Using sonar information on topic %s as source of altitude.", sonar_topic_.c_str());
  }

  // subscribe state
  if (!state_topic_.empty())
  {
    ros::SubscribeOptions ops = ros::SubscribeOptions::create<nav_msgs::Odometry>(
      robot_namespace_+state_topic_, 1,
      boost::bind(&GazeboQuadrotorStateController::StateCallback, this, _1),
      ros::VoidPtr(), &callback_queue_);
    state_subscriber_ = node_handle_->subscribe(ops);
    ROS_INFO_NAMED("quadrotor_state_controller", "Using state information on topic %s as source of state information.", state_topic_.c_str());
  }

  robot_current_state = FLYING_MODEL; //was LANDED_MODEL
  Reset();

  // New Mechanism for Updating every World Cycle
  // Listen to the update event. This event is broadcast every
  // simulation iteration.
  updateConnection = event::Events::ConnectWorldUpdateBegin(
      boost::bind(&GazeboQuadrotorStateController::Update, this));

}

////////////////////////////////////////////////////////////////////////////////
// Callbacks
void GazeboQuadrotorStateController::VelocityCallback(const geometry_msgs::TwistConstPtr& velocity)
{
  velocity_command_ = *velocity;
  velcmd_time=world->GetSimTime();
  ROS_INFO("recieved cmd at %f", velcmd_time.Double());
}

void GazeboQuadrotorStateController::ImuCallback(const sensor_msgs::ImuConstPtr& imu)
{
  pose.rot.Set(imu->orientation.w, imu->orientation.x, imu->orientation.y, imu->orientation.z);
  euler = pose.rot.GetAsEuler();
  angular_velocity = pose.rot.RotateVector(math::Vector3(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z));
}

void GazeboQuadrotorStateController::SonarCallback(const sensor_msgs::RangeConstPtr& sonar_info)
{
  robot_altitude = sonar_info->range;
}

void GazeboQuadrotorStateController::StateCallback(const nav_msgs::OdometryConstPtr& state)
{
  math::Vector3 velocity1(velocity);

  if (imu_topic_.empty()) {
    pose.pos.Set(state->pose.pose.position.x, state->pose.pose.position.y, state->pose.pose.position.z);
    pose.rot.Set(state->pose.pose.orientation.w, state->pose.pose.orientation.x, state->pose.pose.orientation.y, state->pose.pose.orientation.z);
    euler = pose.rot.GetAsEuler();
    angular_velocity.Set(state->twist.twist.angular.x, state->twist.twist.angular.y, state->twist.twist.angular.z);
  }

  velocity.Set(state->twist.twist.linear.x, state->twist.twist.linear.y, state->twist.twist.linear.z);

  // calculate acceleration
  double dt = !state_stamp.isZero() ? (state->header.stamp - state_stamp).toSec() : 0.0;
  state_stamp = state->header.stamp;
  if (dt > 0.0) {
    acceleration = (velocity - velocity1) / dt;
  } else {
    acceleration.Set();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Update the controller
// The simulated uav is for collision avoidance and only has FLYING_MODEL
// Note: this is simplified from the original verion at 
//      (https://github.com/angelsantamaria/tum_simulator.git)
void GazeboQuadrotorStateController::Update()
{
  math::Vector3 force, torque;

  // Get new commands/state
  callback_queue_.callAvailable();

  // Get simulator time
  common::Time sim_time = world->GetSimTime();
  double dt = (sim_time - last_time).Double();
  // Update rate is 200/per second
  if (dt < 0.005) return;

  // Get Pose/Orientation from Gazebo (if no state subscriber is active)
  if (imu_topic_.empty()) {
    pose = link->GetWorldPose();
    angular_velocity = link->GetWorldAngularVel();
    euler = pose.rot.GetAsEuler();
  }
  if (state_topic_.empty()) {
    acceleration = (link->GetWorldLinearVel() - velocity) / dt;
    velocity = link->GetWorldLinearVel();
  }

  // Rotate vectors to coordinate frames relevant for control
  math::Quaternion heading_quaternion(cos(euler.z/2),0,0,sin(euler.z/2));
  math::Vector3 velocity_xy = heading_quaternion.RotateVectorReverse(velocity);
  math::Vector3 acceleration_xy = heading_quaternion.RotateVectorReverse(acceleration);
  math::Vector3 angular_velocity_body = pose.rot.RotateVectorReverse(angular_velocity);

  if( m_drainBattery && ((robot_current_state != LANDED_MODEL)||m_isFlying))
    m_batteryPercentage -= dt / m_maxFlightTime * 100.;

  ardrone_autonomy::Navdata navdata;
  navdata.batteryPercent = m_batteryPercentage;
  navdata.rotX = pose.rot.GetRoll() / M_PI * 180.;
  navdata.rotY = pose.rot.GetPitch() / M_PI * 180.;
  navdata.rotZ = pose.rot.GetYaw() / M_PI * 180.;
  if (!sonar_topic_.empty())
    navdata.altd = int(robot_altitude*1000);
  else
    navdata.altd = pose.pos.z * 1000.f;
  navdata.vx = 1000*velocity_xy.x;
  navdata.vy = 1000*velocity_xy.y;
  navdata.vz = 1000*velocity_xy.z;
  navdata.ax = acceleration_xy.x/10;
  navdata.ay = acceleration_xy.y/10;
  navdata.az = acceleration_xy.z/10 + 1;
  navdata.tm = ros::Time::now().toSec()*1000000; // FIXME what is the real drone sending here?


  navdata.header.stamp = ros::Time::now();
  navdata.header.frame_id = "ardrone_base_link";
  navdata.state = robot_current_state;
  navdata.magX = 0;
  navdata.magY = 0;
  navdata.magZ = 0;
  navdata.pressure = 0;
  navdata.temp = 0;
  navdata.wind_speed = 0.0;
  navdata.wind_angle = 0.0;
  navdata.wind_comp_angle = 0.0;
  navdata.tags_count = 0;
  
  if (state_reset)
  {
    Reset();
    navdata.state=RESET_MODEL;
    state_reset=false;
  }
  
  m_navdataPub.publish( navdata );

  ardrone_autonomy::navdata_raw_measures navdataraw;

  navdataraw.header.stamp = ros::Time::now();
  navdataraw.header.frame_id = "ardrone_base_link";
  navdataraw.drone_time = 0;
  navdataraw.tag = 0;
  navdataraw.size = 0;
  navdataraw.vbat_raw = 0;
  navdataraw.us_debut_echo = 0;
  navdataraw.us_fin_echo = 0;
  navdataraw.us_association_echo = 0;
  navdataraw.us_distance_echo = 0;
  navdataraw.us_courbe_temps= 0;
  navdataraw.us_courbe_valeur = 0;
  navdataraw.us_courbe_ref = 0;
  navdataraw.flag_echo_ini = 0;																																												
  navdataraw.nb_echo = 0;
  navdataraw.sum_echo = 0;
  if (!sonar_topic_.empty())
    navdataraw.alt_temp_raw = int(robot_altitude*1000);
  else
    navdataraw.alt_temp_raw = int(pose.pos.z * 1000.f);
  navdataraw.gradient = 0;

  m_navdatarawPub.publish( navdataraw );
  // save last time stamp
  last_time = sim_time;

}
////////////////////////////////////////////////////////////////////////////////
// Reset the controller
void GazeboQuadrotorStateController::Reset()
{
  // reset state
  pose.Reset();
  velocity.Set();
  angular_velocity.Set();
  acceleration.Set();
  euler.Set();
  state_stamp = ros::Time();
}

////////////////////////////////////////////////////////////////////////////////
// controller callback
void GazeboQuadrotorStateController::TakeoffCallback(const std_msgs::EmptyConstPtr& msg)
{
  if(robot_current_state == LANDED_MODEL)
  {
    m_isFlying = true;
    m_takeoff = true;
    m_batteryPercentage = 100.;
    ROS_INFO("%s","\nQuadrotor takes off!!");
  }
  else if(robot_current_state == LANDING_MODEL)
  {
    m_isFlying = true;
    m_takeoff = true;
    ROS_INFO("%s","\nQuadrotor takes off!!");
  }
}

void GazeboQuadrotorStateController::LandCallback(const std_msgs::EmptyConstPtr& msg)
{
  if((robot_current_state == FLYING_MODEL)||(robot_current_state == TO_FIX_POINT_MODEL)||(robot_current_state == TAKINGOFF_MODEL))
  {
    m_isFlying = false;
    m_takeoff = false;
    ROS_INFO("%s","\nQuadrotor lands!!");
  }

}

void GazeboQuadrotorStateController::ResetCallback(const std_msgs::EmptyConstPtr& msg)
{
  ROS_INFO("%s","\nReset quadrotor!!");
  state_reset=true;
}

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(GazeboQuadrotorStateController)

} // namespace gazebo
