//=================================================================================================
// Copyright (c) 2012, Johannes Meyer, TU Darmstadt
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of the Flight Systems and Automatic Control group,
//       TU Darmstadt, nor the names of its contributors may be used to
//       endorse or promote products derived from this software without
//       specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//=================================================================================================

#include <hector_gazebo_plugins/gazebo_ros_lidar.h>
#include "gazebo/common/Events.hh"
#include "gazebo/physics/physics.hh"
#include "gazebo/sensors/RaySensor.hh"

#include <limits>

namespace gazebo {

GazeboRosLidar::GazeboRosLidar()
{
}

////////////////////////////////////////////////////////////////////////////////
// Destructor
GazeboRosLidar::~GazeboRosLidar()
{
  sensor_->SetActive(false);
  event::Events::DisconnectWorldUpdateBegin(updateConnection);
  node_handle_->shutdown();
  delete node_handle_;
}

////////////////////////////////////////////////////////////////////////////////
// Load the controller
void GazeboRosLidar::Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
{
  // Get then name of the parent sensor
  sensor_ = std::dynamic_pointer_cast<sensors::RaySensor>(_sensor);
  if (!sensor_)
  {
    gzthrow("GazeboRosLiDar requires a Ray Sensor as its parent");
    return;
  }

  // Get the world name.
  std::string worldName = sensor_->WorldName();
  world = physics::get_world(worldName);

  // load parameters
  if (!_sdf->HasElement("robotNamespace"))
    namespace_.clear();
  else
    namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>() + "/";

  if (!_sdf->HasElement("frameId"))
    frame_id_ = "";
  else
    frame_id_ = _sdf->GetElement("frameId")->Get<std::string>();

  if (!_sdf->HasElement("topicName"))
    topic_ = "lidar_collision";
  else
    topic_ = _sdf->GetElement("topicName")->Get<std::string>();

  sensor_model_.Load(_sdf);

  ranges_.header.frame_id = frame_id_;
  ranges_.range_max = sensor_->RangeMax();
  ranges_.range_min = sensor_->RangeMin();

  // start ros node
  if (!ros::isInitialized())
  {
    int argc = 0;
    char** argv = NULL;
    ros::init(argc,argv,"gazebo",ros::init_options::NoSigintHandler|ros::init_options::AnonymousName);
  }

  node_handle_ = new ros::NodeHandle(namespace_);
  publisher_ = node_handle_->advertise<sensor_msgs::LaserScan>(topic_, 1);
  ROS_INFO("lidar sensor pub on %s",topic_.c_str());

  Reset();
  updateConnection = sensor_->LaserShape()->ConnectNewLaserScans(
        boost::bind(&GazeboRosLidar::Update, this));

  // activate RaySensor
  sensor_->SetActive(true);
}

void GazeboRosLidar::Reset()
{
  sensor_model_.reset();
}

////////////////////////////////////////////////////////////////////////////////
// Update the controller
void GazeboRosLidar::Update()
{
  common::Time sim_time = world->GetSimTime();
  double dt = (sim_time - last_time).Double();
//  if (last_time + updatePeriod > sim_time) return;

  // activate RaySensor if it is not yet active
  if (!sensor_->IsActive()) sensor_->SetActive(true);

  ranges_.header.stamp.sec  = (world->GetSimTime()).sec;
  ranges_.header.stamp.nsec = (world->GetSimTime()).nsec;

  // store the current lidar measurement
  ranges_.ranges.clear();//must reset the previously stored ranges
  int num_ranges = sensor_->LaserShape()->GetSampleCount() * sensor_->LaserShape()->GetVerticalSampleCount();
  for(int i = 0; i < num_ranges; ++i) {
    // add Gaussian noise (and limit to min/max range), the noise is caused by the motion of UAV
    // the sensor data consists of another small guassian noise <0,0.01>, sepecified in Gazebo urdf
    double error=sensor_model_.update(dt);
    double range = sensor_->LaserShape()->GetRange(i)+error;
    //ROS_INFO("%f,%f",error,dt);//DEBUG
    if (range < ranges_.range_min) range = ranges_.range_min;
    if (range > ranges_.range_max) range = ranges_.range_max;
    ranges_.ranges.push_back(range);
  }
  publisher_.publish(ranges_);

  // save last time stamp
  last_time = sim_time;
}

// Register this plugin with the simulator
GZ_REGISTER_SENSOR_PLUGIN(GazeboRosLidar)

} // namespace gazebo
