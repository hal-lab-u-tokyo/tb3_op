#include <stdio.h>
#include <iostream>
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/int32.hpp"
#include <array>
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include <string>
#include <fstream>

typedef struct {
  double ranges[360];
} ScanDataType;

static ScanDataType scan_data;
bool is_stop = false;

bool is_forward = true;
float S2R_rot = 2.42;

static void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  int i;
  for (i = 0; i < 360; i++) {
    scan_data.ranges[i] = msg->ranges[i];
  }
  return;
}

static void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg) {
  int i=0;
  return;
}

static geometry_msgs::msg::Twist cmd_vel;

rclcpp::Clock system_clock(RCL_ROS_TIME);
double time_list[2];

static float get_forward_distance(void) {
  int i;
  float min = 100.0f;
  for (i = 0; i < 15; i++) {
    if (scan_data.ranges[i] < min) {
      min = scan_data.ranges[i];
    }
  }
  for (i = (360 - 15); i < 360; i++) {
    if (scan_data.ranges[i] < min) {
      min = scan_data.ranges[i];
    }
  }
  // printf("forward: %lf\n", min);
  return min;
}

static float get_right_distance(void) {
  int i;
  float min = 100.0f;
  for (i = (90 - 30); i < (90 + 30); i++) {
    if (scan_data.ranges[i] < min) {
      min = scan_data.ranges[i];
    }
  }
  // printf("right: %lf\n", min);
  return min;
}

static bool do_forward(void) {
  is_stop = false;
  cmd_vel.linear.x = 0.1;
  //real_cmd_vel.linear.x = 0.2;

  return is_stop;
}

static bool do_stop(void) {
  is_stop = true;
  cmd_vel.linear.x = 0.0;
  //real_cmd_vel.linear.x = 0.0;
  cmd_vel.angular.z = 0.0;
  //real_cmd_vel.angular.z = 0.0;

  return is_stop;
}

static bool turn_right(void) {
  bool is_stop = false;
  // -0.55で現実は90度
  // Unityだと約40度
  cmd_vel.angular.z = -0.55f * S2R_rot ;
  return is_stop;
}

static void move_for_3sec(void) {
  auto node = rclcpp::Node::make_shared("timer");
  if (is_stop == false){
      time_list[1] = system_clock.now().seconds();
  }

  if (time_list[1] - time_list[0] < 3.0){
    //(void)turn_right();
    (void)do_forward();
  }
  else{
    if (is_stop == false){
      printf("%lf\n", time_list[1] - time_list[0] );
    }
    (void)do_stop();
  }
  
  return;
}

static void move_square(void) {
  auto node = rclcpp::Node::make_shared("timer");
  time_list[1] = system_clock.now().seconds();

  if (time_list[1] - time_list[0] < 3.0){
    if (is_forward){
      (void)do_forward();
    }
    else{
      (void)turn_right();
    }
  }
  else{
    printf("forward_time:%lf\n", time_list[1] - time_list[0] );
    time_list[0] = system_clock.now().seconds();
    is_forward = !is_forward;
    (void)do_stop();
  }
  
  return;
}

using namespace std::chrono_literals;

int main(int argc, char **argv) {
  static geometry_msgs::msg::Twist real_cmd_vel;
  char buffer[3][4096];

  if (argc > 1) {
    sprintf(buffer[0], "%s_tb3_node", argv[1]);
    sprintf(buffer[1], "%s_cmd_vel", argv[1]);
    sprintf(buffer[2], "%s_scan", argv[1]);
    printf("START: %s\n", argv[1]);
  }
  else {
    sprintf(buffer[0], "tb3_node");
    sprintf(buffer[1], "cmd_vel");
    sprintf(buffer[2], "scan");
    printf("START\n");
  }
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(buffer[0]);
  auto real_publisher =
      node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
  auto publisher =
      node->create_publisher<geometry_msgs::msg::Twist>(buffer[1], 10);
  auto subscriber = node->create_subscription<sensor_msgs::msg::LaserScan>(
      buffer[2], 1, scanCallback);
  auto cmd_subscriber = node->create_subscription<geometry_msgs::msg::Twist>(
      "cmd_vel", 1, cmdCallback);

  rclcpp::WallRate rate(10ms);
  time_list[0] = system_clock.now().seconds();
  time_list[1] = system_clock.now().seconds();

  while (rclcpp::ok()) {
    //move_for_3sec();
    move_square();
    // adjust here
    real_cmd_vel.linear.x = cmd_vel.linear.x;
    real_cmd_vel.angular.z = cmd_vel.angular.z / S2R_rot ;

    real_publisher->publish(real_cmd_vel);
    publisher->publish(cmd_vel);   

    rclcpp::spin_some(node);
    rate.sleep();
  }
  return 0;
}