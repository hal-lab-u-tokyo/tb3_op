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

int DISTANCE_SEARCH_WIDTH = 30; // degree, unused
bool started = false;
bool stopped = false;
std_msgs::msg::Float64MultiArray dis_arr;
double time_list[2];

typedef struct {
  double ranges[360];
} ScanDataType;

static geometry_msgs::msg::Twist cmd_vel;
static ScanDataType scan_data;

rclcpp::Clock system_clock(RCL_ROS_TIME);

using namespace std::chrono_literals;

static void write_csv(double* time_list){
  std::string filename = "col_time.csv";
  std::ofstream ofs(filename, std::ios::app);
  ofs <<(time_list[1] - time_list[0])<< std::endl;
  return;
}



static void go_forward(void) {
  cmd_vel.linear.x = 0.6f;
  cmd_vel.angular.z = -0.0f;
  return;
}

static void right_forward(void) {
  cmd_vel.linear.x = 0.2f;
  cmd_vel.angular.z = -1.0f;
  return;
}

static void right_rotation(void) {
  cmd_vel.linear.x = 0.0f;
  cmd_vel.angular.z = -1.0f;
  return;
}

static void left_forward(void) {
  cmd_vel.linear.x = 0.2f;
  cmd_vel.angular.z = 1.0f;
  return;
}

static void left_rotation(void) {
  cmd_vel.linear.x = 0.0f;
  cmd_vel.angular.z = 1.0f;
  return;
}

static void tb3_stop(void) {
  cmd_vel.linear.x = 0.0f;
  cmd_vel.angular.z = 0.0f;
  return;
}

static void tb3_backrecover(void) {
  cmd_vel.linear.x = -0.6f;
  cmd_vel.angular.z = 0.0f;
  return;
}


static void moveCallback(const std_msgs::msg::Int32::UniquePtr msg) {
  auto node = rclcpp::Node::make_shared("timer");
  RCLCPP_INFO(node->get_logger(), "sub: '%d'", msg->data);
  if (started == false){
     RCLCPP_INFO(node->get_logger(), "start_time: '%d'", msg->data);
     time_list[0] = system_clock.now().seconds();
     started = true;
  }
  if (msg->data != 99 && msg->data != 100 ){
    stopped = false;
  }


  if (msg->data == 0){
    go_forward();
  }
  else if (msg->data == 1){
    left_forward();
  }
  else if (msg->data == 2){
    right_forward();
  }
  else if (msg->data == 3){
    left_rotation();
  }
  else if (msg->data == 4){
    right_rotation();
  }
  else if (msg->data == 99){
    tb3_stop();
    if (stopped == false){
      printf("collision occured:recovering...\n");
      RCLCPP_INFO(node->get_logger(), "end_time: '%d'", msg->data);
      time_list[1] = system_clock.now().seconds();
      printf("%lf", time_list[1] - time_list[0]);
      write_csv(time_list);
      time_list[0] = time_list[1];
      RCLCPP_INFO(node->get_logger(), "start_time: '%d'", msg->data);
    }
    stopped = true;
  }
  else if (msg->data == 100){
    tb3_backrecover();
  }
  else{
    printf("cmd_dir is not collect value.");
  }

  return;
}


static void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  int i;
  for (i = 0; i < 360; i++) {
    scan_data.ranges[i] = msg->ranges[i];
  }
  return;
}



// static void topic_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
//  {
//     //printf("I heard : %lf\n", msg->linear.x);
//     return;
//  }

int main(int argc, char ** argv)
{
  (void) argc;
  (void) argv;
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

  printf("%s", buffer[1]);

  rclcpp::init(argc, argv);
  //buffer : storage for node name
  auto node = rclcpp::Node::make_shared(buffer[0]);
  auto vel_publisher =
      node->create_publisher<geometry_msgs::msg::Twist>(buffer[1], 1);
  auto real_publisher =
      node->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 1);
  auto dis_publisher =
      node->create_publisher<std_msgs::msg::Float64MultiArray>("dis_arr", 1);
      //node->create_publisher<std::array<float, 60>>("min_arr", 1);
      
      
  //auto subscriber = node->create_subscription<geometry_msgs::msg::Twist>(
  //  buffer[1], 1, topic_callback);
  auto dis_subscriber = node->create_subscription<sensor_msgs::msg::LaserScan>(
    buffer[2], 1, scanCallback);

  auto dir_subscriber = node->create_subscription<std_msgs::msg::Int32>(
    "cmd_dir", 1, moveCallback);

  rclcpp::WallRate rate(10ms);
  
  while (rclcpp::ok()) {
    int i;
    dis_arr.data.resize(360);
    //最小値を求める

    for (i = 0; i < 360; i++) {
      dis_arr.data[i] = scan_data.ranges[i];
    }
    static geometry_msgs::msg::Twist new_cmd_vel;
    new_cmd_vel.linear.x = 0.2 * cmd_vel.linear.x;
    new_cmd_vel.angular.z = 4.0f / 9.0f * cmd_vel.angular.z;

    vel_publisher->publish(cmd_vel);
    real_publisher->publish(new_cmd_vel);
    dis_publisher->publish(dis_arr);
    rclcpp::spin_some(node);
    rate.sleep();
  }
  return 0;
}
