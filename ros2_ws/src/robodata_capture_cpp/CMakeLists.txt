cmake_minimum_required(VERSION 3.16)
project(robodata_capture_cpp CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -O2)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(diagnostic_msgs REQUIRED)

add_executable(stamp_monitor src/stamp_monitor.cpp)
ament_target_dependencies(stamp_monitor rclcpp diagnostic_msgs)
install(TARGETS stamp_monitor DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
