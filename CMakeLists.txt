cmake_minimum_required(VERSION 3.0.2)
project(yolo_trt_ros)

find_package(OpenCV 4.2.0 REQUIRED
    	PATHS /usr/local
	NO_DEFAULT_PATH	
)

find_package(CUDA REQUIRED)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/package.xml ${CMAKE_CURRENT_SOURCE_DIR}/package.xml COPYONLY)

find_package(catkin REQUIRED COMPONENTS
  image_transport
  rospy
  sensor_msgs
  vision_msgs
  cv_bridge
  std_msgs
  message_generation
)

add_message_files(FILES
  Detector2D.msg
  Detector2DArray.msg
  BoundingBox2D.msg
  ObjectHypothesis.msg
)

catkin_python_setup()

generate_messages(DEPENDENCIES std_msgs vision_msgs)
	 
catkin_package(
  CATKIN_DEPENDS 
  nodelet rospy 
  image_transport 
  sensor_msgs 
  #vision_msgs 
  cv_bridge 
  message_runtime
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# enable c++11 (TensorRT requirement)
include_directories(${catkin_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_definitions(-DROS1)

catkin_install_python(PROGRAMS trt_yolo_v3_batch.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
