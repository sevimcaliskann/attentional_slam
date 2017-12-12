# CMakeLists.txt for VOCUS 2 saliency system (Frintrop et al. CVPR 2015)
#
# see http://www.cmake.org/ for a documentation about CMake
#
# compile your programm using the following commands:
# $ cmake CMakeLists.txt
# $ make

cmake_minimum_required(VERSION 3.0.2)

project(vocus2)

add_compile_options(-std=c++11)

# choose if you want to compile with or without debugging flags,
# uncomment only one of the following two commands at a time
#set(CMAKE_BUILD_TYPE release)
#set(CMAKE_BUILD_TYPE debug)

# add a compiler flag for the C++2011 standard
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -fopenmp")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x")

find_package(OpenCV)
find_package(Boost COMPONENTS serialization)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf
  geometry_msgs
  image_transport
  cv_bridge
  std_msgs
  dynamic_reconfigure
)


generate_dynamic_reconfigure_options(
  cfg/vocus_params.cfg
  #...
)


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

catkin_package()

add_library( vocus2l src/VOCUS2.cpp)
#add_dependencies(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})


link_directories(/home/sevim/catkin_ws/devel/lib/vocus2)
add_executable(vocus2 src/main.cpp src/VOCUS2.cpp)
#add_dependencies(vocus2 ${PROJECT_NAME}_gencfg)
add_dependencies(vocus2 ${PROJECT_NAME})
target_link_libraries(vocus2 ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})



