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


find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)

# choose if you want to compile with or without debugging flags,
# uncomment only one of the following two commands at a time
#set(CMAKE_BUILD_TYPE release)
#set(CMAKE_BUILD_TYPE debug)

find_package(OpenCV)
find_package(Boost COMPONENTS serialization)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  tf
  geometry_msgs
  std_msgs
  image_transport
  cv_bridge
  std_msgs
  dynamic_reconfigure
  nodelet
)


generate_dynamic_reconfigure_options(
  cfg/vocus_params.cfg
)


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
include_directories(${catkin_INCLUDE_DIRS})
link_directories(${catkin_INCLUDE_DIRS})


catkin_package(
  #INCLUDE_DIRS include
)
add_library( vocus2l src/VOCUS2.cpp)
add_library( integrate_and_fire_nodelet src/integrate_and_fire_nodelet.h src/integrate_and_fire_nodelet.cpp)
#add_dependencies(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})


link_directories(/home/sevim/catkin_ws/devel/lib/vocus2)
add_executable(vocus2 src/main.cpp src/VOCUS2.cpp)
#add_dependencies(vocus2 ${PROJECT_NAME}_gencfg)
add_executable(integrate_and_fire_node src/integrate_and_fire_node.cpp)


target_link_libraries(vocus2 ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} vocus2l)
target_link_libraries(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(integrate_and_fire_node ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} integrate_and_fire_nodelet)
target_link_libraries(integrate_and_fire_nodelet ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES})


# COMPILE CU FILES
file(GLOB CUDA_FILES "src/" *.cu)
list( APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30; -std=c++11")
CUDA_COMPILE(CU_O ${CUDA_FILES})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
cuda_add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp ${CU_O})
target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} vocus2l)
