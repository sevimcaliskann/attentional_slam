# CMakeLists.txt for VOCUS 2 saliency system (Frintrop et al. CVPR 2015)
#
# see http://www.cmake.org/ for a documentation about CMake
#
# compile your programm using the following commands:
# $ cmake CMakeLists.txt
# $ make

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(vocus2 LANGUAGES CXX CUDA)
include(CTest)

add_compile_options(-std=c++11)


find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)

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
target_compile_features(vocus2l PUBLIC cxx_std_11)
add_library( integrate_and_fire_nodelet src/integrate_and_fire_nodelet.h src/integrate_and_fire_nodelet.cpp)
target_compile_features(integrate_and_fire_nodelet PUBLIC cxx_std_11)
add_library( odometry_nodelet src/odometry_nodelet.h src/odometry_nodelet.cpp)
#add_dependencies(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})


link_directories(/home/sevim/catkin_ws/devel/lib/vocus2)
add_executable(vocus2 src/main.cpp src/VOCUS2.cpp)
target_link_libraries(vocus2l ${OpenCV_LIBS} ${Boost_LIBRARIES})
#add_dependencies(vocus2 ${PROJECT_NAME}_gencfg)
add_executable(integrate_and_fire_node src/integrate_and_fire_node.cpp)
add_executable(odometry_node src/odometry_node.cpp)


target_link_libraries(vocus2 ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} vocus2l)
target_link_libraries(integrate_and_fire_node ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} integrate_and_fire_nodelet)
target_link_libraries(odometry_node ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} odometry_nodelet)
target_link_libraries(integrate_and_fire_nodelet ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES})
target_link_libraries(odometry_nodelet ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES})

add_library(integrate STATIC
  src/integrate.cuh
  )

add_definitions(-D_MWAITXINTRIN_H_INCLUDED)
target_compile_features(integrate PUBLIC cxx_std_11)
set_target_properties(integrate PROPERTIES LINKER_LANGUAGE CUDA)
set_target_properties( integrate
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(integrate ${OpenCV_LIBS} ${catkin_LIBRARIES})



#add_library( vocus2l_cuda_ext src/cuda_ext/VOCUS2_cuda_ext.h src/cuda_ext/VOCUS2_cuda_ext.cpp)
#target_link_libraries(vocus2l_cuda_ext ${OpenCV_LIBS} ${Boost_LIBRARIES})
#cuda_add_executable(main ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_ext/main_cuda_ext.cpp ${CU_O})
#target_link_libraries(main ${OpenCV_LIBS} ${Boost_LIBRARIES} ${catkin_LIBRARIES} vocus2l_cuda_ext)
