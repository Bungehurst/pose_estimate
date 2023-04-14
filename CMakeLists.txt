cmake_minimum_required(VERSION 3.0.2)

project(pose_estimate)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-std=c++14 -mfma -O3")

find_package(catkin REQUIRED COMPONENTS
    cmake_modules
    rosconsole 
    roscpp
)

find_package(OpenCV REQUIRED)

include_directories(
    include/${PROJECT_NAME}
    ${catkin_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS}
)

add_definitions(${EIGEN_DEFINITIONS})

catkin_package()

add_executable(pose_2d2d pose_2d2d.cpp)
target_link_libraries(pose_2d2d 
                        ${catkin_LIBRARIES} 
                        ${OpenCV_LIBRARIES} 
                        )
