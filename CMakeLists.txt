cmake_minimum_required(VERSION 3.0)
project(sift)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
add_library(${PROJECT_NAME} sift.cc)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})