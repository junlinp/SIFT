cmake_minimum_required(VERSION 3.0)
project(sift)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
add_library(${PROJECT_NAME} sift.cc)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

add_executable(${PROJECT_NAME}_test sift_test.cc)
target_link_libraries(${PROJECT_NAME}_test PUBLIC ${PROJECT_NAME} GTest::GTest)
target_compile_definitions(sift_test
    PRIVATE -DTest_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
