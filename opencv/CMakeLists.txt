cmake_minimum_required(VERSION 2.8.12)
project(detector)
add_executable(detector detector.cpp)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4.0.1 REQUIRED)

target_include_directories(detector PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(detector ${OpenCV_LIBS})