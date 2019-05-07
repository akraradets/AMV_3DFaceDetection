cmake_minimum_required(VERSION 2.8.12)
project(detector)
#add_subdirectory(../dlib dlib_build)
add_executable(detector detector.cpp)
#target_link_libraries(detector dlib::dlib)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4.0.1 REQUIRED)

target_include_directories(detector PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(detector ${OpenCV_LIBS})