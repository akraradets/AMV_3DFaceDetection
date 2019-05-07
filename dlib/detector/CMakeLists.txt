cmake_minimum_required(VERSION 2.8.12)
project(detector)
add_subdirectory(../dlib dlib_build)
add_executable(detector detector.cpp)
target_link_libraries(detector dlib::dlib)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV 4.0.1 REQUIRED)

target_include_directories(detector PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(detector ${OpenCV_LIBS})



#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release

#################################################################################
#################################################################################
#  A CMakeLists.txt file can compile more than just one program.  So below we
#  tell it to compile the other dlib example programs using pretty much the
#  same CMake commands we used above.
#################################################################################
#################################################################################

macro(add_example name)
   add_executable(${name} ${name}.cpp)
   target_link_libraries(${name} dlib::dlib )
endmacro()

macro(add_gui_example name)
   if (DLIB_NO_GUI_SUPPORT)
      message("No GUI support, so we won't build the ${name} example.")
   else()
      add_example(${name})
   endif()
endmacro()

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      include_directories(${OpenCV_INCLUDE_DIRS})

   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()