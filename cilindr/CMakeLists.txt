cmake_minimum_required(VERSION 3.10)

find_package(OpenCV REQUIRED)

add_library(cilindr cilindr.h cilindr.cpp)
include_directories(cilindr ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cilindr ${OpenCV_LIBS})
