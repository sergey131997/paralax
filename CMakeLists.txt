cmake_minimum_required(VERSION 3.10)

project(paralax)

find_package(OpenCV REQUIRED)

add_library(cilindr cilindr.cpp cilindr.h)
target_include_directories(cilindr PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(paralax paralax.cpp)
include_directories(paralax ${OpenCV_INCLUDE_DIRS})

target_link_libraries(paralax cilindr ${OpenCV_LIBS})