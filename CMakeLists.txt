cmake_minimum_required(VERSION 3.10)

project(paralax)

find_package(OpenCV REQUIRED)
add_subdirectory(cilindr)

add_executable(paralax paralax.cpp)
include_directories(paralax ${OpenCV_INCLUDE_DIRS})

target_include_directories(paralax PUBLIC
                          "${PROJECT_BINARY_DIR}"
                          "${PROJECT_SOURCE_DIR}/cilindr")

target_link_libraries(paralax cilindr ${OpenCV_LIBS})