cmake_minimum_required(VERSION 2.8)
project( TESTFILES )
set(CMAKE_CXX_FLAGS "-std=c++11")

set(CMAKE_BUILD_TYPE Release)

find_package( OpenCV 3.4 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)
include_directories( "/usr/include/eigen3" )

add_library(${PROJECT_NAME} SHARED
    src/JBF.cc
    src/Kernel.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
add_executable(main main.cpp)
target_link_libraries( main ${PROJECT_NAME})