cmake_minimum_required(VERSION 3.5)
project(dis_ffm)
set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(SOURCE_FILES
    main.cpp ffm.h data.h)

add_executable(dis_ffm ${SOURCE_FILES})