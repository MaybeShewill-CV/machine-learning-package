cmake_minimum_required(VERSION 3.9)
project(road_width_calculator)

set(CMAKE_CXX_STANDARD 11)

set(MAIN_SOURCE_FILE main.cpp)
aux_source_directory(./src DIR_SRCS)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
include_directories(${OpenMP_INCLUDE_DIRS})

SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -g -Wall")

find_package(Boost REQUIRED)
if(!Boost_FOUND)
    message("Boost NOT FOUND")
endif()
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

find_package( OpenCV REQUIRED )
if(!Opencv_FOUND)
    message("OpenCV NOT FOUND")
endif()
include_directories(${Opencv_INCLUDE_DIRS})
link_directories(${Opencv_LIBRARY_DIRS})

include_directories (
        /usr/include
        /usr/local/include
        ./include
        ./include/rapidjson
        ./include/mitab
)

add_executable(task_server ${MAIN_SOURCE_FILE} ${DIR_SRCS})

target_link_libraries(task_server
        ${OpenCV_LIBS}
        libboost_regex.so
        libboost_filesystem.so
        libboost_system.so
        libglog.so
        libsqlite3.so
        libmitab.so
        )