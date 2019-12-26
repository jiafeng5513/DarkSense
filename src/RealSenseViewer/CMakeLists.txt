#EvisionSandbox:主程序

#设置cmake最小版本号
cmake_minimum_required(VERSION 3.13.0)

#查找RealSense
find_path(RS_SDK_DIR Realsense_SDK_Dir)
#设置RS相关的变量
set(RS_INCLUDE_DIR ${RS_SDK_DIR}/include/)
set(RS_LIB ${RS_SDK_DIR}/lib/x64/realsense2.lib)
set(RS_BIN_DIR ${RS_SDK_DIR}/bin/x64/)

message(STATUS "RealSense SDK status:")
message(STATUS "RS_INCLUDE_DIR:${RS_INCLUDE_DIR}")
message(STATUS "RS_LIB: ${RS_LIB}")
message(STATUS "RS_BIN_DIR: ${RS_BIN_DIR}")

# Find OpenCV
find_package(OpenCV REQUIRED)
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
# Add OpenCV headers location to your include paths
include_directories(${OpenCV_INCLUDE_DIRS} ${RS_INCLUDE_DIR})

#包含自定义组件
include_directories("../DarkNet/include/")
#设置工程包含当前目录
set(CMAKE_INCLUDE_CURRENT_DIR ON) 

LINK_DIRECTORIES(../install/lib)

#预处理器定义
add_definitions(-D UNICODE)
add_definitions(-D WIN32)
add_definitions(-D WIN64)
add_definitions(-D OPENCV)

#查找当前文件夹中的所有源代码文件
FILE(GLOB SRC_FILES "./*.cpp") 
#查找当前文件夹中所有的头文件
FILE(GLOB HEAD_FILES "./*.h") 

#文件整理
source_group("Source Files" FILES ${SRC_FILES})
source_group("Header Files" FILES ${HEAD_FILES})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/../install/lib/Debug)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/../install/lib/Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/../install/bin/Debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/../install/bin/Release)
#set(CMAKE_DEBUG_POSTFIX "_d")
#set_target_properties(${TARGET_NAME} PROPERTIES DEBUG_POSTFIX "_d")

#创建可执行文件工程
add_executable(RealSenseViewer ${SRC_FILES} ${HEAD_FILES})

#链接OpenCV
target_link_libraries(RealSenseViewer ${OpenCV_LIBS})
target_link_libraries(RealSenseViewer ${RS_LIB})
target_link_libraries(RealSenseViewer DarkNet)