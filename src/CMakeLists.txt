# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(detectUav)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall -s")  # -s选项自动strip符号

if (DEFINED ENV{DDK_PATH})
    set(INC_PATH $ENV{DDK_PATH})
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(INC_PATH $ENV{ASCEND_TOOLKIT_HOME})
elseif(DEFINED ASCEND_TOOLKIT_HOME)
    set(INC_PATH ${ASCEND_TOOLKIT_HOME})
else()
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()
message(STATUS "set INC_PATH: ${INC_PATH}")

if (DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH $ENV{NPU_HOST_LIB})
elseif(DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(LIB_PATH "${INC_PATH}/runtime/lib64")
elseif(DEFINED ASCEND_TOOLKIT_HOME)
    set(LIB_PATH "${ASCEND_TOOLKIT_HOME}/runtime/lib64")
else()
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64")
endif()
message(STATUS "set LIB_PATH: ${LIB_PATH}")


add_definitions(-DENABLE_DVPP_INTERFACE)
list(APPEND COMMON_DEPEND_LIB avcodec avformat avutil swresample swscale)
aux_source_directory(${PROJECT_SOURCE_DIR}/../common/src aclLite)
list(REMOVE_ITEM aclLite ${PROJECT_SOURCE_DIR}/../common/src/CameraCapture.cpp)

# Header path
include_directories(
        $ENV{SYSROOT}/usr/include/
        $ENV{SYSROOT}/usr/include/jsoncpp/
        ${INC_PATH}/runtime/include/
        ${INC_PATH}/runtime/include/acl/media/
        ../inc/
        ../common/include/
)

find_package(Freetype REQUIRED)

# Enable live555 RTSP streaming (default ON). Requires toolchain to define LIVE555_* paths
option(USE_LIVE555 "Use live555 for RTSP streaming" ON)
if(USE_LIVE555)
    add_definitions(-DUSE_LIVE555)
    include_directories(${LIVE555_INCLUDE_DIRS})
    message(STATUS "USE_LIVE555=ON (includes: ${LIVE555_INCLUDE_DIRS})")
else()
    message(STATUS "USE_LIVE555=OFF (FFmpeg RTSP path)")
endif()

# add host lib path
# 优先使用指定的FFmpeg库路径
link_directories(
        $ENV{SYSROOT}/usr/local/Ascend/thirdpart/aarch64/lib
        ${INC_PATH}/runtime/lib64/stub
        ${INC_PATH}/runtime/lib64
        ${INC_PATH}/thirdpart/lib
)

if(USE_LIVE555)
    # live555 libraries directory from toolchain
    link_directories(${LIVE555_LIBRARY_DIRS})
    # add live555 streamer source
    list(APPEND LIVE555_SRC pushrtsp/Live555Streamer.cpp)
endif()

add_executable(main
        dataInput/dataInput.cpp
        detectPreprocess/detectPreprocess.cpp
        detectInference/detectInference.cpp
        detectPostprocess/detectPostprocess.cpp
        dataOutput/dataOutput.cpp
        pushrtsp/pictortsp.cpp
        pushrtsp/pushrtspthread.cpp
        tracking/tracking.cpp
        hdmiOutput/hdmiOutputThread.cpp
        ${LIVE555_SRC}
        main.cpp)

target_sources(main 
    PUBLIC
        ${aclLite})

target_link_libraries(main ascendcl acl_dvpp acl_dvpp_mpi acl_vo_mpi acl_hdmi_mpi stdc++ pthread ${COMMON_DEPEND_LIB} jsoncpp opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio dl rt X11 Freetype::Freetype)

if(USE_LIVE555)
    # live555 depends on OpenSSL crypto
    target_link_libraries(main ${LIVE555_LIBRARIES} crypto ssl)
endif()

add_executable(test_hdmi_output
        hdmiOutput/hdmiOutputThread.cpp
        test_hdmi_output.cpp)

target_sources(test_hdmi_output 
    PUBLIC
        ${aclLite})

target_link_libraries(test_hdmi_output ascendcl acl_dvpp acl_dvpp_mpi acl_vo_mpi acl_hdmi_mpi stdc++ pthread ${COMMON_DEPEND_LIB} jsoncpp opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio dl rt X11 Freetype::Freetype)

add_executable(test_mixformerv2_om
    tracking/tracking.cpp
        test_mixformerv2_om.cpp)

target_sources(test_mixformerv2_om 
    PUBLIC
        ${aclLite})

target_link_libraries(test_mixformerv2_om ascendcl acl_dvpp acl_dvpp_mpi stdc++ pthread ${COMMON_DEPEND_LIB} jsoncpp opencv_highgui opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio dl rt X11 Freetype::Freetype)

if(USE_LIVE555)
    # live555 depends on OpenSSL crypto
    target_link_libraries(test_mixformerv2_om ${LIVE555_LIBRARIES} crypto ssl)
endif()

install(TARGETS test_mixformerv2_om DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(TARGETS test_hdmi_output DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
