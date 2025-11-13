# 目标系统
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# sysroot
set(CMAKE_SYSROOT $ENV{SYSROOT})

# 交叉编译器 - 使用 GCC 11 以匹配目标系统
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-11)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-11)

# 基本编译与链接参数（关键：--sysroot）
set(CMAKE_EXE_LINKER_FLAGS "--sysroot=/opt/ascend/sysroot -Wl,-rpath-link,/opt/ascend/sysroot/lib/aarch64-linux-gnu:/opt/ascend/sysroot/usr/lib/aarch64-linux-gnu -Wl,-dynamic-linker,/opt/ascend/sysroot/lib/ld-linux-aarch64.so.1")
set(COMMON_FLAGS "--sysroot=${CMAKE_SYSROOT} -fPIC")
set(CMAKE_C_FLAGS   "${COMMON_FLAGS}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}" CACHE STRING "" FORCE)

# 帮助链接器在 sysroot 内解析二/三级依赖
set(RPATH_LINKS "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu:${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
# 如 CANN 路径不同,请据实调整
set(CANN_HOME "${CMAKE_SYSROOT}/usr/local/Ascend/ascend-toolkit/latest")
list(APPEND RPATH_LINKS "${CANN_HOME}/acllib/lib64")
# 添加 BLAS/LAPACK 库路径,以解决 armadillo/arpack/superlu 的依赖
list(APPEND RPATH_LINKS "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu/blas")
list(APPEND RPATH_LINKS "${CMAKE_SYSROOT}/lib/aarch64-linux-gnu/lapack")
list(APPEND RPATH_LINKS "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/blas")
list(APPEND RPATH_LINKS "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/lapack")
list(APPEND RPATH_LINKS "${CMAKE_SYSROOT}/lib")

# 将 CMake 列表转换为冒号分隔的字符串
string(REPLACE ";" ":" RPATH_LINKS_STR "${RPATH_LINKS}")

set(CMAKE_EXE_LINKER_FLAGS
  "--sysroot=${CMAKE_SYSROOT} \
   -Wl,-rpath-link,${RPATH_LINKS_STR} \
   -Wl,-dynamic-linker,${CMAKE_SYSROOT}/lib/ld-linux-aarch64.so.1"
  CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" CACHE STRING "" FORCE)

# 仅在 sysroot 内查找库/头/包
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# OpenCV（若存在 CMake 配置）
set(OpenCV_DIR ${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu/cmake/opencv4 CACHE PATH "")
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

# CANN 根（按实际版本路径修正）
set(ASCEND_TOOLKIT_HOME ${CANN_HOME} CACHE PATH "")