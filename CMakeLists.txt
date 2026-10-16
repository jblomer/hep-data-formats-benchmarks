cmake_minimum_required(VERSION 3.25)

project(RNTupleFmtBench)

find_package(ROOT 6.36 CONFIG REQUIRED)
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

set(CMAKE_CXX_STANDARD "${ROOT_CXX_STANDARD}")
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _BUILD_TYPE_UPPER)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_${_BUILD_TYPE_UPPER}}${ROOT_CXX_FLAGS}${CMAKE_CXX_FLAGS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_library(util SHARED util.cxx util.hxx)
target_link_libraries(util PRIVATE ROOT::ROOTVecOps ROOT::Hist ROOT::Gui)

add_executable(lhcb lhcb.cxx)
target_link_libraries(lhcb PRIVATE util ROOT::RIO ROOT::ROOTDataFrame Arrow::arrow_shared Parquet::parquet_shared)

add_executable(cms cms.cxx)
target_link_libraries(cms PRIVATE util ROOT::RIO ROOT::ROOTDataFrame Arrow::arrow_shared Parquet::parquet_shared)

message(STATUS "ROOT version: ${ROOT_VERSION}")
message(STATUS "ROOT include path: ${ROOT_INCLUDE_DIRS}\n")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}\n")
