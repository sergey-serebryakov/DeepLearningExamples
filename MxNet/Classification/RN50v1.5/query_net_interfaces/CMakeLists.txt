cmake_minimum_required(VERSION 2.6)
project(query_net_interfaces)
set (CMAKE_CXX_STANDARD 11)
add_executable(query_net_interfaces main.cpp)

install(TARGETS query_net_interfaces RUNTIME DESTINATION bin)
