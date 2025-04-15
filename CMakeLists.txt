cmake_minimum_required(VERSION 3.10)
project(QG_Simulation LANGUAGES CPP CUDA)

set(CMAKE_CPP_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 70) # Adjust based on your GPU (70 for V100)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src)  


add_executable(qg_solver 
    src/main.cpp
    src/qg_solver.cpp
)

target_link_libraries(qg_solver
    ${CUDA_LIBRARIES}
    ${CUDA_cufft_LIBRARY}
    ${CUDA_curand_LIBRARY}
)