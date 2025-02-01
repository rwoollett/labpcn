cmake_minimum_required(VERSION 3.16)
project(PerceptronProject VERSION 1.0
        DESCRIPTION "Machine learning - the perceptron algorithm"
        LANGUAGES CXX C)

# set everything up for c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# test that filesystem header actually is there and works
try_compile(HAS_FS "${CMAKE_BINARY_DIR}/temp"
        "${CMAKE_SOURCE_DIR}/cmake/has_stdfs.cc"
        CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON )

#message("Supported features = ${CMAKE_CXX_COMPILE_FEATURES}")
if (HAS_FS)
    message(STATUS "std::filesystem supported")
else()
    message(STATUS "Using std::experimental::filesystem instead of std::filesystem")
    #set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lstdc++fs")
    add_compile_definitions(USE_EXP_FILESYSTEM)
endif()

find_package(Threads)
find_package(OpenMP REQUIRED)

add_executable(Perceptron
        src/main.cpp
        src/TrainingData.h
        src/Parallel.cpp
        src/Parallel.h
        src/Perceptron.cpp
        src/Perceptron.h
        src/io_utility/io_utility.cpp
        src/io_utility/io_utility.h
)

# Download and unpack eigent at configure time
configure_file(cmake/CMakeLists.txt.in eigen-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/eigen-download )
if(result)
    message(FATAL_ERROR "CMake step for eigen failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/eigen-download )
if(result)
    message(FATAL_ERROR "Build step for eigen failed: ${result}")
endif()

add_subdirectory(${CMAKE_BINARY_DIR}/eigen-src )

target_compile_definitions(Perceptron PUBLIC _GLIBCXX_PARALLEL)
target_include_directories(
  Perceptron PRIVATE
    $<BUILD_INTERFACE:${Perceptron_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/eigen-src>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(
  Perceptron PRIVATE 
    Threads::Threads
    OpenMP::OpenMP_CXX
)

include(GNUInstallDirs)
install(TARGETS Perceptron RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
