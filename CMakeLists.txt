cmake_minimum_required(VERSION 3.16..3.21)

# Set app name
set(APP_NAME FaceMeshCpp)

# Set project
project(${APP_NAME})

# Make executable app.
add_executable(${APP_NAME})

# Add source file
add_subdirectory(src)

find_package(OpenCV REQUIRED)
find_package(tensorflowlite REQUIRED)
# find_package(tabulate REQUIRED)

target_include_directories(${APP_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS} ${tensorflowlite_INCLUDE_DIRS})
target_link_libraries(${APP_NAME} ${OpenCV_LIBRARIES} ${tensorflowlite_LIBRARIES})

# Build in multi-process.
# target_compile_options(${APP_NAME} 
#     PRIVATE /MP)
