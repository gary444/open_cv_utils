cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME "SIV")

project( ${PROJECT_NAME} )


set(CMAKE_CXX_FLAGS "-std=c++11")

# for finding open cv on mac
# if (APPLE) 
# 	set (OPENCV_EXTRA_MODULES_PATH "/Library/OpenCV/opencv_contrib/modules")

# 	set(OpenCV_DIR "/Library/OpenCV/opencv-4.4.0/build_opencv")
# endif()


# find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED opencv_optflow opencv_xfeatures2d) # not CUDA optical flow


message("CV version: " ${OpenCV_VERSION})
message("CV libs : " ${OpenCV_LIBS})
# message("optflow: " ${OpenCV_CUDAOPTFLOW_FOUND})
# message("include dirs: " ${OpenCV_INCLUDE_DIRS})
# message("install path: " ${OpenCV_INSTALL_PATH})
# message("modules: " ${OpenCV_LIB_COMPONENTS})

file(GLOB_RECURSE HELPER_FILES ${PROJECT_SOURCE_DIR}/helpers/* ${PROJECT_SOURCE_DIR}/feature_matching/*)

# message(${HELPER_FILES})


add_executable( "ltp" ltp.cpp ${HELPER_FILES} )
target_link_libraries( "ltp" ${OpenCV_LIBS} )


add_executable( "lbp" lbp.cpp ${HELPER_FILES} )
target_link_libraries( "lbp" ${OpenCV_LIBS} )



add_executable( "sift" sift.cpp ${HELPER_FILES} )
target_link_libraries( "sift" ${OpenCV_LIBS} )

