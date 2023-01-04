#设置c++的版本并添加多线程的使用
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#设置输出文件的路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR})

#设置opencv和ncnn的三方库路径
set(OpenCV_DIR "${PROJECT_SOURCE_DIR}/3rdparty/opencv")
set(NCNN_DIR "${PROJECT_SOURCE_DIR}/3rdparty/ncnn")

#添加opencv和ncnn的依赖库路径
link_directories("${NCNN_DIR}/lib")
link_directories("${OpenCV_DIR}/x64/vc16/bin")
link_directories("${OpenCV_DIR}/x64/vc16/lib")

#添加opencv和ncnn的头文件路径
include_directories("${NCNN_DIR}/include/ncnn")
include_directories("${OpenCV_DIR}/include")

#添加所需的opencv和ncnn依赖库
if(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(OPENCV_dep_LIBS opencv_world454)
else()
	set(OPENCV_dep_LIBS opencv_world454d)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
	set(NCNN_dep_LIBS ncnn)
else()
	set(NCNN_dep_LIBS ncnnd)
endif()

#添加工程内头文件路径
include_directories("${PROJECT_SOURCE_DIR}/include")

#查找当前目录的所有源文件
set(REQUIRE_CPP
		${PROJECT_SOURCE_DIR}/src/inference.cpp
		${PROJECT_SOURCE_DIR}/src/tools/des.cpp
		${PROJECT_SOURCE_DIR}/src/tools/des3.cpp
		${PROJECT_SOURCE_DIR}/src/tools/tripleDesTest.cpp
	)

#指定生成目标（生成库）
add_library(inference SHARED ${REQUIRE_CPP})

#链接库
target_link_libraries(inference ${OPENCV_dep_LIBS})
target_link_libraries(inference ${NCNN_dep_LIBS})

if (WIN32)
	target_compile_definitions(inference PRIVATE -DWIN32_LEAN_AND_MEAN -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
else ()
	target_link_libraries(inference stdc++fs)
endif ()