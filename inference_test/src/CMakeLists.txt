
#设置c++的版本并添加多线程的使用
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#设置输出文件的路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

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
    main.cpp 
	inference.cpp 
	tools/des.cpp 
	tools/des3.cpp 
	tools/tripleDesTest.cpp
    ${PROJECT_SOURCE_DIR}/include/tools/decode.h)


#指定生成目标
add_executable(phone_test ${REQUIRE_CPP})

#链接库
target_link_libraries(phone_test ${OPENCV_dep_LIBS})
target_link_libraries(phone_test ${NCNN_dep_LIBS})

if (WIN32)
	target_compile_definitions(phone_test PRIVATE -DWIN32_LEAN_AND_MEAN -D_SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING)
else ()
	target_link_libraries(phone_test)
endif ()

