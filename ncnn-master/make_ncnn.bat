SET WATCH_VC_DIR=C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\VC\Auxiliary\Build
call "%WATCH_VC_DIR%\vcvarsall.bat" x64
if exist build (echo "build folder exist.") else (md build)
cd build

cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=F:/13_Shoot_X86_CPU/ncnn-master/3rdparty/protobuf/include -DProtobuf_LIBRARIES=F:/13_Shoot_X86_CPU/ncnn-master/3rdparty/protobuf/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=F:/13_Shoot_X86_CPU/ncnn-master/3rdparty/protobuf/bin/protoc.exe -DNCNN_VULKAN=OFF ..
nmake
nmake install

pause
exit