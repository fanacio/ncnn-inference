set SLN=.\build\phonedet_test.sln

set VC_DIR=D:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build
call "%VC_DIR%\vcvars64.bat" x64

set OPENCV_4_5=.\third_party\opencv\x64\vc16\bin
set NCNN=.\third_party\ncnn\lib

SET "PATH=%OPENCV_4_5%;%NCNN%;%PATH%"

start devenv.exe %SLN%