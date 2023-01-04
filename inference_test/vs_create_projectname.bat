@echo off

call :prepare_env
call :build_vs

goto :EOF

:prepare_env

echo "call env.bat if exist"
if exist env.bat (call env.bat)

goto :EOF

:build_vs

if defined WATCH_VC_DIR  (
    echo "has WATCH_VC_DIR in env.bat"
)  else  (
    echo "please set the 1 env variables: WATCH_VC_DIR in env.bat, and retry again."
    pause
    exit
)

call "%WATCH_VC_DIR%\vcvarsall.bat" x64



if exist build (echo "build folder exist.") else (md build)
cd build

cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_BUILD_TYPE=Release ..
cd ../

pause
exit