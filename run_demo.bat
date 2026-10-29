@echo off
title Autonomous Driving Perception ^& Autopilot System
cls
echo ================================================================
echo       AUTONOMOUS DRIVING PERCEPTION ^& AUTOPILOT SYSTEM
echo ================================================================
echo.
echo Choose an option to run:
echo.
echo   [1] Real Highway Driving Video (sample_driving.mp4)
echo   [2] Built-in Dynamic Autonomous Road Simulator
echo   [3] Live Webcam Feed
echo   [4] Custom Video File
echo.
set /p opt="Enter option number (1-4) [Default is 1]: "

if "%opt%"=="" set opt=1

cd /d "%~dp0\src"

if "%opt%"=="1" (
    echo.
    echo Launching on Real Highway Driving Video...
    python autonomous_driving_standalone.py --video "..\sample_driving.mp4"
) else if "%opt%"=="2" (
    echo.
    echo Launching Built-in Dynamic Road Simulator...
    python autonomous_driving_standalone.py
) else if "%opt%"=="3" (
    echo.
    echo Launching Live Webcam Feed...
    python autonomous_driving_standalone.py --camera 0
) else if "%opt%"=="4" (
    echo.
    set /p vpath="Enter full path to your video file: "
    python autonomous_driving_standalone.py --video "%vpath%"
) else (
    echo Invalid choice. Running simulator by default...
    python autonomous_driving_standalone.py
)

pause
