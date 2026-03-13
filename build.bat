@echo off
call conda activate cameratracker
pip install pyinstaller
pyinstaller VideoTracker.spec
pause
