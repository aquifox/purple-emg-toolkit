@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -X utf8 check_final_dataset.py
if errorlevel 1 (
  pause
  exit /b 1
)
python -X utf8 train_final_3finger.py
pause
