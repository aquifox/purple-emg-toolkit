@echo off
chcp 65001 >nul
cd /d "%~dp0\.."
python apps\PurpleTeam_EMG_AI_Classifier.py

