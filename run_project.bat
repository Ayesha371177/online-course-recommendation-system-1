@echo off
title CourseMatch - Online Course Recommendation System
cd /d "%~dp0"
echo ========================================================
echo Starting Online Course Recommendation System (Flask)
echo Opening http://127.0.0.1:5000 in your browser...
echo ========================================================
start "" http://127.0.0.1:5000
python app.py
pause
