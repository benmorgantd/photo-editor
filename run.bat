@echo off
:: Change to the directory where the script is located
cd /d "%~dp0"

python photo_editor_gui.py
pause
exit