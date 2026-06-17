@echo off
:: Change to the directory where the script is located
cd /d "%~dp0"

python photo_editor_gui.py

:: Keep the window open if there is an error
exit