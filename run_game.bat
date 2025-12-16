@echo off
REM BISINDO BATTLE - Quick Start Script
REM Windows Batch File

echo ============================================================
echo    BISINDO BATTLE - Game Edukasi Bahasa Isyarat Indonesia
echo ============================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo [INFO] Starting game...
echo.
echo Controls:
echo   Arrow Keys: Navigate menu
echo   Enter: Select
echo   Space: Submit gesture
echo   D: Toggle debug mode (show landmarks)
echo   ESC: Back/Pause
echo.
echo ============================================================
echo.

REM Run game
python game\bisindo_game.py

REM Deactivate on exit
deactivate

echo.
echo Thanks for playing! :)
pause
