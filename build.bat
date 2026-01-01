@echo off
echo Building Pixelmess...
pip install pyinstaller
IF %ERRORLEVEL% NEQ 0 (
    echo Error installing PyInstaller.
    exit /b 1
)

echo Cleaning previous builds...
rmdir /s /q build dist
del /q *.spec

echo Packaging...
pyinstaller --noconsole --onefile --name "Pixelmess" --add-data "logo.png;." --icon "logo.png" app.py

IF %ERRORLEVEL% EQU 0 (
    echo Build Successful!
    echo Executable is located in dist/Pixelmess.exe
) ELSE (
    echo Build Failed.
)
pause
