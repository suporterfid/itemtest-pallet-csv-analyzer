@echo off
set PYTHONIOENCODING=utf-8
REM Usage example:
REM   run.bat "C:\RFID\Tests\CSV" "C:\RFID\Results" "C:\RFID\Pallets\LayoutPallet01.xlsx"
set INPUT_DIR=%1
set OUTPUT_DIR=%2
set LAYOUT_FILE=%3
if "%INPUT_DIR%"=="" (
  echo Usage: run.bat ^<INPUT_DIR^> ^<OUTPUT_DIR^> [LAYOUT_FILE]
  exit /b 1
)
if "%OUTPUT_DIR%"=="" (
  set OUTPUT_DIR=%cd%\output
)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if "%LAYOUT_FILE%"=="" (
  python "%~dp0src\itemtest_analyzer.py" --input "%INPUT_DIR%" --output "%OUTPUT_DIR%"
) else (
  python "%~dp0src\itemtest_analyzer.py" --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --layout "%LAYOUT_FILE%"
)
