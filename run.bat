@echo off
set PYTHONIOENCODING=utf-8
REM Exemplo de uso:
REM   run.bat "C:\RFID\Tests\CSV" "C:\RFID\Resultados" "C:\RFID\Pallets\LayoutPallet01.xlsx"
set INPUT_DIR=%1
set OUTPUT_DIR=%2
set LAYOUT_FILE=%3
if "%INPUT_DIR%"=="" (
  echo Uso: run.bat ^<INPUT_DIR^> ^<OUTPUT_DIR^> [LAYOUT_FILE]
  exit /b 1
)
if "%OUTPUT_DIR%"=="" (
  set OUTPUT_DIR=%cd%\output
)
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if "%LAYOUT_FILE%"=="" (
  python "%~dp0src\analisar_itemtest.py" --input "%INPUT_DIR%" --output "%OUTPUT_DIR%"
) else (
  python "%~dp0src\analisar_itemtest.py" --input "%INPUT_DIR%" --output "%OUTPUT_DIR%" --layout "%LAYOUT_FILE%"
)
