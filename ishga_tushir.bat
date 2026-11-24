@echo off
rem Birinchi marta: virtual muhit yaratadi va kutubxonalarni o'rnatadi. Keyin dasturni ishga tushiradi.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual muhit yaratilmoqda...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo.
        echo XATO: Python topilmadi. https://www.python.org/downloads/ dan Python 3.10-3.13 o'rnating
        echo       va o'rnatishda "Add python.exe to PATH" ni belgilang.
        pause
        exit /b 1
    )
)

if not exist ".venv\.installed_v2" (
    echo Kutubxonalar o'rnatilmoqda ^(birinchi marta bir necha daqiqa olishi mumkin^)...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo XATO: Kutubxonalarni o'rnatib bo'lmadi. Internet va Python versiyasini ^(3.10-3.13, 64-bit^) tekshiring.
        pause
        exit /b 1
    )
    echo ok> ".venv\.installed_v2"
)

".venv\Scripts\python.exe" qol_mishka.py %*
if errorlevel 1 pause
