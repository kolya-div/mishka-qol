@echo off
rem Dasturni bitta papkaga (QolMishka.exe) yig'adi: boshqa kompyuterda Python o'rnatish shart emas.
rem Natija: dist\QolMishka\QolMishka.exe  (butun dist\QolMishka papkasini ko'chiring)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual muhit yaratilmoqda...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo XATO: Python topilmadi. https://www.python.org/downloads/ dan Python 3.10-3.13 o'rnating.
        pause
        exit /b 1
    )
)
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
if errorlevel 1 (
    echo XATO: Kutubxonalarni o'rnatib bo'lmadi.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --name QolMishka --noconsole ^
    --collect-all mediapipe --hidden-import pystray._win32 ^
    --add-data "hand_landmarker.task;." qol_mishka.py
if errorlevel 1 (
    echo XATO: .exe yig'ilmadi.
    pause
    exit /b 1
)
echo.
echo TAYYOR: dist\QolMishka\QolMishka.exe
echo Sozlamalar (settings.json) va xato.log shu papkada saqlanadi.
pause
