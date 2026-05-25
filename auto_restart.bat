@echo off
:: auto_restart.bat
:: Keeps run_live.py running. Restarts after any crash with a 30-second delay.
:: Kill with Ctrl+C twice (once for the bot, once for this wrapper).

set PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
set PYTHONUSERBASE=C:\Users\abhin\AppData\Roaming\Python

:: EDIT THESE:
set ALPACA_KEY=your_paper_key
set ALPACA_SECRET=your_paper_secret

cd /d E:\MyDevelopment\GitHub\quant_strategy

echo ============================================================
echo  Auto-restart wrapper for run_live.py
echo  Ctrl+C to stop
echo ============================================================

:loop
echo [%date% %time%] Starting run_live.py...
C:\Python314\python.exe run_live.py

set EXIT_CODE=%ERRORLEVEL%
echo [%date% %time%] Bot exited with code %EXIT_CODE%

:: Exit code 0 = clean shutdown (Ctrl+C or kill switch)
:: Do not restart on clean shutdown
if %EXIT_CODE%==0 (
    echo Clean shutdown detected. Not restarting.
    goto end
)

echo Crash detected. Restarting in 30 seconds...
echo Press Ctrl+C NOW to cancel restart.
timeout /t 30 /nobreak

goto loop

:end
echo Wrapper stopped.
pause
