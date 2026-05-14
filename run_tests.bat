@echo off
SET PYTHONPATH=E:\MyDevelopment\GitHub\quant_strategy
SET PYTHONUSERBASE=C:\Users\abhin\AppData\Roaming\Python
SET PYTHON=C:\Python314\python.exe

echo ============================================
echo  Quant Strategy - Full Test Suite
echo  E:\MyDevelopment\GitHub\quant_strategy
echo ============================================
%PYTHON% --version
echo.
echo NOTE: ERROR and WARNING log lines inside tests
echo       are EXPECTED - they verify guards fire.
echo       Watch only the X/X tests passed line.
echo ============================================
echo.
echo [1/9] math_guards
%PYTHON% -m modules.math_guards 2>&1 | findstr /i "tests passed FAILED"
echo [2/9] data_fetcher
%PYTHON% -m modules.data_fetcher 2>&1 | findstr /i "tests passed FAILED"
echo [3/9] stationarity
%PYTHON% -m modules.stationarity 2>&1 | findstr /i "tests passed FAILED"
echo [4/9] kalman_filter
%PYTHON% -m modules.kalman_filter 2>&1 | findstr /i "tests passed FAILED"
echo [5/9] ou_estimator
%PYTHON% -m modules.ou_estimator 2>&1 | findstr /i "tests passed FAILED"
echo [6/9] hmm_regime
%PYTHON% -m modules.hmm_regime 2>&1 | findstr /i "tests passed FAILED"
echo [7/9] kelly_sizer
%PYTHON% -m modules.kelly_sizer 2>&1 | findstr /i "tests passed FAILED"
echo [8/9] backtest  (~60s)
%PYTHON% -m modules.backtest 2>&1 | findstr /i "tests passed FAILED"
echo [9/9] walk_forward  (~3min)
%PYTHON% -m modules.walk_forward 2>&1 | findstr /i "tests passed FAILED"
echo.
echo ============================================
echo  Done. All lines should say X/X passed.
echo  Any FAILED line = real problem to fix.
echo ============================================
pause
