@echo off
SET PYTHONPATH=C:\Users\abhin\quant_strategy
SET PYTHONUSERBASE=C:\Users\abhin\AppData\Roaming\Python
SET PYTHON=C:\Python314\python.exe

echo ============================================
echo  Quant Strategy - Full Test Suite
echo ============================================
echo Python: %PYTHON%
%PYTHON% --version
echo.
echo NOTE: ERROR log lines inside tests are EXPECTED.
echo       They verify that guards fire correctly.
echo       Only check the final "X/X tests passed" line.
echo ============================================
echo.

echo [1/7] math_guards
%PYTHON% -m modules.math_guards 2>&1 | findstr /i "tests passed FAILED"
echo.

echo [2/7] data_fetcher
%PYTHON% -m modules.data_fetcher 2>&1 | findstr /i "tests passed FAILED"
echo.

echo [3/7] stationarity
%PYTHON% -m modules.stationarity 2>&1 | findstr /i "tests passed FAILED"
echo.

echo [4/7] kalman_filter
%PYTHON% -m modules.kalman_filter 2>&1 | findstr /i "tests passed FAILED"
echo.

echo [5/7] ou_estimator
%PYTHON% -m modules.ou_estimator 2>&1 | findstr /i "tests passed FAILED"
echo.

echo [6/7] hmm_regime
%PYTHON% -m modules.hmm_regime 2>&1 | findstr /i "tests passed FAILED"
echo.

echo [7/7] kelly_sizer
%PYTHON% -m modules.kelly_sizer 2>&1 | findstr /i "tests passed FAILED"
echo.

echo ============================================
echo  Done. Each line above should say X/X passed
echo  with NO "FAILED" lines.
echo ============================================
pause
