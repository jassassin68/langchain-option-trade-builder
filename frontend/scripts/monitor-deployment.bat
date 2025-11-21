@echo off
REM Deployment Monitoring Script for Windows
setlocal enabledelayedexpansion

set DEPLOYMENT_URL=%1
set CHECK_INTERVAL=60
set MAX_CHECKS=10

if "%DEPLOYMENT_URL%"=="" (
    echo ❌ Usage: %0 ^<deployment-url^>
    echo Example: %0 https://your-app.vercel.app
    exit /b 1
)

echo 🔍 Starting deployment monitoring for: %DEPLOYMENT_URL%
echo Check interval: %CHECK_INTERVAL%s
echo.

set check_count=0
set error_count=0

:monitor_loop
if %check_count% GEQ %MAX_CHECKS% goto end_monitoring

set /a check_count+=1

REM Get current timestamp
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (set mydate=%%a-%%b-%%c)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a:%%b)

echo [%mydate% %mytime%] Check #%check_count%:

REM Check health endpoint
curl -s -o nul -w "  Health: HTTP %%{http_code} (%%{time_total}s)\n" "%DEPLOYMENT_URL%/api/health"
if errorlevel 1 (
    echo   ❌ Health check failed
    set /a error_count+=1
) else (
    echo   ✅ Health check passed
)

REM Check main page
curl -s -o nul -w "  Page:   HTTP %%{http_code} (%%{time_total}s)\n" "%DEPLOYMENT_URL%"
if errorlevel 1 (
    echo   ❌ Page check failed
    set /a error_count+=1
) else (
    echo   ✅ Page check passed
)

echo.

REM Wait before next check
if %check_count% LSS %MAX_CHECKS% (
    timeout /t %CHECK_INTERVAL% /nobreak >nul
)

goto monitor_loop

:end_monitoring
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📈 Monitoring Summary
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Total Checks:     %check_count%
set /a success_count=%check_count%-%error_count%
echo Successful:       %success_count%
echo Failed:           %error_count%
echo.

if %error_count% EQU 0 (
    echo ✅ Deployment is stable and healthy
    exit /b 0
) else if %error_count% LSS 3 (
    echo ⚠️ Deployment has minor issues ^(%error_count% errors^)
    exit /b 0
) else (
    echo ❌ Deployment has significant issues ^(%error_count% errors^)
    exit /b 1
)
