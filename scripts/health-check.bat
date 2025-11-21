@echo off
REM Health Check Script for Windows
setlocal enabledelayedexpansion

REM Configuration
set BACKEND_URL=http://localhost:8000
set TIMEOUT=30
set RETRY_COUNT=5
set RETRY_DELAY=5

echo 🔍 Starting health check for Options Trade Evaluator...
echo Backend URL: %BACKEND_URL%

REM Function to check service health
:check_health
set service_name=%1
set url=%2
set expected_status=%3
if "%expected_status%"=="" set expected_status=200

echo Checking %service_name%...

for /l %%i in (1,1,%RETRY_COUNT%) do (
    curl -s -w "%%{http_code}" -o nul --max-time %TIMEOUT% "%url%" > temp_response.txt 2>nul
    if !errorlevel! equ 0 (
        set /p response=<temp_response.txt
        if "!response!"=="%expected_status%" (
            echo ✅ %service_name% is healthy (HTTP !response!)
            del temp_response.txt 2>nul
            goto :eof
        ) else (
            echo ⚠️ %service_name% returned HTTP !response! (attempt %%i/%RETRY_COUNT%)
        )
    ) else (
        echo ❌ %service_name% is unreachable (attempt %%i/%RETRY_COUNT%)
    )
    
    if %%i lss %RETRY_COUNT% (
        echo Retrying in %RETRY_DELAY% seconds...
        timeout /t %RETRY_DELAY% /nobreak > nul
    )
)

echo ❌ %service_name% health check failed after %RETRY_COUNT% attempts
del temp_response.txt 2>nul
exit /b 1

REM Main health checks
call :check_health "Main API" "%BACKEND_URL%/"
call :check_health "Health Endpoint" "%BACKEND_URL%/api/v1/health"
call :check_health "API Documentation" "%BACKEND_URL%/docs"

REM Check ticker search functionality
echo 🔍 Testing ticker search functionality...
curl -s "%BACKEND_URL%/api/v1/tickers/search?q=AAPL" --max-time %TIMEOUT% > search_response.txt 2>nul
findstr /c:"results" search_response.txt >nul
if !errorlevel! equ 0 (
    echo ✅ Ticker search is working
) else (
    echo ❌ Ticker search failed
    del search_response.txt 2>nul
    exit /b 1
)
del search_response.txt 2>nul

echo.
echo ✅ Health check completed successfully!
echo 🌐 All services are operational

pause