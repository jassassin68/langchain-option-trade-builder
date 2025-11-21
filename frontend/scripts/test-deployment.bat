@echo off
REM Comprehensive Deployment Testing Script for Windows
setlocal enabledelayedexpansion

set DEPLOYMENT_URL=%1
if "%DEPLOYMENT_URL%"=="" set DEPLOYMENT_URL=http://localhost:3000

echo 🧪 Testing deployment at: %DEPLOYMENT_URL%
echo.

set TOTAL_TESTS=0
set PASSED_TESTS=0
set FAILED_TESTS=0

echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🌐 Basic Connectivity Tests
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :test_endpoint "Main page loads" "%DEPLOYMENT_URL%"
call :test_endpoint "Health endpoint responds" "%DEPLOYMENT_URL%/api/health"
call :test_endpoint "Favicon exists" "%DEPLOYMENT_URL%/favicon.ico"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📄 Content Validation Tests
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :test_content "Page contains app name" "%DEPLOYMENT_URL%" "Options Trade Evaluator"
call :test_content "Page has Next.js data" "%DEPLOYMENT_URL%" "__NEXT_DATA__"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 🔒 Security Headers Tests
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

call :test_header "X-Frame-Options header" "%DEPLOYMENT_URL%" "X-Frame-Options"
call :test_header "X-Content-Type-Options header" "%DEPLOYMENT_URL%" "X-Content-Type-Options"

echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo 📊 Test Summary
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo Total Tests:    %TOTAL_TESTS%
echo Passed:         %PASSED_TESTS%
echo Failed:         %FAILED_TESTS%
echo.

if %FAILED_TESTS% EQU 0 (
    echo ✅ All tests passed! Deployment is ready.
    exit /b 0
) else if %FAILED_TESTS% LEQ 2 (
    echo ⚠️ Some tests failed, but deployment may be acceptable.
    exit /b 0
) else (
    echo ❌ Multiple tests failed. Deployment needs attention.
    exit /b 1
)

:test_endpoint
set /a TOTAL_TESTS+=1
echo Testing: %~1...
curl -s -o nul -w "%%{http_code}" "%~2" | findstr "200" >nul
if errorlevel 1 (
    echo   ✗ FAILED
    set /a FAILED_TESTS+=1
) else (
    echo   ✓ PASSED
    set /a PASSED_TESTS+=1
)
goto :eof

:test_content
set /a TOTAL_TESTS+=1
echo Testing: %~1...
curl -s "%~2" | findstr /C:"%~3" >nul
if errorlevel 1 (
    echo   ✗ FAILED
    set /a FAILED_TESTS+=1
) else (
    echo   ✓ PASSED
    set /a PASSED_TESTS+=1
)
goto :eof

:test_header
set /a TOTAL_TESTS+=1
echo Testing: %~1...
curl -I -s "%~2" | findstr /I "%~3" >nul
if errorlevel 1 (
    echo   ✗ FAILED
    set /a FAILED_TESTS+=1
) else (
    echo   ✓ PASSED
    set /a PASSED_TESTS+=1
)
goto :eof
