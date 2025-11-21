@echo off
REM Vercel Deployment Script for Windows
setlocal enabledelayedexpansion

set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=production

echo 🚀 Deploying to Vercel (Environment: %ENVIRONMENT%)...

REM Check if Vercel CLI is installed
vercel --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Vercel CLI not found. Installing...
    npm install -g vercel
)

REM Set environment-specific variables
if "%ENVIRONMENT%"=="development" (
    set ENV_FILE=.env.development
    set VERCEL_ENV=development
) else if "%ENVIRONMENT%"=="dev" (
    set ENV_FILE=.env.development
    set VERCEL_ENV=development
) else if "%ENVIRONMENT%"=="staging" (
    set ENV_FILE=.env.staging
    set VERCEL_ENV=preview
) else if "%ENVIRONMENT%"=="production" (
    set ENV_FILE=.env.production
    set VERCEL_ENV=production
) else if "%ENVIRONMENT%"=="prod" (
    set ENV_FILE=.env.production
    set VERCEL_ENV=production
) else (
    echo ❌ Invalid environment: %ENVIRONMENT%
    echo Valid options: development, staging, production
    exit /b 1
)

REM Check if environment file exists
if not exist "%ENV_FILE%" (
    echo ❌ Environment file %ENV_FILE% not found
    exit /b 1
)

echo ✅ Using environment file: %ENV_FILE%

REM Load environment variables (simplified for Windows)
for /f "usebackq tokens=1,2 delims==" %%a in ("%ENV_FILE%") do (
    if not "%%a"=="" if not "%%a:~0,1%"=="#" (
        set %%a=%%b
    )
)

REM Validate required environment variables
if "%NEXT_PUBLIC_API_URL%"=="" (
    echo ❌ NEXT_PUBLIC_API_URL is not set in %ENV_FILE%
    exit /b 1
)

echo 🔧 Configuration:
echo   API URL: %NEXT_PUBLIC_API_URL%
echo   App Name: %NEXT_PUBLIC_APP_NAME%

REM Run pre-deployment checks
echo 🔍 Running pre-deployment checks...

echo   - TypeScript type checking...
call npm run type-check
if errorlevel 1 (
    echo ❌ TypeScript type checking failed
    exit /b 1
)

echo   - ESLint checking...
call npm run lint
if errorlevel 1 (
    echo ❌ ESLint checking failed
    exit /b 1
)

echo   - Running tests...
call npm run test -- --passWithNoTests
if errorlevel 1 (
    echo ❌ Tests failed
    exit /b 1
)

echo   - Testing build...
call npm run build
if errorlevel 1 (
    echo ❌ Build test failed
    exit /b 1
)

echo ✅ Pre-deployment checks passed

REM Deploy to Vercel
echo 🚀 Deploying to Vercel...

if "%VERCEL_ENV%"=="production" (
    vercel --prod --confirm --env NEXT_PUBLIC_API_URL="%NEXT_PUBLIC_API_URL%" --env NEXT_PUBLIC_APP_NAME="%NEXT_PUBLIC_APP_NAME%"
) else (
    vercel --confirm --env NEXT_PUBLIC_API_URL="%NEXT_PUBLIC_API_URL%" --env NEXT_PUBLIC_APP_NAME="%NEXT_PUBLIC_APP_NAME%"
)

echo ✅ Deployment completed successfully!

pause