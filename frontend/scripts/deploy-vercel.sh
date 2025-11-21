#!/bin/bash

# Vercel Deployment Script
set -e

# Configuration
ENVIRONMENT=${1:-production}
PROJECT_NAME="options-trade-evaluator-frontend"

echo "🚀 Deploying to Vercel (Environment: $ENVIRONMENT)..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Set environment-specific variables
case $ENVIRONMENT in
    "development"|"dev")
        ENV_FILE=".env.development"
        VERCEL_ENV="development"
        ;;
    "staging")
        ENV_FILE=".env.staging"
        VERCEL_ENV="preview"
        ;;
    "production"|"prod")
        ENV_FILE=".env.production"
        VERCEL_ENV="production"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "Valid options: development, staging, production"
        exit 1
        ;;
esac

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file $ENV_FILE not found"
    exit 1
fi

echo "✅ Using environment file: $ENV_FILE"

# Load environment variables
export $(cat $ENV_FILE | grep -v '^#' | xargs)

# Validate required environment variables
if [ -z "$NEXT_PUBLIC_API_URL" ]; then
    echo "❌ NEXT_PUBLIC_API_URL is not set in $ENV_FILE"
    exit 1
fi

echo "🔧 Configuration:"
echo "  API URL: $NEXT_PUBLIC_API_URL"
echo "  App Name: $NEXT_PUBLIC_APP_NAME"

# Run pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Type checking
echo "  - TypeScript type checking..."
npm run type-check || {
    echo "❌ TypeScript type checking failed"
    exit 1
}

# Linting
echo "  - ESLint checking..."
npm run lint || {
    echo "❌ ESLint checking failed"
    exit 1
}

# Testing
echo "  - Running tests..."
npm run test -- --passWithNoTests || {
    echo "❌ Tests failed"
    exit 1
}

# Build test
echo "  - Testing build..."
npm run build || {
    echo "❌ Build test failed"
    exit 1
}

echo "✅ Pre-deployment checks passed"

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."

if [ "$VERCEL_ENV" = "production" ]; then
    # Production deployment
    vercel --prod --confirm --env NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" --env NEXT_PUBLIC_APP_NAME="$NEXT_PUBLIC_APP_NAME"
else
    # Preview deployment
    vercel --confirm --env NEXT_PUBLIC_API_URL="$NEXT_PUBLIC_API_URL" --env NEXT_PUBLIC_APP_NAME="$NEXT_PUBLIC_APP_NAME"
fi

echo "✅ Deployment completed successfully!"

# Get deployment URL
DEPLOYMENT_URL=$(vercel ls $PROJECT_NAME --limit 1 | grep https | awk '{print $2}')
if [ ! -z "$DEPLOYMENT_URL" ]; then
    echo "🌐 Deployment URL: $DEPLOYMENT_URL"
    
    # Run post-deployment verification
    echo "🧪 Running post-deployment verification..."
    sleep 10  # Wait for deployment to be ready
    
    if curl -f "$DEPLOYMENT_URL" > /dev/null 2>&1; then
        echo "✅ Deployment verification successful"
    else
        echo "⚠️ Deployment verification failed - site may still be starting up"
    fi
else
    echo "⚠️ Could not retrieve deployment URL"
fi