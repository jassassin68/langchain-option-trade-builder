#!/bin/bash

# Build and Test Script
set -e

ENVIRONMENT=${1:-development}

echo "🔨 Building and testing frontend (Environment: $ENVIRONMENT)..."

# Set environment file
case $ENVIRONMENT in
    "development"|"dev")
        ENV_FILE=".env.development"
        ;;
    "staging")
        ENV_FILE=".env.staging"
        ;;
    "production"|"prod")
        ENV_FILE=".env.production"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        exit 1
        ;;
esac

# Load environment variables
if [ -f "$ENV_FILE" ]; then
    export $(cat $ENV_FILE | grep -v '^#' | xargs)
    echo "✅ Loaded environment from $ENV_FILE"
else
    echo "⚠️ Environment file $ENV_FILE not found, using defaults"
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm ci

# Type checking
echo "🔍 Running TypeScript type checking..."
npm run type-check

# Linting
echo "🧹 Running ESLint..."
npm run lint

# Testing
echo "🧪 Running tests..."
npm run test -- --passWithNoTests --coverage

# Build
echo "🔨 Building application..."
npm run build

# Analyze bundle (if analyzer is available)
if npm list --depth=0 @next/bundle-analyzer > /dev/null 2>&1; then
    echo "📊 Analyzing bundle size..."
    ANALYZE=true npm run build
fi

echo "✅ Build and test completed successfully!"

# Display build information
if [ -d ".next" ]; then
    echo ""
    echo "📈 Build Information:"
    echo "  Build directory: .next"
    
    if [ -f ".next/BUILD_ID" ]; then
        BUILD_ID=$(cat .next/BUILD_ID)
        echo "  Build ID: $BUILD_ID"
    fi
    
    # Show bundle sizes
    if [ -d ".next/static" ]; then
        echo "  Static assets:"
        du -sh .next/static/* 2>/dev/null || echo "    No static assets found"
    fi
fi