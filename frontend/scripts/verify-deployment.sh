#!/bin/bash

# Deployment Verification Script
set -e

DEPLOYMENT_URL=${1:-"http://localhost:3000"}
TIMEOUT=${TIMEOUT:-30}
RETRY_COUNT=${RETRY_COUNT:-5}
RETRY_DELAY=${RETRY_DELAY:-5}

echo "🔍 Verifying deployment at: $DEPLOYMENT_URL"

# Function to check endpoint
check_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo "Checking $name..."
    
    for i in $(seq 1 $RETRY_COUNT); do
        if response=$(curl -s -w "%{http_code}" -o /tmp/verify_response --max-time $TIMEOUT "$url" 2>/dev/null); then
            if [ "$response" = "$expected_status" ]; then
                echo "✅ $name is working (HTTP $response)"
                return 0
            else
                echo "⚠️ $name returned HTTP $response (attempt $i/$RETRY_COUNT)"
            fi
        else
            echo "❌ $name is unreachable (attempt $i/$RETRY_COUNT)"
        fi
        
        if [ $i -lt $RETRY_COUNT ]; then
            echo "Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    done
    
    echo "❌ $name verification failed after $RETRY_COUNT attempts"
    return 1
}

# Function to check page content
check_content() {
    local name=$1
    local url=$2
    local expected_text=$3
    
    echo "Checking $name content..."
    
    if response=$(curl -s --max-time $TIMEOUT "$url" 2>/dev/null); then
        if echo "$response" | grep -q "$expected_text"; then
            echo "✅ $name contains expected content"
            return 0
        else
            echo "❌ $name does not contain expected content: '$expected_text'"
            return 1
        fi
    else
        echo "❌ Failed to fetch $name"
        return 1
    fi
}

# Function to check performance
check_performance() {
    local url=$1
    
    echo "🚀 Checking performance..."
    
    # Measure response time
    response_time=$(curl -o /dev/null -s -w "%{time_total}" --max-time $TIMEOUT "$url" 2>/dev/null || echo "timeout")
    
    if [ "$response_time" != "timeout" ]; then
        # Convert to milliseconds
        response_time_ms=$(echo "$response_time * 1000" | bc 2>/dev/null || echo "unknown")
        echo "📊 Response time: ${response_time_ms}ms"
        
        # Check if response time is acceptable (< 3 seconds)
        if (( $(echo "$response_time < 3.0" | bc -l 2>/dev/null || echo 0) )); then
            echo "✅ Response time is acceptable"
        else
            echo "⚠️ Response time is slow (>${response_time}s)"
        fi
    else
        echo "❌ Performance check timed out"
    fi
}

# Main verification
echo "🧪 Starting deployment verification..."

# Check main page
check_endpoint "Main page" "$DEPLOYMENT_URL"

# Check if it's a Next.js app
check_content "Next.js app" "$DEPLOYMENT_URL" "Options Trade Evaluator\|__NEXT_DATA__"

# Check static assets (if accessible)
check_endpoint "Static assets" "$DEPLOYMENT_URL/_next/static/css" 404  # 404 is expected for directory

# Check favicon
check_endpoint "Favicon" "$DEPLOYMENT_URL/favicon.ico"

# Performance check
check_performance "$DEPLOYMENT_URL"

# Check responsive design (basic check)
echo "📱 Checking responsive design..."
if response=$(curl -s -H "User-Agent: Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)" --max-time $TIMEOUT "$DEPLOYMENT_URL" 2>/dev/null); then
    if echo "$response" | grep -q "viewport"; then
        echo "✅ Responsive design meta tag found"
    else
        echo "⚠️ Responsive design meta tag not found"
    fi
else
    echo "❌ Failed to check responsive design"
fi

# Check security headers (if accessible)
echo "🔒 Checking security headers..."
headers=$(curl -I -s --max-time $TIMEOUT "$DEPLOYMENT_URL" 2>/dev/null || echo "")
if [ ! -z "$headers" ]; then
    if echo "$headers" | grep -q "X-Frame-Options"; then
        echo "✅ X-Frame-Options header found"
    else
        echo "⚠️ X-Frame-Options header missing"
    fi
    
    if echo "$headers" | grep -q "X-Content-Type-Options"; then
        echo "✅ X-Content-Type-Options header found"
    else
        echo "⚠️ X-Content-Type-Options header missing"
    fi
else
    echo "⚠️ Could not check security headers"
fi

echo ""
echo "✅ Deployment verification completed!"
echo "🌐 Site is accessible at: $DEPLOYMENT_URL"

# Cleanup
rm -f /tmp/verify_response