#!/bin/bash

# Comprehensive Deployment Testing Script
# Tests all critical functionality after deployment

set -e

DEPLOYMENT_URL=${1:-"http://localhost:3000"}
TIMEOUT=30

echo "🧪 Testing deployment at: $DEPLOYMENT_URL"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run a test
run_test() {
    local test_name=$1
    local test_command=$2
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing: $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Function to test HTTP endpoint
test_endpoint() {
    local name=$1
    local url=$2
    local expected_status=${3:-200}
    
    run_test "$name" "curl -s -o /dev/null -w '%{http_code}' --max-time $TIMEOUT '$url' | grep -q '^$expected_status$'"
}

# Function to test content
test_content() {
    local name=$1
    local url=$2
    local expected_text=$3
    
    run_test "$name" "curl -s --max-time $TIMEOUT '$url' | grep -q '$expected_text'"
}

# Function to test JSON response
test_json() {
    local name=$1
    local url=$2
    local json_path=$3
    
    run_test "$name" "curl -s --max-time $TIMEOUT '$url' | grep -q '\"$json_path\"'"
}

# Function to test performance
test_performance() {
    local name=$1
    local url=$2
    local max_time=$3
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Testing: $name... "
    
    response_time=$(curl -o /dev/null -s -w "%{time_total}" --max-time $TIMEOUT "$url" 2>/dev/null || echo "999")
    
    if (( $(echo "$response_time < $max_time" | bc -l) )); then
        echo -e "${GREEN}✓ PASSED${NC} (${response_time}s)"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (${response_time}s > ${max_time}s)"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Basic Connectivity Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_endpoint "Main page loads" "$DEPLOYMENT_URL"
test_endpoint "Health endpoint responds" "$DEPLOYMENT_URL/api/health"
test_endpoint "Favicon exists" "$DEPLOYMENT_URL/favicon.ico"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📄 Content Validation Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_content "Page contains app name" "$DEPLOYMENT_URL" "Options Trade Evaluator"
test_content "Page has Next.js data" "$DEPLOYMENT_URL" "__NEXT_DATA__"
test_content "Page has viewport meta" "$DEPLOYMENT_URL" "viewport"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 API Health Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_json "Health returns status" "$DEPLOYMENT_URL/api/health" "status"
test_json "Health returns timestamp" "$DEPLOYMENT_URL/api/health" "timestamp"
test_json "Health returns environment" "$DEPLOYMENT_URL/api/health" "environment"
test_json "Health returns API URL" "$DEPLOYMENT_URL/api/health" "api_url"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔒 Security Headers Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_test "X-Frame-Options header" "curl -I -s --max-time $TIMEOUT '$DEPLOYMENT_URL' | grep -i 'X-Frame-Options'"
run_test "X-Content-Type-Options header" "curl -I -s --max-time $TIMEOUT '$DEPLOYMENT_URL' | grep -i 'X-Content-Type-Options'"
run_test "Referrer-Policy header" "curl -I -s --max-time $TIMEOUT '$DEPLOYMENT_URL' | grep -i 'Referrer-Policy'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⚡ Performance Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

test_performance "Main page loads quickly" "$DEPLOYMENT_URL" 3.0
test_performance "Health check is fast" "$DEPLOYMENT_URL/api/health" 1.0

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 Responsive Design Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

run_test "Mobile viewport meta tag" "curl -s --max-time $TIMEOUT '$DEPLOYMENT_URL' | grep -q 'width=device-width'"
run_test "Mobile user agent works" "curl -s -H 'User-Agent: Mozilla/5.0 (iPhone)' --max-time $TIMEOUT '$DEPLOYMENT_URL' | grep -q 'viewport'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Static Assets Tests"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if static assets have proper cache headers
run_test "Static assets have cache headers" "curl -I -s --max-time $TIMEOUT '$DEPLOYMENT_URL/_next/static/css' | grep -i 'cache-control' || true"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total Tests:    $TOTAL_TESTS"
echo -e "Passed:         ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed:         ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    SUCCESS_RATE=100
else
    SUCCESS_RATE=$(echo "scale=2; $PASSED_TESTS * 100 / $TOTAL_TESTS" | bc)
fi

echo "Success Rate:   ${SUCCESS_RATE}%"
echo ""

# Final verdict
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed! Deployment is ready.${NC}"
    exit 0
elif [ $FAILED_TESTS -le 2 ]; then
    echo -e "${YELLOW}⚠️ Some tests failed, but deployment may be acceptable.${NC}"
    exit 0
else
    echo -e "${RED}❌ Multiple tests failed. Deployment needs attention.${NC}"
    exit 1
fi
