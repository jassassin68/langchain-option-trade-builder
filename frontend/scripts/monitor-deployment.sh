#!/bin/bash

# Deployment Monitoring Script
# Monitors deployment health and performance metrics

set -e

DEPLOYMENT_URL=${1:-""}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
ALERT_THRESHOLD=${ALERT_THRESHOLD:-5000}  # 5 seconds
MAX_CHECKS=${MAX_CHECKS:-10}

if [ -z "$DEPLOYMENT_URL" ]; then
    echo "❌ Usage: $0 <deployment-url>"
    echo "Example: $0 https://your-app.vercel.app"
    exit 1
fi

echo "🔍 Starting deployment monitoring for: $DEPLOYMENT_URL"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "Alert threshold: ${ALERT_THRESHOLD}ms"
echo ""

# Function to check health
check_health() {
    local url="$1/api/health"
    local response=$(curl -s -w "\n%{http_code}\n%{time_total}" "$url" 2>/dev/null || echo -e "\n000\n0")
    
    local body=$(echo "$response" | head -n -2)
    local status=$(echo "$response" | tail -n 2 | head -n 1)
    local time=$(echo "$response" | tail -n 1)
    
    echo "$status|$time|$body"
}

# Function to check main page
check_main_page() {
    local url="$1"
    local response=$(curl -s -w "\n%{http_code}\n%{time_total}" "$url" 2>/dev/null || echo -e "\n000\n0")
    
    local status=$(echo "$response" | tail -n 2 | head -n 1)
    local time=$(echo "$response" | tail -n 1)
    
    echo "$status|$time"
}

# Function to format time
format_time() {
    local time=$1
    local time_ms=$(echo "$time * 1000" | bc 2>/dev/null || echo "0")
    printf "%.0f" "$time_ms"
}

# Monitoring loop
check_count=0
error_count=0
total_response_time=0
max_response_time=0
min_response_time=999999

echo "📊 Monitoring started at $(date)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

while [ $check_count -lt $MAX_CHECKS ]; do
    check_count=$((check_count + 1))
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Check health endpoint
    health_result=$(check_health "$DEPLOYMENT_URL")
    health_status=$(echo "$health_result" | cut -d'|' -f1)
    health_time=$(echo "$health_result" | cut -d'|' -f2)
    health_body=$(echo "$health_result" | cut -d'|' -f3-)
    
    # Check main page
    page_result=$(check_main_page "$DEPLOYMENT_URL")
    page_status=$(echo "$page_result" | cut -d'|' -f1)
    page_time=$(echo "$page_result" | cut -d'|' -f2)
    
    # Calculate metrics
    health_time_ms=$(format_time "$health_time")
    page_time_ms=$(format_time "$page_time")
    
    # Update statistics
    total_response_time=$(echo "$total_response_time + $page_time" | bc)
    
    if (( $(echo "$page_time > $max_response_time" | bc -l) )); then
        max_response_time=$page_time
    fi
    
    if (( $(echo "$page_time < $min_response_time" | bc -l) )); then
        min_response_time=$page_time
    fi
    
    # Determine status
    if [ "$health_status" = "200" ] && [ "$page_status" = "200" ]; then
        status_icon="✅"
        status_text="HEALTHY"
    elif [ "$health_status" = "200" ] || [ "$page_status" = "200" ]; then
        status_icon="⚠️"
        status_text="DEGRADED"
        error_count=$((error_count + 1))
    else
        status_icon="❌"
        status_text="DOWN"
        error_count=$((error_count + 1))
    fi
    
    # Check if response time exceeds threshold
    if [ "$page_time_ms" -gt "$ALERT_THRESHOLD" ]; then
        status_icon="🐌"
        status_text="SLOW"
    fi
    
    # Display check result
    echo "[$timestamp] Check #$check_count: $status_icon $status_text"
    echo "  Health: HTTP $health_status (${health_time_ms}ms)"
    echo "  Page:   HTTP $page_status (${page_time_ms}ms)"
    
    # Parse and display backend status if available
    if [ "$health_status" = "200" ] && [ ! -z "$health_body" ]; then
        backend_status=$(echo "$health_body" | grep -o '"backend_status":"[^"]*"' | cut -d'"' -f4 2>/dev/null || echo "unknown")
        if [ ! -z "$backend_status" ]; then
            echo "  Backend: $backend_status"
        fi
    fi
    
    echo ""
    
    # Wait before next check (except for last iteration)
    if [ $check_count -lt $MAX_CHECKS ]; then
        sleep $CHECK_INTERVAL
    fi
done

# Calculate final statistics
avg_response_time=$(echo "scale=3; $total_response_time / $check_count" | bc)
avg_response_time_ms=$(format_time "$avg_response_time")
max_response_time_ms=$(format_time "$max_response_time")
min_response_time_ms=$(format_time "$min_response_time")
success_rate=$(echo "scale=2; ($check_count - $error_count) * 100 / $check_count" | bc)

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 Monitoring Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Total Checks:     $check_count"
echo "Successful:       $((check_count - error_count))"
echo "Failed:           $error_count"
echo "Success Rate:     ${success_rate}%"
echo ""
echo "Response Times:"
echo "  Average:        ${avg_response_time_ms}ms"
echo "  Minimum:        ${min_response_time_ms}ms"
echo "  Maximum:        ${max_response_time_ms}ms"
echo ""

# Final verdict
if [ $error_count -eq 0 ]; then
    echo "✅ Deployment is stable and healthy"
    exit 0
elif [ $error_count -lt 3 ]; then
    echo "⚠️ Deployment has minor issues ($error_count errors)"
    exit 0
else
    echo "❌ Deployment has significant issues ($error_count errors)"
    exit 1
fi
