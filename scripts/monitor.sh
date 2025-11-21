#!/bin/bash

# Monitoring Script for Options Trade Evaluator
set -e

# Configuration
COMPOSE_FILE=${COMPOSE_FILE:-"docker-compose.yml"}
LOG_LINES=${LOG_LINES:-50}
WATCH_INTERVAL=${WATCH_INTERVAL:-5}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check container status
check_containers() {
    print_status $BLUE "📊 Container Status:"
    docker-compose -f $COMPOSE_FILE ps
    echo ""
}

# Function to show resource usage
show_resources() {
    print_status $BLUE "💻 Resource Usage:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
    echo ""
}

# Function to show recent logs
show_logs() {
    local service=$1
    print_status $BLUE "📝 Recent logs for $service:"
    docker-compose -f $COMPOSE_FILE logs --tail=$LOG_LINES $service
    echo ""
}

# Function to check service health
check_health() {
    print_status $BLUE "🔍 Health Checks:"
    
    # Check if health check script exists and run it
    if [ -f "scripts/health-check.sh" ]; then
        bash scripts/health-check.sh
    else
        # Basic health check
        backend_url="http://localhost:8000"
        if curl -f "$backend_url/api/v1/health" > /dev/null 2>&1; then
            print_status $GREEN "✅ Backend API is healthy"
        else
            print_status $RED "❌ Backend API is not responding"
        fi
    fi
    echo ""
}

# Function to show disk usage
show_disk_usage() {
    print_status $BLUE "💾 Docker Disk Usage:"
    docker system df
    echo ""
    
    print_status $BLUE "📁 Volume Usage:"
    docker volume ls -q | xargs docker volume inspect | grep -E "(Name|Mountpoint)" | paste - -
    echo ""
}

# Function to show network information
show_network() {
    print_status $BLUE "🌐 Network Information:"
    docker network ls | grep -E "(options|app-network|bridge)"
    echo ""
}

# Main monitoring function
monitor() {
    clear
    print_status $GREEN "🚀 Options Trade Evaluator - System Monitor"
    print_status $YELLOW "Compose file: $COMPOSE_FILE"
    print_status $YELLOW "Refresh interval: ${WATCH_INTERVAL}s"
    echo "========================================================"
    
    check_containers
    check_health
    show_resources
    show_disk_usage
}

# Interactive monitoring
interactive_monitor() {
    while true; do
        monitor
        
        print_status $BLUE "Commands: [r]efresh [l]ogs [h]ealth [q]uit [s]hell"
        read -t $WATCH_INTERVAL -n 1 -s input || input=""
        
        case $input in
            r|R)
                continue
                ;;
            l|L)
                echo ""
                print_status $YELLOW "Select service for logs:"
                print_status $BLUE "1) backend  2) postgres  3) redis  4) nginx"
                read -n 1 service_choice
                case $service_choice in
                    1) show_logs "backend" ;;
                    2) show_logs "postgres" ;;
                    3) show_logs "redis" ;;
                    4) show_logs "nginx" ;;
                    *) print_status $RED "Invalid choice" ;;
                esac
                read -p "Press Enter to continue..."
                ;;
            h|H)
                echo ""
                check_health
                read -p "Press Enter to continue..."
                ;;
            s|S)
                echo ""
                print_status $YELLOW "Opening shell in backend container..."
                docker-compose -f $COMPOSE_FILE exec backend /bin/bash
                ;;
            q|Q)
                print_status $GREEN "Monitoring stopped."
                exit 0
                ;;
        esac
    done
}

# Command line options
case "${1:-}" in
    --once)
        monitor
        ;;
    --logs)
        service=${2:-backend}
        show_logs $service
        ;;
    --health)
        check_health
        ;;
    --resources)
        show_resources
        ;;
    --help)
        echo "Usage: $0 [--once|--logs [service]|--health|--resources|--help]"
        echo ""
        echo "Options:"
        echo "  --once      Run monitoring once and exit"
        echo "  --logs      Show logs for specified service (default: backend)"
        echo "  --health    Run health checks only"
        echo "  --resources Show resource usage only"
        echo "  --help      Show this help message"
        echo ""
        echo "Interactive mode (default): Run continuous monitoring with commands"
        ;;
    *)
        interactive_monitor
        ;;
esac