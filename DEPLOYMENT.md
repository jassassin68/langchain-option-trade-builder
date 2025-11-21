# Deployment Guide

This guide covers the deployment configuration for the Options Trade Evaluator backend application.

## Overview

The application supports multiple deployment environments:
- **Development**: Local development with hot reload
- **Production**: Optimized production deployment with security features

## Prerequisites

- Docker and Docker Compose installed
- Environment variables configured
- Required API keys (OpenAI, Alpha Vantage, Tradier)

## Environment Configuration

### Development Environment

1. Copy the development environment template:
   ```bash
   cp .env.development .env.development.local
   ```

2. Update the API keys in `.env.development.local`:
   ```bash
   OPENAI_API_KEY=your_actual_dev_key
   ALPHA_VANTAGE_API_KEY=your_actual_dev_key
   TRADIER_API_KEY=your_actual_dev_key
   ```

### Production Environment

1. Copy the production environment template:
   ```bash
   cp .env.production .env.production.local
   ```

2. Update all configuration values in `.env.production.local`:
   - Change all `CHANGE_ME_*` values to secure production values
   - Use strong passwords for database and Redis
   - Use production API keys
   - Configure proper domain names if using HTTPS

## Deployment Methods

### Development Deployment

#### Using Scripts (Recommended)

**Linux/macOS:**
```bash
./scripts/deploy-dev.sh
```

**Windows:**
```cmd
scripts\deploy-dev.bat
```

#### Manual Deployment

```bash
# Load development environment
export $(cat .env.development | grep -v '^#' | xargs)

# Start services
docker-compose --env-file .env.development up --build -d

# Check status
docker-compose ps
```

### Production Deployment

#### Using Scripts (Recommended)

**Linux/macOS:**
```bash
./scripts/deploy-prod.sh
```

#### Manual Deployment

```bash
# Load production environment
export $(cat .env.production | grep -v '^#' | xargs)

# Start production services
docker-compose -f docker-compose.prod.yml --env-file .env.production up --build -d

# Check status
docker-compose -f docker-compose.prod.yml ps
```

## Docker Configuration

### Multi-Stage Dockerfile

The production Dockerfile uses multi-stage builds for optimization:

1. **Builder Stage**: Installs dependencies and creates virtual environment
2. **Production Stage**: Copies only necessary files and runs as non-root user

Key features:
- Non-root user for security
- Health checks built-in
- Optimized for production performance
- Multi-worker configuration

### Docker Compose Files

- `docker-compose.yml`: Development environment with hot reload
- `docker-compose.prod.yml`: Production environment with Nginx reverse proxy

## Services

### Backend (FastAPI)
- **Development**: Hot reload enabled, debug mode on
- **Production**: Multi-worker, optimized settings, security headers

### PostgreSQL
- **Development**: Exposed on port 5432 for debugging
- **Production**: Internal networking only, health checks

### Redis
- **Development**: No password for simplicity
- **Production**: Password protected, persistent storage

### Nginx (Production Only)
- Reverse proxy with rate limiting
- SSL/TLS termination ready
- Gzip compression
- Security headers

## Monitoring and Health Checks

### Health Check Script

**Linux/macOS:**
```bash
./scripts/health-check.sh
```

**Windows:**
```cmd
scripts\health-check.bat
```

### Monitoring Script

```bash
./scripts/monitor.sh
```

Interactive monitoring with options:
- `r`: Refresh status
- `l`: View logs
- `h`: Run health checks
- `s`: Open shell in backend container
- `q`: Quit

### Manual Health Checks

```bash
# Check API health
curl http://localhost:8000/api/v1/health

# Check service status
docker-compose ps

# View logs
docker-compose logs backend

# Check resource usage
docker stats
```

## Security Considerations

### Production Security Features

1. **Non-root container execution**
2. **Security headers via Nginx**
3. **Rate limiting on API endpoints**
4. **Environment variable validation**
5. **Health check endpoints**
6. **Resource limits**

### Environment Variables Security

- Never commit `.env.production` to version control
- Use strong, unique passwords
- Rotate API keys regularly
- Use secrets management in cloud deployments

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in environment files
2. **Permission errors**: Ensure Docker has proper permissions
3. **API key errors**: Verify API keys are valid and have proper permissions
4. **Database connection**: Check PostgreSQL is running and accessible

### Debug Commands

```bash
# View detailed logs
docker-compose logs -f backend

# Access backend container
docker-compose exec backend /bin/bash

# Check database connection
docker-compose exec postgres psql -U postgres -d options_db

# Check Redis connection
docker-compose exec redis redis-cli ping
```

### Performance Tuning

1. **Database**: Adjust connection pool settings
2. **Redis**: Configure memory limits and persistence
3. **Backend**: Tune worker count based on CPU cores
4. **Nginx**: Adjust worker processes and connections

## Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec postgres pg_dump -U postgres options_db > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U postgres options_db < backup.sql
```

### Volume Backup

```bash
# Backup volumes
docker run --rm -v options_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz -C /data .
```

## Scaling Considerations

### Horizontal Scaling

1. **Load Balancer**: Add load balancer in front of multiple backend instances
2. **Database**: Consider read replicas for heavy read workloads
3. **Cache**: Use Redis Cluster for distributed caching
4. **API Rate Limiting**: Implement distributed rate limiting

### Vertical Scaling

1. **CPU**: Increase worker count for backend
2. **Memory**: Adjust container memory limits
3. **Storage**: Use faster storage for database
4. **Network**: Optimize network configuration

## Cloud Deployment

### AWS Deployment

- Use ECS or EKS for container orchestration
- RDS for managed PostgreSQL
- ElastiCache for managed Redis
- ALB for load balancing

### Google Cloud Deployment

- Use Cloud Run or GKE
- Cloud SQL for PostgreSQL
- Memorystore for Redis
- Cloud Load Balancing

### Azure Deployment

- Use Container Instances or AKS
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Azure Load Balancer

## Maintenance

### Regular Tasks

1. **Update dependencies**: Regularly update Docker images and Python packages
2. **Monitor logs**: Check for errors and performance issues
3. **Database maintenance**: Run VACUUM and ANALYZE on PostgreSQL
4. **Security updates**: Keep base images and dependencies updated
5. **Backup verification**: Regularly test backup and restore procedures

### Monitoring Metrics

- API response times
- Database connection pool usage
- Redis memory usage
- Container resource utilization
- Error rates and types