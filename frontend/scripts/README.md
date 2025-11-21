# Deployment Scripts

This directory contains scripts for deploying, testing, and monitoring the Options Trade Evaluator frontend.

## Available Scripts

### Deployment Scripts

#### `deploy-vercel.sh` / `deploy-vercel.bat`

Deploys the application to Vercel with pre-deployment checks.

**Usage:**
```bash
# Linux/Mac
./scripts/deploy-vercel.sh [environment]

# Windows
.\scripts\deploy-vercel.bat [environment]

# Examples
./scripts/deploy-vercel.sh production
./scripts/deploy-vercel.sh staging
./scripts/deploy-vercel.sh development
```

**Features:**
- Validates environment configuration
- Runs TypeScript type checking
- Executes ESLint validation
- Runs test suite
- Performs build verification
- Deploys to specified environment
- Verifies deployment success

**Environment Options:**
- `production` / `prod` - Deploy to production
- `staging` - Deploy to staging/preview
- `development` / `dev` - Deploy to development

### Build and Test Scripts

#### `build-and-test.sh`

Builds and tests the application locally before deployment.

**Usage:**
```bash
./scripts/build-and-test.sh [environment]

# Examples
./scripts/build-and-test.sh production
./scripts/build-and-test.sh development
```

**Features:**
- Installs dependencies
- Runs TypeScript type checking
- Executes ESLint
- Runs test suite with coverage
- Builds application
- Analyzes bundle size (if available)

### Verification Scripts

#### `verify-deployment.sh`

Verifies that a deployment is working correctly.

**Usage:**
```bash
./scripts/verify-deployment.sh <deployment-url>

# Example
./scripts/verify-deployment.sh https://your-app.vercel.app
```

**Features:**
- Checks main page accessibility
- Verifies health endpoint
- Tests static assets
- Validates page content
- Checks performance
- Verifies responsive design
- Tests security headers

**Configuration:**
```bash
# Set custom timeout (default: 30 seconds)
TIMEOUT=60 ./scripts/verify-deployment.sh https://your-app.vercel.app

# Set retry count (default: 5)
RETRY_COUNT=10 ./scripts/verify-deployment.sh https://your-app.vercel.app

# Set retry delay (default: 5 seconds)
RETRY_DELAY=10 ./scripts/verify-deployment.sh https://your-app.vercel.app
```

#### `test-deployment.sh` / `test-deployment.bat`

Runs comprehensive tests on a deployed application.

**Usage:**
```bash
# Linux/Mac
./scripts/test-deployment.sh [deployment-url]

# Windows
.\scripts\test-deployment.bat [deployment-url]

# Examples
./scripts/test-deployment.sh https://your-app.vercel.app
./scripts/test-deployment.sh  # Uses http://localhost:3000
```

**Test Categories:**
1. **Basic Connectivity**
   - Main page loads
   - Health endpoint responds
   - Favicon exists

2. **Content Validation**
   - App name present
   - Next.js data included
   - Viewport meta tag

3. **API Health**
   - Health endpoint returns proper JSON
   - All required fields present
   - Backend status reported

4. **Security Headers**
   - X-Frame-Options
   - X-Content-Type-Options
   - Referrer-Policy

5. **Performance**
   - Page load time < 3 seconds
   - Health check < 1 second

6. **Responsive Design**
   - Mobile viewport configured
   - Mobile user agent works

7. **Static Assets**
   - Proper cache headers
   - Assets accessible

### Monitoring Scripts

#### `monitor-deployment.sh` / `monitor-deployment.bat`

Continuously monitors a deployment for health and performance.

**Usage:**
```bash
# Linux/Mac
./scripts/monitor-deployment.sh <deployment-url>

# Windows
.\scripts\monitor-deployment.bat <deployment-url>

# Example
./scripts/monitor-deployment.sh https://your-app.vercel.app
```

**Features:**
- Periodic health checks
- Response time monitoring
- Backend connectivity status
- Error rate tracking
- Performance metrics
- Summary statistics

**Configuration:**
```bash
# Set check interval (default: 60 seconds)
CHECK_INTERVAL=30 ./scripts/monitor-deployment.sh https://your-app.vercel.app

# Set alert threshold (default: 5000ms)
ALERT_THRESHOLD=3000 ./scripts/monitor-deployment.sh https://your-app.vercel.app

# Set maximum checks (default: 10)
MAX_CHECKS=20 ./scripts/monitor-deployment.sh https://your-app.vercel.app
```

**Output:**
- Real-time check results
- Response times
- Backend status
- Summary statistics:
  - Total checks
  - Success rate
  - Average/min/max response times

## NPM Scripts

You can also run these scripts via npm:

```bash
# Deployment
npm run deploy:dev
npm run deploy:staging
npm run deploy:prod

# Build and test
npm run build:test

# Verification
npm run verify:deployment -- https://your-app.vercel.app

# Testing
npm run test:deployment -- https://your-app.vercel.app

# Monitoring
npm run monitor:deployment -- https://your-app.vercel.app
```

## Workflow Examples

### Complete Deployment Workflow

```bash
# 1. Build and test locally
npm run build:test

# 2. Deploy to staging
npm run deploy:staging

# 3. Verify staging deployment
npm run verify:deployment -- https://staging.your-app.vercel.app

# 4. Run comprehensive tests
npm run test:deployment -- https://staging.your-app.vercel.app

# 5. If all good, deploy to production
npm run deploy:prod

# 6. Verify production deployment
npm run verify:deployment -- https://your-app.vercel.app

# 7. Monitor production for a while
npm run monitor:deployment -- https://your-app.vercel.app
```

### Quick Deployment

```bash
# Deploy directly to production (includes pre-checks)
npm run deploy:prod
```

### Post-Deployment Monitoring

```bash
# Run continuous monitoring
npm run monitor:deployment -- https://your-app.vercel.app

# Or run periodic tests
while true; do
  npm run test:deployment -- https://your-app.vercel.app
  sleep 300  # Wait 5 minutes
done
```

## Exit Codes

All scripts follow standard exit code conventions:

- `0` - Success
- `1` - Failure
- `2` - Partial success (warnings)

This allows for easy integration with CI/CD pipelines:

```bash
# Example CI/CD usage
if npm run deploy:prod; then
  echo "Deployment successful"
  npm run verify:deployment -- $DEPLOYMENT_URL
else
  echo "Deployment failed"
  exit 1
fi
```

## Requirements

### Linux/Mac Scripts

- `bash` shell
- `curl` command
- `bc` calculator (for performance calculations)
- `grep` for text matching
- Vercel CLI (for deployment scripts)

### Windows Scripts

- Windows Command Prompt or PowerShell
- `curl` command (included in Windows 10+)
- Vercel CLI (for deployment scripts)

## Troubleshooting

### Script Permission Issues (Linux/Mac)

If you get permission denied errors:

```bash
chmod +x scripts/*.sh
```

### Vercel CLI Not Found

Install Vercel CLI globally:

```bash
npm install -g vercel
```

### Curl Not Found (Windows)

Curl is included in Windows 10 and later. For older versions:

1. Download curl from https://curl.se/windows/
2. Add to PATH
3. Or use Git Bash which includes curl

### Script Fails with "Command not found"

Ensure you're running the script from the frontend directory:

```bash
cd frontend
./scripts/deploy-vercel.sh production
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Vercel

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup Node.js
        uses: actions/setup-node@v2
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: npm ci
        working-directory: ./frontend
      
      - name: Build and test
        run: npm run build:test
        working-directory: ./frontend
      
      - name: Deploy to Vercel
        run: npm run deploy:prod
        working-directory: ./frontend
        env:
          VERCEL_TOKEN: ${{ secrets.VERCEL_TOKEN }}
      
      - name: Verify deployment
        run: npm run verify:deployment -- ${{ secrets.VERCEL_URL }}
        working-directory: ./frontend
```

### GitLab CI Example

```yaml
deploy:
  stage: deploy
  script:
    - cd frontend
    - npm ci
    - npm run build:test
    - npm run deploy:prod
    - npm run verify:deployment -- $VERCEL_URL
  only:
    - main
```

## Best Practices

1. **Always test locally first**: Run `npm run build:test` before deploying
2. **Use staging environment**: Test in staging before production
3. **Verify after deployment**: Always run verification scripts
4. **Monitor production**: Use monitoring scripts after major deployments
5. **Keep scripts updated**: Update scripts when adding new features
6. **Document changes**: Update this README when modifying scripts

## Support

For issues with:
- **Scripts**: Check script output and logs
- **Vercel**: See [Vercel Documentation](https://vercel.com/docs)
- **Deployment**: Review [DEPLOYMENT_GUIDE.md](../DEPLOYMENT_GUIDE.md)
