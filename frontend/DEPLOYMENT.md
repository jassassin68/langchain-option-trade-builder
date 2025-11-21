# Frontend Deployment Guide

This guide covers the deployment configuration for the Options Trade Evaluator frontend application built with Next.js.

## 📚 Documentation

- **[Quick Start Guide](./DEPLOYMENT_QUICKSTART.md)** - Get deployed in 5 minutes
- **[Comprehensive Guide](./DEPLOYMENT_GUIDE.md)** - Detailed deployment instructions
- **[Scripts Documentation](./scripts/README.md)** - Deployment scripts reference

## Overview

The frontend supports multiple deployment environments:
- **Development**: Local development with hot reload
- **Staging**: Preview deployments for testing
- **Production**: Optimized production deployment

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Vercel CLI (for Vercel deployments)
- Environment variables configured

## Environment Configuration

### Environment Files

The application uses different environment files for each deployment stage:

- `.env.development` - Development environment
- `.env.staging` - Staging environment  
- `.env.production` - Production environment
- `.env.local` - Local overrides (not committed to git)

### Required Environment Variables

```bash
# API Configuration
NEXT_PUBLIC_API_URL=https://your-api-domain.com

# Application Configuration
NEXT_PUBLIC_APP_NAME=Options Trade Evaluator
```

### Setting Up Environment Files

1. **Development Environment:**
   ```bash
   cp .env.development .env.local
   # Edit .env.local with your local API URL
   ```

2. **Production Environment:**
   ```bash
   cp .env.production .env.production.local
   # Edit .env.production.local with production values
   ```

## Deployment Methods

### Vercel Deployment (Recommended)

#### Automatic Deployment

1. **Connect Repository:**
   - Connect your GitHub repository to Vercel
   - Vercel will automatically deploy on push to main branch

2. **Configure Environment Variables:**
   ```bash
   # In Vercel dashboard, add environment variables:
   NEXT_PUBLIC_API_URL=https://your-production-api.com
   NEXT_PUBLIC_APP_NAME=Options Trade Evaluator
   ```

#### Manual Deployment

**Using Scripts (Recommended):**

```bash
# Development deployment
npm run deploy:dev

# Staging deployment  
npm run deploy:staging

# Production deployment
npm run deploy:prod
```

**Using Vercel CLI directly:**

```bash
# Install Vercel CLI
npm install -g vercel

# Login to Vercel
vercel login

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

### Other Deployment Platforms

#### Netlify

1. **Build Configuration:**
   ```toml
   # netlify.toml
   [build]
     command = "npm run build"
     publish = ".next"
   
   [build.environment]
     NEXT_PUBLIC_API_URL = "https://your-api-domain.com"
     NEXT_PUBLIC_APP_NAME = "Options Trade Evaluator"
   ```

#### AWS Amplify

1. **Build Configuration:**
   ```yaml
   # amplify.yml
   version: 1
   frontend:
     phases:
       preBuild:
         commands:
           - npm ci
       build:
         commands:
           - npm run build
     artifacts:
       baseDirectory: .next
       files:
         - '**/*'
   ```

#### Docker Deployment

1. **Create Dockerfile:**
   ```dockerfile
   FROM node:18-alpine AS deps
   WORKDIR /app
   COPY package*.json ./
   RUN npm ci --only=production
   
   FROM node:18-alpine AS builder
   WORKDIR /app
   COPY . .
   COPY --from=deps /app/node_modules ./node_modules
   RUN npm run build
   
   FROM node:18-alpine AS runner
   WORKDIR /app
   ENV NODE_ENV production
   
   COPY --from=builder /app/public ./public
   COPY --from=builder /app/.next/standalone ./
   COPY --from=builder /app/.next/static ./.next/static
   
   EXPOSE 3000
   CMD ["node", "server.js"]
   ```

## Build Optimization

### Next.js Configuration

The `next.config.js` includes several optimizations:

- **Compression**: Gzip compression enabled
- **Image Optimization**: WebP and AVIF support
- **Bundle Splitting**: Optimized chunk splitting
- **Security Headers**: Security headers for production
- **Static Asset Caching**: Long-term caching for static assets

### Bundle Analysis

```bash
# Analyze bundle size
npm run build:analyze

# View bundle analyzer report
open .next/analyze/client.html
```

### Performance Optimizations

1. **Image Optimization:**
   - Use Next.js Image component
   - Enable WebP/AVIF formats
   - Implement lazy loading

2. **Code Splitting:**
   - Dynamic imports for large components
   - Route-based code splitting
   - Vendor chunk optimization

3. **Caching Strategy:**
   - Static assets: 1 year cache
   - API responses: No cache
   - Build assets: Immutable cache

## Testing and Quality Assurance

### Pre-deployment Checks

```bash
# Run all checks
npm run build:test

# Individual checks
npm run type-check    # TypeScript checking
npm run lint         # ESLint checking  
npm run test         # Jest tests
npm run build        # Build verification
```

### Deployment Verification

```bash
# Verify deployment
bash scripts/verify-deployment.sh https://your-deployment-url.com

# Local verification
npm run start
bash scripts/verify-deployment.sh http://localhost:3000
```

### Automated Testing

```bash
# Unit tests
npm run test

# Coverage report
npm run test:coverage

# Watch mode for development
npm run test:watch
```

## Monitoring and Analytics

### Performance Monitoring

1. **Web Vitals:**
   - Core Web Vitals tracking
   - Real User Monitoring (RUM)
   - Performance budgets

2. **Error Tracking:**
   - Runtime error monitoring
   - Build error tracking
   - User experience errors

### Analytics Integration

```javascript
// Add to next.config.js for Google Analytics
const nextConfig = {
  env: {
    NEXT_PUBLIC_GA_ID: process.env.NEXT_PUBLIC_GA_ID,
  },
};
```

## Security Considerations

### Security Headers

The application includes security headers:
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`

### Environment Variable Security

- Never expose sensitive data in `NEXT_PUBLIC_*` variables
- Use server-side environment variables for secrets
- Validate environment variables at build time

### Content Security Policy

```javascript
// Add to next.config.js
const nextConfig = {
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
          },
        ],
      },
    ];
  },
};
```

## Troubleshooting

### Common Issues

1. **Build Failures:**
   ```bash
   # Clear Next.js cache
   rm -rf .next
   
   # Clear node modules
   rm -rf node_modules package-lock.json
   npm install
   ```

2. **Environment Variable Issues:**
   ```bash
   # Check environment variables
   npm run build -- --debug
   
   # Verify environment loading
   console.log(process.env.NEXT_PUBLIC_API_URL);
   ```

3. **Performance Issues:**
   ```bash
   # Analyze bundle
   npm run build:analyze
   
   # Check for large dependencies
   npx bundle-analyzer .next/static/chunks/*.js
   ```

### Debug Commands

```bash
# Verbose build output
npm run build -- --debug

# Check TypeScript issues
npm run type-check

# Lint with auto-fix
npm run lint:fix

# Test with coverage
npm run test:coverage
```

## Rollback Procedures

### Vercel Rollback

```bash
# List deployments
vercel ls

# Promote previous deployment
vercel promote <deployment-url>
```

### Git-based Rollback

```bash
# Revert to previous commit
git revert HEAD

# Push to trigger new deployment
git push origin main
```

## Maintenance

### Regular Tasks

1. **Dependency Updates:**
   ```bash
   # Check outdated packages
   npm outdated
   
   # Update dependencies
   npm update
   
   # Security audit
   npm audit
   ```

2. **Performance Monitoring:**
   - Monitor Core Web Vitals
   - Check bundle size growth
   - Review error rates

3. **Security Updates:**
   - Regular dependency updates
   - Security header validation
   - Environment variable rotation

### Monitoring Checklist

- [ ] Deployment successful
- [ ] All pages loading correctly
- [ ] API integration working
- [ ] Performance metrics acceptable
- [ ] Error rates within normal range
- [ ] Security headers present
- [ ] Mobile responsiveness working

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy Frontend

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: npm ci
      - run: npm run type-check
      - run: npm run lint
      - run: npm run test
      - run: npm run build

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      - uses: amondnet/vercel-action@v20
        with:
          vercel-token: ${{ secrets.VERCEL_TOKEN }}
          vercel-org-id: ${{ secrets.ORG_ID }}
          vercel-project-id: ${{ secrets.PROJECT_ID }}
          vercel-args: '--prod'
```

This deployment guide provides comprehensive instructions for deploying the Options Trade Evaluator frontend across different platforms and environments.