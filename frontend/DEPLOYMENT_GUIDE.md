# Frontend Deployment Guide

This guide covers deploying the Options Trade Evaluator frontend to Vercel.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Configuration](#environment-configuration)
- [Vercel Setup](#vercel-setup)
- [Deployment Process](#deployment-process)
- [Verification](#verification)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install globally
   ```bash
   npm install -g vercel
   ```
3. **Backend API**: Ensure your backend API is deployed and accessible
4. **Environment Variables**: Prepare your production API URL

## Environment Configuration

### Required Environment Variables

The following environment variables must be configured in Vercel:

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://api.yourdomain.com` |
| `NEXT_PUBLIC_APP_NAME` | Application name | `Options Trade Evaluator` |

### Setting Environment Variables in Vercel

#### Via Vercel Dashboard:

1. Go to your project settings
2. Navigate to "Environment Variables"
3. Add each variable for Production, Preview, and Development environments

#### Via Vercel CLI:

```bash
# Set production environment variable
vercel env add NEXT_PUBLIC_API_URL production

# Set preview environment variable
vercel env add NEXT_PUBLIC_API_URL preview

# Set development environment variable
vercel env add NEXT_PUBLIC_API_URL development
```

### Environment Files

The project includes environment files for different stages:

- `.env.development` - Local development
- `.env.staging` - Staging/preview deployments
- `.env.production` - Production deployments
- `.env.local.example` - Template for local setup

**Important**: Never commit `.env.local` or files containing secrets to version control.

## Vercel Setup

### Initial Project Setup

1. **Link your project to Vercel:**
   ```bash
   cd frontend
   vercel link
   ```

2. **Configure project settings:**
   - Framework Preset: Next.js
   - Build Command: `npm run build`
   - Output Directory: `.next`
   - Install Command: `npm ci`

3. **Set environment variables** (as described above)

### Vercel Configuration

The project includes a `vercel.json` configuration file with:

- **Build settings**: Optimized for Next.js
- **Security headers**: X-Frame-Options, CSP, etc.
- **Cache policies**: Aggressive caching for static assets
- **CORS configuration**: For API routes
- **Redirects**: Health check endpoint

## Deployment Process

### Automated Deployment Scripts

The project includes deployment scripts for different environments:

#### Linux/Mac:

```bash
# Deploy to development
npm run deploy:dev

# Deploy to staging
npm run deploy:staging

# Deploy to production
npm run deploy:prod
```

#### Windows:

```bash
# Deploy to production
.\scripts\deploy-vercel.bat production

# Deploy to staging
.\scripts\deploy-vercel.bat staging
```

### Manual Deployment

#### Development/Preview Deployment:

```bash
cd frontend
vercel
```

#### Production Deployment:

```bash
cd frontend
vercel --prod
```

### Deployment Workflow

The deployment script performs the following steps:

1. **Pre-deployment checks:**
   - TypeScript type checking
   - ESLint validation
   - Test execution
   - Build verification

2. **Deployment:**
   - Uploads code to Vercel
   - Builds the application
   - Deploys to specified environment

3. **Post-deployment:**
   - Retrieves deployment URL
   - Runs basic verification
   - Reports status

## Verification

### Automated Verification

Run the deployment verification script:

```bash
# Linux/Mac
./scripts/verify-deployment.sh https://your-app.vercel.app

# Windows
.\scripts\verify-deployment.bat https://your-app.vercel.app
```

### Comprehensive Testing

Run the full test suite:

```bash
# Linux/Mac
./scripts/test-deployment.sh https://your-app.vercel.app

# Windows
.\scripts\test-deployment.bat https://your-app.vercel.app
```

This tests:
- Basic connectivity
- Content validation
- API health endpoints
- Security headers
- Performance metrics
- Responsive design
- Static assets

### Manual Verification Checklist

- [ ] Main page loads successfully
- [ ] Health endpoint returns 200: `/api/health`
- [ ] Ticker search functionality works
- [ ] Analysis results display correctly
- [ ] Error handling works properly
- [ ] Mobile responsive design works
- [ ] Security headers are present
- [ ] Backend API connectivity confirmed

## Monitoring

### Continuous Monitoring

Run the monitoring script to track deployment health:

```bash
# Linux/Mac
./scripts/monitor-deployment.sh https://your-app.vercel.app

# Windows
.\scripts\monitor-deployment.bat https://your-app.vercel.app
```

This monitors:
- Health endpoint status
- Page load times
- Backend connectivity
- Error rates
- Response times

### Vercel Analytics

Enable Vercel Analytics for production insights:

1. Go to your project in Vercel Dashboard
2. Navigate to "Analytics"
3. Enable Web Analytics
4. View real-time metrics and performance data

### Custom Monitoring

The health endpoint (`/api/health`) provides:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "version": "1.0.0",
  "environment": "production",
  "api_url": "https://api.yourdomain.com",
  "backend_status": "healthy",
  "uptime": 12345,
  "memory": {
    "used": 50,
    "total": 100
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Build Failures

**Problem**: Build fails during deployment

**Solutions**:
- Check TypeScript errors: `npm run type-check`
- Check ESLint errors: `npm run lint`
- Verify all dependencies are in `package.json`
- Check build logs in Vercel dashboard

#### 2. Environment Variables Not Working

**Problem**: API calls fail or use wrong URL

**Solutions**:
- Verify variables are set in Vercel dashboard
- Ensure variables start with `NEXT_PUBLIC_` for client-side access
- Redeploy after changing environment variables
- Check browser console for actual values

#### 3. API Connection Issues

**Problem**: Frontend can't connect to backend

**Solutions**:
- Verify `NEXT_PUBLIC_API_URL` is correct
- Check CORS configuration on backend
- Ensure backend is deployed and accessible
- Test backend health endpoint directly
- Check network tab in browser DevTools

#### 4. Slow Performance

**Problem**: Pages load slowly

**Solutions**:
- Check Vercel Analytics for bottlenecks
- Verify static assets are cached properly
- Check backend API response times
- Enable compression in backend
- Review bundle size with `npm run build:analyze`

#### 5. Security Header Issues

**Problem**: Security headers not appearing

**Solutions**:
- Verify `vercel.json` configuration
- Check `next.config.js` headers
- Test with: `curl -I https://your-app.vercel.app`
- Redeploy if configuration changed

### Getting Help

1. **Check Vercel Logs**: View deployment and runtime logs in dashboard
2. **Review Documentation**: [Vercel Next.js Docs](https://vercel.com/docs/frameworks/nextjs)
3. **Test Locally**: Run `npm run build && npm start` to test production build
4. **Contact Support**: Use Vercel support for platform issues

## Best Practices

### Before Deployment

- [ ] Run all tests locally
- [ ] Verify environment variables
- [ ] Test production build locally
- [ ] Review security headers
- [ ] Check bundle size
- [ ] Update documentation

### After Deployment

- [ ] Run verification scripts
- [ ] Test all critical paths
- [ ] Monitor for errors
- [ ] Check performance metrics
- [ ] Verify backend connectivity
- [ ] Test on multiple devices

### Continuous Deployment

For automatic deployments:

1. Connect your Git repository to Vercel
2. Configure branch deployments:
   - `main` → Production
   - `staging` → Preview
   - Feature branches → Preview
3. Enable automatic deployments on push
4. Set up deployment notifications

## Performance Optimization

### Build Optimization

The project includes several optimizations:

- **Code splitting**: Automatic via Next.js
- **Image optimization**: WebP and AVIF formats
- **Static generation**: Pre-rendered pages
- **Bundle analysis**: Available via `npm run build:analyze`

### Caching Strategy

- **Static assets**: 1 year cache (`immutable`)
- **Images**: 1 week cache with revalidation
- **API routes**: No cache
- **Pages**: ISR with revalidation

### CDN Configuration

Vercel automatically provides:
- Global CDN distribution
- Edge caching
- Automatic HTTPS
- DDoS protection

## Security Considerations

### Headers

The deployment includes security headers:
- `X-Frame-Options: DENY`
- `X-Content-Type-Options: nosniff`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy`: Configured for trusted sources

### Environment Variables

- Never expose sensitive keys in client-side code
- Use `NEXT_PUBLIC_` prefix only for non-sensitive values
- Rotate API keys regularly
- Use Vercel's encrypted environment variables

### CORS

- Configure backend to allow your Vercel domain
- Use specific origins instead of wildcards in production
- Implement rate limiting on backend

## Rollback Procedure

If a deployment has issues:

1. **Via Vercel Dashboard:**
   - Go to Deployments
   - Find previous working deployment
   - Click "Promote to Production"

2. **Via CLI:**
   ```bash
   vercel rollback
   ```

3. **Via Git:**
   ```bash
   git revert HEAD
   git push origin main
   ```

## Additional Resources

- [Next.js Deployment Documentation](https://nextjs.org/docs/deployment)
- [Vercel Documentation](https://vercel.com/docs)
- [Vercel CLI Reference](https://vercel.com/docs/cli)
- [Next.js Performance Best Practices](https://nextjs.org/docs/advanced-features/measuring-performance)
