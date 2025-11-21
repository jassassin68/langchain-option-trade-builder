# Deployment Quick Start

Quick reference for deploying the Options Trade Evaluator frontend to Vercel.

## Prerequisites Checklist

- [ ] Vercel account created
- [ ] Vercel CLI installed: `npm install -g vercel`
- [ ] Backend API deployed and accessible
- [ ] Environment variables prepared

## 5-Minute Deployment

### 1. Set Environment Variables

Create `.env.production` with your backend API URL:

```bash
NEXT_PUBLIC_API_URL=https://your-backend-api.com
NEXT_PUBLIC_APP_NAME=Options Trade Evaluator
```

### 2. Deploy to Vercel

```bash
cd frontend
npm run deploy:prod
```

That's it! The script will:
- ✓ Run all pre-deployment checks
- ✓ Deploy to Vercel
- ✓ Verify the deployment

### 3. Verify Deployment

```bash
npm run verify:deployment -- https://your-app.vercel.app
```

## Common Commands

```bash
# Deploy to different environments
npm run deploy:dev        # Development
npm run deploy:staging    # Staging
npm run deploy:prod       # Production

# Test deployment
npm run test:deployment -- https://your-app.vercel.app

# Monitor deployment
npm run monitor:deployment -- https://your-app.vercel.app

# Build and test locally
npm run build:test
```

## Vercel Dashboard Setup

1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Import your Git repository
3. Set environment variables:
   - `NEXT_PUBLIC_API_URL` → Your backend URL
   - `NEXT_PUBLIC_APP_NAME` → App name
4. Deploy!

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Run `npm run type-check` and `npm run lint` |
| API not connecting | Check `NEXT_PUBLIC_API_URL` in Vercel dashboard |
| Slow performance | Check backend response times |
| 404 errors | Verify routes in `next.config.js` |

## Need More Help?

See [DEPLOYMENT_GUIDE.md](./DEPLOYMENT_GUIDE.md) for detailed instructions.
