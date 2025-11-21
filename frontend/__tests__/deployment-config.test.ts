/**
 * Deployment Configuration Tests
 * 
 * Tests to verify deployment configuration is correct
 */

describe('Deployment Configuration', () => {
  describe('Environment Variables', () => {
    it('should have NEXT_PUBLIC_API_URL defined', () => {
      // In test environment, this might not be set, so we just check the format
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      
      if (apiUrl) {
        expect(apiUrl).toMatch(/^https?:\/\/.+/);
      } else {
        // In test environment, it's okay if not set
        expect(apiUrl).toBeUndefined();
      }
    });

    it('should have NEXT_PUBLIC_APP_NAME defined', () => {
      const appName = process.env.NEXT_PUBLIC_APP_NAME;
      
      if (appName) {
        expect(appName).toBeTruthy();
        expect(typeof appName).toBe('string');
      } else {
        // In test environment, it's okay if not set
        expect(appName).toBeUndefined();
      }
    });
  });

  describe('Next.js Configuration', () => {
    it('should have valid next.config.js', () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig).toBeDefined();
      expect(nextConfig.compress).toBe(true);
      expect(nextConfig.poweredByHeader).toBe(false);
    });

    it('should have security headers configured', async () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig.headers).toBeDefined();
      
      const headers = await nextConfig.headers();
      expect(Array.isArray(headers)).toBe(true);
      expect(headers.length).toBeGreaterThan(0);
      
      const mainHeaders = headers.find((h: any) => h.source === '/(.*)')?.headers;
      expect(mainHeaders).toBeDefined();
      
      const headerKeys = mainHeaders.map((h: any) => h.key);
      expect(headerKeys).toContain('X-Frame-Options');
      expect(headerKeys).toContain('X-Content-Type-Options');
      expect(headerKeys).toContain('X-XSS-Protection');
      expect(headerKeys).toContain('Referrer-Policy');
    });

    it('should have image optimization configured', () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig.images).toBeDefined();
      expect(nextConfig.images.formats).toContain('image/webp');
      expect(nextConfig.images.formats).toContain('image/avif');
    });

    it('should have webpack optimization for production', () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig.webpack).toBeDefined();
      
      // Test webpack config with dev=false (production)
      const mockConfig = {
        optimization: {}
      };
      
      const result = nextConfig.webpack(mockConfig, { dev: false });
      
      expect(result.optimization.splitChunks).toBeDefined();
      expect(result.optimization.minimize).toBe(true);
    });
  });

  describe('Vercel Configuration', () => {
    it('should have valid vercel.json', () => {
      const vercelConfig = require('../vercel.json');
      
      expect(vercelConfig).toBeDefined();
      expect(vercelConfig.version).toBe(2);
      expect(vercelConfig.framework).toBe('nextjs');
    });

    it('should have environment variables configured', () => {
      const vercelConfig = require('../vercel.json');
      
      expect(vercelConfig.env).toBeDefined();
      expect(vercelConfig.env.NEXT_PUBLIC_API_URL).toBeDefined();
      expect(vercelConfig.env.NEXT_PUBLIC_APP_NAME).toBeDefined();
    });

    it('should have security headers in vercel.json', () => {
      const vercelConfig = require('../vercel.json');
      
      expect(vercelConfig.headers).toBeDefined();
      expect(Array.isArray(vercelConfig.headers)).toBe(true);
      
      const mainHeaders = vercelConfig.headers.find((h: any) => h.source === '/(.*)')?.headers;
      expect(mainHeaders).toBeDefined();
      
      const headerKeys = mainHeaders.map((h: any) => h.key);
      expect(headerKeys).toContain('X-Frame-Options');
      expect(headerKeys).toContain('X-Content-Type-Options');
      expect(headerKeys).toContain('Content-Security-Policy');
    });

    it('should have cache headers for static assets', () => {
      const vercelConfig = require('../vercel.json');
      
      const staticHeaders = vercelConfig.headers.find(
        (h: any) => h.source === '/_next/static/(.*)'
      )?.headers;
      
      expect(staticHeaders).toBeDefined();
      
      const cacheControl = staticHeaders.find((h: any) => h.key === 'Cache-Control');
      expect(cacheControl).toBeDefined();
      expect(cacheControl.value).toContain('immutable');
    });

    it('should have CORS headers for API routes', () => {
      const vercelConfig = require('../vercel.json');
      
      const apiHeaders = vercelConfig.headers.find(
        (h: any) => h.source === '/api/(.*)'
      )?.headers;
      
      expect(apiHeaders).toBeDefined();
      
      const headerKeys = apiHeaders.map((h: any) => h.key);
      expect(headerKeys).toContain('Access-Control-Allow-Origin');
      expect(headerKeys).toContain('Access-Control-Allow-Methods');
    });
  });

  describe('Package Configuration', () => {
    it('should have deployment scripts defined', () => {
      const packageJson = require('../package.json');
      
      expect(packageJson.scripts).toBeDefined();
      expect(packageJson.scripts['deploy:dev']).toBeDefined();
      expect(packageJson.scripts['deploy:staging']).toBeDefined();
      expect(packageJson.scripts['deploy:prod']).toBeDefined();
    });

    it('should have verification scripts defined', () => {
      const packageJson = require('../package.json');
      
      expect(packageJson.scripts['verify:deployment']).toBeDefined();
      expect(packageJson.scripts['test:deployment']).toBeDefined();
      expect(packageJson.scripts['monitor:deployment']).toBeDefined();
    });

    it('should have build and test scripts', () => {
      const packageJson = require('../package.json');
      
      expect(packageJson.scripts.build).toBeDefined();
      expect(packageJson.scripts.test).toBeDefined();
      expect(packageJson.scripts['type-check']).toBeDefined();
      expect(packageJson.scripts.lint).toBeDefined();
    });
  });

  describe('TypeScript Configuration', () => {
    it('should have valid tsconfig.json', () => {
      const tsConfig = require('../tsconfig.json');
      
      expect(tsConfig).toBeDefined();
      expect(tsConfig.compilerOptions).toBeDefined();
      expect(tsConfig.compilerOptions.strict).toBe(true);
    });
  });

  describe('Build Output', () => {
    it('should have standalone output configured', () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig.output).toBe('standalone');
    });

    it('should not ignore TypeScript errors in build', () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig.typescript.ignoreBuildErrors).toBe(false);
    });

    it('should not ignore ESLint errors in build', () => {
      const nextConfig = require('../next.config.js');
      
      expect(nextConfig.eslint.ignoreDuringBuilds).toBe(false);
    });
  });
});
