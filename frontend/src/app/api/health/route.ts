import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Check if the application is running properly
    const healthData = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: process.env.npm_package_version || '1.0.0',
      environment: process.env.NODE_ENV || 'development',
      api_url: process.env.NEXT_PUBLIC_API_URL || 'not configured',
      app_name: process.env.NEXT_PUBLIC_APP_NAME || 'Options Trade Evaluator',
      uptime: process.uptime(),
      memory: {
        used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
        total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
      },
    };

    // Optional: Check backend API connectivity
    let backendStatus = 'unknown';
    if (process.env.NEXT_PUBLIC_API_URL) {
      try {
        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/v1/health`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          // Short timeout for health check
          signal: AbortSignal.timeout(5000),
        });
        
        backendStatus = response.ok ? 'healthy' : 'unhealthy';
      } catch (error) {
        backendStatus = 'unreachable';
      }
    }

    return NextResponse.json({
      ...healthData,
      backend_status: backendStatus,
    }, {
      status: 200,
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    });
  } catch (error) {
    return NextResponse.json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error instanceof Error ? error.message : 'Unknown error',
    }, {
      status: 500,
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0',
      },
    });
  }
}