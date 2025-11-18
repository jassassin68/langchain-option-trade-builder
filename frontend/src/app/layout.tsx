import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Options Trade Evaluator - AI-Powered Trading Analysis',
  description: 'Get comprehensive options trading analysis with AI-powered technical indicators, fundamental screening, and strategy recommendations for informed trading decisions.',
  keywords: 'options trading, stock analysis, AI trading, technical analysis, fundamental analysis, trading strategies',
  authors: [{ name: 'Options Trade Evaluator' }],
  viewport: 'width=device-width, initial-scale=1',
  robots: 'index, follow',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full antialiased`}>
        {children}
      </body>
    </html>
  )
}