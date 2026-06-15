import React from 'react';

export default function Home({ setPage }) {
  return (
    <div>
      <div className="hero-card">
        <div className="hero-content">
          <span className="hero-badge">Enterprise AI</span>
          <h1 className="hero-title">MarketPulse AI</h1>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Real-Time Financial Sentiment Intelligence
          </h2>
          <p className="hero-subtitle">
            An enterprise-grade platform that bridges the gap between global news streams and actionable market data. 
            Ingest financial news, parse tickers, extract key entities, and analyze sentiment using specialized deep learning ensembles.
          </p>
          <div style={{ display: 'flex', gap: '12px' }}>
            <button className="btn btn-primary" onClick={() => setPage('Dashboard')}>
              📊 Launch Dashboard
            </button>
            <button className="btn btn-secondary" onClick={() => setPage('Market Data')}>
              📈 Check Market Data
            </button>
          </div>
        </div>
        <img 
          src="https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=2000&auto=format&fit=crop" 
          alt="Financial Markets Graph" 
          className="hero-image-placeholder"
        />
      </div>

      <div className="grid-2" style={{ marginBottom: '32px' }}>
        <div className="card">
          <h3 className="card-title">⚠️ The Problem Statement</h3>
          <p style={{ marginBottom: '12px', color: 'var(--text-secondary)' }}>
            1. **Information Overload:** Thousands of financial articles are published every single day. Retail and institutional traders cannot possibly read or process this massive volume of text.
          </p>
          <p style={{ marginBottom: '12px', color: 'var(--text-secondary)' }}>
            2. **Nuance Misinterpretation:** Standard lexicon-based algorithms fail to understand nuanced financial terminology. For instance, "Company X cuts debt" is a 'cut' (negative in plain english) but is financially positive.
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            3. **Delayed Reaction Time:** By the time a human trader manually reads, digests, and maps a news report to the stock chart, algorithmic trading bots have already priced it in.
          </p>
        </div>

        <div className="card">
          <h3 className="card-title">💡 The Solution</h3>
          <p style={{ marginBottom: '12px', color: 'var(--text-secondary)' }}>
            An automated NLP pipeline that ingests raw, real-time data from RSS feeds, WallStreetBets, and web scrapers.
          </p>
          <p style={{ marginBottom: '12px', color: 'var(--text-secondary)' }}>
            It feeds unstructured text into **FinBERT** (a custom BERT model fine-tuned on financial jargon) to achieve state-of-the-art sentiment accuracy, categorizes topics automatically, and aligns that data against Yahoo Finance charting.
          </p>
          <p style={{ color: 'var(--text-secondary)' }}>
            <strong>Lazy Loading Neural Networks:</strong> To keep resource demands light, heavy networks (FinBERT, Summarizers, NER) are loaded dynamically on request, preventing out-of-memory errors.
          </p>
        </div>
      </div>

      <div className="card" style={{ marginBottom: '32px' }}>
        <h3 className="card-title" style={{ marginBottom: '16px' }}>🚀 How To Use This Platform</h3>
        <div className="grid-3">
          <div style={{ padding: '12px', borderLeft: '3px solid var(--primary)' }}>
            <h4 style={{ marginBottom: '6px' }}>1. Analyze Topics</h4>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Head over to the <strong>Dashboard</strong>. Search for a stock ticker (e.g., AAPL, NVDA) or macro topic (e.g., Inflation) to scrape, analyze, and classify sentiment.
            </p>
          </div>
          <div style={{ padding: '12px', borderLeft: '3px solid var(--positive)' }}>
            <h4 style={{ marginBottom: '6px' }}>2. Cross-Reference Markets</h4>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Visit <strong>Market Data</strong> to view real-time technical stock prices mapped alongside live retail sentiment charts compiled directly from WallStreetBets.
            </p>
          </div>
          <div style={{ padding: '12px', borderLeft: '3px solid var(--neutral)' }}>
            <h4 style={{ marginBottom: '6px' }}>3. Perform Deep Dives</h4>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              Browse analyzed headlines in <strong>Scraped Articles</strong>. Select any article to trigger full page scraping, AI Summarization, and Named Entity Extraction (NER).
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="card-title" style={{ marginBottom: '16px' }}>🛠️ Key Features & Technologies</h3>
        <div className="grid-3" style={{ fontSize: '0.95rem' }}>
          <div>
            <h4 style={{ color: 'var(--primary)', marginBottom: '8px' }}>Python & FastAPI</h4>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <li>Asynchronous APIs</li>
              <li>Feedparser (RSS news extraction)</li>
              <li>BeautifulSoup4 (Custom scrapers)</li>
              <li>yFinance integration</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--positive-dark)', marginBottom: '8px' }}>Deep Learning & NLP</h4>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <li>FinBERT Sentiment Engine</li>
              <li>BERT Named Entity Recognition (NER)</li>
              <li>BART Neural Summarizer</li>
              <li>Zero-Shot Topic Modeling</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--neutral-dark)', marginBottom: '8px' }}>React UI (Light Theme)</h4>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <li>Single Page Router</li>
              <li>Interactive SVG Candlesticks</li>
              <li>Real-time progress bars</li>
              <li>Responsive CSS Flexbox/Grid</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
