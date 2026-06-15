import React from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  Cpu, 
  Zap, 
  ArrowRight, 
  Search, 
  LineChart, 
  ShieldCheck, 
  Database,
  Sparkles
} from 'lucide-react';

export default function Home({ setPage }) {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.05
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 100 } }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}
    >
      {/* Premium Glowing Hero */}
      <motion.div variants={itemVariants} className="hero-card">
        <div className="hero-content">
          <span className="hero-badge">
            <Sparkles size={12} /> Enterprise AI Intelligence
          </span>
          <h1 className="hero-title">MarketPulse AI</h1>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '16px' }}>
            Real-Time Financial Sentiment Intelligence Terminal
          </h2>
          <p className="hero-subtitle">
            An advanced cognitive platform bridging the gap between raw web chatter, news networks, and technical price data. 
            Automate your research pipeline with fine-tuned natural language transformer models.
          </p>
          <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            <button className="btn btn-primary" onClick={() => setPage('Dashboard')}>
              Launch Dashboard <ArrowRight size={16} />
            </button>
            <button className="btn btn-secondary" onClick={() => setPage('Market Data')}>
              Check Market Data
            </button>
          </div>
        </div>
        
        <div className="hero-image-container">
          <div className="hero-glow-back"></div>
          <img 
            src="https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=600&auto=format&fit=crop" 
            alt="Financial Markets Graph" 
            className="hero-image-placeholder"
          />
        </div>
      </motion.div>

      {/* Grid: Problem Statement vs. Solution */}
      <div className="grid-2">
        <motion.div variants={itemVariants} className="card card-hover">
          <h3 className="card-title" style={{ color: 'var(--negative-dark)' }}>
            <Zap size={18} /> The Information Challenge
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '8px' }}>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              <strong>1. Mass Sentiment Overload:</strong> Over 10,000 headlines and countless social media articles are generated hourly. Humans cannot scale to read or categorize this data in real time.
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              <strong>2. Semantic Blind Spots:</strong> Standard word-matching rules misinterpret context. Phrases like <em>"Fed cuts interest rates"</em> contain negative words but represent bullish catalysts.
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              <strong>3. Fragmentation Friction:</strong> Traders waste crucial minutes switching between charts, news tickers, and Reddit, losing arbitrage opportunities to bots.
            </p>
          </div>
        </motion.div>

        <motion.div variants={itemVariants} className="card card-hover">
          <h3 className="card-title" style={{ color: 'var(--positive-dark)' }}>
            <Cpu size={18} /> Neural Network Solution
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '8px' }}>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              <strong>1. Cognitive Automation:</strong> Automated ingestion systems query global RSS networks and Reddit WSB streams to feed models continuously.
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              <strong>2. Context-Aware NLP:</strong> Leveraging <strong>FinBERT</strong>, pre-trained on finance-specific datasets, to accurately label market sentiment.
            </p>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.95rem' }}>
              <strong>3. Unified Terminal:</strong> Merging technical candlesticks with NLP scores and named entities in a single, high-fidelity browser dashboard.
            </p>
          </div>
        </motion.div>
      </div>

      {/* Feature Walkthrough */}
      <motion.div variants={itemVariants} className="card">
        <h3 className="card-title" style={{ marginBottom: '24px' }}>
          <Database size={18} /> Automated Ingestion & Pipeline Stages
        </h3>
        <div className="grid-3">
          <div style={{ padding: '16px', borderLeft: '3px solid var(--primary)', background: 'rgba(99, 102, 241, 0.02)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0' }}>
            <h4 style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>
              <Search size={16} /> 1. Query & Scrape
            </h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Scan global RSS feeds for target tickers or macro keywords. Scrape article text body on-demand.
            </p>
          </div>
          
          <div style={{ padding: '16px', borderLeft: '3px solid var(--positive-dark)', background: 'rgba(16, 185, 129, 0.02)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0' }}>
            <h4 style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>
              <TrendingUp size={16} /> 2. Model Inference
            </h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Compute sentiment compound scores, extract organizational entities, and summarize paragraphs.
            </p>
          </div>
          
          <div style={{ padding: '16px', borderLeft: '3px solid var(--neutral-dark)', background: 'rgba(245, 158, 11, 0.02)', borderRadius: '0 var(--radius-sm) var(--radius-sm) 0' }}>
            <h4 style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1rem' }}>
              <LineChart size={16} /> 3. Technical Overlay
            </h4>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Overlay news catalysts against historical trading candlesticks and social metrics.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Grid: Technology stack */}
      <motion.div variants={itemVariants} className="card">
        <h3 className="card-title" style={{ marginBottom: '20px' }}>
          <ShieldCheck size={18} /> Deep Learning Stack
        </h3>
        <div className="grid-3" style={{ fontSize: '0.9rem' }}>
          <div>
            <h4 style={{ color: 'var(--primary)', marginBottom: '10px', fontWeight: 600 }}>Python API Server</h4>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px', color: 'var(--text-secondary)' }}>
              <li>FastAPI Async Service</li>
              <li>BeautifulSoup4 & RSS Parsers</li>
              <li>yFinance Integration</li>
              <li>CPU-optimized inference engine</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--positive-dark)', marginBottom: '10px', fontWeight: 600 }}>NLP Neural Models</h4>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px', color: 'var(--text-secondary)' }}>
              <li>ProsusAI/finbert (Sentiment)</li>
              <li>DistilBERT-MNLI (Topics)</li>
              <li>BART-Text-Summarization (TL;DR)</li>
              <li>BERT-Base-NER (Entities)</li>
            </ul>
          </div>
          <div>
            <h4 style={{ color: 'var(--text-primary)', marginBottom: '10px', fontWeight: 600 }}>SaaS React Front-end</h4>
            <ul style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '8px', color: 'var(--text-secondary)' }}>
              <li>Framer Motion Animations</li>
              <li>Lucide Vector Icons</li>
              <li>Glassmorphic CSS Theme</li>
              <li>Responsive Chart Engine</li>
            </ul>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
