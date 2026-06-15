import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowLeft, 
  Sparkles, 
  Cpu, 
  Layers, 
  Clock, 
  ExternalLink,
  Bot,
  Building,
  User,
  MapPin,
  Tag
} from 'lucide-react';

export default function ArticleView({ article, setPage }) {
  const [loadingScrape, setLoadingScrape] = useState(false);
  const [scrapedData, setScrapedData] = useState(null);
  const [scrapeError, setScrapeError] = useState(null);

  const [loadingSummary, setLoadingSummary] = useState(false);
  const [summary, setSummary] = useState('');
  const [summaryError, setSummaryError] = useState(null);

  const [loadingEntities, setLoadingEntities] = useState(false);
  const [entities, setEntities] = useState(null);
  const [entitiesError, setEntitiesError] = useState(null);

  useEffect(() => {
    if (!article) return;
    
    const triggerScrape = async () => {
      setLoadingScrape(true);
      setScrapeError(null);
      try {
        const apiHost = import.meta.env.VITE_API_URL || '';
        const res = await fetch(`${apiHost}/api/scrape-article`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: article.link })
        });
        if (!res.ok) {
          throw new Error('Failed to scrape full article content.');
        }
        const data = await res.json();
        setScrapedData(data);
      } catch (err) {
        setScrapeError(err.message);
      } finally {
        setLoadingScrape(false);
      }
    };

    triggerScrape();
    setSummary('');
    setEntities(null);
  }, [article]);

  if (!article) {
    return (
      <div className="card" style={{ padding: '32px', textAlign: 'center' }}>
        <h3>No Article Selected</h3>
        <button className="btn btn-primary" onClick={() => setPage('Scraped Articles')} style={{ marginTop: '16px' }}>
          📰 View Articles
        </button>
      </div>
    );
  }

  const handleSummarize = async () => {
    if (!scrapedData || !scrapedData.content) return;
    setLoadingSummary(true);
    setSummaryError(null);
    setSummary('');
    try {
      const apiHost = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${apiHost}/api/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: scrapedData.content })
      });
      if (!res.ok) {
        throw new Error('Summarization neural network encountered an error.');
      }
      const data = await res.json();
      setSummary(data.summary);
    } catch (err) {
      setSummaryError(err.message);
    } finally {
      setLoadingSummary(false);
    }
  };

  const handleNER = async () => {
    if (!scrapedData || !scrapedData.content) return;
    setLoadingEntities(true);
    setEntitiesError(null);
    setEntities(null);
    try {
      const apiHost = import.meta.env.VITE_API_URL || '';
      const res = await fetch(`${apiHost}/api/extract-entities`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: scrapedData.content })
      });
      if (!res.ok) {
        throw new Error('Named entity extraction engine encountered an error.');
      }
      const data = await res.json();
      setEntities(data);
    } catch (err) {
      setEntitiesError(err.message);
    } finally {
      setLoadingEntities(false);
    }
  };

  const isPositive = article.FinBERT_Label === 'Positive';
  const isNegative = article.FinBERT_Label === 'Negative';
  
  let sentimentLabel = 'Neutral Noise';
  let sentimentColor = 'var(--neutral-dark)';
  let sentimentBg = 'var(--neutral-light)';
  let sentimentGlow = '0 0 15px var(--neutral-glow)';
  if (isPositive) {
    sentimentLabel = 'Bullish Catalyst';
    sentimentColor = 'var(--positive-dark)';
    sentimentBg = 'var(--positive-light)';
    sentimentGlow = '0 0 15px var(--positive-glow)';
  } else if (isNegative) {
    sentimentLabel = 'Bearish Risk';
    sentimentColor = 'var(--negative-dark)';
    sentimentBg = 'var(--negative-light)';
    sentimentGlow = '0 0 15px var(--negative-glow)';
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <button className="btn btn-secondary" onClick={() => setPage('Scraped Articles')} style={{ padding: '10px 16px', fontSize: '0.85rem' }}>
          <ArrowLeft size={16} /> Back to Repository
        </button>
      </div>

      <AnimatePresence mode="wait">
        {loadingScrape ? (
          <motion.div 
            key="scrape-loading"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="loading-container"
          >
            <div className="spinner"></div>
            <h3>Executing HTML Web Scraper</h3>
            <p style={{ color: 'var(--text-secondary)' }}>
              Resolving remote asset domain, bypassing cookies, and parsing semantic text bodies...
            </p>
          </motion.div>
        ) : (
          <motion.div 
            key="scraped-content"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="scraped-view"
          >
            {/* Left Column: Full Content */}
            <div style={{ minWidth: 0 }}>
              <div className="card">
                <h1 style={{ fontSize: '1.75rem', lineHeight: '1.3', marginBottom: '16px', color: 'var(--text-primary)' }}>
                  {article.title}
                </h1>
                
                <div style={{ display: 'flex', gap: '16px', alignItems: 'center', fontSize: '0.85rem', color: 'var(--text-secondary)', flexWrap: 'wrap', marginBottom: '20px' }}>
                  <span>Source: <strong>{article.source}</strong></span>
                  <span>|</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Clock size={14} /> {new Date(article.published).toLocaleString()}
                  </span>
                  <span>|</span>
                  <a href={article.link} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', fontWeight: 600, display: 'inline-flex', alignItems: 'center', gap: '4px', textDecoration: 'none' }}>
                    Original URL <ExternalLink size={12} />
                  </a>
                </div>

                <img 
                  src={scrapedData?.image_url || "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop"} 
                  alt="Article Banner" 
                  className="full-article-image" 
                  onError={(e) => {
                    e.target.src = "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop";
                  }}
                />

                <h3 style={{ fontSize: '1.25rem', margin: '28px 0 16px 0', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px' }}>
                  Extracted Body Paragraphs
                </h3>
                
                <div className="full-article-content">
                  {scrapedData?.content || article.summary || "No body content resolved for this endpoint."}
                </div>
              </div>
            </div>

            {/* Right Column: AI Panel */}
            <div className="ml-panel">
              {/* Sentiment Overview Card */}
              <div style={{ 
                padding: '20px', 
                backgroundColor: sentimentBg, 
                borderLeft: `4px solid ${sentimentColor}`, 
                borderRadius: 'var(--radius-sm)',
                boxShadow: sentimentGlow
              }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 700, letterSpacing: '0.05em' }}>NLP CLASSIFICATION</div>
                <div style={{ fontSize: '1.4rem', fontWeight: 800, color: sentimentColor, margin: '2px 0' }}>
                  {sentimentLabel}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  Model Confidence: <strong>{article.FinBERT_Confidence}%</strong>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '2px' }}>
                  Extracted Topic: <strong>{article.Topic || 'Macro/General'}</strong>
                </div>
              </div>

              {/* Summarization Pipeline */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <h4 style={{ fontSize: '0.95rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Bot size={16} style={{ color: 'var(--primary)' }} /> AI Summary Engine
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  Distills full text body into cohesive TL;DR bullet summaries using the BART transformer pipeline.
                </p>
                
                <button 
                  className="btn btn-primary" 
                  onClick={handleSummarize} 
                  disabled={loadingSummary || !scrapedData?.content || scrapedData.content.length < 200}
                  style={{ width: '100%', padding: '12px' }}
                >
                  {loadingSummary ? 'Running Inference...' : '✨ Generate AI Summary'}
                </button>

                {loadingSummary && (
                  <div className="pipeline-running-step">
                    <span style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>Condensing paragraph buffers...</span>
                    <div className="pulse-glow"></div>
                  </div>
                )}

                {summaryError && (
                  <div style={{ fontSize: '0.85rem', color: 'var(--negative-dark)' }}>
                    ⚠️ {summaryError}
                  </div>
                )}

                {summary && (
                  <motion.div 
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    style={{ 
                      backgroundColor: 'rgba(16, 185, 129, 0.05)', 
                      border: '1px solid var(--positive-dark)', 
                      borderRadius: 'var(--radius-sm)', 
                      padding: '16px', 
                      fontSize: '0.88rem', 
                      color: 'var(--positive-dark)',
                      lineHeight: '1.6',
                      fontWeight: 500
                    }}
                  >
                    {summary}
                  </motion.div>
                )}
              </div>

              {/* NER Token Classification */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <h4 style={{ fontSize: '0.95rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Layers size={16} style={{ color: 'var(--primary)' }} /> Token Classifier (NER)
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                  Resolves individual tokens to extract names of organizations, persons, locations, and miscellaneous groups.
                </p>

                <button 
                  className="btn btn-secondary" 
                  onClick={handleNER} 
                  disabled={loadingEntities || !scrapedData?.content}
                  style={{ width: '100%', padding: '12px' }}
                >
                  {loadingEntities ? 'Running NER...' : '🧬 Extract Entities'}
                </button>

                {loadingEntities && (
                  <div className="pipeline-running-step">
                    <span style={{ fontSize: '0.82rem', color: 'var(--text-primary)' }}>Evaluating token sequences...</span>
                    <div className="pulse-glow" style={{ backgroundColor: 'var(--neutral)' }}></div>
                  </div>
                )}

                {entitiesError && (
                  <div style={{ fontSize: '0.85rem', color: 'var(--negative-dark)' }}>
                    ⚠️ {entitiesError}
                  </div>
                )}

                {entities && (
                  <motion.div 
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '6px' }}
                  >
                    {Object.entries(entities).map(([tag, words]) => {
                      if (words.length === 0) return null;
                      
                      let Icon = Tag;
                      let label = 'Other';
                      if (tag === 'ORG') { Icon = Building; label = 'Organizations'; }
                      else if (tag === 'PER') { Icon = User; label = 'People'; }
                      else if (tag === 'LOC') { Icon = MapPin; label = 'Locations'; }

                      return (
                        <div key={tag}>
                          <div style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '6px', textTransform: 'uppercase' }}>
                            <Icon size={12} /> {label}
                          </div>
                          <div className="entity-list">
                            {words.map((w, i) => (
                              <span key={i} className={`entity-badge entity-${tag}`}>{w}</span>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                    {Object.values(entities).every(arr => arr.length === 0) && (
                      <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                        No clear corporate entities extracted.
                      </div>
                    )}
                  </motion.div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
