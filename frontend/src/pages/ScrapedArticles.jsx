import React from 'react';
import { motion } from 'framer-motion';
import { 
  Newspaper, 
  Calendar, 
  Globe, 
  Sparkles, 
  ChevronRight 
} from 'lucide-react';

export default function ScrapedArticles({ newsData, setPage, setSelectedArticle }) {
  if (!newsData || newsData.length === 0) {
    return (
      <motion.div 
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        className="loading-container" 
        style={{ padding: '64px 24px', backgroundColor: 'var(--bg-secondary)' }}
      >
        <Newspaper size={32} style={{ color: 'var(--text-muted)' }} />
        <h3 style={{ marginTop: '12px' }}>Cache Repository Empty</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '24px', maxWidth: '460px' }}>
          We haven't ingested or analyzed any headlines in this session yet. Please navigate to the Dashboard to run a sentiment sweep.
        </p>
        <button className="btn btn-primary" onClick={() => setPage('Dashboard')}>
          📊 Go to Dashboard
        </button>
      </motion.div>
    );
  }

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.05 }
    }
  };

  const cardVariants = {
    hidden: { opacity: 0, x: -10 },
    show: { opacity: 1, x: 0, transition: { type: 'spring', stiffness: 100 } }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ marginBottom: '6px', fontSize: '1.8rem' }}>📰 Ingested Article Repository</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          Browse through the headlines currently cached in your active session. Select **Deep Dive** to scrape full texts and run neural summary models.
        </p>
      </div>

      <motion.div 
        variants={containerVariants}
        initial="hidden"
        animate="show"
        className="articles-list"
      >
        {newsData.map((article, idx) => {
          const isPositive = article.FinBERT_Label === 'Positive';
          const isNegative = article.FinBERT_Label === 'Negative';
          
          let sentimentLabel = 'Neutral';
          let badgeClass = 'badge-neutral';
          if (isPositive) {
            sentimentLabel = 'Bullish';
            badgeClass = 'badge-positive';
          } else if (isNegative) {
            sentimentLabel = 'Bearish';
            badgeClass = 'badge-negative';
          }

          return (
            <motion.div 
              variants={cardVariants}
              className="article-card-row" 
              key={idx}
            >
              <div className="article-info">
                <h3 style={{ fontSize: '1.15rem', marginBottom: '8px', fontWeight: 700, color: 'var(--text-primary)', lineHeight: '1.4' }}>
                  {article.title}
                </h3>
                
                <div className="article-meta">
                  <span className="badge badge-topic">🏷️ {article.Topic || 'General'}</span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Globe size={14} /> Source: <strong>{article.source}</strong>
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Calendar size={14} /> {new Date(article.published).toLocaleDateString()}
                  </span>
                </div>

                <div style={{ marginTop: '16px', fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>AI Index:</span>
                  <span className={`badge ${badgeClass}`}>{sentimentLabel}</span> 
                  <span style={{ color: 'var(--text-muted)' }}>({article.FinBERT_Confidence}% confidence)</span>
                </div>
              </div>

              <div className="article-action">
                <button 
                  className="btn btn-secondary" 
                  onClick={() => {
                    setSelectedArticle(article);
                    setPage('Article View');
                  }}
                  style={{ whiteSpace: 'nowrap', padding: '10px 18px', fontSize: '0.85rem' }}
                >
                  <Sparkles size={14} style={{ color: 'var(--primary)' }} /> Deep Dive <ChevronRight size={14} />
                </button>
              </div>
            </motion.div>
          );
        })}
      </motion.div>
    </div>
  );
}
