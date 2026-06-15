import React from 'react';

export default function ScrapedArticles({ newsData, setPage, setSelectedArticle }) {
  if (!newsData || newsData.length === 0) {
    return (
      <div className="loading-container" style={{ padding: '48px 24px' }}>
        <h3 style={{ marginBottom: '8px' }}>No News Articles in Cache</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '16px', maxWidth: '500px' }}>
          We haven't ingested or analyzed any headlines in this session yet. Please navigate to the Dashboard to run a sentiment sweep.
        </p>
        <button className="btn btn-primary" onClick={() => setPage('Dashboard')}>
          📊 Go to Dashboard
        </button>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ marginBottom: '8px', fontSize: '1.8rem' }}>📰 Scraped Article Repository</h2>
      <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
        Browse through the headlines currently loaded in this session. Click on <strong>Deep Dive Analysis</strong> to execute the custom web scraper and run AI Summarization or Entity Extraction on the full text.
      </p>

      <div className="articles-list">
        {newsData.map((article, idx) => {
          const isPositive = article.FinBERT_Label === 'Positive';
          const isNegative = article.FinBERT_Label === 'Negative';
          
          let sentimentColor = 'var(--neutral)';
          if (isPositive) sentimentColor = 'var(--positive)';
          if (isNegative) sentimentColor = 'var(--negative)';

          return (
            <div className="article-card-row" key={idx}>
              <div className="article-info">
                <h3 style={{ fontSize: '1.2rem', marginBottom: '8px', fontWeight: 600, color: 'var(--text-primary)' }}>
                  {article.title}
                </h3>
                
                <div className="article-meta">
                  <span className="badge badge-topic">🏷️ {article.Topic || 'General'}</span>
                  <span>🏢 Source: <strong>{article.source}</strong></span>
                  <span>📅 Published: {new Date(article.published).toLocaleString()}</span>
                </div>

                <div style={{ marginTop: '12px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  AI Sentiment Index: <span style={{ color: sentimentColor, fontWeight: 700 }}>{article.FinBERT_Label}</span> ({article.FinBERT_Confidence}% confidence)
                </div>
              </div>

              <div className="article-action">
                <button 
                  className="btn btn-secondary" 
                  onClick={() => {
                    setSelectedArticle(article);
                    setPage('Article View');
                  }}
                  style={{ whiteSpace: 'nowrap' }}
                >
                  🔍 Deep Dive Analysis
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
