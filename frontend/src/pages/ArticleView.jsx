import React, { useState, useEffect } from 'react';

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
        const res = await fetch('/api/scrape-article', {
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
    // Reset secondary states
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
      const res = await fetch('/api/summarize', {
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
      const res = await fetch('/api/extract-entities', {
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
  
  let sentimentColor = 'var(--neutral-dark)';
  let sentimentBg = 'var(--neutral-light)';
  if (isPositive) {
    sentimentColor = 'var(--positive-dark)';
    sentimentBg = 'var(--positive-light)';
  } else if (isNegative) {
    sentimentColor = 'var(--negative-dark)';
    sentimentBg = 'var(--negative-light)';
  }

  return (
    <div>
      <div>
        <button className="btn btn-secondary" onClick={() => setPage('Scraped Articles')} style={{ marginBottom: '16px' }}>
          ⬅ Back to Articles
        </button>
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid var(--border-color)', margin: '16px 0' }} />

      {loadingScrape && (
        <div className="loading-container">
          <div className="spinner"></div>
          <h3>Executing Web Scraper</h3>
          <p style={{ color: 'var(--text-muted)' }}>
            Retrieving HTML body elements and resolving original asset paths...
          </p>
        </div>
      )}

      {scrapeError && (
        <div className="alert alert-warning">
          <span>⚠️ <strong>Scrape Warning:</strong> {scrapeError}. Showing headline details only.</span>
        </div>
      )}

      {!loadingScrape && (
        <div className="scraped-view">
          {/* Left Column: Full Content */}
          <div className="card">
            <h1 style={{ fontSize: '1.8rem', lineHeight: '1.25', marginBottom: '12px' }}>{article.title}</h1>
            <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginBottom: '16px' }}>
              Source: <a href={article.link} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', fontWeight: 600 }}>{article.source}</a> | Published: {new Date(article.published).toLocaleString()}
            </p>

            <img 
              src={scrapedData?.image_url || "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop"} 
              alt="Article Graphic" 
              className="full-article-image" 
              onError={(e) => {
                e.target.src = "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=1000&auto=format&fit=crop";
              }}
            />

            <h3 style={{ fontSize: '1.2rem', margin: '24px 0 12px 0' }}>Extracted Content</h3>
            <div className="full-article-content">
              {scrapedData?.content || article.summary || "No description text available."}
            </div>
          </div>

          {/* Right Column: AI Sidebar */}
          <div className="ml-panel">
            {/* Quick Metrics */}
            <div style={{ padding: '16px', backgroundColor: sentimentBg, borderLeft: `5px solid ${sentimentColor}`, borderRadius: 'var(--radius-sm)' }}>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>ANALYZED SENTIMENT</div>
              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: sentimentColor }}>
                {article.FinBERT_Label}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px' }}>
                Confidence: <strong>{article.FinBERT_Confidence}%</strong> | Topic: <strong>{article.Topic || 'General'}</strong>
              </div>
            </div>

            {/* AI Summarization */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <h4 style={{ fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                🤖 AI Summarization (TL;DR)
              </h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Compresses long articles into bullet highlights using the DistilBART neural summarizer.
              </p>
              
              <button 
                className="btn btn-primary" 
                onClick={handleSummarize} 
                disabled={loadingSummary || !scrapedData?.content || scrapedData.content.length < 200}
                style={{ fontSize: '0.9rem', padding: '10px 16px' }}
              >
                {loadingSummary ? 'Writing Summary...' : '✨ Generate AI Summary'}
              </button>

              {loadingSummary && (
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', margin: '8px 0' }}>
                  <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px', margin: 0 }}></div>
                  <span style={{ fontSize: '0.85rem' }}>Reading & summarizing text...</span>
                </div>
              )}

              {summaryError && (
                <div style={{ fontSize: '0.85rem', color: 'var(--negative-dark)' }}>
                  ⚠️ {summaryError}
                </div>
              )}

              {summary && (
                <div style={{ 
                  backgroundColor: '#ecfdf5', 
                  border: '1px solid #10b981', 
                  borderRadius: 'var(--radius-sm)', 
                  padding: '14px', 
                  fontSize: '0.9rem', 
                  color: 'var(--positive-dark)',
                  lineHeight: '1.5'
                }}>
                  {summary}
                </div>
              )}
            </div>

            {/* Named Entity Extraction */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <h4 style={{ fontSize: '1rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '6px' }}>
                🔍 Named Entity Recognition (NER)
              </h4>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Extracts names of Organizations, Persons, Locations, and Misc from text using the BERT-NER model.
              </p>

              <button 
                className="btn btn-secondary" 
                onClick={handleNER} 
                disabled={loadingEntities || !scrapedData?.content}
                style={{ fontSize: '0.9rem', padding: '10px 16px' }}
              >
                {loadingEntities ? 'Extracting...' : '🧬 Extract Entities'}
              </button>

              {loadingEntities && (
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', margin: '8px 0' }}>
                  <div className="spinner" style={{ width: '20px', height: '20px', borderWidth: '2px', margin: 0 }}></div>
                  <span style={{ fontSize: '0.85rem' }}>Running NER pipeline...</span>
                </div>
              )}

              {entitiesError && (
                <div style={{ fontSize: '0.85rem', color: 'var(--negative-dark)' }}>
                  ⚠️ {entitiesError}
                </div>
              )}

              {entities && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '6px' }}>
                  {Object.entries(entities).map(([tag, words]) => {
                    if (words.length === 0) return null;
                    return (
                      <div key={tag}>
                        <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', marginBottom: '4px' }}>
                          {tag === 'ORG' ? 'ORGANIZATIONS' : tag === 'PER' ? 'PEOPLE' : tag === 'LOC' ? 'LOCATIONS' : 'MISCELLANEOUS'}
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
                    <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                      No clear entities extracted.
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
