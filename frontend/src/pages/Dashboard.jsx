import React, { useState } from 'react';

export default function Dashboard({ searchPrefix, newsData, setNewsData, query, setQuery, loading, setLoading, error, setError }) {
  const [limit, setLimit] = useState(15);

  const handleFetch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/analyze?query=${encodeURIComponent(query)}&limit=${limit}`);
      if (!response.ok) {
        throw new Error('Failed to run sentiment analysis pipelines.');
      }
      const data = await response.json();
      setNewsData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const total = newsData.length;
  const positiveCount = newsData.filter(a => a.FinBERT_Label === 'Positive').length;
  const negativeCount = newsData.filter(a => a.FinBERT_Label === 'Negative').length;
  const neutralCount = newsData.filter(a => a.FinBERT_Label === 'Neutral').length;

  const posPct = total > 0 ? Math.round((positiveCount / total) * 100) : 0;
  const negPct = total > 0 ? Math.round((negativeCount / total) * 100) : 0;
  const neuPct = total > 0 ? Math.round((neutralCount / total) * 100) : 0;

  // Topic classification distribution
  const topicsMap = {};
  newsData.forEach(art => {
    topicsMap[art.Topic] = (topicsMap[art.Topic] || 0) + 1;
  });
  const sortedTopics = Object.entries(topicsMap).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <h2 style={{ marginBottom: '8px', fontSize: '1.8rem' }}>📊 Financial News & Intelligence Dashboard</h2>
      <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
        Scans global RSS feeds, feeds title and summary into the FinBERT Transformer model, and runs Zero-Shot Classifier for topic extraction.
      </p>

      <div className="search-container">
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Ticker or Macro Query</label>
          <input 
            type="text" 
            className="input-field"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g. NVDA, AI Regulations, Inflation, Federal Reserve"
            onKeyDown={(e) => e.key === 'Enter' && handleFetch()}
          />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', width: '120px' }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Max Articles</label>
          <select 
            className="input-field" 
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            style={{ padding: '12px' }}
          >
            <option value={10}>10</option>
            <option value={15}>15</option>
            <option value={25}>25</option>
            <option value={50}>50</option>
          </select>
        </div>
        <div style={{ alignSelf: 'flex-end' }}>
          <button className="btn btn-primary" onClick={handleFetch} disabled={loading} style={{ height: '48px' }}>
            🚀 Fetch & Analyze
          </button>
        </div>
      </div>

      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <h3>Downloading RSS News & Running Inference</h3>
          <p style={{ color: 'var(--text-muted)', maxWidth: '500px', marginTop: '8px' }}>
            Summoning neural networks... This might take up to 10-15 seconds if models are loading into memory for the first time.
          </p>
        </div>
      )}

      {error && (
        <div className="alert alert-warning">
          <span>⚠️ <strong>Error:</strong> {error}</span>
        </div>
      )}

      {!loading && !error && total > 0 && (
        <div>
          <h3 style={{ marginBottom: '16px', fontSize: '1.3rem' }}>📈 Executive Intelligence Summary ({query.toUpperCase()})</h3>
          
          <div className="grid-4" style={{ marginBottom: '32px' }}>
            <div className="metric-card">
              <span className="metric-label">Articles Processed</span>
              <span className="metric-value">{total}</span>
              <span className="metric-change" style={{ color: 'var(--text-muted)' }}>Last 24 hours</span>
            </div>
            
            <div className="metric-card bullish">
              <span className="metric-label">Bullish Sentiment</span>
              <span className="metric-value">{positiveCount}</span>
              <span className="metric-change up">▲ {posPct}% of total</span>
            </div>

            <div className="metric-card bearish">
              <span className="metric-label">Bearish Sentiment</span>
              <span className="metric-value">{negativeCount}</span>
              <span className="metric-change down">▼ {negPct}% of total</span>
            </div>

            <div className="metric-card neutral">
              <span className="metric-label">Neutral Noise</span>
              <span className="metric-value">{neutralCount}</span>
              <span className="metric-change" style={{ color: 'var(--neutral-dark)' }}>● {neuPct}% of total</span>
            </div>
          </div>

          <div className="grid-2" style={{ marginBottom: '32px' }}>
            {/* Pie Chart Card using SVG */}
            <div className="card">
              <h4 style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                🍩 FinBERT Sentiment Ratio
              </h4>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '32px', height: '220px' }}>
                <svg width="160" height="160" viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)' }}>
                  {/* Background Circle */}
                  <circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--bg-accent)" strokeWidth="3" />
                  
                  {/* Positive Slice */}
                  {positiveCount > 0 && (
                    <circle 
                      cx="18" cy="18" r="15.915" fill="none" stroke="var(--positive)" strokeWidth="3"
                      strokeDasharray={`${posPct} ${100 - posPct}`}
                      strokeDashoffset="0"
                    />
                  )}

                  {/* Negative Slice */}
                  {negativeCount > 0 && (
                    <circle 
                      cx="18" cy="18" r="15.915" fill="none" stroke="var(--negative)" strokeWidth="3"
                      strokeDasharray={`${negPct} ${100 - negPct}`}
                      strokeDashoffset={`-${posPct}`}
                    />
                  )}

                  {/* Neutral Slice */}
                  {neutralCount > 0 && (
                    <circle 
                      cx="18" cy="18" r="15.915" fill="none" stroke="var(--neutral)" strokeWidth="3"
                      strokeDasharray={`${neuPct} ${100 - neuPct}`}
                      strokeDashoffset={`-${posPct + negPct}`}
                    />
                  )}
                  
                  {/* Center Hole for Donut Effect */}
                  <circle cx="18" cy="18" r="11" fill="var(--bg-secondary)" />
                </svg>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--positive)' }}></span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Positive: {posPct}%</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--negative)' }}></span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Negative: {negPct}%</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--neutral)' }}></span>
                    <span style={{ fontSize: '0.9rem', fontWeight: 600 }}>Neutral: {neuPct}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Dominant Topics Progress Bar Grid */}
            <div className="card">
              <h4 style={{ marginBottom: '16px' }}>🏷️ Dominant Market Topics</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', justifyContent: 'center', height: '220px', padding: '10px 0' }}>
                {sortedTopics.length > 0 ? (
                  sortedTopics.map(([topic, count]) => {
                    const topicPct = Math.round((count / total) * 100);
                    return (
                      <div key={topic}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '4px' }}>
                          <span>{topic}</span>
                          <span style={{ color: 'var(--text-muted)' }}>{count} ({topicPct}%)</span>
                        </div>
                        <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--bg-accent)', borderRadius: '9999px', overflow: 'hidden' }}>
                          <div style={{ width: `${topicPct}%`, height: '100%', background: 'linear-gradient(90deg, var(--primary) 0%, #06b6d4 100%)', borderRadius: '9999px' }}></div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p style={{ color: 'var(--text-muted)' }}>No topics classified.</p>
                )}
              </div>
            </div>
          </div>

          {/* Raw Structured Table */}
          <div className="card" style={{ marginBottom: '24px' }}>
            <h4 style={{ marginBottom: '16px' }}>🗃️ Raw Structured Intelligence Data</h4>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Article Headline</th>
                    <th>Source</th>
                    <th>Topic Category</th>
                    <th>FinBERT Sentiment</th>
                    <th>Conf. Score</th>
                    <th>Published</th>
                  </tr>
                </thead>
                <tbody>
                  {newsData.map((art, idx) => (
                    <tr key={idx}>
                      <td style={{ fontWeight: 600, maxWidth: '400px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        <a href={art.link} target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none' }}>
                          {art.title}
                        </a>
                      </td>
                      <td>{art.source}</td>
                      <td>
                        <span className="badge badge-topic">{art.Topic}</span>
                      </td>
                      <td>
                        <span className={`badge ${
                          art.FinBERT_Label === 'Positive' ? 'badge-positive' : 
                          art.FinBERT_Label === 'Negative' ? 'badge-negative' : 'badge-neutral'
                        }`}>
                          {art.FinBERT_Label}
                        </span>
                      </td>
                      <td style={{ fontFamily: 'monospace', fontWeight: 'bold' }}>{art.FinBERT_Confidence}%</td>
                      <td style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                        {new Date(art.published).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!loading && total === 0 && !error && (
        <div className="loading-container" style={{ padding: '48px 24px', backgroundColor: '#f8fafc' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.05rem' }}>
            No analysis results loaded in session. Please search a ticker or topic above to execute the dynamic AI ingestion pipelines.
          </p>
        </div>
      )}
    </div>
  );
}
