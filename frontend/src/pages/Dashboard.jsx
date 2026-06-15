import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Search, 
  Layers, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Cpu, 
  ExternalLink,
  ChevronRight,
  AlertTriangle
} from 'lucide-react';

export default function Dashboard({ newsData, setNewsData, query, setQuery, loading, setLoading, error, setError }) {
  const [limit, setLimit] = useState(15);

  const handleFetch = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      // Prepend dynamic API URL if configured
      const apiHost = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${apiHost}/api/analyze?query=${encodeURIComponent(query)}&limit=${limit}`);
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

  const cardVariants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 80 } }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.08
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ marginBottom: '6px', fontSize: '1.8rem' }}>📊 News & Sentiment Intelligence</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          Queries Google RSS feeds, normalizes contents, and computes FinBERT sentiment and Zero-Shot topic labels.
        </p>
      </div>

      {/* Query Bar */}
      <div className="search-container">
        <div className="input-wrapper">
          <label className="input-label">Ticker or Search Query</label>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={18} style={{ position: 'absolute', left: '16px', color: 'var(--text-secondary)' }} />
            <input 
              type="text" 
              className="input-field"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g. NVDA, AI Regulations, Inflation, Federal Reserve"
              style={{ paddingLeft: '48px' }}
              onKeyDown={(e) => e.key === 'Enter' && handleFetch()}
            />
          </div>
        </div>
        
        <div className="input-wrapper" style={{ maxWidth: '140px' }}>
          <label className="input-label">Max Headlines</label>
          <select 
            className="input-field" 
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
          >
            <option value={10}>10 Articles</option>
            <option value={15}>15 Articles</option>
            <option value={25}>25 Articles</option>
            <option value={50}>50 Articles</option>
          </select>
        </div>

        <button className="btn btn-primary" onClick={handleFetch} disabled={loading} style={{ height: '49px' }}>
          <Activity size={16} /> Fetch & Analyze
        </button>
      </div>

      {loading && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="loading-container"
        >
          <div className="spinner"></div>
          <h3>Running AI Inference Pipelines</h3>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '500px' }}>
            Downloading articles and running FinBERT transformer inference... This may take a few seconds if models are lazy-loading for the first time.
          </p>
        </motion.div>
      )}

      {error && (
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="alert alert-warning"
        >
          <AlertTriangle size={18} />
          <span><strong>Inference Error:</strong> {error}</span>
        </motion.div>
      )}

      {!loading && !error && total > 0 && (
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate="show"
          style={{ display: 'flex', flexDirection: 'column', gap: '32px' }}
        >
          {/* Grid: Metrics */}
          <div className="grid-4">
            <motion.div variants={cardVariants} className="metric-card">
              <span className="metric-label">Headlines Analyzed</span>
              <span className="metric-value">{total}</span>
              <span className="metric-change" style={{ color: 'var(--text-secondary)' }}>
                <Layers size={14} style={{ marginRight: '4px' }} /> Active Session
              </span>
            </motion.div>
            
            <motion.div variants={cardVariants} className="metric-card bullish">
              <span className="metric-label">Positive Indicators</span>
              <span className="metric-value">{positiveCount}</span>
              <span className="metric-change up">
                <TrendingUp size={14} /> {posPct}% Compound
              </span>
            </motion.div>

            <motion.div variants={cardVariants} className="metric-card bearish">
              <span className="metric-label">Bearish Sentiment</span>
              <span className="metric-value">{negativeCount}</span>
              <span className="metric-change down">
                <TrendingDown size={14} /> {negPct}% Compound
              </span>
            </motion.div>

            <motion.div variants={cardVariants} className="metric-card neutral">
              <span className="metric-label">Neutral Noise</span>
              <span className="metric-value">{neutralCount}</span>
              <span className="metric-change" style={{ color: 'var(--neutral-dark)' }}>
                ● {neuPct}% Noise Ratio
              </span>
            </motion.div>
          </div>

          {/* Grid: Charts */}
          <div className="grid-2">
            {/* Pie Chart Widget */}
            <motion.div variants={cardVariants} className="card">
              <h4 className="card-title">
                <Cpu size={16} /> FinBERT Sentiment Ratio
              </h4>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '32px', height: '240px', flexWrap: 'wrap' }}>
                <div style={{ position: 'relative', width: '150px', height: '150px' }}>
                  <svg width="150" height="150" viewBox="0 0 36 36" style={{ transform: 'rotate(-90deg)', width: '100%', height: '100%' }}>
                    {/* Background track */}
                    <circle cx="18" cy="18" r="15.915" fill="none" stroke="var(--bg-accent)" strokeWidth="3.2" />
                    
                    {/* Positive Slice */}
                    {positiveCount > 0 && (
                      <motion.circle 
                        cx="18" cy="18" r="15.915" fill="none" stroke="var(--positive)" strokeWidth="3.2"
                        strokeDasharray={`${posPct} ${100 - posPct}`}
                        strokeDashoffset="0"
                        initial={{ strokeDasharray: `0 100` }}
                        animate={{ strokeDasharray: `${posPct} ${100 - posPct}` }}
                        transition={{ duration: 1, ease: 'easeOut' }}
                      />
                    )}

                    {/* Negative Slice */}
                    {negativeCount > 0 && (
                      <motion.circle 
                        cx="18" cy="18" r="15.915" fill="none" stroke="var(--negative)" strokeWidth="3.2"
                        strokeDasharray={`${negPct} ${100 - negPct}`}
                        strokeDashoffset={`-${posPct}`}
                        initial={{ strokeDasharray: `0 100`, strokeDashoffset: 0 }}
                        animate={{ strokeDasharray: `${negPct} ${100 - negPct}`, strokeDashoffset: -posPct }}
                        transition={{ duration: 1, ease: 'easeOut', delay: 0.1 }}
                      />
                    )}

                    {/* Neutral Slice */}
                    {neutralCount > 0 && (
                      <motion.circle 
                        cx="18" cy="18" r="15.915" fill="none" stroke="var(--neutral)" strokeWidth="3.2"
                        strokeDasharray={`${neuPct} ${100 - neuPct}`}
                        strokeDashoffset={`-${posPct + negPct}`}
                        initial={{ strokeDasharray: `0 100`, strokeDashoffset: 0 }}
                        animate={{ strokeDasharray: `${neuPct} ${100 - neuPct}`, strokeDashoffset: -(posPct + negPct) }}
                        transition={{ duration: 1, ease: 'easeOut', delay: 0.2 }}
                      />
                    )}
                    
                    {/* Center Ring */}
                    <circle cx="18" cy="18" r="12" fill="var(--bg-secondary)" />
                  </svg>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--positive)', boxShadow: '0 0 8px var(--positive-glow)' }}></span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Positive: {posPct}%</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--negative)', boxShadow: '0 0 8px var(--negative-glow)' }}></span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Negative: {negPct}%</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', backgroundColor: 'var(--neutral)', boxShadow: '0 0 8px var(--neutral-glow)' }}></span>
                    <span style={{ fontSize: '0.85rem', fontWeight: 600 }}>Neutral: {neuPct}%</span>
                  </div>
                </div>
              </div>
            </motion.div>

            {/* Dominant Topics */}
            <motion.div variants={cardVariants} className="card">
              <h4 className="card-title">
                <Layers size={16} /> Market Topic Classifications
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px', justifyContent: 'center', height: '240px' }}>
                {sortedTopics.length > 0 ? (
                  sortedTopics.map(([topic, count]) => {
                    const topicPct = Math.round((count / total) * 100);
                    return (
                      <div key={topic}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, marginBottom: '6px' }}>
                          <span>{topic}</span>
                          <span style={{ color: 'var(--text-secondary)' }}>{count} ({topicPct}%)</span>
                        </div>
                        <div style={{ width: '100%', height: '6px', backgroundColor: 'var(--bg-primary)', borderRadius: '9999px', overflow: 'hidden', border: '1px solid var(--border-color)' }}>
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${topicPct}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                            style={{ 
                              height: '100%', 
                              background: 'linear-gradient(90deg, var(--primary) 0%, #06b6d4 100%)', 
                              borderRadius: '9999px' 
                            }}
                          ></motion.div>
                        </div>
                      </div>
                    );
                  })
                ) : (
                  <p style={{ color: 'var(--text-muted)' }}>No topics classified.</p>
                )}
              </div>
            </motion.div>
          </div>

          {/* Table: Ingested Data */}
          <motion.div variants={cardVariants} className="card">
            <h4 className="card-title" style={{ marginBottom: '20px' }}>
              Structured Catalyst Matrix
            </h4>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Headline Title</th>
                    <th>Source</th>
                    <th>Category</th>
                    <th>Sentiment</th>
                    <th>Confidence</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {newsData.map((art, idx) => (
                    <tr key={idx}>
                      <td style={{ fontWeight: 600, maxWidth: '380px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        <a href={art.link} target="_blank" rel="noreferrer" style={{ color: 'inherit', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          {art.title} <ExternalLink size={12} style={{ color: 'var(--text-muted)', opacity: 0.6 }} />
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
                      <td style={{ fontFamily: 'monospace', fontWeight: 700, color: 'var(--text-primary)' }}>{art.FinBERT_Confidence}%</td>
                      <td style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                        {new Date(art.published).toLocaleDateString()}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </motion.div>
        </motion.div>
      )}

      {!loading && total === 0 && !error && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="loading-container" 
          style={{ padding: '64px 24px', backgroundColor: 'rgba(12, 14, 20, 0.4)' }}
        >
          <Activity size={32} style={{ color: 'var(--text-muted)' }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: '1rem', maxWidth: '480px' }}>
            No intelligence profile loaded. Search a stock ticker or keyword query above to fetch articles and execute natural language processing models.
          </p>
        </motion.div>
      )}
    </div>
  );
}
