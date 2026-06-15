import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  LineChart, 
  TrendingUp, 
  TrendingDown, 
  MessageSquare, 
  DollarSign, 
  Activity, 
  Search,
  AlertTriangle
} from 'lucide-react';

export default function MarketData() {
  const [ticker, setTicker] = useState('AAPL');
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleLoad = async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const apiHost = 'https://market-sentiment-analyser-1.onrender.com';
      const response = await fetch(`${apiHost}/api/market-data?ticker=${encodeURIComponent(ticker)}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to load asset market data.');
      }
      const data = await response.json();
      setMarketData(data);
    } catch (err) {
      setError(err.message);
      setMarketData(null);
    } finally {
      setLoading(false);
    }
  };

  // Custom SVG Candlestick Chart Renderer with Framer Motion animations
  const renderCandlestickChart = (data) => {
    if (!data || data.length === 0) return null;

    const width = 760;
    const height = 300;
    const padding = { top: 20, bottom: 30, left: 10, right: 60 };

    const prices = data.flatMap(d => [d.High, d.Low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    const yMin = minPrice - priceRange * 0.05;
    const yMax = maxPrice + priceRange * 0.05;
    const yRange = yMax - yMin;

    const getX = (index) => {
      const chartWidth = width - padding.left - padding.right;
      return padding.left + (index / (data.length - 1)) * chartWidth;
    };

    const getY = (val) => {
      if (yRange === 0) return height / 2;
      const chartHeight = height - padding.top - padding.bottom;
      return height - padding.bottom - ((val - yMin) / yRange) * chartHeight;
    };

    const rectWidth = Math.max(5, ((width - padding.left - padding.right) / data.length) * 0.7);

    // Gridlines (4 levels)
    const gridLines = [];
    for (let i = 0; i <= 4; i++) {
      const priceVal = yMin + (i / 4) * yRange;
      gridLines.push({
        y: getY(priceVal),
        price: priceVal.toFixed(2)
      });
    }

    // Generate path for MA5 Line
    let ma5Path = '';
    data.forEach((d, idx) => {
      if (d.MA5 !== null && d.MA5 !== undefined) {
        const x = getX(idx);
        const y = getY(d.MA5);
        if (ma5Path === '') {
          ma5Path = `M ${x} ${y}`;
        } else {
          ma5Path += ` L ${x} ${y}`;
        }
      }
    });

    return (
      <div style={{ position: 'relative', width: '100%', overflow: 'hidden' }}>
        <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} style={{ overflow: 'visible' }}>
          {/* Horizontal Gridlines */}
          {gridLines.map((line, idx) => (
            <g key={idx}>
              <line 
                x1={padding.left} 
                y1={line.y} 
                x2={width - padding.right} 
                y2={line.y} 
                stroke="var(--border-color)" 
                strokeWidth="1" 
                strokeDasharray="4 4"
              />
              <text 
                x={width - padding.right + 12} 
                y={line.y + 4} 
                fill="var(--text-secondary)" 
                fontSize="0.75rem" 
                fontFamily="monospace"
              >
                ${line.price}
              </text>
            </g>
          ))}

          {/* Candlesticks (Wick + Body) with Framer Motion entry */}
          {data.map((d, idx) => {
            const x = getX(idx);
            const yHigh = getY(d.High);
            const yLow = getY(d.Low);
            const yOpen = getY(d.Open);
            const yClose = getY(d.Close);
            const isBullish = d.Close >= d.Open;
            const candleHeight = Math.max(1, Math.abs(yOpen - yClose));
            const color = isBullish ? 'var(--positive)' : 'var(--negative)';

            return (
              <g key={idx}>
                {/* Wick */}
                <motion.line 
                  x1={x} 
                  y1={yLow} 
                  x2={x} 
                  y2={yLow}
                  animate={{ y1: yHigh }}
                  transition={{ duration: 0.6, ease: 'easeOut', delay: idx * 0.01 }}
                  stroke={color} 
                  strokeWidth="1.5"
                />
                {/* Body Rect */}
                <motion.rect 
                  x={x - rectWidth / 2} 
                  y={isBullish ? yClose : yOpen} 
                  width={rectWidth} 
                  height={0} 
                  animate={{ height: candleHeight }}
                  transition={{ duration: 0.5, ease: 'backOut', delay: idx * 0.01 + 0.1 }}
                  fill={color}
                  stroke={color}
                  strokeWidth="1"
                  rx="1.5"
                  style={{ transformOrigin: `${x}px ${isBullish ? yClose : yOpen}px` }}
                />
              </g>
            );
          })}

          {/* Moving Average Line drawing animation */}
          {ma5Path && (
            <motion.path 
              d={ma5Path} 
              fill="none" 
              stroke="#fbbf24" 
              strokeWidth="2.5" 
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1.2, ease: 'easeInOut', delay: 0.4 }}
            />
          )}

          {/* X Axis Time Labels */}
          {data.map((d, idx) => {
            if (idx % 6 === 0 || idx === data.length - 1) {
              const x = getX(idx);
              const labelDate = new Date(d.Date);
              const formattedDate = `${labelDate.getMonth() + 1}/${labelDate.getDate()}`;
              return (
                <text 
                  key={idx} 
                  x={x} 
                  y={height - 6} 
                  textAnchor="middle" 
                  fill="var(--text-secondary)" 
                  fontSize="0.72rem"
                >
                  {formattedDate}
                </text>
              );
            }
            return null;
          })}
        </svg>
        <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginTop: '16px', fontSize: '0.8rem', fontWeight: 600 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: 'var(--positive)' }}></span>
            <span>Bullish</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '10px', height: '10px', borderRadius: '2px', backgroundColor: 'var(--negative)' }}></span>
            <span>Bearish</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '16px', height: '3px', backgroundColor: '#fbbf24', borderRadius: '2px', display: 'inline-block' }}></span>
            <span>5-Day SMA</span>
          </div>
        </div>
      </div>
    );
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 80 } }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ marginBottom: '6px', fontSize: '1.8rem' }}>📈 Market Geometry & WSB Buzz</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          Pairs technical stock movement algorithms (yFinance) directly with social chatter trends scraped from Reddit's WallStreetBets.
        </p>
      </div>

      {/* Asset Search */}
      <div className="search-container">
        <div className="input-wrapper">
          <label className="input-label">Stock Ticker Symbol</label>
          <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Search size={18} style={{ position: 'absolute', left: '16px', color: 'var(--text-secondary)' }} />
            <input 
              type="text" 
              className="input-field"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL, TSLA, MSFT, NVDA, BTC-USD"
              style={{ paddingLeft: '48px' }}
              onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
            />
          </div>
        </div>
        <button className="btn btn-primary" onClick={handleLoad} disabled={loading} style={{ height: '49px' }}>
          <LineChart size={16} /> Load Market Data
        </button>
      </div>

      {loading && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="loading-container"
        >
          <div className="spinner"></div>
          <h3>Connecting to Financial Protocols</h3>
          <p style={{ color: 'var(--text-secondary)' }}>
            Retrieving candle history and crawling Reddit social networks...
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
          <span><strong>Assets Query Failure:</strong> {error}</span>
        </motion.div>
      )}

      {!loading && marketData && (
        <motion.div 
          variants={containerVariants}
          initial="hidden"
          animate="show"
          className="scraped-view"
        >
          {/* Chart Section */}
          <div style={{ minWidth: 0 }}>
            <motion.div variants={itemVariants} className="card" style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Activity size={18} style={{ color: 'var(--primary)' }} /> {marketData.ticker} Technical Matrix
              </h3>
              
              {/* Technical indicators */}
              <div className="grid-3" style={{ marginBottom: '28px' }}>
                <div style={{ padding: '16px', backgroundColor: 'rgba(5, 7, 10, 0.3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 700, letterSpacing: '0.05em' }}>CURRENT CLOSE</div>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', margin: '4px 0' }}>
                    ${marketData.metrics.latest_close.toFixed(2)}
                  </div>
                  <div style={{ 
                    fontSize: '0.85rem', 
                    fontWeight: 700, 
                    color: marketData.metrics.pct_change >= 0 ? 'var(--positive-dark)' : 'var(--negative-dark)',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    {marketData.metrics.pct_change >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />} 
                    {marketData.metrics.pct_change >= 0 ? '+' : ''}{marketData.metrics.pct_change.toFixed(2)}%
                  </div>
                </div>

                <div style={{ padding: '16px', backgroundColor: 'rgba(5, 7, 10, 0.3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 700, letterSpacing: '0.05em' }}>30-DAY HIGH</div>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', margin: '4px 0' }}>
                    ${marketData.metrics.period_high.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Resistance Ceiling
                  </div>
                </div>

                <div style={{ padding: '16px', backgroundColor: 'rgba(5, 7, 10, 0.3)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 700, letterSpacing: '0.05em' }}>30-DAY LOW</div>
                  <div style={{ fontSize: '1.6rem', fontWeight: 800, color: 'var(--text-primary)', margin: '4px 0' }}>
                    ${marketData.metrics.period_low.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                    Support Floor
                  </div>
                </div>
              </div>

              {/* Candlestick Graphic */}
              {renderCandlestickChart(marketData.historical_data)}
            </motion.div>

            {/* Historical Table */}
            <motion.div variants={itemVariants} className="card">
              <h4 className="card-title">Historical Price Matrix</h4>
              <div className="table-container" style={{ maxHeight: '280px', overflowY: 'auto' }}>
                <table>
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Open</th>
                      <th>High</th>
                      <th>Low</th>
                      <th>Close</th>
                      <th>Volume</th>
                      <th>5-Day SMA</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...marketData.historical_data].reverse().map((row, idx) => (
                      <tr key={idx}>
                        <td style={{ fontWeight: 600 }}>{row.Date}</td>
                        <td>${row.Open.toFixed(2)}</td>
                        <td style={{ color: 'var(--positive-dark)' }}>${row.High.toFixed(2)}</td>
                        <td style={{ color: 'var(--negative-dark)' }}>${row.Low.toFixed(2)}</td>
                        <td style={{ fontWeight: 700, color: 'var(--text-primary)' }}>${row.Close.toFixed(2)}</td>
                        <td style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>{row.Volume.toLocaleString()}</td>
                        <td style={{ color: '#fbbf24', fontWeight: 600 }}>{row.MA5 ? `$${row.MA5.toFixed(2)}` : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </motion.div>
          </div>

          {/* Social column */}
          <motion.div variants={itemVariants} className="ml-panel">
            <h3 style={{ fontSize: '1.15rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <MessageSquare size={18} style={{ color: 'var(--primary)' }} /> r/WallStreetBets Index
            </h3>

            {marketData.reddit_sentiment.total_mentions > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', justifyContent: 'space-between' }}>
                    <span>Analyzed Mentions:</span>
                    <strong>{marketData.reddit_sentiment.total_mentions} posts</strong>
                  </div>
                  
                  <div className="sentiment-progress-bar">
                    <motion.div 
                      className="sentiment-progress-fill" 
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.round(marketData.reddit_sentiment.bullish_ratio * 100)}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                    ></motion.div>
                  </div>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700 }}>
                    <span style={{ color: 'var(--positive-dark)' }}>
                      Bullish: {Math.round(marketData.reddit_sentiment.bullish_ratio * 100)}%
                    </span>
                    <span style={{ color: 'var(--negative-dark)' }}>
                      Bearish: {100 - Math.round(marketData.reddit_sentiment.bullish_ratio * 100)}%
                    </span>
                  </div>
                </div>

                <h4 style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginTop: '8px' }}>Retail Feed</h4>
                <div style={{ maxHeight: '430px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px', paddingRight: '4px' }}>
                  {marketData.reddit_sentiment.posts.map((post, idx) => (
                    <div 
                      key={idx} 
                      className={`reddit-post-card ${
                        post.sentiment === 'Bullish' ? 'bullish' : 
                        post.sentiment === 'Bearish' ? 'bearish' : 'neutral'
                      }`}
                    >
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)', lineHeight: '1.4' }}>{post.title}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                        <span>VADER Score: <strong style={{ 
                          color: post.sentiment === 'Bullish' ? 'var(--positive-dark)' : 
                                 post.sentiment === 'Bearish' ? 'var(--negative-dark)' : 'var(--neutral-dark)' 
                        }}>{post.score}</strong></span>
                        <span>{new Date(post.published).toLocaleDateString()}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '32px 16px', color: 'var(--text-muted)' }}>
                <MessageSquare size={24} style={{ margin: '0 auto 12px auto', opacity: 0.4 }} />
                <p style={{ fontSize: '0.85rem' }}>No recent chatter registered for this asset on WallStreetBets.</p>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}

      {!loading && !marketData && !error && (
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="loading-container" 
          style={{ padding: '64px 24px', backgroundColor: 'rgba(12, 14, 20, 0.4)' }}
        >
          <Activity size={32} style={{ color: 'var(--text-muted)' }} />
          <p style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>
            No stock ticker loaded. Enter an asset ticker (e.g. AAPL, TSLA) and query to load technical analysis charts.
          </p>
        </motion.div>
      )}
    </div>
  );
}
