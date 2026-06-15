import React, { useState } from 'react';

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
      const response = await fetch(`/api/market-data?ticker=${encodeURIComponent(ticker)}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to load market data.');
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

  // Custom SVG Candlestick Chart Renderer
  const renderCandlestickChart = (data) => {
    if (!data || data.length === 0) return null;

    const width = 760;
    const height = 300;
    const padding = { top: 20, bottom: 30, left: 10, right: 60 };

    const prices = data.flatMap(d => [d.High, d.Low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice;
    
    // Add 5% padding to top and bottom of Y-axis
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

    const rectWidth = Math.max(4, ((width - padding.left - padding.right) / data.length) * 0.7);

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
      <div style={{ position: 'relative', width: '100%' }}>
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
                x={width - padding.right + 8} 
                y={line.y + 4} 
                fill="var(--text-muted)" 
                fontSize="0.75rem" 
                fontFamily="monospace"
              >
                ${line.price}
              </text>
            </g>
          ))}

          {/* Candlesticks (Wick + Body) */}
          {data.map((d, idx) => {
            const x = getX(idx);
            const yHigh = getY(d.High);
            const yLow = getY(d.Low);
            const yOpen = getY(d.Open);
            const yClose = getY(d.Close);
            const isBullish = d.Close >= d.Open;

            return (
              <g key={idx}>
                {/* Wick */}
                <line 
                  x1={x} 
                  y1={yHigh} 
                  x2={x} 
                  y2={yLow} 
                  stroke={isBullish ? 'var(--positive)' : 'var(--negative)'} 
                  strokeWidth="1.5"
                />
                {/* Body Rect */}
                <rect 
                  x={x - rectWidth / 2} 
                  y={Math.min(yOpen, yClose)} 
                  width={rectWidth} 
                  height={Math.max(1, Math.abs(yOpen - yClose))} 
                  fill={isBullish ? 'var(--positive)' : 'var(--negative)'}
                  stroke={isBullish ? 'var(--positive)' : 'var(--negative)'}
                  strokeWidth="1"
                  rx="1"
                />
              </g>
            );
          })}

          {/* Moving Average Line */}
          {ma5Path && (
            <path 
              d={ma5Path} 
              fill="none" 
              stroke="#f59e0b" 
              strokeWidth="2.5" 
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* X Axis Time Labels */}
          {data.map((d, idx) => {
            // Label every 6th tick to prevent overlapping
            if (idx % 6 === 0 || idx === data.length - 1) {
              const x = getX(idx);
              const labelDate = new Date(d.Date);
              const formattedDate = `${labelDate.getMonth() + 1}/${labelDate.getDate()}`;
              return (
                <text 
                  key={idx} 
                  x={x} 
                  y={height - 8} 
                  textAnchor="middle" 
                  fill="var(--text-muted)" 
                  fontSize="0.75rem"
                >
                  {formattedDate}
                </text>
              );
            }
            return null;
          })}
        </svg>
        <div style={{ display: 'flex', gap: '16px', justifyContent: 'center', marginTop: '12px', fontSize: '0.8rem', fontWeight: 600 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '6px', backgroundColor: 'var(--positive)', display: 'inline-block' }}></span>
            <span>Bullish Candle</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '12px', height: '6px', backgroundColor: 'var(--negative)', display: 'inline-block' }}></span>
            <span>Bearish Candle</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '16px', height: '3px', backgroundColor: '#f59e0b', display: 'inline-block' }}></span>
            <span>5-Day SMA</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div>
      <h2 style={{ marginBottom: '8px', fontSize: '1.8rem' }}>📈 Real-Time Market Data & Social Sentiment</h2>
      <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
        Cross-references recent stock price geometry (Yahoo Finance) against retail/social media buzz indices harvested from Reddit.
      </p>

      <div className="search-container">
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>Stock Ticker Symbol</label>
          <input 
            type="text" 
            className="input-field"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="e.g. AAPL, TSLA, MSFT, NVDA, BTC-USD"
            onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
          />
        </div>
        <div style={{ alignSelf: 'flex-end' }}>
          <button className="btn btn-primary" onClick={handleLoad} disabled={loading} style={{ height: '48px', padding: '0 32px' }}>
            🔍 Query Financial Assets
          </button>
        </div>
      </div>

      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <h3>Connecting to Financial Servers</h3>
          <p style={{ color: 'var(--text-muted)', marginTop: '8px' }}>
            Loading market candlesticks and scraping Reddit sentiment tags...
          </p>
        </div>
      )}

      {error && (
        <div className="alert alert-warning">
          <span>⚠️ <strong>Load Failure:</strong> {error}</span>
        </div>
      )}

      {!loading && marketData && (
        <div className="scraped-view">
          {/* Main Chart Column */}
          <div>
            <div className="card" style={{ marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.3rem', marginBottom: '16px' }}>
                📊 {marketData.ticker} Price Action (Last 30 Days)
              </h3>
              
              {/* Asset Metrics Row */}
              <div className="grid-3" style={{ marginBottom: '24px' }}>
                <div style={{ padding: '16px', backgroundColor: 'var(--bg-accent)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>REAL-TIME CLOSE</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    ${marketData.metrics.latest_close.toFixed(2)}
                  </div>
                  <div style={{ 
                    fontSize: '0.9rem', 
                    fontWeight: 700, 
                    color: marketData.metrics.pct_change >= 0 ? 'var(--positive-dark)' : 'var(--negative-dark)',
                    marginTop: '4px'
                  }}>
                    {marketData.metrics.pct_change >= 0 ? '▲' : '▼'} {marketData.metrics.pct_change.toFixed(2)}%
                  </div>
                </div>

                <div style={{ padding: '16px', backgroundColor: 'var(--bg-accent)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>30-DAY PERIOD HIGH</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    ${marketData.metrics.period_high.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Resistance Level
                  </div>
                </div>

                <div style={{ padding: '16px', backgroundColor: 'var(--bg-accent)', borderRadius: 'var(--radius-sm)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>30-DAY PERIOD LOW</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                    ${marketData.metrics.period_low.toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '6px' }}>
                    Support Level
                  </div>
                </div>
              </div>

              {/* Chart Graphic SVG */}
              {renderCandlestickChart(marketData.historical_data)}
            </div>

            {/* Historical Data Table */}
            <div className="card">
              <h4 style={{ marginBottom: '16px' }}>📅 30-Day Price Matrix</h4>
              <div className="table-container" style={{ maxHeight: '300px', overflowY: 'auto' }}>
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
                        <td style={{ fontWeight: 700 }}>${row.Close.toFixed(2)}</td>
                        <td style={{ fontFamily: 'monospace' }}>{row.Volume.toLocaleString()}</td>
                        <td style={{ color: '#b45309' }}>{row.MA5 ? `$${row.MA5.toFixed(2)}` : 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Social Sentiment Column */}
          <div className="ml-panel">
            <h3 style={{ fontSize: '1.2rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
              🗣️ Reddit WallStreetBets Pulse
            </h3>

            {marketData.reddit_sentiment.total_mentions > 0 ? (
              <div>
                <div style={{ fontSize: '0.9rem', marginBottom: '8px' }}>
                  Total Mentions Analyzed: <strong>{marketData.reddit_sentiment.total_mentions}</strong>
                </div>
                
                <div className="sentiment-progress-bar">
                  <div 
                    className="sentiment-progress-fill" 
                    style={{ width: `${Math.round(marketData.reddit_sentiment.bullish_ratio * 100)}%` }}
                  ></div>
                </div>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 700, marginBottom: '24px' }}>
                  <span style={{ color: 'var(--positive-dark)' }}>
                    Bullish: {Math.round(marketData.reddit_sentiment.bullish_ratio * 100)}%
                  </span>
                  <span style={{ color: 'var(--negative-dark)' }}>
                    Bearish: {100 - Math.round(marketData.reddit_sentiment.bullish_ratio * 100)}%
                  </span>
                </div>

                <h4 style={{ fontSize: '0.95rem', marginBottom: '12px' }}>Latest Retail Chatter:</h4>
                <div style={{ maxHeight: '420px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {marketData.reddit_sentiment.posts.map((post, idx) => (
                    <div 
                      key={idx} 
                      className={`reddit-post-card ${
                        post.sentiment === 'Bullish' ? 'bullish' : 
                        post.sentiment === 'Bearish' ? 'bearish' : 'neutral'
                      }`}
                    >
                      <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{post.title}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '6px' }}>
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
              <div style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)' }}>
                <p>No retail mentions found on WallStreetBets this week.</p>
                <p style={{ fontSize: '0.8rem', marginTop: '6px' }}>It may be flying under retail radar.</p>
              </div>
            )}
          </div>
        </div>
      )}

      {!loading && !marketData && !error && (
        <div className="loading-container" style={{ padding: '48px 24px', backgroundColor: '#f8fafc' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: '1.05rem' }}>
            No stock ticker loaded. Enter an asset ticker (e.g. AAPL) and click the load button to render stock metrics.
          </p>
        </div>
      )}
    </div>
  );
}
