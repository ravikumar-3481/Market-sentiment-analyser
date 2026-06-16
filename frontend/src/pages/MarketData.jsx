import React, { useState, useEffect, useRef } from 'react';
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
import { apiFetch, getApiSettings } from '../utils/api';

// ASCII Candlestick Chart Renderer helper
const renderAsciiChart = (historyData) => {
  if (!historyData || historyData.length === 0) {
    return 'No historical data loaded. Run "load &lt;ticker&gt;" first.';
  }
  const data = historyData.slice(-20);
  const heights = 10;
  const prices = data.flatMap(d => [d.High, d.Low]);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const priceRange = maxPrice - minPrice || 1;

  let output = '';
  for (let r = heights; r >= 0; r--) {
    const rowPrice = minPrice + (r / heights) * priceRange;
    let rowStr = `$${rowPrice.toFixed(2).padStart(8)} | `;
    
    for (let c = 0; c < data.length; c++) {
      const d = data[c];
      const isBullish = d.Close >= d.Open;
      const highY = Math.round(((d.High - minPrice) / priceRange) * heights);
      const lowY = Math.round(((d.Low - minPrice) / priceRange) * heights);
      const openY = Math.round(((d.Open - minPrice) / priceRange) * heights);
      const closeY = Math.round(((d.Close - minPrice) / priceRange) * heights);
      
      const bodyMax = Math.max(openY, closeY);
      const bodyMin = Math.min(openY, closeY);
      
      let char = ' ';
      
      if (r === highY && r > bodyMax) {
        char = '╷';
      } else if (r === lowY && r < bodyMin) {
        char = '╵';
      } else if (r <= highY && r > bodyMax) {
        char = '│';
      } else if (r >= lowY && r < bodyMin) {
        char = '│';
      } else if (r >= bodyMin && r <= bodyMax) {
        char = isBullish ? '█' : '░';
      }
      
      const colorStyle = isBullish ? 'color: #9ece6a' : 'color: #f7768e';
      rowStr += `<span style="${colorStyle}">${char}</span> `;
    }
    output += rowStr + '\n';
  }
  
  output += ' '.repeat(10) + '+' + '-'.repeat(data.length * 2) + '\n';
  
  let dateRow = ' '.repeat(10) + '  ';
  data.forEach((d, idx) => {
    if (idx % 5 === 0) {
      const parts = d.Date.split('-');
      dateRow += parts[1] + '/' + parts[2];
    } else {
      dateRow += '   ';
    }
  });
  output += dateRow + '\n';
  
  return output;
};

export default function MarketData() {
  const [ticker, setTicker] = useState('AAPL');
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Layout Tab state: 'visualizer' | 'terminal'
  const [activeTab, setActiveTab] = useState('visualizer');

  // Timeframe selector state
  const [timeframe, setTimeframe] = useState('1d');

  // Drawing tools state
  const [drawingTool, setDrawingTool] = useState('cursor'); // 'cursor' | 'trendline'
  const [trendlines, setTrendlines] = useState([]);
  const [drawingStart, setDrawingStart] = useState(null);
  const [drawingPreview, setDrawingPreview] = useState(null);

  // Crosshair tracker state
  const [crosshair, setCrosshair] = useState(null); // { x, y, price, date }

  // Index ticker state
  const [indices, setIndices] = useState([
    { name: 'NIFTY 50', value: 24178.15, change: -0.62 },
    { name: 'SENSEX', value: 77378.19, change: -0.66 },
    { name: 'CRUDE OIL', value: 8022.00, change: -0.47 },
    { name: 'NATURAL GAS', value: 260.20, change: -1.40 },
    { name: 'BANKNIFTY', value: 55310.55, change: -1.31 }
  ]);

  // Real-time WebSocket states
  const [livePrice, setLivePrice] = useState(null);
  const [liveChange, setLiveChange] = useState(null);
  const [liveHigh, setLiveHigh] = useState(null);
  const [liveLow, setLiveLow] = useState(null);
  const [liveVolume, setLiveVolume] = useState(null);
  const [optionsChain, setOptionsChain] = useState([]);

  // Simulated Trading State
  const [cash, setCash] = useState(() => {
    const saved = localStorage.getItem('marketpulse_sim_cash');
    return saved ? parseFloat(saved) : 100000.00;
  });
  const [holdings, setHoldings] = useState(() => {
    const saved = localStorage.getItem('marketpulse_sim_holdings');
    return saved ? JSON.parse(saved) : {};
  });

  const [terminalInput, setTerminalInput] = useState('');
  const [terminalHistory, setTerminalHistory] = useState([
    { text: '=======================================================', type: 'system' },
    { text: '     MARKETPULSE AI - REAL-TIME TRADING SYSTEM v1.2.0', type: 'system' },
    { text: '     Simulated Trading Balance: $100,000.00 USD', type: 'system' },
    { text: '=======================================================', type: 'system' },
    { text: 'Type <span style="color: #00ffc8; font-weight: bold">help</span> to list simulated trading commands.', type: 'system' },
    { text: 'Type <span style="color: #00ffc8; font-weight: bold">load &lt;ticker&gt;</span> to connect WebSocket streams.', type: 'system' },
    { text: '', type: 'system' }
  ]);

  const terminalEndRef = useRef(null);
  const socketRef = useRef(null);

  // Fluctuate indices randomly to simulate trading ticker activity
  useEffect(() => {
    const interval = setInterval(() => {
      setIndices(prev => prev.map(ind => {
        const move = randomMove(0.0003);
        const newValue = roundNum(ind.value * (1 + move), 2);
        const newChange = roundNum(ind.change + move * 100, 2);
        return { ...ind, value: newValue, change: newChange };
      }));
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  const randomMove = (factor) => {
    return (Math.random() - 0.5) * 2 * factor;
  };

  const roundNum = (num, decimals) => {
    return Math.round(num * Math.pow(10, decimals)) / Math.pow(10, decimals);
  };

  // Connect WebSockets when ticker shifts
  useEffect(() => {
    if (!ticker) return;

    // Connect to WebSocket router
    const { host } = getApiSettings();
    const cleanHost = host.replace(/^http:\/\//, '').replace(/^https:\/\//, '');
    const wsProto = host.startsWith('https') ? 'wss' : 'ws';
    const wsUrl = `${wsProto}://${cleanHost}/ws/trades/${ticker}`;

    if (socketRef.current) {
      socketRef.current.close();
    }

    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setLivePrice(data.price);
      setLiveChange(data.pct_change);
      setLiveHigh(data.high);
      setLiveLow(data.low);
      setLiveVolume(data.volume);
      setOptionsChain(data.options_chain);

      // Append tick updates to the last historical candlestick
      setMarketData(prev => {
        if (!prev || !prev.historical_data || prev.historical_data.length === 0) return prev;
        const updated = { ...prev };
        const lastIndex = updated.historical_data.length - 1;
        const lastRow = { ...updated.historical_data[lastIndex] };
        
        lastRow.Close = data.price;
        lastRow.High = Math.max(lastRow.High, data.price);
        lastRow.Low = Math.min(lastRow.Low, data.price);
        lastRow.Volume = data.volume;
        
        updated.metrics = {
          ...updated.metrics,
          latest_close: data.price,
          pct_change: data.pct_change,
          period_high: Math.max(updated.metrics.period_high || 0, data.high),
          period_low: Math.min(updated.metrics.period_low || 99999, data.low)
        };
        
        updated.historical_data[lastIndex] = lastRow;
        return updated;
      });
    };

    ws.onclose = () => {
      // Clean socket reference
    };

    return () => {
      if (ws) ws.close();
    };
  }, [ticker]);

  // Persist simulated states
  useEffect(() => {
    localStorage.setItem('marketpulse_sim_cash', cash.toString());
  }, [cash]);

  useEffect(() => {
    localStorage.setItem('marketpulse_sim_holdings', JSON.stringify(holdings));
  }, [holdings]);

  // Smooth scroll terminal
  useEffect(() => {
    if (terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [terminalHistory]);

  const handleLoad = async () => {
    if (!ticker.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await apiFetch(`/api/market-data?ticker=${encodeURIComponent(ticker)}`);
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Failed to load asset market data.');
      }
      const data = await response.json();
      setMarketData(data);
      // Log connection notification to terminal
      setTerminalHistory(prev => [
        ...prev,
        { text: `[System] WebSocket feed synchronized for: ${ticker.toUpperCase()}`, type: 'system' }
      ]);
    } catch (err) {
      setError(err.message);
      setMarketData(null);
      setTerminalHistory(prev => [
        ...prev,
        { text: `[System Error] GUI search failed: ${err.message}`, type: 'error' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const runTerminalCommand = async (cmdText) => {
    const cleanCmd = cmdText.trim();
    if (!cleanCmd) return;
    
    setTerminalHistory(prev => [...prev, { text: `marketpulse-ai&gt; ${cleanCmd}`, type: 'command' }]);
    
    const parts = cleanCmd.split(/\s+/);
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);
    
    switch (cmd) {
      case 'clear':
        setTerminalHistory([]);
        break;
        
      case 'help':
        setTerminalHistory(prev => [
          ...prev,
          { text: '=======================================================', type: 'output' },
          { text: '     MARKETPULSE AI - STOCK TRADING SIMULATOR', type: 'output' },
          { text: '=======================================================', type: 'output' },
          { text: 'Available commands:', type: 'output' },
          { text: '  <span style="color: #00e1ff">load &lt;ticker&gt;</span>   : Fetch market and register WebSocket live ticks (e.g. load TSLA)', type: 'output' },
          { text: '  <span style="color: #00e1ff">buy &lt;shares&gt;</span>    : Purchase stock shares at current close price (e.g. buy 10)', type: 'output' },
          { text: '  <span style="color: #00e1ff">sell &lt;shares&gt;</span>   : Sell stock shares at current close price (e.g. sell 5)', type: 'output' },
          { text: '  <span style="color: #00e1ff">portfolio</span>       : Show cash balance, positions, cost vs current, and profit/loss', type: 'output' },
          { text: '  <span style="color: #00e1ff">quote</span>           : Display latest bid, ask, spread, high, low, and volume', type: 'output' },
          { text: '  <span style="color: #00e1ff">ascii</span>           : Render retro text-based ASCII candlestick chart', type: 'output' },
          { text: '  <span style="color: #00e1ff">sentiment</span>       : Show Reddit WallStreetBets mentions index & VADER chatter', type: 'output' },
          { text: '  <span style="color: #00e1ff">clear</span>           : Clear the terminal screen', type: 'output' },
          { text: '  <span style="color: #00e1ff">help</span>            : Show this documentation reference', type: 'output' },
          { text: '=======================================================', type: 'output' }
        ]);
        break;
        
      case 'load':
        if (args.length === 0) {
          setTerminalHistory(prev => [...prev, { text: 'Error: Ticker symbol required. Syntax: load <ticker> (e.g. load AAPL)', type: 'error' }]);
          break;
        }
        const targetTicker = args[0].toUpperCase();
        setTerminalHistory(prev => [...prev, { text: `Connecting to exchange gateway... Fetching data for ${targetTicker}...`, type: 'system' }]);
        setLoading(true);
        setError(null);
        try {
          const response = await apiFetch(`/api/market-data?ticker=${encodeURIComponent(targetTicker)}`);
          if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.detail || 'Asset not found.');
          }
          const data = await response.json();
          setMarketData(data);
          setTicker(targetTicker);
          setTerminalHistory(prev => [
            ...prev,
            { text: `SUCCESS: ${targetTicker} loaded successfully. Close: $${data.metrics?.latest_close?.toFixed(2) ?? 'N/A'}`, type: 'success' }
          ]);
        } catch (err) {
          setError(err.message);
          setMarketData(null);
          setTerminalHistory(prev => [...prev, { text: `CONNECTION ERROR: ${err.message}`, type: 'error' }]);
        } finally {
          setLoading(false);
        }
        break;
        
      case 'buy': {
        if (!marketData) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No active stock loaded. Run "load <ticker>" first.', type: 'error' }]);
          break;
        }
        if (args.length === 0) {
          setTerminalHistory(prev => [...prev, { text: 'Error: Share quantity required. Syntax: buy <qty> (e.g. buy 10)', type: 'error' }]);
          break;
        }
        const qty = parseInt(args[0], 10);
        if (isNaN(qty) || qty <= 0) {
          setTerminalHistory(prev => [...prev, { text: 'Error: Invalid share quantity. Please enter a positive integer.', type: 'error' }]);
          break;
        }
        const price = livePrice || marketData.metrics?.latest_close;
        if (!price) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No pricing data available for transaction.', type: 'error' }]);
          break;
        }
        const cost = price * qty;
        if (cost > cash) {
          setTerminalHistory(prev => [
            ...prev,
            { text: `Error: Insufficient cash balance. Order cost: $${cost.toFixed(2)} | Available cash: $${cash.toFixed(2)}`, type: 'error' }
          ]);
          break;
        }
        
        setCash(prev => prev - cost);
        setHoldings(prev => {
          const current = prev[marketData.ticker] || { shares: 0, avgCost: 0 };
          const newShares = current.shares + qty;
          const newAvgCost = ((current.shares * current.avgCost) + cost) / newShares;
          return {
            ...prev,
            [marketData.ticker]: { shares: newShares, avgCost: newAvgCost }
          };
        });
        
        setTerminalHistory(prev => [
          ...prev,
          { text: `TRANSACTION SUCCESS: Bought ${qty} shares of ${marketData.ticker} at $${price.toFixed(2)} per share. Total cost: $${cost.toFixed(2)}.`, type: 'success' }
        ]);
        break;
      }
      
      case 'sell': {
        if (!marketData) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No active stock loaded. Run "load <ticker>" first.', type: 'error' }]);
          break;
        }
        if (args.length === 0) {
          setTerminalHistory(prev => [...prev, { text: 'Error: Share quantity required. Syntax: sell <qty> (e.g. sell 5)', type: 'error' }]);
          break;
        }
        const qty = parseInt(args[0], 10);
        if (isNaN(qty) || qty <= 0) {
          setTerminalHistory(prev => [...prev, { text: 'Error: Invalid share quantity. Please enter a positive integer.', type: 'error' }]);
          break;
        }
        const activeHolding = holdings[marketData.ticker];
        if (!activeHolding || activeHolding.shares < qty) {
          const owned = activeHolding ? activeHolding.shares : 0;
          setTerminalHistory(prev => [
            ...prev,
            { text: `Error: Insufficient holdings. Attempted to sell ${qty} shares of ${marketData.ticker} | Owned: ${owned} shares`, type: 'error' }
          ]);
          break;
        }
        
        const price = livePrice || marketData.metrics?.latest_close;
        if (!price) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No pricing data available for transaction.', type: 'error' }]);
          break;
        }
        const proceeds = price * qty;
        
        setCash(prev => prev + proceeds);
        setHoldings(prev => {
          const current = prev[marketData.ticker];
          const newShares = current.shares - qty;
          const updated = { ...prev };
          if (newShares <= 0) {
            delete updated[marketData.ticker];
          } else {
            updated[marketData.ticker] = { shares: newShares, avgCost: current.avgCost };
          }
          return updated;
        });
        
        setTerminalHistory(prev => [
          ...prev,
          { text: `TRANSACTION SUCCESS: Sold ${qty} shares of ${marketData.ticker} at $${price.toFixed(2)} per share. Total proceeds: $${proceeds.toFixed(2)}.`, type: 'success' }
        ]);
        break;
      }
      
      case 'portfolio':
      case 'p': {
        const ownedTickers = Object.keys(holdings);
        let outputText = '====================================================================\n';
        outputText += `              PORTFOLIO STATEMENT (SIMULATED PAPER TRADING)\n`;
        outputText += '====================================================================\n';
        outputText += `CASH BALANCE: $${cash.toFixed(2)}\n\n`;
        
        if (ownedTickers.length === 0) {
          outputText += 'NO ACTIVE POSITIONS. Use "buy <qty>" to trade stocks.\n';
        } else {
          outputText += `TICKER   SHARES   AVG COST   MARKET PRICE   INVESTED VALUE   MARKET VALUE   RETURN (%)\n`;
          outputText += `------   ------   --------   ------------   --------------   ------------   ----------\n`;
          
          let totalInvested = 0;
          let totalMarketValue = 0;
          
          ownedTickers.forEach(t => {
            const h = holdings[t];
            const currentPrice = (marketData && marketData.ticker === t) ? (livePrice || marketData.metrics?.latest_close) : h.avgCost;
            const investedVal = h.shares * h.avgCost;
            const marketVal = h.shares * currentPrice;
            const profitLoss = marketVal - investedVal;
            const profitLossPct = investedVal > 0 ? (profitLoss / investedVal) * 100 : 0;
            
            totalInvested += investedVal;
            totalMarketValue += marketVal;
            
            const colorStyle = profitLoss >= 0 ? 'color: #9ece6a' : 'color: #f7768e';
            const sign = profitLoss >= 0 ? '+' : '';
            
            outputText += `${t.padEnd(6)}   ${h.shares.toString().padEnd(6)}   $${h.avgCost.toFixed(2).padEnd(8)}   $${currentPrice.toFixed(2).padEnd(12)}   $${investedVal.toFixed(2).padEnd(14)}   $${marketVal.toFixed(2).padEnd(12)}   <span style="${colorStyle}">${sign}${profitLossPct.toFixed(2)}%</span>\n`;
          });
          
          const totalPL = totalMarketValue - totalInvested;
          const totalPLPct = totalInvested > 0 ? (totalPL / totalInvested) * 100 : 0;
          const totalColorStyle = totalPL >= 0 ? 'color: #9ece6a' : 'color: #f7768e';
          const totalSign = totalPL >= 0 ? '+' : '';
          
          outputText += `--------------------------------------------------------------------\n`;
          outputText += `TOTAL INVESTED: $${totalInvested.toFixed(2)}\n`;
          outputText += `PORTFOLIO VALUE: $${totalMarketValue.toFixed(2)}\n`;
          outputText += `TOTAL UNREALIZED RETURN: <span style="${totalColorStyle}">${totalSign}$${totalPL.toFixed(2)} (${totalPLPct.toFixed(2)}%)</span>\n`;
        }
        outputText += '====================================================================';
        
        setTerminalHistory(prev => [...prev, { text: outputText, type: 'output' }]);
        break;
      }
      
      case 'quote':
      case 'q': {
        if (!marketData) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No active stock loaded. Run "load <ticker>" first.', type: 'error' }]);
          break;
        }
        const m = marketData.metrics;
        const close = livePrice || m?.latest_close || 0;
        const bid = close - 0.05;
        const ask = close + 0.05;
        const spread = ask - bid;
        const lastVol = liveVolume || marketData.historical_data?.slice(-1)[0]?.Volume || 0;
        
        let outputText = '============================================\n';
        outputText += `             STOCK REAL-TIME QUOTE: ${marketData.ticker}\n`;
        outputText += '============================================\n';
        outputText += `  LAST PRICE : $${close.toFixed(2)}\n`;
        outputText += `  BID PRICE  : $${bid.toFixed(2)}\n`;
        outputText += `  ASK PRICE  : $${ask.toFixed(2)}\n`;
        outputText += `  SPREAD     : $${spread.toFixed(2)}\n`;
        outputText += `  30D HIGH   : $${(liveHigh || m?.period_high || 0).toFixed(2)}\n`;
        outputText += `  30D LOW    : $${(liveLow || m?.period_low || 0).toFixed(2)}\n`;
        outputText += `  CHANGE     : ${(liveChange !== null ? liveChange : (m?.pct_change || 0)) >= 0 ? '+' : ''}${(liveChange !== null ? liveChange : (m?.pct_change || 0)).toFixed(2)}%\n`;
        outputText += `  VOLUME     : ${lastVol.toLocaleString()}\n`;
        outputText += '============================================';
        
        setTerminalHistory(prev => [...prev, { text: outputText, type: 'output' }]);
        break;
      }
      
      case 'ascii':
      case 'chart': {
        if (!marketData || !marketData.historical_data) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No active stock loaded. Run "load <ticker>" first.', type: 'error' }]);
          break;
        }
        const asciiChart = renderAsciiChart(marketData.historical_data);
        setTerminalHistory(prev => [...prev, { text: asciiChart, type: 'output' }]);
        break;
      }
      
      case 'sentiment':
      case 'wsb': {
        if (!marketData) {
          setTerminalHistory(prev => [...prev, { text: 'Error: No active stock loaded. Run "load <ticker>" first.', type: 'error' }]);
          break;
        }
        const s = marketData.reddit_sentiment;
        if (!s || s.total_mentions === 0) {
          setTerminalHistory(prev => [...prev, { text: `r/WallStreetBets sentiment for ${marketData.ticker}: No recent chatter found.`, type: 'output' }]);
          break;
        }
        let outputText = '=========================================================\n';
        outputText += `         r/WallStreetBets CHATTER SENTIMENT: ${marketData.ticker}\n`;
        outputText += '=========================================================\n';
        outputText += `  TOTAL MENTIONS : ${s.total_mentions} posts analyzed\n`;
        outputText += `  BULLISH RATIO  : ${Math.round(s.bullish_ratio * 100)}%\n`;
        outputText += `  BEARISH RATIO  : ${100 - Math.round(s.bullish_ratio * 100)}%\n`;
        outputText += `  BREAKDOWN      : ${s.pos_count} Bullish | ${s.neg_count} Bearish | ${s.neu_count} Neutral\n\n`;
        outputText += `RECENT DISCUSSIONS:\n`;
        s.posts.slice(0, 3).forEach((p, idx) => {
          outputText += `  ${idx + 1}. [${p.sentiment}] ${p.title} (VADER: ${p.score})\n`;
        });
        outputText += '=========================================================';
        
        setTerminalHistory(prev => [...prev, { text: outputText, type: 'output' }]);
        break;
      }
      
      default:
        setTerminalHistory(prev => [
          ...prev, 
          { text: `Error: Command not recognized: '${cmd}'. Type 'help' to see list of valid commands.`, type: 'error' }
        ]);
        break;
    }
  };

  // Crosshair coordinates calculation helper
  const handleMouseMove = (e) => {
    if (!marketData || !marketData.historical_data) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const width = 760;
    const height = 300;
    const padding = { top: 20, bottom: 30, left: 10, right: 60 };

    const mouseX = (e.clientX - rect.left) * (width / rect.width);
    const mouseY = (e.clientY - rect.top) * (height / rect.height);

    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;

    // Resolve index
    const colRatio = (mouseX - padding.left) / chartWidth;
    let index = Math.round(colRatio * (marketData.historical_data.length - 1));
    index = Math.max(0, Math.min(marketData.historical_data.length - 1, index));

    const item = marketData.historical_data[index];
    if (!item) return;

    // Resolve price
    const prices = marketData.historical_data.flatMap(d => [d.High, d.Low]);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const priceRange = maxPrice - minPrice || 1;
    const yMin = minPrice - priceRange * 0.05;
    const yMax = maxPrice + priceRange * 0.05;
    const yRange = yMax - yMin;

    const rowRatio = (height - padding.bottom - mouseY) / chartHeight;
    const price = yMin + rowRatio * yRange;

    setCrosshair({
      x: padding.left + (index / (marketData.historical_data.length - 1)) * chartWidth,
      y: mouseY,
      price: price.toFixed(2),
      date: item.Date
    });
  };

  const handleMouseLeave = () => {
    setCrosshair(null);
    if (drawingTool === 'trendline' && drawingStart) {
      setDrawingPreview(null);
    }
  };

  // Click handler to draw trendlines on the SVG chart
  const handleChartClick = (e) => {
    if (drawingTool !== 'trendline' || !crosshair) return;

    if (!drawingStart) {
      // First click: set line origin
      setDrawingStart({ x: crosshair.x, y: crosshair.y });
    } else {
      // Second click: save trendline
      setTrendlines(prev => [...prev, {
        x1: drawingStart.x,
        y1: drawingStart.y,
        x2: crosshair.x,
        y2: crosshair.y
      }]);
      setDrawingStart(null);
      setDrawingPreview(null);
    }
  };

  const handleChartMouseMove = (e) => {
    handleMouseMove(e);
    if (drawingTool === 'trendline' && drawingStart && crosshair) {
      setDrawingPreview({
        x1: drawingStart.x,
        y1: drawingStart.y,
        x2: crosshair.x,
        y2: crosshair.y
      });
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
      const divisor = data.length > 1 ? data.length - 1 : 1;
      return padding.left + (index / divisor) * chartWidth;
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
        <svg 
          viewBox={`0 0 ${width} ${height}`} 
          width="100%" 
          height={height} 
          style={{ overflow: 'visible', cursor: drawingTool === 'trendline' ? 'crosshair' : 'default' }}
          onMouseMove={handleChartMouseMove}
          onMouseLeave={handleMouseLeave}
          onClick={handleChartClick}
        >
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

          {/* Candlesticks (Wick + Body) */}
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
                <line 
                  x1={x} 
                  y1={yLow} 
                  x2={x} 
                  y2={yHigh}
                  stroke={color} 
                  strokeWidth="1.5"
                />
                {/* Body Rect */}
                <rect 
                  x={x - rectWidth / 2} 
                  y={isBullish ? yClose : yOpen} 
                  width={rectWidth} 
                  height={candleHeight} 
                  fill={color}
                  stroke={color}
                  strokeWidth="1"
                  rx="1.5"
                />
              </g>
            );
          })}

          {/* Moving Average Line */}
          {ma5Path && (
            <path 
              d={ma5Path} 
              fill="none" 
              stroke="var(--neutral)" 
              strokeWidth="2" 
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Drawn Trendlines */}
          {trendlines.map((line, idx) => (
            <line 
              key={idx}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke="#4f46e5"
              strokeWidth="2.5"
              strokeLinecap="round"
            />
          ))}

          {/* Live Trendline Preview */}
          {drawingPreview && (
            <line 
              x1={drawingPreview.x1}
              y1={drawingPreview.y1}
              x2={drawingPreview.x2}
              y2={drawingPreview.y2}
              stroke="#4f46e5"
              strokeWidth="1.5"
              strokeDasharray="4 4"
            />
          )}

          {/* Crosshair Guides */}
          {crosshair && (
            <g>
              {/* Vertical crosshair line */}
              <line 
                x1={crosshair.x} 
                y1={padding.top} 
                x2={crosshair.x} 
                y2={height - padding.bottom} 
                stroke="var(--text-muted)" 
                strokeWidth="1" 
                strokeDasharray="3 3"
              />
              {/* Horizontal crosshair line */}
              <line 
                x1={padding.left} 
                y1={crosshair.y} 
                x2={width - padding.right} 
                y2={crosshair.y} 
                stroke="var(--text-muted)" 
                strokeWidth="1" 
                strokeDasharray="3 3"
              />
              
              {/* Floating Price Tag */}
              <rect 
                x={width - padding.right + 2} 
                y={crosshair.y - 10} 
                width="56" 
                height="20" 
                fill="var(--text-primary)" 
                rx="3"
              />
              <text 
                x={width - padding.right + 8} 
                y={crosshair.y + 4} 
                fill="var(--bg-secondary)" 
                fontSize="0.7rem" 
                fontFamily="monospace"
                fontWeight="700"
              >
                ${crosshair.price}
              </text>

              {/* Floating Date Tag */}
              <rect 
                x={crosshair.x - 30} 
                y={height - padding.bottom + 2} 
                width="60" 
                height="18" 
                fill="var(--text-primary)" 
                rx="3"
              />
              <text 
                x={crosshair.x} 
                y={height - padding.bottom + 14} 
                fill="var(--bg-secondary)" 
                fontSize="0.7rem" 
                textAnchor="middle"
              >
                {crosshair.date.split('-').slice(1).join('/')}
              </text>
            </g>
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
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      
      {/* 1. Top Ribbon Ticker Marquee */}
      <div style={{ 
        display: 'flex', 
        gap: '24px', 
        overflowX: 'auto', 
        padding: '10px 20px', 
        backgroundColor: '#0c0f12', 
        borderBottom: '1.5px solid #1c242b',
        borderRadius: '8px',
        fontSize: '0.8rem',
        fontFamily: 'monospace',
        fontWeight: 'bold',
        whiteSpace: 'nowrap'
      }}>
        {indices.map((ind, idx) => (
          <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ color: '#8892b0' }}>{ind.name}</span>
            <span style={{ color: '#fff' }}>{ind.value.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
            <span style={{ 
              color: ind.change >= 0 ? '#00ff66' : '#ff3366', 
              display: 'flex', 
              alignItems: 'center',
              gap: '2px'
            }}>
              {ind.change >= 0 ? '▲' : '▼'} {ind.change.toFixed(2)}%
            </span>
            {idx < indices.length - 1 && <span style={{ color: '#333e48', marginLeft: '12px' }}>|</span>}
          </div>
        ))}
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '8px' }}>
        <div>
          <h2 style={{ marginBottom: '4px', fontSize: '1.7rem' }}>📈 Market Geometry & WSB Buzz</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            Real-time interactive stock terminal equipped with WebSocket price triggers and options matrices.
          </p>
        </div>

        {/* Navigation Tabs */}
        <div style={{ display: 'flex', gap: '8px', backgroundColor: 'var(--bg-accent)', padding: '4px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <button 
            onClick={() => setActiveTab('visualizer')} 
            style={{
              background: activeTab === 'visualizer' ? 'var(--bg-secondary)' : 'none',
              border: 'none',
              color: activeTab === 'visualizer' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: 600,
              padding: '6px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.85rem',
              transition: 'all 0.2s ease',
              boxShadow: activeTab === 'visualizer' ? 'var(--shadow-sm)' : 'none'
            }}
          >
            📊 Chart Visualizer
          </button>
          <button 
            onClick={() => setActiveTab('terminal')} 
            style={{
              background: activeTab === 'terminal' ? 'var(--bg-secondary)' : 'none',
              border: 'none',
              color: activeTab === 'terminal' ? 'var(--text-primary)' : 'var(--text-muted)',
              fontWeight: 600,
              padding: '6px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '0.85rem',
              transition: 'all 0.2s ease',
              boxShadow: activeTab === 'terminal' ? 'var(--shadow-sm)' : 'none'
            }}
          >
            💼 Trading Terminal
          </button>
        </div>
      </div>

      {/* TECHNICAL VISUALIZER TAB */}
      {activeTab === 'visualizer' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          
          {/* Asset Search & Timeframe Ribbon */}
          <div className="search-container" style={{ gap: '16px', flexWrap: 'wrap' }}>
            <div className="input-wrapper" style={{ flex: 1, minWidth: '220px' }}>
              <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
                <Search size={18} style={{ position: 'absolute', left: '16px', color: 'var(--text-secondary)' }} />
                <input 
                  type="text" 
                  className="input-field"
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  placeholder="e.g. AAPL, TSLA, NVDA"
                  style={{ paddingLeft: '48px' }}
                  onKeyDown={(e) => e.key === 'Enter' && handleLoad()}
                />
              </div>
            </div>
            
            {/* Timeframe selector */}
            <div style={{ display: 'flex', backgroundColor: 'var(--bg-accent)', padding: '3px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
              {['1m', '5m', '1h', '1d', '1w', '1mo'].map(tf => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  style={{
                    backgroundColor: timeframe === tf ? 'var(--bg-secondary)' : 'transparent',
                    border: 'none',
                    borderRadius: '4px',
                    color: timeframe === tf ? 'var(--primary)' : 'var(--text-muted)',
                    fontWeight: timeframe === tf ? '700' : '500',
                    fontSize: '0.8rem',
                    padding: '6px 12px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                >
                  {tf}
                </button>
              ))}
            </div>

            <button className="btn btn-primary" onClick={handleLoad} disabled={loading} style={{ height: '49px' }}>
              <LineChart size={16} /> Sync Feed
            </button>
          </div>

          {loading && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="loading-container"
            >
              <div className="spinner"></div>
              <h3>Syncing WebSocket Order Books</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                Connecting to live trade channels and parsing order parameters...
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
              style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '20px' }}
              className="market-layout-responsive"
            >
              {/* Chart Visualizer Core */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', minWidth: 0 }}>
                <motion.div variants={itemVariants} className="card" style={{ padding: '20px', position: 'relative' }}>
                  {/* Chart Header Bar */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
                    <h3 style={{ fontSize: '1.15rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <Activity size={18} style={{ color: 'var(--primary)' }} /> {marketData.ticker} Real-Time Chart
                    </h3>
                    
                    {/* Live Trade Status Indicator */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', color: 'var(--positive-dark)', fontWeight: 'bold' }}>
                      <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--positive)', display: 'inline-block', animation: 'pulse 1.5s infinite' }}></span>
                      LIVE FEED CONNECTED
                    </div>
                  </div>

                  {/* Layout splitting: Left toolbars + Center chart */}
                  <div style={{ display: 'flex', gap: '12px' }}>
                    
                    {/* Drawing Toolbar */}
                    <div style={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      gap: '8px', 
                      backgroundColor: 'var(--bg-accent)', 
                      padding: '6px', 
                      borderRadius: '8px', 
                      border: '1px solid var(--border-color)',
                      height: 'fit-content'
                    }}>
                      <button 
                        onClick={() => setDrawingTool('cursor')} 
                        title="Pointer Tool"
                        style={{
                          backgroundColor: drawingTool === 'cursor' ? 'var(--primary)' : 'transparent',
                          color: drawingTool === 'cursor' ? '#fff' : 'var(--text-secondary)',
                          border: 'none', borderRadius: '6px', padding: '8px', cursor: 'pointer'
                        }}
                      >
                        🖱️
                      </button>
                      <button 
                        onClick={() => setDrawingTool('trendline')} 
                        title="Draw Trendline"
                        style={{
                          backgroundColor: drawingTool === 'trendline' ? 'var(--primary)' : 'transparent',
                          color: drawingTool === 'trendline' ? '#fff' : 'var(--text-secondary)',
                          border: 'none', borderRadius: '6px', padding: '8px', cursor: 'pointer'
                        }}
                      >
                        📈
                      </button>
                      <button 
                        onClick={() => setTrendlines([])} 
                        title="Clear Drawings"
                        style={{
                          backgroundColor: 'transparent',
                          color: 'var(--text-secondary)',
                          border: 'none', borderRadius: '6px', padding: '8px', cursor: 'pointer'
                        }}
                      >
                        🧹
                      </button>
                    </div>

                    {/* Candlestick SVG Graphic Canvas */}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      {renderCandlestickChart(marketData.historical_data)}
                    </div>
                  </div>
                </motion.div>

                {/* Historical Price List */}
                <motion.div variants={itemVariants} className="card">
                  <h4 className="card-title">Exchange Price Logs</h4>
                  <div className="table-container" style={{ maxHeight: '240px', overflowY: 'auto' }}>
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
                            <td>${row.Open?.toFixed(2) ?? 'N/A'}</td>
                            <td style={{ color: 'var(--positive-dark)' }}>${row.High?.toFixed(2) ?? 'N/A'}</td>
                            <td style={{ color: 'var(--negative-dark)' }}>${row.Low?.toFixed(2) ?? 'N/A'}</td>
                            <td style={{ fontWeight: 700, color: 'var(--text-primary)' }}>${row.Close?.toFixed(2) ?? 'N/A'}</td>
                            <td style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>{row.Volume?.toLocaleString() ?? 'N/A'}</td>
                            <td style={{ color: 'var(--neutral)', fontWeight: 600 }}>{row.MA5 ? `$${row.MA5.toFixed(2)}` : 'N/A'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              </div>

              {/* Right Sidebar Option Chain */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                
                {/* 1. Quick Metrics Summary */}
                <motion.div variants={itemVariants} className="card" style={{ padding: '16px' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em' }}>LIVE PRICE TICK</div>
                  <div style={{ fontSize: '1.75rem', fontWeight: 800, color: 'var(--text-primary)', margin: '4px 0', display: 'flex', alignItems: 'baseline', gap: '8px' }}>
                    ${(livePrice || marketData.metrics?.latest_close || 0).toFixed(2)}
                    <span style={{ 
                      fontSize: '0.85rem', 
                      fontWeight: 700, 
                      color: (liveChange !== null ? liveChange : (marketData.metrics?.pct_change || 0)) >= 0 ? 'var(--positive)' : 'var(--negative)'
                    }}>
                      {(liveChange !== null ? liveChange : (marketData.metrics?.pct_change || 0)) >= 0 ? '+' : ''}
                      {(liveChange !== null ? liveChange : (marketData.metrics?.pct_change || 0)).toFixed(2)}%
                    </span>
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'var(--text-secondary)', borderTop: '1px solid var(--border-color)', paddingTop: '10px', marginTop: '6px' }}>
                    <span>High: <strong>${(liveHigh || marketData.metrics?.period_high || 0).toFixed(2)}</strong></span>
                    <span>Low: <strong>${(liveLow || marketData.metrics?.period_low || 0).toFixed(2)}</strong></span>
                  </div>
                </motion.div>

                {/* 2. Live Options Chain Matrix */}
                <motion.div variants={itemVariants} className="card" style={{ padding: '16px' }}>
                  <h4 style={{ fontSize: '0.95rem', marginBottom: '14px', borderBottom: '1px solid var(--border-color)', paddingBottom: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span>📋 Option Chain (LTP)</span>
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Real-time</span>
                  </h4>
                  
                  <div style={{ overflowX: 'auto' }}>
                    <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr style={{ borderBottom: '1.5px solid var(--border-color)', color: 'var(--text-muted)' }}>
                          <th style={{ textAlign: 'left', paddingBottom: '6px' }}>Call LTP</th>
                          <th style={{ textAlign: 'center', paddingBottom: '6px' }}>Strike</th>
                          <th style={{ textAlign: 'right', paddingBottom: '6px' }}>Put LTP</th>
                        </tr>
                      </thead>
                      <tbody>
                        {optionsChain.map((row, idx) => (
                          <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                            {/* Call option value */}
                            <td style={{ padding: '8px 0', color: row.call_change >= 0 ? 'var(--positive-dark)' : 'var(--negative-dark)', fontWeight: '600' }}>
                              ${row.call_ltp.toFixed(2)}
                            </td>
                            {/* Strike level */}
                            <td style={{ padding: '8px 0', textAlign: 'center', fontWeight: 'bold', backgroundColor: 'var(--bg-accent)', color: 'var(--text-primary)' }}>
                              {row.strike}
                            </td>
                            {/* Put option value */}
                            <td style={{ padding: '8px 0', textAlign: 'right', color: row.put_change >= 0 ? 'var(--positive-dark)' : 'var(--negative-dark)', fontWeight: '600' }}>
                              ${row.put_ltp.toFixed(2)}
                            </td>
                          </tr>
                        ))}
                        {optionsChain.length === 0 && (
                          <tr>
                            <td colSpan="3" style={{ textAlign: 'center', padding: '24px 0', color: 'var(--text-muted)', fontSize: '0.75rem' }}>
                              Connect feed to calculate options premiums
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </motion.div>
              </div>
            </motion.div>
          )}

          {!loading && !marketData && !error && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="loading-container" 
              style={{ padding: '64px 24px', backgroundColor: 'var(--bg-secondary)' }}
            >
              <Activity size={32} style={{ color: 'var(--text-muted)' }} />
              <p style={{ color: 'var(--text-secondary)', fontSize: '1rem' }}>
                No active stock streams loaded. Enter a ticker symbol (e.g. AAPL, TSLA) to establish real-time connections.
              </p>
            </motion.div>
          )}
        </div>
      )}

      {/* STOCK TRADING TERMINAL TAB */}
      {activeTab === 'terminal' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Portfolio Summary Bar */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            flexWrap: 'wrap', 
            gap: '16px', 
            padding: '16px 24px', 
            backgroundColor: 'var(--bg-secondary)', 
            borderRadius: 'var(--radius-md)', 
            border: '1px solid var(--border-color)',
            boxShadow: 'var(--shadow-sm)'
          }}>
            <div style={{ display: 'flex', gap: '32px', flexWrap: 'wrap' }}>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em', display: 'block', marginBottom: '2px' }}>ACTIVE SYMBOL</span>
                <strong style={{ fontSize: '1.3rem', color: 'var(--text-primary)' }}>{ticker || 'NONE'}</strong>
              </div>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em', display: 'block', marginBottom: '2px' }}>AVAILABLE CASH</span>
                <strong style={{ fontSize: '1.3rem', color: 'var(--positive-dark)' }}>
                  ${cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </strong>
              </div>
              <div>
                <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em', display: 'block', marginBottom: '2px' }}>YOUR POSITION</span>
                <strong style={{ fontSize: '1.3rem', color: 'var(--primary)' }}>
                  {holdings[ticker] ? `${holdings[ticker].shares} shares @ $${holdings[ticker].avgCost.toFixed(2)}` : '0 shares'}
                </strong>
              </div>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontStyle: 'italic', fontWeight: 500 }}>
              Use <code style={{ backgroundColor: 'var(--bg-accent)', padding: '2px 6px', borderRadius: '4px' }}>buy &lt;shares&gt;</code> to trade stocks.
            </div>
          </div>

          {/* Retro Bloomberg-style Terminal Box */}
          <div style={{ 
            backgroundColor: '#0c0f12', 
            color: '#00ff66', 
            fontFamily: 'Consolas, monaco, "Courier New", Courier, monospace', 
            padding: '24px', 
            borderRadius: '16px', 
            boxShadow: '0 20px 40px rgba(0,0,0,0.3)', 
            border: '2px solid #222e35',
            display: 'flex',
            flexDirection: 'column',
            gap: '16px'
          }}>
            {/* Terminal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid #1a242a', paddingBottom: '12px' }}>
              <div style={{ display: 'flex', gap: '6px' }}>
                <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#ff5f56', display: 'inline-block' }}></span>
                <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#ffbd2e', display: 'inline-block' }}></span>
                <span style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: '#27c93f', display: 'inline-block' }}></span>
              </div>
              <span style={{ fontSize: '0.78rem', color: '#ffb900', fontWeight: 700, letterSpacing: '0.1em' }}>⚡ SECURE FINANCIAL EXCHANGE TRADING TERMINAL ⚡</span>
              <span style={{ width: '40px' }}></span>
            </div>

            {/* Terminal History output */}
            <div style={{ 
              height: '440px', 
              overflowY: 'auto', 
              display: 'flex', 
              flexDirection: 'column', 
              gap: '6px',
              fontSize: '0.88rem',
              lineHeight: '1.45',
              paddingRight: '6px',
              color: '#d1f4ff'
            }}>
              {terminalHistory.map((line, idx) => (
                <div 
                  key={idx} 
                  style={{ 
                    whiteSpace: 'pre-wrap',
                    color: 
                      line.type === 'command' ? '#ffd700' : 
                      line.type === 'system' ? '#00e1ff' : 
                      line.type === 'success' ? '#00ff66' : 
                      line.type === 'error' ? '#ff3366' : '#d1f4ff'
                  }}
                  dangerouslySetInnerHTML={{ __html: line.text }}
                />
              ))}
              {loading && (
                <div style={{ color: '#00e1ff' }}>
                  &gt; Fetching market protocols and order books...
                </div>
              )}
              <div ref={terminalEndRef} />
            </div>

            {/* Prompt input */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderTop: '1px solid #1a242a', paddingTop: '16px', marginTop: '4px' }}>
              <span style={{ color: '#ffb900', fontWeight: 700, letterSpacing: '0.05em' }}>marketpulse-ai&gt;</span>
              <input 
                type="text" 
                value={terminalInput}
                onChange={(e) => setTerminalInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    runTerminalCommand(terminalInput);
                    setTerminalInput('');
                  }
                }}
                placeholder="Enter trading command (e.g., 'help', 'load TSLA', 'buy 10', 'portfolio', 'ascii')..."
                style={{ 
                  flex: 1, 
                  background: 'none', 
                  border: 'none', 
                  outline: 'none', 
                  color: '#ffffff', 
                  fontFamily: 'Consolas, monaco, "Courier New", Courier, monospace',
                  fontSize: '0.9rem',
                  caretColor: '#ffd700'
                }}
                autoFocus
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
