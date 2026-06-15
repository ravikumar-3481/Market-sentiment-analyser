import React, { useState } from 'react';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import ScrapedArticles from './pages/ScrapedArticles';
import MarketData from './pages/MarketData';
import ArticleView from './pages/ArticleView';
import About from './pages/About';

export default function App() {
  const [page, setPage] = useState('Home');
  
  // Shared States
  const [newsData, setNewsData] = useState([]);
  const [selectedArticle, setSelectedArticle] = useState(null);
  const [query, setQuery] = useState('NVDA');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const renderActivePage = () => {
    switch (page) {
      case 'Home':
        return <Home setPage={setPage} />;
      case 'Dashboard':
        return (
          <Dashboard 
            newsData={newsData} 
            setNewsData={setNewsData} 
            query={query} 
            setQuery={setQuery} 
            loading={loading} 
            setLoading={setLoading}
            error={error}
            setError={setError}
          />
        );
      case 'Scraped Articles':
        return (
          <ScrapedArticles 
            newsData={newsData} 
            setPage={setPage} 
            setSelectedArticle={setSelectedArticle} 
          />
        );
      case 'Market Data':
        return <MarketData />;
      case 'Article View':
        return <ArticleView article={selectedArticle} setPage={setPage} />;
      case 'About':
        return <About />;
      default:
        return <Home setPage={setPage} />;
    }
  };

  const navItems = [
    { name: 'Home', icon: '🏠' },
    { name: 'Dashboard', icon: '📊' },
    { name: 'Scraped Articles', icon: '📰' },
    { name: 'Market Data', icon: '📈' },
    { name: 'About', icon: '👨‍💻' }
  ];

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="logo" onClick={() => setPage('Home')} style={{ cursor: 'pointer' }}>
          <span>🤖</span>
          <span>MarketPulse AI</span>
        </div>
        
        <nav className="nav-links">
          {navItems.map((item) => (
            <button
              key={item.name}
              className={`nav-button ${page === item.name || (item.name === 'Scraped Articles' && page === 'Article View') ? 'active' : ''}`}
              onClick={() => {
                setPage(item.name);
                if (item.name === 'Scraped Articles' && selectedArticle) {
                  // Keep it on article view or switch back
                }
              }}
            >
              <span>{item.icon}</span>
              <span>{item.name}</span>
            </button>
          ))}
        </nav>
      </header>

      {/* Main View Area */}
      <main style={{ minHeight: 'calc(100vh - 200px)' }}>
        {renderActivePage()}
      </main>

      {/* Footer */}
      <footer style={{ 
        marginTop: '64px', 
        padding: '24px 0', 
        borderTop: '1px solid var(--border-color)', 
        textAlign: 'center', 
        fontSize: '0.85rem',
        color: 'var(--text-muted)'
      }}>
        <p>© {new Date().getFullYear()} MarketPulse AI. Enterprise Financial Analysis Platform.</p>
        <p style={{ marginTop: '4px' }}>Powered by FastAPI & React (Vite).</p>
      </footer>
    </div>
  );
}
