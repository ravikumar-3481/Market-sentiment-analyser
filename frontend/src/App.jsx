import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Home as HomeIcon, 
  LayoutDashboard, 
  Newspaper, 
  LineChart, 
  User,
  Bot,
  Terminal,
  Settings as SettingsIcon,
  Menu,
  X
} from 'lucide-react';

import Home from './pages/Home';
import Dashboard from './pages/Dashboard';
import ScrapedArticles from './pages/ScrapedArticles';
import MarketData from './pages/MarketData';
import ArticleView from './pages/ArticleView';
import About from './pages/About';
import Settings from './pages/Settings';

export default function App() {
  const [page, setPage] = useState('Home');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  
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
      case 'Settings':
        return <Settings />;
      case 'About':
        return <About />;
      default:
        return <Home setPage={setPage} />;
    }
  };

  const navItems = [
    { name: 'Home', icon: HomeIcon },
    { name: 'Dashboard', icon: LayoutDashboard },
    { name: 'Scraped Articles', icon: Newspaper },
    { name: 'Market Data', icon: LineChart },
    { name: 'Settings', icon: SettingsIcon },
    { name: 'About', icon: User }
  ];

  const pageVariants = {
    initial: { opacity: 0, y: 15 },
    animate: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
    exit: { opacity: 0, y: -15, transition: { duration: 0.2, ease: 'easeIn' } }
  };

  const sidebarVariants = {
    open: { x: 0, transition: { type: 'spring', stiffness: 300, damping: 30 } },
    closed: { x: '-100%', transition: { type: 'spring', stiffness: 300, damping: 30 } }
  };

  const navigateToPage = (targetPage) => {
    setPage(targetPage);
    setMobileMenuOpen(false);
  };

  return (
    <div className="app-container">
      {/* Mobile Top Navbar */}
      <header className="mobile-header">
        <div className="logo" style={{ marginBottom: 0 }}>
          <div className="logo-icon">
            <Bot size={20} />
          </div>
          <span className="logo-text">MarketPulse AI</span>
        </div>
        <button 
          className="mobile-menu-toggle" 
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          aria-label="Toggle Navigation Menu"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </header>

      <div className="saas-layout">
        {/* Sidebar Panel - Desktop View */}
        <aside className="saas-sidebar desktop-sidebar">
          <div className="logo">
            <div className="logo-icon">
              <Bot size={20} />
            </div>
            <span className="logo-text">MarketPulse AI</span>
          </div>
          
          <nav className="sidebar-links">
            {navItems.map((item) => {
              const IconComponent = item.icon;
              const isActive = page === item.name || (item.name === 'Scraped Articles' && page === 'Article View');
              return (
                <button
                  key={item.name}
                  className={`sidebar-button ${isActive ? 'active' : ''}`}
                  onClick={() => navigateToPage(item.name)}
                >
                  <IconComponent size={18} />
                  <span>{item.name}</span>
                </button>
              );
            })}
          </nav>

          <div style={{ marginTop: 'auto', padding: '16px 8px 0 8px', borderTop: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.8rem' }}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--positive)' }}></div>
            <span style={{ color: 'var(--text-secondary)' }}>v1.1.0 (FastAPI Secure)</span>
          </div>
        </aside>

        {/* Mobile Slide-out Drawer */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <>
              {/* Dark backdrop overlay */}
              <motion.div 
                className="drawer-backdrop"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setMobileMenuOpen(false)}
              />
              <motion.aside 
                className="saas-sidebar mobile-drawer"
                variants={sidebarVariants}
                initial="closed"
                animate="open"
                exit="closed"
              >
                <div className="logo" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <div className="logo-icon">
                      <Bot size={20} />
                    </div>
                    <span className="logo-text">MarketPulse AI</span>
                  </div>
                  <button className="mobile-drawer-close" onClick={() => setMobileMenuOpen(false)}>
                    <X size={20} />
                  </button>
                </div>
                
                <nav className="sidebar-links">
                  {navItems.map((item) => {
                    const IconComponent = item.icon;
                    const isActive = page === item.name || (item.name === 'Scraped Articles' && page === 'Article View');
                    return (
                      <button
                        key={item.name}
                        className={`sidebar-button ${isActive ? 'active' : ''}`}
                        onClick={() => navigateToPage(item.name)}
                      >
                        <IconComponent size={18} />
                        <span>{item.name}</span>
                      </button>
                    );
                  })}
                </nav>

                <div style={{ marginTop: 'auto', padding: '16px 8px 0 8px', borderTop: '1px solid var(--border-color)', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.8rem' }}>
                  <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--positive)' }}></div>
                  <span style={{ color: 'var(--text-secondary)' }}>v1.1.0 (FastAPI Secure)</span>
                </div>
              </motion.aside>
            </>
          )}
        </AnimatePresence>

        {/* Main Content Area */}
        <main className="saas-main">
          <AnimatePresence mode="wait">
            <motion.div
              key={page}
              variants={pageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              style={{ flex: 1, display: 'flex', flexDirection: 'column' }}
            >
              {renderActivePage()}
            </motion.div>
          </AnimatePresence>

          {/* Footer inside main layout */}
          <footer className="footer">
            <p>© {new Date().getFullYear()} MarketPulse AI. Enterprise Financial Analysis Platform.</p>
            <p style={{ marginTop: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '6px' }}>
              <Terminal size={12} /> Powered by FastAPI & React (Vite).
            </p>
          </footer>
        </main>
      </div>
    </div>
  );
}
