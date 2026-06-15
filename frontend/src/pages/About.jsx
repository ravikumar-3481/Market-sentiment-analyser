import React from 'react';

export default function About() {
  return (
    <div>
      <h2 style={{ marginBottom: '8px', fontSize: '1.8rem' }}>👨‍💻 Developer Profile & Architecture</h2>
      <p style={{ color: 'var(--text-muted)', marginBottom: '24px' }}>
        Learn about the engineering behind MarketPulse AI and connect with the author.
      </p>

      <div className="scraped-view" style={{ gridTemplateColumns: '1fr 2fr' }}>
        {/* Left Column: Developer Profile */}
        <div className="card" style={{ textAlign: 'center', height: 'fit-content' }}>
          <img 
            src="https://profileravi.netlify.app/og/img.webp" 
            alt="Ravi Vishwakarma" 
            style={{ 
              width: '180px', 
              height: '180px', 
              borderRadius: '50%', 
              objectFit: 'cover', 
              margin: '0 auto 16px auto', 
              border: '4px solid var(--primary-light)',
              boxShadow: 'var(--shadow-md)'
            }}
            onError={(e) => {
              e.target.src = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=500&auto=format&fit=crop";
            }}
          />
          <h3 style={{ fontSize: '1.4rem', marginBottom: '4px' }}>Ravi Vishwakarma</h3>
          <p style={{ color: 'var(--primary)', fontWeight: 700, fontSize: '0.9rem', marginBottom: '16px' }}>
            Machine Learning Engineer & Developer
          </p>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '24px', textAlign: 'left', lineHeight: '1.5' }}>
            Specialized in deploying Deep Learning models into lightweight web environments, developing robust data acquisition pipelines, and crafting clean, premium user interfaces.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <a href="https://profileravi.netlify.app/" target="_blank" rel="noreferrer" className="btn btn-primary" style={{ fontSize: '0.9rem', textDecoration: 'none' }}>
              🌐 Portfolio Website
            </a>
            <a href="https://linkedin.com/in/ravi-vishwakarma-2a8264253" target="_blank" rel="noreferrer" className="btn btn-secondary" style={{ fontSize: '0.9rem', textDecoration: 'none' }}>
              💼 LinkedIn Profile
            </a>
            <a href="https://github.com/RAVIVISHWAKARMA2003" target="_blank" rel="noreferrer" className="btn btn-secondary" style={{ fontSize: '0.9rem', textDecoration: 'none' }}>
              💻 GitHub Repository
            </a>
          </div>
        </div>

        {/* Right Column: System Architecture */}
        <div className="card">
          <h3 style={{ fontSize: '1.40rem', marginBottom: '12px' }}>Enterprise AI Pipeline Architecture</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '20px' }}>
            MarketPulse AI operates as a distributed financial pipeline, connecting raw internet feed protocols directly to local GPU/CPU transformers.
          </p>

          <div className="pipeline-container">
            <div className="pipeline-step">
              <div className="pipeline-num">1</div>
              <div className="pipeline-details">
                <h4>Ingestion Layer</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  Fetches content asynchronously from Google RSS search protocols, scrapes HTML body elements via <code>BeautifulSoup4</code>, and triggers stock price feeds using <code>yFinance</code>.
                </p>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="pipeline-num">2</div>
              <div className="pipeline-details">
                <h4>Processing & Filtering</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  Strips raw HTML markup, normalizes strings, and formats timestamps. Merges news channels and Reddit WallStreetBets RSS data feeds cleanly under unified JSON models.
                </p>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="pipeline-num">3</div>
              <div className="pipeline-details">
                <h4>AI Inference Engines</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  Processes titles, summaries, and body text through 4 distinct Hugging Face neural networks:
                </p>
                <ul style={{ paddingLeft: '20px', marginTop: '8px', fontSize: '0.85rem', color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <li><code>ProsusAI/finbert</code> (Financial sentiment classifier)</li>
                  <li><code>typeform/distilbert-base-uncased-mnli</code> (Zero-shot topic classifier)</li>
                  <li><code>Falconsai/text_summarization</code> (BART-based summarizer)</li>
                  <li><code>dslim/bert-base-NER</code> (Token classifier for Named Entity Recognition)</li>
                </ul>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="pipeline-num">4</div>
              <div className="pipeline-details">
                <h4>Resource Management</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  To avoid Out-Of-Memory (OOM) failures or sluggish startup times, the backend implements lazy-loading. Deep learning models are instantiated in CPU/GPU VRAM strictly when the corresponding endpoint is triggered for the first time.
                </p>
              </div>
            </div>

            <div className="pipeline-step">
              <div className="pipeline-num">5</div>
              <div className="pipeline-details">
                <h4>Visualization Layer</h4>
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                  Outputs structural metrics to a clean React application, rendering customized SVG-based technical candlestick price action, retail VADER indices, and semantic entity badges.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
