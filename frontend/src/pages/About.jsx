import React from 'react';
import { motion } from 'framer-motion';
import { 
  Terminal, 
  Cpu, 
  Globe, 
  Linkedin, 
  Github, 
  Database,
  Layers,
  Cpu as EngineIcon,
  HardDrive,
  MonitorPlay
} from 'lucide-react';

export default function About() {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { staggerChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 15 },
    show: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 80 } }
  };

  const pipelineSteps = [
    {
      num: 1,
      icon: Database,
      title: "Asynchronous Ingestion Layer",
      desc: "Queries real-time RSS search structures (Google News) and scrapes remote HTML text nodes using BeautifulSoup4, while downloading historical market ticks via yFinance."
    },
    {
      num: 2,
      icon: Layers,
      title: "Data Normalization & Cleaning",
      desc: "Runs clean-up regex pipelines on raw HTML nodes. Normalizes timestamps and hashes unique article contents inside unified JSON schemas."
    },
    {
      num: 3,
      icon: EngineIcon,
      title: "Deep Learning Transformers",
      desc: "Routes ingested paragraph vectors into 4 specific neural networks: ProsusAI/finbert (sentiment modeling), DistilBERT-MNLI (zero-shot classification), DistilBART (summarization), and BERT-NER (token entity identification)."
    },
    {
      num: 4,
      icon: HardDrive,
      title: "Lazy VRAM Resource Management",
      desc: "In order to operate within restrictive cloud compute quotas, heavy networks are initialized strictly on-demand. VRAM footprint is allocated dynamically when the first query is triggered."
    },
    {
      num: 5,
      icon: MonitorPlay,
      title: "SaaS Visualization Terminal",
      desc: "Processes semantic scores, named entities, and technical candlestick indices into interactive SVG chart paths and real-time dashboard progress structures."
    }
  ];

  return (
    <motion.div 
      variants={containerVariants}
      initial="hidden"
      animate="show"
      style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}
    >
      <div>
        <h2 style={{ marginBottom: '6px', fontSize: '1.8rem' }}>👨‍💻 Developer & System Core</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          Detailed breakdown of the enterprise NLP architecture powering the MarketPulse platform.
        </p>
      </div>

      <div className="scraped-view" style={{ gridTemplateColumns: '1fr 1.8fr' }}>
        {/* Left Column: Profile Card */}
        <motion.div variants={itemVariants} className="card" style={{ height: 'fit-content', display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
          <img 
            src="https://profileravi.netlify.app/og/img.webp" 
            alt="Ravi Vishwakarma" 
            style={{ 
              width: '150px', 
              height: '150px', 
              borderRadius: '50%', 
              objectFit: 'cover', 
              margin: '0 auto 16px auto', 
              border: '3px solid var(--primary)',
              boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)'
            }}
            onError={(e) => {
              e.target.src = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=500&auto=format&fit=crop";
            }}
          />
          <h3 style={{ fontSize: '1.3rem', marginBottom: '4px', color: 'var(--text-primary)' }}>Ravi Vishwakarma</h3>
          <p style={{ color: 'var(--primary)', fontWeight: 700, fontSize: '0.85rem', marginBottom: '20px', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Machine Learning Engineer
          </p>
          <p style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: '28px', textAlign: 'left', lineHeight: '1.6' }}>
            Specializing in integrating heavy neural network models into highly responsive UI web ecosystems, designing low-latency crawling arrays, and writing clean, scalable terminal systems.
          </p>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', width: '100%' }}>
            <a href="https://profileravi.netlify.app/" target="_blank" rel="noreferrer" className="btn btn-primary" style={{ fontSize: '0.85rem', padding: '12px', textDecoration: 'none' }}>
              <Globe size={14} /> Portfolio Site
            </a>
            <a href="https://linkedin.com/in/ravi-vishwakarma-2a8264253" target="_blank" rel="noreferrer" className="btn btn-secondary" style={{ fontSize: '0.85rem', padding: '12px', textDecoration: 'none' }}>
              <Linkedin size={14} /> LinkedIn Connection
            </a>
            <a href="https://github.com/RAVIVISHWAKARMA2003" target="_blank" rel="noreferrer" className="btn btn-secondary" style={{ fontSize: '0.85rem', padding: '12px', textDecoration: 'none' }}>
              <Github size={14} /> GitHub Workpace
            </a>
          </div>
        </motion.div>

        {/* Right Column: Timelines */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <motion.div variants={itemVariants} className="card">
            <h3 style={{ fontSize: '1.25rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Terminal size={18} style={{ color: 'var(--primary)' }} /> AI Pipeline Architecture
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '24px', lineHeight: '1.5' }}>
              Unstructured streaming logs are routed dynamically across localized model caches to run parallel inference arrays:
            </p>

            <div className="pipeline-container" style={{ gap: '16px' }}>
              {pipelineSteps.map((step) => {
                const StepIcon = step.icon;
                return (
                  <motion.div 
                    key={step.num}
                    variants={itemVariants} 
                    className="pipeline-step"
                    style={{ gap: '16px', padding: '16px' }}
                  >
                    <div className="pipeline-num" style={{ width: '36px', height: '36px', fontSize: '0.95rem' }}>
                      <StepIcon size={16} />
                    </div>
                    <div className="pipeline-details">
                      <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: 'var(--text-primary)' }}>{step.title}</h4>
                      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '4px', lineHeight: '1.5' }}>
                        {step.desc}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        </div>
      </div>
    </motion.div>
  );
}
