import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Server, CheckCircle, Wifi, WifiOff, RefreshCw } from 'lucide-react';
import { getApiSettings, saveApiSettings, DEFAULT_API_HOST } from '../utils/api';

export default function Settings() {
  const [apiHost, setApiHost] = useState('');
  const [saveSuccess, setSaveSuccess] = useState(false);
  
  // Connection testing states
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null); // 'success' | 'error' | null
  const [testError, setTestError] = useState('');

  useEffect(() => {
    const { host } = getApiSettings();
    setApiHost(host);
  }, []);

  const handleSave = (e) => {
    e.preventDefault();
    saveApiSettings(apiHost);
    setSaveSuccess(true);
    setTestResult(null); // Reset test results on save
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const handleReset = () => {
    setApiHost(DEFAULT_API_HOST);
    saveApiSettings(DEFAULT_API_HOST);
    setSaveSuccess(true);
    setTestResult(null);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const testConnection = async () => {
    setTesting(true);
    setTestResult(null);
    setTestError('');
    
    try {
      // Use clean host URL
      const hostToTest = apiHost.trim().replace(/\/+$/, '');
      const response = await fetch(`${hostToTest}/api/health`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (response.ok) {
        await response.json();
        setTestResult('success');
      } else {
        throw new Error(`HTTP Error ${response.status}`);
      }
    } catch (err) {
      setTestResult('error');
      setTestError(err.message || 'Unable to connect to host. Make sure backend is running.');
    } finally {
      setTesting(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', maxWidth: '800px' }}>
      <div>
        <h2 style={{ marginBottom: '6px', fontSize: '1.8rem' }}>⚙️ SaaS Environment Settings</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          Configure API connection endpoints and verify the health of your intelligence services.
        </p>
      </div>

      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Active API Host Status */}
          <div style={{ padding: '20px', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius)', border: '1px solid var(--border-color)', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.05em', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                <Server size={14} /> ACTIVE API HOST GATEWAY
              </div>
              <code style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--primary)', fontFamily: 'monospace' }}>{apiHost || 'http://127.0.0.1:8000'}</code>
            </div>
            <button 
              type="button" 
              className="btn btn-primary" 
              onClick={testConnection} 
              disabled={testing || !apiHost}
              style={{ display: 'flex', alignItems: 'center', gap: '8px', height: '44px', padding: '0 20px' }}
            >
              {testing ? <RefreshCw size={14} className="spinner" /> : <Wifi size={14} />}
              Test Connectivity
            </button>
          </div>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', margin: 0 }}>
            The application dynamically connects to the endpoint specified above to fetch real-time charts, VADER scores, and headlines analysis.
          </p>
        </div>

        {/* Success Toast */}
        {saveSuccess && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="alert alert-info"
            style={{ marginTop: '20px', border: '1px solid var(--primary)' }}
          >
            <CheckCircle size={18} style={{ color: 'var(--primary)' }} />
            <span>Settings saved successfully in your local browser workspace.</span>
          </motion.div>
        )}

        {/* Test Connection Results */}
        {testResult === 'success' && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="alert"
            style={{ 
              marginTop: '20px', 
              backgroundColor: 'var(--positive-light)', 
              borderColor: 'rgba(16, 185, 129, 0.3)',
              color: 'var(--positive-dark)'
            }}
          >
            <Wifi size={18} />
            <span><strong>Connection Active:</strong> Successfully reached the FastAPI backend. Status is healthy!</span>
          </motion.div>
        )}

        {testResult === 'error' && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="alert alert-warning"
            style={{ marginTop: '20px' }}
          >
            <WifiOff size={18} />
            <span><strong>Connection Offline:</strong> {testError}</span>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
