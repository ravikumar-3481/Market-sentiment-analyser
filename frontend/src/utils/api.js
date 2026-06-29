/**
 * centralizes backend fetch request operations, resolving API hosts
 * from localStorage with dynamic fallback configurations.
 */

export const DEFAULT_API_HOST = 'https://market-sentiment-analyser-1.onrender.com';


export const getApiSettings = () => {
  const host = localStorage.getItem('marketpulse_api_host') || DEFAULT_API_HOST;
  return { host };
};

export const saveApiSettings = (host) => {
  // Trim and remove trailing slashes from host
  const cleanedHost = host.trim().replace(/\/+$/, '');
  localStorage.setItem('marketpulse_api_host', cleanedHost);
  return { host: cleanedHost };
};

export const apiFetch = async (endpoint, options = {}) => {
  const { host } = getApiSettings();
  
  // Normalise endpoint to prevent double-slashes
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const url = `${host}${path}`;
  
  // Build headers dictionary
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  
  const response = await fetch(url, {
    ...options,
    headers,
  });
  
  return response;
};
