import { App } from './app/app';
import './styles.css';

const app = new App();
declare global {
  interface Window { smoothit: App }
}
window.smoothit = app;
void app.boot();

// Offline support: after the first visit the app shell, the models and the WASM runtime are served
// from the cache, so the installed PWA opens without a network.
if ('serviceWorker' in navigator && import.meta.env.PROD && location.protocol === 'https:') {
  window.addEventListener('load', () => void navigator.serviceWorker.register('./sw.js').catch(() => undefined));
}
