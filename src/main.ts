import { App } from './app/app';
import './styles.css';

const app = new App();
declare global {
  interface Window { smoothit: App }
}
window.smoothit = app;
void app.boot();

// Offline support for the browser/PWA build: after the first visit the app shell, the models and
// the WASM runtime are served from the cache. Native builds already ship everything in the bundle
// (and a service worker cannot see the WebView's local asset server), so they skip it.
const native = !!(window as unknown as { Capacitor?: { isNativePlatform?: () => boolean } }).Capacitor?.isNativePlatform?.();
if ('serviceWorker' in navigator && import.meta.env.PROD && location.protocol === 'https:' && !native) {
  window.addEventListener('load', () => void navigator.serviceWorker.register('./sw.js').catch(() => undefined));
}
