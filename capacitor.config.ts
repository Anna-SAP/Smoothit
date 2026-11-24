import type { CapacitorConfig } from '@capacitor/cli';

// Native shells: the same web build is packaged into an iOS (WKWebView) and an Android (WebView)
// app. Camera access goes through the standard web camera API inside the shell.
const config: CapacitorConfig = {
  appId: 'app.smoothit.camera',
  appName: 'Smoothit',
  webDir: 'dist',
  backgroundColor: '#0c0b0b',
  ios: { contentInset: 'never', backgroundColor: '#0c0b0b' },
  android: { backgroundColor: '#0c0b0b' },
};

export default config;
