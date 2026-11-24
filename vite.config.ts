import { resolve } from 'node:path';
import basicSsl from '@vitejs/plugin-basic-ssl';
import { defineConfig } from 'vitest/config';

// Relative base so the same build runs from a web server, GitHub Pages, and inside the
// Capacitor iOS/Android shells (served from the app bundle).
// `npm run dev:https` serves over a self-signed certificate so a phone on the same network can open
// the dev server: browsers only expose the camera to secure origins.
export default defineConfig({
  base: './',
  plugins: process.env.SMOOTHIT_HTTPS ? [basicSsl()] : [],
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        lab: resolve(__dirname, 'lab.html'),
      },
    },
  },
  server: { host: true },
  test: { include: ['tests/**/*.test.ts'] },
});
