import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';

// Relative base so the same build runs from a web server, GitHub Pages, and inside the
// Capacitor iOS/Android shells (served from the app bundle).
export default defineConfig({
  base: './',
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
