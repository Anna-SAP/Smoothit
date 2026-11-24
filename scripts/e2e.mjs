// End-to-end smoke test of the real app in headless Chromium with a fake camera fed from a video
// file: boots the app, waits for the live pipeline, takes a photo and saves screenshots + the JPEG.
//
//   npm run dev   (in another terminal)
//   node scripts/e2e.mjs --video face.y4m --out test-results [--url http://localhost:5173] [--look film]
import { mkdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join, resolve } from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const opt = { out: 'test-results', url: 'http://localhost:5173', video: '', look: '', name: 'e2e', lang: 'zh-CN' };
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i += 2) opt[args[i].slice(2)] = args[i + 1];
if (!opt.video) throw new Error('--video <file.y4m|.mjpeg> is required');
mkdirSync(opt.out, { recursive: true });

const browser = await chromium.launch({
  args: [
    '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist',
    '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
    `--use-file-for-fake-video-capture=${resolve(opt.video)}`,
  ],
});
const context = await browser.newContext({
  viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, locale: opt.lang,
  permissions: ['camera'],
});
const page = await context.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error' && !m.text().includes('XNNPACK')) errors.push(m.text()); });

const t0 = Date.now();
await page.goto(`${opt.url}/?delegate=CPU&debug=1`);
await page.waitForFunction(() => window.smoothit?.state.frame >= 4, null, { timeout: 180_000 });
if (opt.look) await page.click(`#looks [data-look="${opt.look}"]`);
await page.waitForFunction((f) => window.smoothit.state.frame >= f, (await page.evaluate(() => window.smoothit.state.frame)) + 4, { timeout: 120_000 });
const live = await page.evaluate(() => window.smoothit.state);
await page.screenshot({ path: join(opt.out, `${opt.name}-1-live.png`) });

await page.click('#shutter');
await page.waitForSelector('#review:not([hidden])', { timeout: 180_000 });
await page.waitForFunction(() => document.getElementById('reviewImg').naturalWidth > 0);
await page.screenshot({ path: join(opt.out, `${opt.name}-2-review.png`) });
const photo = await page.evaluate(async () => {
  const img = document.getElementById('reviewImg');
  const buf = await (await fetch(img.src)).arrayBuffer();
  let bin = '';
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return { b64: btoa(bin), width: img.naturalWidth, height: img.naturalHeight };
});
writeFileSync(join(opt.out, `${opt.name}-photo.jpg`), Buffer.from(photo.b64, 'base64'));

await page.click('#reviewClose');
await page.click('#galleryBtn');
await page.waitForSelector('#gallery:not([hidden])');
const galleryCount = await page.evaluate(() => document.querySelectorAll('#galleryGrid button').length);
await page.screenshot({ path: join(opt.out, `${opt.name}-3-gallery.png`) });

const summary = { seconds: (Date.now() - t0) / 1000, live, photo: { width: photo.width, height: photo.height }, galleryCount, errors };
writeFileSync(join(opt.out, `${opt.name}-summary.json`), JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 1));
await browser.close();
if (errors.length || !live.hasFace || photo.width < 600 || galleryCount < 1) process.exit(1);
