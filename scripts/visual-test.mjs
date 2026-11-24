// Visual regression helper: runs still images through the real capture pipeline in headless
// Chromium (software WebGL) and writes before/after JPEGs plus the Director's diagnostics.
//
//   npm run dev            (in another terminal)
//   node scripts/visual-test.mjs --out /tmp/out [--look studio] [--url http://localhost:5173] a.jpg b.jpg
import { mkdirSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { basename, extname, join, resolve } from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const args = process.argv.slice(2);
const opt = { out: 'test-results', look: 'studio', url: 'http://localhost:5173', delegate: 'CPU', maxLong: 1600, debug: 'none', overrides: '{}', tag: '' };
const files = [];
for (let i = 0; i < args.length; i++) {
  if (args[i].startsWith('--')) opt[args[i].slice(2)] = args[++i];
  else files.push(resolve(args[i]));
}
mkdirSync(opt.out, { recursive: true });

const browser = await chromium.launch({
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[browser ${m.type()}]`, m.text()); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.route('**/__img/*', (route) => {
  const idx = Number(new URL(route.request().url()).pathname.split('/').pop());
  return route.fulfill({ path: files[idx] });
});
await page.goto(`${opt.url}/lab.html?delegate=${opt.delegate}`);
await page.evaluate(() => window.smoothitLab.ready);

const summary = {};
for (let i = 0; i < files.length; i++) {
  const name = basename(files[i], extname(files[i]));
  for (const look of String(opt.look).split(',')) {
    const info = await page.evaluate(
      ([url, lk, maxLong, debug, overrides]) => window.smoothitLab.run(url, lk, { maxLong, debug, overrides }),
      [`/__img/${i}`, look, Number(opt.maxLong), opt.debug, JSON.parse(opt.overrides)],
    );
    for (const id of ['before', 'after']) {
      const dataUrl = await page.evaluate((cid) => document.getElementById(cid).toDataURL('image/jpeg', 0.93), id);
      const suffix = id === 'after' ? `after-${look}${opt.tag}` : 'before';
      writeFileSync(join(opt.out, `${name}.${suffix}.jpg`), Buffer.from(dataUrl.split(',')[1], 'base64'));
    }
    summary[`${name}/${look}`] = info;
    console.log(name, look, JSON.stringify(info));
  }
}
writeFileSync(join(opt.out, 'summary.json'), JSON.stringify(summary, null, 2));
await browser.close();
