// Development lab: runs a still image through the same code path as a capture and shows
// before / after. Also the entry point of the automated visual tests (window.smoothitLab).
import { processStill } from './capture/still';
import { type DebugView, Engine } from './engine/engine';
import { LOOKS, lookById } from './engine/looks';
import { Vision } from './vision/vision';

const log = document.getElementById('log') as HTMLPreElement;
const lookSel = document.getElementById('look') as HTMLSelectElement;
for (const l of LOOKS) lookSel.add(new Option(`${l.name.en} / ${l.name.zh}`, l.id));

const query = new URLSearchParams(location.search);
const engine = new Engine(document.createElement('canvas'));
let visionPromise: Promise<Vision> | null = null;
const getVision = () =>
  (visionPromise ??= Vision.create({
    baseUrl: import.meta.env.BASE_URL,
    mode: 'IMAGE',
    delegate: (query.get('delegate') as 'GPU' | 'CPU' | null) ?? undefined,
  }));

function show(id: string, pixels: Uint8Array, w: number, h: number) {
  const c = document.getElementById(id) as HTMLCanvasElement;
  c.width = w;
  c.height = h;
  c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(pixels), w, h), 0, 0);
}

type Overrides = { [k: string]: number | number[] | Overrides };
function merge<T>(base: T, over: Overrides | undefined): T {
  if (!over) return base;
  const out = { ...base } as Record<string, unknown>;
  for (const [k, v] of Object.entries(over)) {
    out[k] = v !== null && typeof v === 'object' && !Array.isArray(v) ? merge(out[k], v) : v;
  }
  return out as T;
}

async function run(
  url: string, lookId: string,
  opts: { mirror?: boolean; maxLong?: number; overrides?: Overrides; debug?: DebugView } = {},
) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const bitmap = await createImageBitmap(img);
  const vision = await getVision();
  engine.debugView = opts.debug ?? 'none';
  const t0 = performance.now();
  const res = processStill(engine, vision, bitmap, {
    look: merge(lookById(lookId), opts.overrides), mirror: opts.mirror ?? false, maxLongSide: opts.maxLong ?? 2048, withOriginal: true,
  });
  const ms = performance.now() - t0;
  show('after', res.pixels, res.width, res.height);
  if (res.original) show('before', res.original, res.width, res.height);
  const info = { ms: Math.round(ms), delegate: vision.delegate, size: [res.width, res.height], hints: res.direction.hints, ...res.direction.diagnostics };
  log.textContent = JSON.stringify(info, null, 1);
  return info;
}

(document.getElementById('file') as HTMLInputElement).addEventListener('change', (e) => {
  const f = (e.target as HTMLInputElement).files?.[0];
  if (f) void run(URL.createObjectURL(f), lookSel.value);
});

declare global {
  interface Window { smoothitLab: { run: typeof run; ready: Promise<unknown> } }
}
window.smoothitLab = { run, ready: getVision() };
if (query.get('img')) void run(query.get('img')!, query.get('look') ?? 'studio');
