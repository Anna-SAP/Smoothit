// Downloads the on-device models and copies the MediaPipe WASM runtime into public/.
// Run automatically before `dev` and `build`. Everything runs on-device; nothing is fetched at runtime.
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const MODELS = [
  {
    file: 'face_landmarker.task',
    url: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    sha256: '64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff',
  },
  {
    file: 'selfie_multiclass_256x256.tflite',
    url: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
    sha256: 'c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0',
  },
];

const sha = (buf) => createHash('sha256').update(buf).digest('hex');

async function fetchModel(m) {
  const dest = join(root, 'public/models', m.file);
  if (existsSync(dest)) {
    if (!m.sha256 || sha(readFileSync(dest)) === m.sha256) return console.log(`ok       ${m.file}`);
    console.log(`checksum mismatch, re-downloading ${m.file}`);
  }
  const res = await fetch(m.url);
  if (!res.ok) throw new Error(`${m.url} -> HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (m.sha256 && sha(buf) !== m.sha256) throw new Error(`checksum mismatch for ${m.file}: ${sha(buf)}`);
  writeFileSync(dest, buf);
  console.log(`fetched  ${m.file} (${(buf.length / 1e6).toFixed(1)} MB, sha256 ${sha(buf)})`);
}

mkdirSync(join(root, 'public/models'), { recursive: true });
mkdirSync(join(root, 'public/wasm'), { recursive: true });
for (const f of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm']) {
  cpSync(join(root, 'node_modules/@mediapipe/tasks-vision/wasm', f), join(root, 'public/wasm', f));
}
console.log('copied   MediaPipe WASM runtime');
for (const m of MODELS) await fetchModel(m);
