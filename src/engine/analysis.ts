// CPU-side scene analysis on a small copy of the frame: what a photographer's eye and light meter
// would read before touching any setting. Everything here is cheap (a ~190x256 image).
import { EYE_A_LOWER, EYE_A_UPPER, EYE_B_LOWER, EYE_B_UPPER, SKIN_SAMPLE_VERTS } from '../vision/topology';
import type { AnalysisPixels, SegmentationMask } from './engine';
import type { FaceGeometry } from './faceModel';
import { SH_TERMS, describeLighting, fitLighting, type LightDescription } from './lighting';

const SRGB_TO_LIN = (() => {
  const t = new Float32Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    t[i] = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  }
  return t;
})();

export interface Analysis {
  hasFace: boolean;
  /** Mean skin colour, linear RGB. */
  skinLin: [number, number, number];
  /**
   * Colour of the lit half of the skin, linear RGB. Camera tone curves inflate channel ratios in
   * shadows, so chromaticity for white balance is read from well-exposed skin only.
   */
  skinLitLin: [number, number, number];
  /** Skin tone in gamma space as the shaders want it: luma, B-Y, R-Y. */
  skinYcc: [number, number, number];
  /** Spread of skin chroma (gamma space). */
  skinChromaSigma: number;
  /** Linear luminance of the brightest fifth of the skin (the lit side). */
  skinHiLin: number;
  /** Fraction of skin samples with a clipped channel. */
  skinClipped: number;
  /** Linear luminance of the eye whites, or 0 when not measurable. */
  scleraLin: number;
  /** Fitted normalised SH lighting on the face. */
  sh: Float64Array;
  light: LightDescription;
  /** Scene luma percentiles (gamma, 0..1). */
  p05: number;
  p50: number;
  p95: number;
  /** Fraction of pixels at or near clipping. */
  clipped: number;
  /** Mean linear luminance of the whole frame, and of the background only. */
  sceneLin: number;
  backgroundLin: number;
  /**
   * Log2 chromaticity (R/G, B/G) of the background's mid-tones: a second, independent witness of
   * the illuminant colour. `sceneChromaWeight` is 0 when there was too little to measure.
   */
  sceneChroma: [number, number];
  sceneChromaWeight: number;
}

export function emptyAnalysis(): Analysis {
  const sh = new Float64Array(SH_TERMS);
  sh[0] = 1;
  return {
    hasFace: false, skinLin: [0.4, 0.26, 0.2], skinLitLin: [0.4, 0.26, 0.2], skinYcc: [0.6, -0.08, 0.1], skinChromaSigma: 0.04,
    skinHiLin: 0.4, skinClipped: 0, scleraLin: 0, sceneChroma: [0, 0], sceneChromaWeight: 0,
    sh, light: describeLighting(sh), p05: 0.05, p50: 0.45, p95: 0.9, clipped: 0, sceneLin: 0.2, backgroundLin: 0.2,
  };
}

const lumaOf = (r: number, g: number, b: number) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

function pointInPolygon(x: number, y: number, poly: number[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 2; i < poly.length; j = i, i += 2) {
    const xi = poly[i], yi = poly[i + 1], xj = poly[j], yj = poly[j + 1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

export function analyze(img: AnalysisPixels, face: FaceGeometry | null, seg: SegmentationMask | null): Analysis {
  const out = emptyAnalysis();
  const { data, width: w, height: h } = img;

  // --- scene statistics ---
  const hist = new Uint32Array(256);
  let sceneSum = 0, bgSum = 0, bgCount = 0;
  let gr = 0, gg = 0, gbl = 0, greyCount = 0;
  const fb = face
    ? {
        x0: (face.bbox.x0 - 0.08) * w, x1: (face.bbox.x1 + 0.08) * w,
        y0: (face.bbox.y0 - 0.15) * h, y1: (face.bbox.y1 + 0.1) * h,
      }
    : null;
  for (let y = 0, k = 0; y < h; y++) {
    for (let x = 0; x < w; x++, k += 4) {
      const r = data[k], g = data[k + 1], b = data[k + 2];
      hist[Math.round(lumaOf(r, g, b))]++;
      const lin = lumaOf(SRGB_TO_LIN[r], SRGB_TO_LIN[g], SRGB_TO_LIN[b]);
      sceneSum += lin;
      let isBg: boolean;
      if (seg) {
        const sx = Math.min(seg.width - 1, ((x + 0.5) / w * seg.width) | 0);
        const sy = Math.min(seg.height - 1, ((y + 0.5) / h * seg.height) | 0);
        isBg = seg.data[(sy * seg.width + sx) * 4] < 64;
      } else {
        isBg = !fb || x < fb.x0 || x > fb.x1 || y < fb.y0 || y > fb.y1;
      }
      if (isBg) {
        bgSum += lin;
        bgCount++;
        // Mid-tone, not strongly coloured: usable for a grey-world estimate.
        const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        if (mx < 245 && mn > 20 && mx - mn < 0.45 * mx) {
          gr += SRGB_TO_LIN[r]; gg += SRGB_TO_LIN[g]; gbl += SRGB_TO_LIN[b];
          greyCount++;
        }
      }
    }
  }
  const total = w * h;
  const pct = (p: number) => {
    let acc = 0;
    for (let i = 0; i < 256; i++) {
      acc += hist[i];
      if (acc >= p * total) return i / 255;
    }
    return 1;
  };
  out.p05 = pct(0.05);
  out.p50 = pct(0.5);
  out.p95 = pct(0.95);
  let clip = 0;
  for (let i = 250; i < 256; i++) clip += hist[i];
  out.clipped = clip / total;
  out.sceneLin = sceneSum / total;
  out.backgroundLin = bgCount > total * 0.02 ? bgSum / bgCount : out.sceneLin;
  if (greyCount > total * 0.03 && gg > 0 && gr > 0 && gbl > 0) {
    out.sceneChroma = [Math.log2(gr / gg), Math.log2(gbl / gg)];
    out.sceneChromaWeight = Math.min(1, greyCount / (total * 0.25));
  }
  if (!face) return out;

  // --- skin samples at mesh vertices (3x3 mean) ---
  const sx = w / face.width, sy = h / face.height;
  const verts = SKIN_SAMPLE_VERTS;
  const lum = new Float32Array(verts.length);
  const samples: { l: number; r: number; g: number; b: number; gr: number; gg: number; gb: number }[] = [];
  let clippedSkin = 0;
  for (let k = 0; k < verts.length; k++) {
    const v = verts[k] * 3;
    const cx = Math.round(face.px[v] * sx), cy = Math.round(face.px[v + 1] * sy);
    if (cx < 1 || cy < 1 || cx >= w - 1 || cy >= h - 1) { lum[k] = NaN; continue; }
    let r = 0, g = 0, b = 0, gr = 0, gg = 0, gb = 0;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const i = ((cy + dy) * w + cx + dx) * 4;
        r += SRGB_TO_LIN[data[i]]; g += SRGB_TO_LIN[data[i + 1]]; b += SRGB_TO_LIN[data[i + 2]];
        gr += data[i]; gg += data[i + 1]; gb += data[i + 2];
      }
    }
    r /= 9; g /= 9; b /= 9;
    lum[k] = lumaOf(r, g, b);
    if (Math.max(gr, gg, gb) / 9 > 249) clippedSkin++;
    samples.push({ l: lum[k], r, g, b, gr: gr / (9 * 255), gg: gg / (9 * 255), gb: gb / (9 * 255) });
  }
  if (samples.length < 24) return out;
  out.hasFace = true;

  // Trimmed statistics: drop the darkest and brightest 15% (hair, shadows, specular shine).
  samples.sort((a, b) => a.l - b.l);
  const lo = Math.floor(samples.length * 0.15), hi = Math.ceil(samples.length * 0.85);
  let r = 0, g = 0, b = 0, gy = 0, cb = 0, cr = 0;
  for (let i = lo; i < hi; i++) {
    const s = samples[i];
    r += s.r; g += s.g; b += s.b;
    const y = lumaOf(s.gr, s.gg, s.gb);
    gy += y; cb += s.gb - y; cr += s.gr - y;
  }
  const n = hi - lo;
  let hiSum = 0;
  const hiStart = Math.floor(samples.length * 0.8);
  for (let i = hiStart; i < samples.length; i++) hiSum += samples[i].l;
  out.skinHiLin = hiSum / (samples.length - hiStart);
  out.skinClipped = clippedSkin / samples.length;
  out.skinLin = [r / n, g / n, b / n];
  {
    const a0 = Math.floor(samples.length * 0.5), a1 = Math.ceil(samples.length * 0.9);
    let lr = 0, lg = 0, lb = 0;
    for (let i = a0; i < a1; i++) { lr += samples[i].r; lg += samples[i].g; lb += samples[i].b; }
    out.skinLitLin = [lr / (a1 - a0), lg / (a1 - a0), lb / (a1 - a0)];
  }
  out.skinYcc = [gy / n, cb / n, cr / n];
  let varC = 0;
  for (let i = lo; i < hi; i++) {
    const s = samples[i];
    const y = lumaOf(s.gr, s.gg, s.gb);
    varC += (s.gb - y - out.skinYcc[1]) ** 2 + (s.gr - y - out.skinYcc[2]) ** 2;
  }
  out.skinChromaSigma = Math.sqrt(varC / n);

  // --- lighting ---
  out.sh = fitLighting(face.normals, lum, verts);
  out.light = describeLighting(out.sh);

  // --- eye whites: the brightest, least saturated pixels inside the eye openings ---
  const sclera: number[] = [];
  for (const [lower, upper] of [[EYE_A_LOWER, EYE_A_UPPER], [EYE_B_LOWER, EYE_B_UPPER]]) {
    const ring = [...lower, ...upper.slice(1, -1).reverse()];
    const poly: number[] = [];
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const i of ring) {
      const x = face.px[i * 3] * sx, y = face.px[i * 3 + 1] * sy;
      poly.push(x, y);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
    for (let y = Math.max(0, Math.floor(y0)); y <= Math.min(h - 1, Math.ceil(y1)); y++) {
      for (let x = Math.max(0, Math.floor(x0)); x <= Math.min(w - 1, Math.ceil(x1)); x++) {
        if (!pointInPolygon(x + 0.5, y + 0.5, poly)) continue;
        const i = (y * w + x) * 4;
        const mx = Math.max(data[i], data[i + 1], data[i + 2]), mn = Math.min(data[i], data[i + 1], data[i + 2]);
        if (mx > 0 && (mx - mn) / mx < 0.3) {
          sclera.push(lumaOf(SRGB_TO_LIN[data[i]], SRGB_TO_LIN[data[i + 1]], SRGB_TO_LIN[data[i + 2]]));
        }
      }
    }
  }
  if (sclera.length >= 8) {
    sclera.sort((a, b) => b - a);
    const top = Math.max(3, Math.floor(sclera.length * 0.25));
    let s = 0;
    for (let i = 0; i < top; i++) s += sclera[i];
    out.scleraLin = s / top;
  }
  return out;
}
