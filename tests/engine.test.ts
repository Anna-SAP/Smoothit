import { describe, expect, it } from 'vitest';
import { analyze } from '../src/engine/analysis';
import { frameSignals, scoreFrame } from '../src/engine/bestshot';
import { CropSmoother, composeCrop, maxCrop } from '../src/engine/composition';
import { Director, estimateWhiteBalance, faceExposureStops } from '../src/engine/director';
import { buildWarpField, perspectiveDisplacements } from '../src/engine/geometry';
import {
  SH_TERMS, chooseKeySide, describeLighting, evalSH, fitLighting, relightGains, relightNeed, solveSPD,
} from '../src/engine/lighting';
import { LOOKS } from '../src/engine/looks';
import { MESH_STRIDE, MESH_VERTEX_COUNT, buildFaceRenderData, underEyeWeights } from '../src/engine/meshBuffer';
import { WARP_COLS, WARP_ROWS } from '../src/engine/params';
import { NOSE_TIP, NUM_MESH_VERTS, Region, SKIN_SAMPLE_VERTS, TRIANGLES, TRI_REGION } from '../src/vision/topology';
import { synthFace } from './synthFace';

const deg = (d: number) => (d * Math.PI) / 180;

describe('topology', () => {
  it('covers the mesh and fills eyes and mouth', () => {
    expect(TRIANGLES.length % 3).toBe(0);
    expect(TRI_REGION.length).toBe(TRIANGLES.length / 3);
    expect(Math.max(...TRIANGLES)).toBeLessThan(NUM_MESH_VERTS);
    const count = (r: number) => TRI_REGION.filter((x) => x === r).length;
    expect(count(Region.Skin)).toBeGreaterThan(600);
    expect(count(Region.Eye)).toBe(28);
    expect(count(Region.Mouth)).toBe(18);
    expect(count(Region.Lip)).toBeGreaterThan(40);
    expect(count(Region.Brow)).toBeGreaterThan(8);
    expect(SKIN_SAMPLE_VERTS.length).toBeGreaterThan(150);
  });
});

describe('face geometry', () => {
  it('recovers camera distance from iris size', () => {
    for (const d of [250, 350, 500, 900]) {
      const face = synthFace({ distanceMm: d });
      expect(Math.abs(face.distanceMm - d) / d).toBeLessThan(0.04);
    }
  });

  it('recovers head pose with the engine sign conventions', () => {
    expect(Math.abs(synthFace({ distanceMm: 400 }).yaw)).toBeLessThan(deg(4));
    const right = synthFace({ distanceMm: 400, yaw: deg(20) });
    expect(right.yaw).toBeGreaterThan(deg(12));
    expect(right.yaw).toBeLessThan(deg(28));
    const up = synthFace({ distanceMm: 400, pitch: deg(15) });
    const level = synthFace({ distanceMm: 400 });
    expect(up.pitch - level.pitch).toBeGreaterThan(deg(9));
    expect(up.pitch - level.pitch).toBeLessThan(deg(21));
  });

  it('gives camera-facing unit normals, and mirroring flips their x', () => {
    const face = synthFace({ distanceMm: 400, yaw: deg(15) });
    const mirrored = synthFace({ distanceMm: 400, yaw: deg(15), mirror: true });
    let facing = 0;
    for (let i = 0; i < NUM_MESH_VERTS; i++) {
      const n = face.normals.subarray(i * 3, i * 3 + 3);
      expect(Math.hypot(n[0], n[1], n[2])).toBeCloseTo(1, 4);
      if (n[2] > 0) facing++;
      expect(mirrored.normals[i * 3]).toBeCloseTo(-n[0], 3);
      expect(mirrored.normals[i * 3 + 2]).toBeCloseTo(n[2], 3);
    }
    expect(facing / NUM_MESH_VERTS).toBeGreaterThan(0.9);
    expect(face.normals[NOSE_TIP * 3 + 2]).toBeGreaterThan(0.7);
    expect(mirrored.yaw).toBeCloseTo(-face.yaw, 2);
  });
});

describe('lens-geometry correction', () => {
  const full = { targetDistanceMm: 1600, nearStrength: 1, farStrength: 1, pitchGain: 0, maxPitchFix: 0 };

  it('re-projects a 30 cm selfie to what a 1.6 m portrait lens would draw', () => {
    const near = synthFace({ distanceMm: 300 });
    // Ground truth: the same head from 1.6 m, zoomed so the frontal plane keeps its size.
    const mmPerPx = near.distanceMm / near.focalPx;
    const planeOffset = (near.frontPlaneZ - near.px[33 * 3 + 2]) * mmPerPx; // frontal plane relative to the eye corner
    const far = synthFace({ distanceMm: 1600, focalPx: (near.focalPx * (1600 + planeOffset)) / (300 + planeOffset) });
    const disp = perspectiveDisplacements(near, full);
    // Compare shapes up to a translation (where the face sits in the frame is the framing's job).
    const residual = (corrected: boolean) => {
      const rx: number[] = [], ry: number[] = [];
      for (let i = 0; i < NUM_MESH_VERTS; i++) {
        rx.push(far.px[i * 3] - near.px[i * 3] - (corrected ? disp[i * 2] : 0));
        ry.push(far.px[i * 3 + 1] - near.px[i * 3 + 1] - (corrected ? disp[i * 2 + 1] : 0));
      }
      const mx = rx.reduce((a, b) => a + b) / rx.length, my = ry.reduce((a, b) => a + b) / ry.length;
      return Math.sqrt(rx.reduce((s2, v, i) => s2 + (v - mx) ** 2 + (ry[i] - my) ** 2, 0) / rx.length);
    };
    // The "selfie effect" itself: at 30 cm the nose is drawn markedly wider relative to the face.
    const noseToFace = (f: typeof near) => (f.px[358 * 3] - f.px[129 * 3]) / (f.px[454 * 3] - f.px[234 * 3]);
    expect(noseToFace(near) / noseToFace(far)).toBeGreaterThan(1.12);
    expect(noseToFace(near) / noseToFace(far)).toBeLessThan(1.35);
    expect(residual(false)).toBeGreaterThan(3);                   // the selfie really was distorted
    expect(residual(true) / residual(false)).toBeLessThan(0.25);  // and at least 75% of that is gone
  });

  it('by default shrinks the protruding nose, keeps the outline, and leaves distant faces alone', () => {
    const near = synthFace({ distanceMm: 300 });
    const disp = perspectiveDisplacements(near);
    const ax = (near.eyeMid[0] + near.noseTip[0]) / 2, ay = (near.eyeMid[1] + near.noseTip[1]) / 2;
    const mmPerPx = near.distanceMm / near.focalPx;
    let tip = 0, nTip = 0;
    for (let i = 0; i < NUM_MESH_VERTS; i++) {
      const dz = (near.px[i * 3 + 2] - near.frontPlaneZ) * mmPerPx;
      const rx = near.px[i * 3] - ax, ry = near.px[i * 3 + 1] - ay;
      const r = Math.hypot(rx, ry);
      if (dz < -15 && r > 10) { tip += (disp[i * 2] * rx + disp[i * 2 + 1] * ry) / r / r; nTip++; }
    }
    expect(nTip).toBeGreaterThan(5);
    expect(tip / nTip).toBeLessThan(-0.03); // the nose is drawn at least 3% smaller
    // The face outline is deliberately left almost as shot (a true dolly-zoom would widen it ~15%).
    const widthBefore = near.px[454 * 3] - near.px[234 * 3];
    const widthAfter = widthBefore + disp[454 * 2] - disp[234 * 2];
    expect(widthAfter / widthBefore).toBeGreaterThan(1);
    expect(widthAfter / widthBefore).toBeLessThan(1.03);
    const far = perspectiveDisplacements(synthFace({ distanceMm: 1700 }));
    expect(Math.max(...far.map(Math.abs))).toBe(0);
  });

  it('undoes part of a low camera angle but never touches a high one', () => {
    const opts = { ...full, nearStrength: 0, farStrength: 0, pitchGain: 0.5, maxPitchFix: deg(7) };
    const low = synthFace({ distanceMm: 350, pitch: deg(20) });
    const d = perspectiveDisplacements(low, opts);
    // Forehead grows away from the anchor (up = negative y), chin is pulled in (also negative y).
    expect(d[10 * 2 + 1]).toBeLessThan(-0.3);
    expect(d[152 * 2 + 1]).toBeLessThan(-0.3);
    const high = perspectiveDisplacements(synthFace({ distanceMm: 350, pitch: deg(-20) }), opts);
    expect(Math.max(...high.map(Math.abs))).toBe(0);
  });

  it('builds a smooth warp that vanishes away from the face', () => {
    const face = synthFace({ distanceMm: 300 });
    const crop = { x: 0, y: 0, w: 1, h: 1 };
    const zero = buildWarpField(face, new Float32Array(NUM_MESH_VERTS * 2), crop);
    expect(Math.max(...zero.map(Math.abs))).toBe(0);
    const field = buildWarpField(face, perspectiveDisplacements(face), crop);
    expect(field.length).toBe((WARP_COLS + 1) * (WARP_ROWS + 1) * 2);
    expect(Math.max(...field.map(Math.abs))).toBeGreaterThan(0.001);
    expect(Math.max(...field.map(Math.abs))).toBeLessThan(0.05);
    for (const k of [0, WARP_COLS * 2, field.length - 2]) expect(field[k]).toBe(0); // frame corners
    // No fold-overs: neighbouring grid columns must stay ordered after the warp.
    for (let j = 0; j <= WARP_ROWS; j++) {
      for (let i = 0; i < WARP_COLS; i++) {
        const k = (j * (WARP_COLS + 1) + i) * 2;
        expect(1 / WARP_COLS - field[k + 2] + field[k]).toBeGreaterThan(0);
      }
    }
  });
});

describe('lighting', () => {
  const face = synthFace({ distanceMm: 400 });
  const light = (c: number[]) => {
    const lum = new Float32Array(SKIN_SAMPLE_VERTS.length);
    SKIN_SAMPLE_VERTS.forEach((v, k) => {
      lum[k] = 0.3 * Math.max(0.02, evalSH(c, face.normals[v * 3], face.normals[v * 3 + 1], face.normals[v * 3 + 2]));
    });
    return lum;
  };

  it('solves SPD systems', () => {
    const x = solveSPD(new Float64Array([4, 1, 1, 3]), new Float64Array([1, 2]), 2);
    expect(x[0]).toBeCloseTo(1 / 11, 6);
    expect(x[1]).toBeCloseTo(7 / 11, 6);
  });

  it('measures the direction and harshness of the light on a face', () => {
    const fromLeft = fitLighting(face.normals, light([1, -1.1, 0.1, 0.2, 0, 0, 0, 0, 0]));
    const d = describeLighting(fromLeft);
    expect(d.side).toBeLessThan(-0.5);
    expect(d.contrast).toBeGreaterThan(3);
    expect(relightNeed(d)).toBeGreaterThan(0.3);
    expect(chooseKeySide(face, d)).toBe(-1);

    const flat = describeLighting(fitLighting(face.normals, light([1, 0, 0, 0.1, 0, 0, 0, 0, 0])));
    expect(Math.abs(flat.side)).toBeLessThan(0.1);
    expect(flat.contrast).toBeLessThan(1.3);
    expect(relightNeed(flat)).toBe(0);

    const below = describeLighting(fitLighting(face.normals, light([1, 0, -0.7, 0.2, 0, 0, 0, 0, 0])));
    expect(below.elevation).toBeLessThan(-0.4);
    expect(relightNeed(below)).toBeGreaterThan(0.8);
  });

  it('survives outliers such as cast shadows and specular highlights', () => {
    const truth = [1, 0.5, 0.2, 0.2, 0, 0, 0, 0, 0];
    const lum = light(truth);
    for (let k = 0; k < lum.length; k += 9) lum[k] *= k % 2 ? 3 : 0.15;
    const fit = fitLighting(face.normals, lum);
    expect(describeLighting(fit).side).toBeGreaterThan(0.25);
    expect(fit[1] / fit[0]).toBeGreaterThan(0.3);
    expect(fit[1] / fit[0]).toBeLessThan(0.7);
  });

  it('relights harsh side light toward the studio ratio without changing mean exposure', () => {
    const coeffs = fitLighting(face.normals, light([1, -0.85, 0, 0.2, 0, 0, 0, 0, 0]));
    const look = LOOKS[0];
    const gains = relightGains(face, coeffs, look.light, -1);
    let shadowSide = 0, litSide = 0, ns = 0, nl = 0, mean = 0, wsum = 0;
    for (let i = 0; i < NUM_MESH_VERTS; i++) {
      const nx = face.normals[i * 3], nz = face.normals[i * 3 + 2];
      if (nz < 0.5) continue;
      if (nx > 0.3) { shadowSide += gains[i]; ns++; }
      if (nx < -0.3) { litSide += gains[i]; nl++; }
      mean += nz * gains[i];
      wsum += nz;
    }
    expect(shadowSide / ns).toBeGreaterThan(0.35); // shadows are opened by a good part of a stop
    expect(litSide / nl).toBeLessThan(0.05);       // the lit side is not brightened
    expect(Math.min(...gains)).toBeGreaterThanOrEqual(-0.4 * 0.9 - 1e-6);
    expect(Math.max(...gains)).toBeLessThanOrEqual(0.9 + 1e-6);
    expect(Math.abs(mean / wsum)).toBeLessThan(0.35);
  });

  it('leaves light that already matches the target almost untouched', () => {
    const look = LOOKS[0].light;
    const ce = Math.cos(look.elevation);
    const l = [-Math.sin(look.azimuth) * ce, Math.sin(look.elevation), Math.cos(look.azimuth) * ce];
    const fill = 1 / look.ratio;
    const lum = new Float32Array(SKIN_SAMPLE_VERTS.length);
    SKIN_SAMPLE_VERTS.forEach((v, k) => {
      const n = face.normals.subarray(v * 3, v * 3 + 3);
      lum[k] = fill + (1 - fill) * Math.max(0, (n[0] * l[0] + n[1] * l[1] + n[2] * l[2] + look.wrap) / (1 + look.wrap));
    });
    const gains = relightGains(face, fitLighting(face.normals, lum), look, -1);
    let sum = 0, n = 0;
    for (let i = 0; i < NUM_MESH_VERTS; i++) if (face.normals[i * 3 + 2] > 0.5) { sum += Math.abs(gains[i]); n++; }
    expect(sum / n).toBeLessThan(0.12);
  });
});

describe('composition', () => {
  it('keeps the exact aspect ratio and stays inside the frame', () => {
    for (const [w, h] of [[1080, 1440], [1080, 1920], [1920, 1080], [1440, 1440]]) {
      for (const d of [250, 400, 800, 1500]) {
        const face = synthFace({ distanceMm: d, width: w, height: h, yaw: deg(10) });
        const c = composeCrop(face, w, h);
        expect((c.w * w) / (c.h * h)).toBeCloseTo(3 / 4, 5);
        expect(c.x).toBeGreaterThanOrEqual(0);
        expect(c.y).toBeGreaterThanOrEqual(0);
        expect(c.x + c.w).toBeLessThanOrEqual(1 + 1e-9);
        expect(c.y + c.h).toBeLessThanOrEqual(1 + 1e-9);
        const m = maxCrop(w, h, 3 / 4);
        expect(m.w / c.w).toBeGreaterThanOrEqual(1 - 1e-9);
        expect(m.w / c.w).toBeLessThanOrEqual(1.6 + 1e-9);
      }
    }
  });

  it('places the eyes on the upper third when there is room to reframe', () => {
    const face = synthFace({ distanceMm: 700, width: 1080, height: 1920 });
    const c = composeCrop(face, 1080, 1920);
    const eyeY = (face.eyeMid[1] / 1920 - c.y) / c.h;
    expect(eyeY).toBeGreaterThan(0.33);
    expect(eyeY).toBeLessThan(0.45);
    expect((face.eyeMid[0] / 1080 - c.x) / c.w).toBeCloseTo(0.5, 1);
  });

  it('zooms in on a small face and gives lead room to a turned one', () => {
    const farFace = synthFace({ distanceMm: 1500 });
    expect(composeCrop(farFace, 1080, 1440).w).toBeLessThan(0.7);
    const turned = synthFace({ distanceMm: 900, yaw: deg(25), width: 1920, height: 1080 });
    const c = composeCrop(turned, 1920, 1080);
    expect((turned.eyeMid[0] / 1920 - c.x) / c.w).toBeLessThan(0.49); // facing right -> sits left of centre
  });

  it('falls back to a centred crop without a face', () => {
    const c = composeCrop(null, 1920, 1080);
    expect(c.x + c.w / 2).toBeCloseTo(0.5, 6);
    expect(c.y + c.h / 2).toBeCloseTo(0.5, 6);
  });

  it('smooths framing over time and ignores jitter', () => {
    const s = new CropSmoother();
    const a = { x: 0.1, y: 0.1, w: 0.6, h: 0.8 };
    expect(s.update(a, 0.016)).toEqual(a);
    const jitter = s.update({ ...a, x: 0.102 }, 0.016);
    expect(jitter.x).toBeCloseTo(0.1, 9);
    const b = { x: 0.3, y: 0.15, w: 0.6, h: 0.8 };
    const first = s.update(b, 0.016);
    expect(first.x).toBeGreaterThan(0.1);
    expect(first.x).toBeLessThan(0.15);
    let cur = first;
    for (let i = 0; i < 200; i++) cur = s.update(b, 0.016);
    expect(Math.abs(cur.x - b.x)).toBeLessThan(0.006);
  });
});

describe('metering', () => {
  it('leaves neutral skin alone and removes a green tint', () => {
    const neutral = estimateWhiteBalance([0.54, 0.305, 0.223]);
    for (const g of neutral) expect(g).toBeCloseTo(1, 1);
    const green = estimateWhiteBalance([0.54 * 0.85, 0.305 * 1.15, 0.223 * 0.85]);
    expect(green[1] / green[0]).toBeLessThan(0.93);
    expect(green[1] / green[2]).toBeLessThan(0.93);
  });

  it('does not mistake deep or fair skin tones for a colour cast', () => {
    for (const skin of [[0.171, 0.084, 0.058], [0.871, 0.644, 0.546], [0.4, 0.2, 0.12]] as [number, number, number][]) {
      const wb = estimateWhiteBalance(skin);
      expect(Math.abs(Math.log2(wb[0] / wb[1]))).toBeLessThan(0.12);
      expect(Math.abs(Math.log2(wb[2] / wb[1]))).toBeLessThan(0.15);
    }
  });

  it('preserves skin luminance', () => {
    const skin: [number, number, number] = [0.3, 0.3, 0.45];
    const wb = estimateWhiteBalance(skin);
    const y = (c: number[]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    expect(y(skin.map((v, i) => v * wb[i]))).toBeCloseTo(y(skin), 5);
  });

  it('exposes for the face, respecting skin tone and highlights', () => {
    expect(faceExposureStops(0.08, 0.1)).toBeGreaterThan(1);           // light skin, under-exposed
    expect(faceExposureStops(0.33, 0.35)).toBeCloseTo(0, 0);           // already right
    // Same skin luminance, but bright eye whites reveal genuinely dark skin: push far less.
    expect(faceExposureStops(0.12, 0.45)).toBeLessThan(faceExposureStops(0.12, 0.14) - 0.3);
    // The lit side must not be driven into clipping.
    expect(faceExposureStops(0.15, 0.2, 0.5)).toBeLessThan(0.3);
    // Clipped skin cannot be recovered by pulling exposure down hard.
    expect(faceExposureStops(0.8, 0.8, 0.95, 0.4)).toBeGreaterThanOrEqual(-0.3);
  });
});

describe('scene analysis', () => {
  const W = 192, H = 256;
  const paint = (face: ReturnType<typeof synthFace>, skin: [number, number, number], bg: [number, number, number]) => {
    const data = new Uint8Array(W * H * 4);
    const b = face.bbox;
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const inside = x / W > b.x0 && x / W < b.x1 && y / H > b.y0 && y / H < b.y1;
        data.set([...(inside ? skin : bg), 255], (y * W + x) * 4);
      }
    }
    return { data, width: W, height: H };
  };

  it('reads skin tone and scene brightness', () => {
    const face = synthFace({ distanceMm: 350 });
    const a = analyze(paint(face, [200, 150, 130], [40, 40, 40]), face, null);
    expect(a.hasFace).toBe(true);
    expect(a.skinLin[0]).toBeCloseTo(0.578, 2);
    expect(a.skinLin[1]).toBeCloseTo(0.305, 2);
    expect(a.skinYcc[0]).toBeCloseTo(0.629, 2);
    expect(a.skinChromaSigma).toBeLessThan(0.005);
    expect(a.light.contrast).toBeLessThan(1.1);
    expect(a.backgroundLin).toBeLessThan(0.03);
    expect(a.skinClipped).toBe(0);
  });

  it('reports no face when there is none', () => {
    const face = synthFace({ distanceMm: 350 });
    expect(analyze(paint(face, [10, 10, 10], [10, 10, 10]), null, null).hasFace).toBe(false);
  });
});

describe('best shot', () => {
  const face = synthFace({ distanceMm: 400 });
  it('prefers open eyes and still frames', () => {
    const open = scoreFrame(frameSignals(face, face, { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 }));
    const blink = scoreFrame(frameSignals(face, face, { eyeBlinkLeft: 0.9, eyeBlinkRight: 0.8 }));
    const moved = synthFace({ distanceMm: 400, yaw: deg(6) });
    const blurry = scoreFrame(frameSignals(moved, face, { eyeBlinkLeft: 0.05, eyeBlinkRight: 0.05 }));
    const smile = scoreFrame(frameSignals(face, face, { mouthSmileLeft: 0.7, mouthSmileRight: 0.7 }));
    expect(open).toBeGreaterThan(blink + 0.5);
    expect(open).toBeGreaterThan(blurry + 0.2);
    expect(smile).toBeGreaterThan(open);
    expect(scoreFrame(frameSignals(null, null, null))).toBeLessThan(blink);
  });
});

describe('render data', () => {
  const face = synthFace({ distanceMm: 400 });
  it('packs the mesh and a region of interest that contains the face', () => {
    const data = buildFaceRenderData(face, null, underEyeWeights(face));
    expect(data.vertexCount).toBe(MESH_VERTEX_COUNT);
    expect(data.vertices.length).toBe(MESH_VERTEX_COUNT * MESH_STRIDE);
    const [x, y, w, h] = data.roi;
    expect(x).toBeLessThan(face.bbox.x0);
    expect(y).toBeLessThan(face.bbox.y0);
    expect(x + w).toBeGreaterThan(face.bbox.x1);
    expect(y + h).toBeGreaterThan(face.bbox.y1);
    expect(data.iodUv).toBeCloseTo(face.iod / face.width, 6);
  });

  it('puts the under-eye weight below the eyes, not on them', () => {
    const w = underEyeWeights(face);
    let cy = 0, sum = 0;
    for (let i = 0; i < NUM_MESH_VERTS; i++) { cy += w[i] * face.px[i * 3 + 1]; sum += w[i]; }
    cy /= sum;
    expect(cy).toBeGreaterThan(face.eyeMid[1] + 0.15 * face.iod);
    expect(cy).toBeLessThan(face.eyeMid[1] + 0.5 * face.iod);
  });
});

describe('director', () => {
  it('produces complete, finite render parameters for every look', () => {
    const face = synthFace({ distanceMm: 320, yaw: deg(8) });
    for (const look of LOOKS) {
      const director = new Director();
      director.look = look;
      const d = director.update({ face, analysis: null, srcW: face.width, srcH: face.height, dt: Infinity });
      const flat = JSON.stringify(d.params, (_k, v) => (ArrayBuffer.isView(v) ? Array.from(v as Float32Array) : v));
      expect(flat).not.toMatch(/null,|NaN/);
      expect(d.params.face).not.toBeNull();
      expect(d.params.warp).not.toBeNull();
      expect(d.params.grade.vigCenter[0]).toBeGreaterThan(0.2);
      expect(d.hints).not.toContain('noFace');
    }
  });

  it('hints only at what software cannot fix', () => {
    const director = new Director();
    const close = synthFace({ distanceMm: 200 });
    expect(director.update({ face: close, analysis: null, srcW: 1080, srcH: 1440, dt: Infinity }).hints).toContain('tooClose');
    expect(director.update({ face: null, analysis: null, srcW: 1080, srcH: 1440, dt: 0.03 }).hints).toContain('noFace');
    const low = synthFace({ distanceMm: 400, pitch: deg(30) });
    expect(new Director().update({ face: low, analysis: null, srcW: 1080, srcH: 1440, dt: Infinity }).hints).toContain('lowAngle');
  });

  it('exposes an SH vector of the right size', () => {
    expect(describeLighting(new Float64Array(SH_TERMS).fill(0).map((_, i) => (i === 0 ? 1 : 0))).contrast).toBeCloseTo(1, 6);
  });
});
