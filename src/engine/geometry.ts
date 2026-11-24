// Lens-geometry correction.
//
// A selfie is shot from 25-50 cm with a ~24 mm-equivalent lens; a studio portrait from 1.5-3 m with
// 85-135 mm. Magnification falls off with distance, so at arm's length the nose (closest) is drawn
// noticeably larger than the ears (farthest) - the "selfie effect". Because the mesh gives us metric
// depth per vertex, we can re-project the face as if the camera had been dollied back to a portrait
// distance and zoomed in to keep the face the same size, then warp the image to match.
import { NUM_MESH_VERTS } from '../vision/topology';
import type { FaceGeometry } from './faceModel';
import { type Crop, WARP_COLS, WARP_ROWS } from './params';

export interface PerspectiveOptions {
  /** Virtual camera distance in mm (a classic portrait working distance). */
  targetDistanceMm: number;
  /** 0..1 strength for points in front of the face's frontal plane (nose, lips). */
  nearStrength: number;
  /** 0..1 strength for points behind it (temples, jaw, ears). */
  farStrength: number;
  /** Fraction of a low camera angle to undo, and its cap in radians. */
  pitchGain: number;
  maxPitchFix: number;
}

export const DEFAULT_PERSPECTIVE: PerspectiveOptions = {
  targetDistanceMm: 1600,
  nearStrength: 0.75,
  // A true dolly-zoom also widens the face outline. That is optically correct but nobody's idea of
  // flattering, so the far side gets only a token amount.
  farStrength: 0.12,
  pitchGain: 0.45,
  maxPitchFix: (7 * Math.PI) / 180,
};

/**
 * Forward displacement (pixels) of every mesh vertex that takes the captured projection to the
 * projection a camera at `targetDistanceMm` would have produced.
 */
export function perspectiveDisplacements(face: FaceGeometry, opt: PerspectiveOptions = DEFAULT_PERSPECTIVE): Float32Array {
  const disp = new Float32Array(NUM_MESH_VERTS * 2);
  const d = face.distanceMm;
  if (!(d > 120) || d >= opt.targetDistanceMm) return disp; // unknown, or already far enough
  const D = opt.targetDistanceMm;
  const f = face.focalPx;
  const mmPerPx = d / f;
  // Anchor: the middle of the face (between the eye line and the nose tip), on its frontal plane.
  const cx = (face.eyeMid[0] + face.noseTip[0]) / 2, cy = (face.eyeMid[1] + face.noseTip[1]) / 2;
  // Undo part of a low camera angle (pitch > 0) by rotating the head model about the anchor.
  const delta = Math.min(opt.maxPitchFix, Math.max(0, face.pitch - 0.05) * opt.pitchGain);
  const cos = Math.cos(delta), sin = Math.sin(delta);
  const fNew = (f * D) / d;
  for (let i = 0; i < NUM_MESH_VERTS; i++) {
    const x = face.px[i * 3], y = face.px[i * 3 + 1];
    const dz = (face.px[i * 3 + 2] - face.frontPlaneZ) * mmPerPx; // mm, negative = in front of the face plane
    // Back-project to metric coordinates relative to the anchor.
    const vx = ((x - cx) * (d + dz)) / f;
    const vy = ((y - cy) * (d + dz)) / f;
    // Pure dolly-zoom.
    const px = cx + (fNew * vx) / (D + dz);
    const py = cy + (fNew * vy) / (D + dz);
    const s = dz < 0 ? opt.nearStrength : opt.farStrength;
    let ox = (px - x) * s, oy = (py - y) * s;
    if (delta > 0) {
      // Image y is down, z is away: a positive delta brings the forehead toward the camera.
      const ry = vy * cos - dz * sin;
      const rz = vy * sin + dz * cos;
      ox += (fNew * vx) / (D + rz) - (fNew * vx) / (D + dz);
      oy += (fNew * ry) / (D + rz) - (fNew * vy) / (D + dz);
    }
    disp[i * 2] = ox;
    disp[i * 2 + 1] = oy;
  }
  return disp;
}

/**
 * Turn sparse forward displacements at the mesh vertices into a smooth backward-warp field sampled
 * on the output warp grid (Gaussian kernel regression; decays to zero away from the face so hair
 * and background are left alone).
 * @returns per grid vertex: displacement in source uv to SUBTRACT from the sampling position.
 */
export function buildWarpField(face: FaceGeometry, disp: Float32Array, crop: Crop, out?: Float32Array): Float32Array {
  const n = (WARP_COLS + 1) * (WARP_ROWS + 1) * 2;
  const field = out ?? new Float32Array(n);
  field.fill(0);
  const { width: W, height: H, px } = face;
  const sigma = 0.11 * face.faceWidth;
  const inv2s2 = 1 / (2 * sigma * sigma);
  const reach = 3.2 * sigma;
  const bx0 = face.bbox.x0 * W - reach, bx1 = face.bbox.x1 * W + reach;
  const by0 = face.bbox.y0 * H - reach, by1 = face.bbox.y1 * H + reach;
  const cut = reach * reach;
  const EPS = 3; // background prior: field fades once fewer than ~3 vertices are within reach

  const sample = (qx: number, qy: number, o: [number, number]) => {
    let sx = 0, sy = 0, sw = 0;
    for (let i = 0; i < NUM_MESH_VERTS; i++) {
      const dx = qx - px[i * 3], dy = qy - px[i * 3 + 1];
      const r2 = dx * dx + dy * dy;
      if (r2 > cut) continue;
      const w = Math.exp(-r2 * inv2s2);
      sx += w * disp[i * 2];
      sy += w * disp[i * 2 + 1];
      sw += w;
    }
    o[0] = sx / (sw + EPS);
    o[1] = sy / (sw + EPS);
  };

  const d0: [number, number] = [0, 0], d1: [number, number] = [0, 0];
  for (let j = 0, k = 0; j <= WARP_ROWS; j++) {
    const qy = (crop.y + (j / WARP_ROWS) * crop.h) * H;
    for (let i = 0; i <= WARP_COLS; i++, k += 2) {
      const qx = (crop.x + (i / WARP_COLS) * crop.w) * W;
      if (qx < bx0 || qx > bx1 || qy < by0 || qy > by1) continue;
      // Invert the forward map with one fixed-point refinement: src = q - d(q - d(q)).
      sample(qx, qy, d0);
      sample(qx - d0[0], qy - d0[1], d1);
      field[k] = d1[0] / W;
      field[k + 1] = d1[1] / H;
    }
  }
  return field;
}
