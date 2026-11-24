// Lighting estimation and relighting.
//
// Skin is close to a diffuse reflector, and the appearance of a convex diffuse object under any
// distant lighting is captured to ~99% by nine spherical-harmonic coefficients (Ramamoorthi &
// Hanrahan 2001; Basri & Jacobs 2003). We have a normal for every mesh vertex and can measure the
// skin's brightness there, so we can solve for those coefficients, i.e. *measure the light the
// selfie was actually taken in*. Dividing a studio lighting pattern by the measured one gives a
// per-vertex exposure correction (a "ratio image") that re-lights the face.
import { NUM_MESH_VERTS, SKIN_SAMPLE_VERTS } from '../vision/topology';
import type { FaceGeometry } from './faceModel';

export const SH_TERMS = 9;

export function shBasis(nx: number, ny: number, nz: number, out: Float64Array): void {
  out[0] = 1;
  out[1] = nx;
  out[2] = ny;
  out[3] = nz;
  out[4] = nx * ny;
  out[5] = nx * nz;
  out[6] = ny * nz;
  out[7] = nx * nx - ny * ny;
  out[8] = 3 * nz * nz - 1;
}

export function evalSH(c: ArrayLike<number>, nx: number, ny: number, nz: number): number {
  return (
    c[0] + c[1] * nx + c[2] * ny + c[3] * nz + c[4] * nx * ny + c[5] * nx * nz + c[6] * ny * nz +
    c[7] * (nx * nx - ny * ny) + c[8] * (3 * nz * nz - 1)
  );
}

/** Solve the symmetric positive-definite system A x = b in place (Cholesky). */
export function solveSPD(A: Float64Array, b: Float64Array, n: number): Float64Array {
  for (let j = 0; j < n; j++) {
    let s = A[j * n + j];
    for (let k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
    const d = Math.sqrt(Math.max(s, 1e-12));
    A[j * n + j] = d;
    for (let i = j + 1; i < n; i++) {
      let t = A[i * n + j];
      for (let k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = t / d;
    }
  }
  const x = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= A[i * n + k] * x[k];
    x[i] = s / A[i * n + i];
  }
  for (let i = n - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < n; k++) s -= A[k * n + i] * x[k];
    x[i] = s / A[i * n + i];
  }
  return x;
}

/**
 * Fit normalised SH lighting (mean brightness ~1) to skin luminance samples.
 * Ridge-regularised (the visible normals only span a hemisphere) and robust to cast shadows,
 * specular highlights and stray hair via iteratively re-weighted least squares.
 * @param lum linear-light luminance for each index in `verts` (NaN = no sample)
 */
export function fitLighting(normals: Float32Array, lum: Float32Array, verts: Uint16Array = SKIN_SAMPLE_VERTS): Float64Array {
  const n = SH_TERMS;
  // Only surfaces that face the camera take part: toward the silhouette the mesh normals are
  // unreliable and the samples pick up hair, ears and background.
  const facing = (k: number) => Math.max(0, normals[verts[k] * 3 + 2] - 0.3);
  let mean = 0, count = 0;
  for (let k = 0; k < verts.length; k++) if (lum[k] === lum[k] && facing(k) > 0) { mean += lum[k]; count++; }
  const flat = new Float64Array(n);
  flat[0] = 1;
  if (count < 24 || mean <= 0) return flat;
  mean /= count;

  const basis = new Float64Array(n);
  const weights = new Float64Array(verts.length).fill(1);
  let coeffs: Float64Array = flat;
  const ridge = [0, 0.004, 0.004, 0.02, 0.06, 0.06, 0.06, 0.06, 0.1];
  for (let iter = 0; iter < 3; iter++) {
    const A = new Float64Array(n * n), b = new Float64Array(n);
    for (let k = 0; k < verts.length; k++) {
      const y = lum[k] / mean;
      const fw = facing(k);
      if (y !== y || fw <= 0) continue;
      const v = verts[k] * 3;
      shBasis(normals[v], normals[v + 1], normals[v + 2], basis);
      const w = weights[k] * fw;
      for (let i = 0; i < n; i++) {
        b[i] += w * basis[i] * y;
        for (let j = 0; j <= i; j++) A[i * n + j] += w * basis[i] * basis[j];
      }
    }
    for (let i = 0; i < n; i++) {
      A[i * n + i] += ridge[i] * count;
      for (let j = 0; j < i; j++) A[j * n + i] = A[i * n + j];
    }
    coeffs = solveSPD(A, b, n);
    // Huber weights on the residuals.
    for (let k = 0; k < verts.length; k++) {
      const y = lum[k] / mean;
      if (y !== y) continue;
      const v = verts[k] * 3;
      const r = Math.abs(y - evalSH(coeffs, normals[v], normals[v + 1], normals[v + 2]));
      weights[k] = r < 0.15 ? 1 : 0.15 / r;
    }
  }
  return coeffs;
}

export interface LightDescription {
  /** Unit direction the light mainly comes from, in normal space (x right, y up, z to camera). */
  dir: [number, number, number];
  /** Brightness ratio between the most and least lit of four probe directions on the face (>= 1). */
  contrast: number;
  /** -1..1: light from image-left .. image-right. */
  side: number;
  /** -1..1: light from below .. above. */
  elevation: number;
}

export function describeLighting(c: ArrayLike<number>): LightDescription {
  const len = Math.hypot(c[1], c[2], c[3]) || 1;
  const dir: [number, number, number] = [c[1] / len, c[2] / len, c[3] / len];
  // Compare surfaces turned 40 degrees left/right/up/down of the camera axis: inside the range of
  // normals a face actually shows, so the fit is being interpolated, not extrapolated.
  const s = Math.sin(0.7), z = Math.cos(0.7);
  const at = (x: number, y: number) => Math.max(0.03, evalSH(c, x * s, y * s, z));
  const hx = at(1, 0), lx = at(-1, 0), hy = at(0, 1), ly = at(0, -1);
  return {
    dir,
    contrast: Math.max(hx, lx, hy, ly) / Math.min(hx, lx, hy, ly),
    side: Math.max(-1, Math.min(1, Math.log2(hx / lx) / 1.5)),
    elevation: Math.max(-1, Math.min(1, Math.log2(hy / ly) / 1.5)),
  };
}

export interface LightTarget {
  /** Key-light azimuth off the camera axis (radians, always positive; side is chosen automatically). */
  azimuth: number;
  /** Key-light elevation above the camera axis (radians). */
  elevation: number;
  /** Key-to-fill ratio, e.g. 2 means the lit side is one stop brighter than the shadow side. */
  ratio: number;
  /** Wrap factor: 0 = hard Lambert terminator, 1 = very large, soft source. */
  wrap: number;
}

/**
 * How much the measured light needs correcting, 0..1. Good light is left alone: a photographer does
 * not re-light a face that already sits in soft window light.
 */
export function relightNeed(light: LightDescription): number {
  const ss = (a: number, b: number, x: number) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  const harsh = ss(1.2, 2.6, Math.log2(light.contrast)); // beyond ~2.3:1, fully "harsh" at 6:1
  const fromBelow = ss(0.1, 0.5, -light.elevation);       // phone-screen / monitor underlighting
  const steepTop = ss(0.55, 1.0, light.elevation);         // overhead lamp: dark eye sockets
  return Math.max(harsh, fromBelow, steepTop);
}

/** Choose which side the virtual key light goes on. */
export function chooseKeySide(face: FaceGeometry, light: LightDescription): 1 | -1 {
  // Follow the real light when it has a clear direction: enhancing it looks natural, fighting it
  // cannot remove the shadows it already cast.
  if (Math.abs(light.side) > 0.12) return light.side > 0 ? 1 : -1;
  // Otherwise use "short lighting": light the cheek turned away from the camera, which slims.
  if (Math.abs(face.yaw) > 0.08) return face.yaw > 0 ? 1 : -1;
  return -1;
}

/**
 * Per-vertex exposure correction (stops) that moves the measured lighting toward the target.
 * @param maxStops clamp on the correction magnitude
 */
export function relightGains(
  face: FaceGeometry, coeffs: ArrayLike<number>, target: LightTarget, side: 1 | -1, maxStops = 0.9,
): Float32Array {
  const out = new Float32Array(NUM_MESH_VERTS);
  const n = face.normals;
  const ce = Math.cos(target.elevation);
  const lx = side * Math.sin(target.azimuth) * ce, ly = Math.sin(target.elevation), lz = Math.cos(target.azimuth) * ce;
  const fill = 1 / Math.max(1.01, target.ratio);
  const T = new Float32Array(NUM_MESH_VERTS), E = new Float32Array(NUM_MESH_VERTS);
  let mt = 0, me = 0, wsum = 0;
  for (let i = 0; i < NUM_MESH_VERTS; i++) {
    const nx = n[i * 3], ny = n[i * 3 + 1], nz = n[i * 3 + 2];
    const lambert = Math.max(0, (nx * lx + ny * ly + nz * lz + target.wrap) / (1 + target.wrap));
    T[i] = fill + (1 - fill) * lambert;
    E[i] = evalSH(coeffs, nx, ny, nz);
    const w = Math.max(0, nz); // weight by how much of the surface the camera sees
    mt += w * T[i];
    me += w * Math.max(E[i], 0.05);
    wsum += w;
  }
  mt /= wsum || 1;
  me /= wsum || 1;
  for (let i = 0; i < NUM_MESH_VERTS; i++) {
    // Normalise both patterns to the same mean, so relighting never changes overall exposure.
    const e = Math.max(E[i] / me, 0.3);
    const facing = Math.min(1, Math.max(0, n[i * 3 + 2] / 0.35)); // fade out at grazing angles
    const stops = Math.log2(T[i] / mt / e) * facing;
    // Adding light (fill) is safe; taking it away quickly looks muddy, so darkening is limited.
    out[i] = Math.max(-0.4 * maxStops, Math.min(maxStops, stops));
  }
  return out;
}
