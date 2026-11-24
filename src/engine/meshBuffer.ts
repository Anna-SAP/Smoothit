// Packs the face mesh into the vertex layout the G-buffer pass expects.
import { NUM_MESH_VERTS, Region, TRIANGLES, TRI_REGION } from '../vision/topology';
import type { FaceGeometry } from './faceModel';
import type { FaceRenderData } from './params';

/** Floats per vertex: pos(3) + [gain stops, dodge](2) + region one-hot [eye, lip, mouth, skin](4). */
export const MESH_STRIDE = 9;
export const MESH_VERTEX_COUNT = TRIANGLES.length;

/**
 * Under-eye dodge weight per vertex, derived geometrically from the eye positions so it adapts to
 * any face shape: a soft blob centred roughly 0.3 inter-ocular distances below each eye.
 */
export function underEyeWeights(face: FaceGeometry): Float32Array {
  const w = new Float32Array(NUM_MESH_VERTS);
  const ax = face.eyeB[0] - face.eyeA[0], ay = face.eyeB[1] - face.eyeA[1];
  const len = Math.hypot(ax, ay) || 1;
  const ux = ax / len, uy = ay / len;
  // "Down" is perpendicular to the eye line, on the chin's side.
  let dx = -uy, dy = ux;
  if (dx * (face.chin[0] - face.eyeMid[0]) + dy * (face.chin[1] - face.eyeMid[1]) < 0) { dx = -dx; dy = -dy; }
  const eyeW = 0.42 * face.iod;
  for (let i = 0; i < NUM_MESH_VERTS; i++) {
    const x = face.px[i * 3], y = face.px[i * 3 + 1];
    let best = 0;
    for (const e of [face.eyeA, face.eyeB]) {
      const u = ((x - e[0]) * ux + (y - e[1]) * uy) / eyeW;
      const v = ((x - e[0]) * dx + (y - e[1]) * dy) / eyeW;
      const g = Math.exp(-((u / 0.75) ** 2)) * Math.exp(-(((v - 0.72) / 0.24) ** 2));
      if (g > best) best = g;
    }
    w[i] = best;
  }
  return w;
}

/**
 * @param gainStops per-vertex relight gain in stops (468), or null for none
 * @param dodge per-vertex under-eye weight (468), or null
 * @param out reusable buffer of MESH_VERTEX_COUNT * MESH_STRIDE floats
 */
export function buildFaceRenderData(
  face: FaceGeometry, gainStops: Float32Array | null, dodge: Float32Array | null, out?: Float32Array,
): FaceRenderData {
  const v = out ?? new Float32Array(MESH_VERTEX_COUNT * MESH_STRIDE);
  const { px, width, height } = face;
  let k = 0;
  for (let t = 0; t < TRIANGLES.length; t++) {
    const i = TRIANGLES[t];
    const r = TRI_REGION[(t / 3) | 0];
    v[k++] = px[i * 3] / width;
    v[k++] = px[i * 3 + 1] / height;
    v[k++] = px[i * 3 + 2] / width;
    v[k++] = gainStops ? gainStops[i] : 0;
    v[k++] = dodge && r === Region.Skin ? dodge[i] : 0;
    v[k++] = r === Region.Eye ? 1 : 0;
    v[k++] = r === Region.Lip ? 1 : 0;
    v[k++] = r === Region.Mouth ? 1 : 0;
    v[k++] = r === Region.Skin ? 1 : 0;
  }
  // Square-ish region of interest around the face with generous margins (forehead, jaw, feathering).
  const b = face.bbox;
  const cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
  const hw = ((b.x1 - b.x0) / 2) * 1.35, hh = ((b.y1 - b.y0) / 2) * 1.35;
  return {
    roi: [cx - hw, cy - hh, hw * 2, hh * 2],
    vertices: v,
    vertexCount: MESH_VERTEX_COUNT,
    iodUv: face.iod / width,
  };
}
