// Per-frame face geometry reconstructed from the 478 MediaPipe landmarks.
//
// Coordinate conventions used throughout the engine:
//   image space  : x right, y DOWN, origin top-left, in source pixels (or uv = pixels / size)
//   normal space : x right, y UP, z toward the camera
// MediaPipe's landmark z is "relative depth, smaller = closer", on the same scale as x.
import {
  CHEEK_A, CHEEK_B, CHIN, EYE_A_LOWER, EYE_B_LOWER, FACE_OVAL, FOREHEAD_TOP, IRIS_A, IRIS_B, NOSE_TIP, NUM_LANDMARKS,
  NUM_MESH_VERTS, SKIN_SAMPLE_VERTS, SURFACE_TRIANGLES,
} from '../vision/topology';

/** Mean human iris (horizontal visible iris diameter) in mm — nearly constant across adults. */
export const IRIS_DIAMETER_MM = 11.7;
/** Focal length in pixels as a fraction of the frame's long side for a ~24 mm-equivalent front camera. */
export const FRONT_CAMERA_FOCAL_PER_LONG_SIDE = 0.693;

export interface FaceGeometry {
  /** Frame size the pixel coordinates refer to. */
  width: number;
  height: number;
  /** Landmarks in pixels: x, y in image space, z = relative depth in pixels (smaller = closer). */
  px: Float32Array;
  /** Unit normals of the 468 mesh vertices in normal space. */
  normals: Float32Array;
  /** Centres of the two eye openings (pixels). A is image-left. */
  eyeA: [number, number];
  eyeB: [number, number];
  /** Midpoint between the eyes (pixels). */
  eyeMid: [number, number];
  /**
   * Depth (pixels) of the face's frontal plane: the median over camera-facing skin (forehead,
   * cheeks, chin). The nose protrudes in front of it; temples, jaw and ears recede behind it.
   */
  frontPlaneZ: number;
  /** Nose tip (pixels). */
  noseTip: [number, number];
  /** Inter-ocular distance in pixels. */
  iod: number;
  /** Mean iris diameter in pixels (0 if unavailable). */
  irisPx: number;
  /** Bounding box of the face mesh in uv. */
  bbox: { x0: number; y0: number; x1: number; y1: number };
  /** Face centre (pixels), chin and top-of-forehead points. */
  center: [number, number];
  chin: [number, number];
  foreheadTop: [number, number];
  /** Width across the face oval at cheek height (pixels). */
  faceWidth: number;
  /** Head pose in radians. yaw > 0: face turned toward image right. pitch > 0: chin up / camera below. */
  yaw: number;
  pitch: number;
  roll: number;
  /** Estimated camera-to-eye distance in mm (from iris size), or 0 when unknown. */
  distanceMm: number;
  /** Focal length in pixels assumed for this frame. */
  focalPx: number;
}

function centroid(px: Float32Array, idx: number[]): [number, number, number] {
  let x = 0, y = 0, z = 0;
  for (const i of idx) {
    x += px[i * 3];
    y += px[i * 3 + 1];
    z += px[i * 3 + 2];
  }
  const n = idx.length;
  return [x / n, y / n, z / n];
}

/** Area-weighted vertex normals for the canonical mesh, oriented toward the camera on average. */
export function computeNormals(px: Float32Array): Float32Array {
  const n = new Float32Array(NUM_MESH_VERTS * 3);
  const T = SURFACE_TRIANGLES;
  for (let t = 0; t < T.length; t += 3) {
    const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
    // Convert to normal space: y up, z toward camera.
    const ax = px[a], ay = -px[a + 1], az = -px[a + 2];
    const e1x = px[b] - ax, e1y = -px[b + 1] - ay, e1z = -px[b + 2] - az;
    const e2x = px[c] - ax, e2y = -px[c + 1] - ay, e2z = -px[c + 2] - az;
    const nx = e1y * e2z - e1z * e2y;
    const ny = e1z * e2x - e1x * e2z;
    const nz = e1x * e2y - e1y * e2x;
    n[a] += nx; n[a + 1] += ny; n[a + 2] += nz;
    n[b] += nx; n[b + 1] += ny; n[b + 2] += nz;
    n[c] += nx; n[c + 1] += ny; n[c + 2] += nz;
  }
  // The mesh is coarse and creased (nose ridge, lip line). Lighting computed from raw normals would
  // inherit those creases as visible seams, so relax the normals over the mesh a few times.
  const tmp = new Float32Array(n.length);
  for (let iter = 0; iter < 3; iter++) {
    tmp.set(n);
    for (let t = 0; t < T.length; t += 3) {
      const a = T[t] * 3, b = T[t + 1] * 3, c = T[t + 2] * 3;
      for (let k = 0; k < 3; k++) {
        const sum = (n[a + k] + n[b + k] + n[c + k]) * 0.5;
        tmp[a + k] += sum; tmp[b + k] += sum; tmp[c + k] += sum;
      }
    }
    for (let i = 0; i < n.length; i += 3) {
      const l = Math.hypot(tmp[i], tmp[i + 1], tmp[i + 2]) || 1;
      // Keep the area weighting: rescale to the vertex's original magnitude.
      const m = Math.hypot(n[i], n[i + 1], n[i + 2]) / l;
      n[i] = tmp[i] * m; n[i + 1] = tmp[i + 1] * m; n[i + 2] = tmp[i + 2] * m;
    }
  }

  // Winding of the canonical mesh is consistent, but its sign depends on mirroring: fix it globally.
  let sumZ = 0;
  for (let i = 2; i < n.length; i += 3) sumZ += n[i];
  const sign = sumZ >= 0 ? 1 : -1;
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] = (sign * n[i]) / l;
    n[i + 1] = (sign * n[i + 1]) / l;
    n[i + 2] = (sign * n[i + 2]) / l;
  }
  return n;
}

/**
 * Build face geometry from normalised landmarks.
 * @param lm 478 x (x, y, z) normalised landmarks, already in the (possibly mirrored) frame's space.
 */
export function buildFaceGeometry(lm: ArrayLike<number>, width: number, height: number): FaceGeometry {
  const px = new Float32Array(NUM_LANDMARKS * 3);
  const count = Math.min(NUM_LANDMARKS, Math.floor(lm.length / 3));
  for (let i = 0; i < count; i++) {
    px[i * 3] = lm[i * 3] * width;
    px[i * 3 + 1] = lm[i * 3 + 1] * height;
    px[i * 3 + 2] = lm[i * 3 + 2] * width;
  }
  const normals = computeNormals(px);

  const cA = centroid(px, EYE_A_LOWER.concat([159, 158, 160]));
  const cB = centroid(px, EYE_B_LOWER.concat([386, 385, 387]));
  const eyeA: [number, number] = [cA[0], cA[1]];
  const eyeB: [number, number] = [cB[0], cB[1]];
  const eyeMid: [number, number] = [(cA[0] + cB[0]) / 2, (cA[1] + cB[1]) / 2];
  const iod = Math.hypot(cB[0] - cA[0], cB[1] - cA[1]);

  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const i of FACE_OVAL) {
    x0 = Math.min(x0, px[i * 3]); x1 = Math.max(x1, px[i * 3]);
    y0 = Math.min(y0, px[i * 3 + 1]); y1 = Math.max(y1, px[i * 3 + 1]);
  }
  const center: [number, number] = [(x0 + x1) / 2, (y0 + y1) / 2];
  const chin: [number, number] = [px[CHIN * 3], px[CHIN * 3 + 1]];
  const foreheadTop: [number, number] = [px[FOREHEAD_TOP * 3], px[FOREHEAD_TOP * 3 + 1]];
  const faceWidth = Math.hypot(
    px[CHEEK_B * 3] - px[CHEEK_A * 3], px[CHEEK_B * 3 + 1] - px[CHEEK_A * 3 + 1], px[CHEEK_B * 3 + 2] - px[CHEEK_A * 3 + 2],
  );

  const frontal: number[] = [];
  for (const i of SKIN_SAMPLE_VERTS) if (normals[i * 3 + 2] > 0.75) frontal.push(px[i * 3 + 2]);
  frontal.sort((a, b) => a - b);
  const frontPlaneZ = frontal.length ? frontal[frontal.length >> 1] : centroid(px, [33, 133, 362, 263])[2];
  const noseTip: [number, number] = [px[NOSE_TIP * 3], px[NOSE_TIP * 3 + 1]];

  // Head pose from the mean surface normal (self-consistent with the engine's conventions).
  let mx = 0, my = 0, mz = 0;
  for (let i = 0; i < normals.length; i += 3) {
    mx += normals[i]; my += normals[i + 1]; mz += normals[i + 2];
  }
  const yaw = Math.atan2(mx, mz);
  const pitch = Math.atan2(my, Math.hypot(mx, mz));
  const roll = Math.atan2(cB[1] - cA[1], cB[0] - cA[0]);

  // Iris size -> metric distance. Only present when the model outputs the 10 iris landmarks.
  let irisPx = 0;
  if (count >= NUM_LANDMARKS) {
    const diam = (iris: { ring: number[] }) => {
      const [r0, r1, r2, r3] = iris.ring;
      const h = Math.hypot(px[r0 * 3] - px[r2 * 3], px[r0 * 3 + 1] - px[r2 * 3 + 1]);
      const v = Math.hypot(px[r1 * 3] - px[r3 * 3], px[r1 * 3 + 1] - px[r3 * 3 + 1]);
      return Math.max(h, v); // the lids often clip the vertical extent, foreshortening clips the other
    };
    // The eye nearer the camera is less foreshortened; take the larger of the two.
    irisPx = Math.max(diam(IRIS_A), diam(IRIS_B));
  }
  const focalPx = FRONT_CAMERA_FOCAL_PER_LONG_SIDE * Math.max(width, height);
  const distanceMm = irisPx > 2 ? (focalPx * IRIS_DIAMETER_MM) / irisPx : 0;

  return {
    width, height, px, normals, eyeA, eyeB, eyeMid, frontPlaneZ, noseTip, iod, irisPx,
    bbox: { x0: x0 / width, y0: y0 / height, x1: x1 / width, y1: y1 / height },
    center, chin, foreheadTop, faceWidth, yaw, pitch, roll, distanceMm, focalPx,
  };
}
