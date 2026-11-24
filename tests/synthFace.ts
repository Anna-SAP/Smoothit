// Test helper: renders MediaPipe's generic canonical face through an ideal pinhole camera to
// synthesise landmark sets with exactly known distance, pose and focal length.
import { readFileSync } from 'node:fs';
import { FRONT_CAMERA_FOCAL_PER_LONG_SIDE, IRIS_DIAMETER_MM, buildFaceGeometry, type FaceGeometry } from '../src/engine/faceModel';

const model: number[] = JSON.parse(readFileSync(new URL('./fixtures/canonicalFace.json', import.meta.url), 'utf8')).vertices;

export interface SynthOptions {
  /** Camera to eye-plane distance in mm. */
  distanceMm: number;
  width?: number;
  height?: number;
  /** Head rotation in radians, engine conventions: yaw > 0 turns toward image right, pitch > 0 chin up. */
  yaw?: number;
  pitch?: number;
  /** Focal length in pixels (defaults to the engine's front-camera assumption). */
  focalPx?: number;
  /** Mirror horizontally, as the engine does for the front camera. */
  mirror?: boolean;
}

/** Model-space points in mm (x right, y up, z toward the viewer), including 10 synthetic iris points. */
function modelPoints(): number[][] {
  const pts: number[][] = [];
  for (let i = 0; i < model.length; i += 3) pts.push([model[i] * 10, model[i + 1] * 10, model[i + 2] * 10]);
  const mid = (a: number, b: number) => pts[a].map((v, k) => (v + pts[b][k]) / 2);
  const r = IRIS_DIAMETER_MM / 2;
  for (const c of [mid(33, 133), mid(362, 263)]) {
    pts.push(c, [c[0] + r, c[1], c[2]], [c[0], c[1] + r, c[2]], [c[0] - r, c[1], c[2]], [c[0], c[1] - r, c[2]]);
  }
  return pts;
}

export function synthLandmarks(opt: SynthOptions): { landmarks: Float32Array; width: number; height: number; focalPx: number } {
  const width = opt.width ?? 1080, height = opt.height ?? 1440;
  const f = opt.focalPx ?? FRONT_CAMERA_FOCAL_PER_LONG_SIDE * Math.max(width, height);
  const yaw = opt.yaw ?? 0, pitch = opt.pitch ?? 0;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  const pts = modelPoints().map(([x, y, z]) => {
    // Pitch about x (positive tips the forehead back = chin up), then yaw about y (toward +x).
    const y1 = y * cp + z * sp, z1 = -y * sp + z * cp;
    return [x * cy + z1 * sy, y1, -x * sy + z1 * cy];
  });
  const eyeZ = [33, 133, 362, 263].reduce((s, i) => s + pts[i][2], 0) / 4;
  const z0 = opt.distanceMm + eyeZ; // depth of the model origin
  const out = new Float32Array(pts.length * 3);
  pts.forEach(([x, y, z], i) => {
    const depth = z0 - z;
    let u = (width / 2 + (f * x) / depth) / width;
    if (opt.mirror) u = 1 - u;
    out[i * 3] = u;
    out[i * 3 + 1] = (height / 2 - (f * y) / depth) / height;
    out[i * 3 + 2] = ((depth - z0) * (f / z0)) / width; // MediaPipe convention: relative, x-scaled
  });
  return { landmarks: out, width, height, focalPx: f };
}

export function synthFace(opt: SynthOptions): FaceGeometry {
  const { landmarks, width, height } = synthLandmarks(opt);
  return buildFaceGeometry(landmarks, width, height);
}
