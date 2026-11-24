// Frame selection. A photographer shoots many frames and keeps the one with open eyes, a settled
// expression and no motion blur; pressing the shutter here scores a short burst the same way.
import type { FaceGeometry } from './faceModel';

export interface FrameSignals {
  /** 0 (open) .. 1 (closed), max over both eyes. */
  blink: number;
  /** 0..1 smile intensity. */
  smile: number;
  /** Mean landmark motion since the previous frame as a fraction of the inter-ocular distance. */
  motion: number;
  hasFace: boolean;
}

export function frameSignals(
  face: FaceGeometry | null, prev: FaceGeometry | null, blend: Record<string, number> | null,
): FrameSignals {
  if (!face) return { blink: 0, smile: 0, motion: 0, hasFace: false };
  const b = blend ?? {};
  const blink = Math.max(b.eyeBlinkLeft ?? 0, b.eyeBlinkRight ?? 0);
  const smile = ((b.mouthSmileLeft ?? 0) + (b.mouthSmileRight ?? 0)) / 2;
  let motion = 0;
  if (prev && prev.px.length === face.px.length) {
    let sum = 0, n = 0;
    for (let i = 0; i < 468 * 3; i += 3 * 6) {
      sum += Math.hypot(face.px[i] - prev.px[i], face.px[i + 1] - prev.px[i + 1]);
      n++;
    }
    motion = sum / n / (face.iod || 1);
  }
  return { blink, smile, motion, hasFace: true };
}

/** Higher is better. Blinks and blur dominate; a pleasant expression breaks ties. */
export function scoreFrame(s: FrameSignals): number {
  if (!s.hasFace) return -1;
  const blinkPenalty = Math.max(0, s.blink - 0.25) * 2.2;
  const motionPenalty = Math.min(1.5, s.motion * 9);
  return 1 - blinkPenalty - motionPenalty + 0.15 * Math.min(1, s.smile * 1.5);
}
