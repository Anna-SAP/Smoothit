// Automatic framing. The sensor frame is treated as a negative that is always a little larger than
// the picture: the engine picks the crop a portrait photographer would have framed in-camera.
import type { FaceGeometry } from './faceModel';
import type { Crop } from './params';

export interface CompositionOptions {
  /** Output aspect ratio, width / height. */
  aspect: number;
  /** Desired head height (crown to chin) as a fraction of the output height. */
  headFraction: number;
  /** Largest digital zoom the reframing may use. */
  maxZoom: number;
}

export const DEFAULT_COMPOSITION: CompositionOptions = { aspect: 3 / 4, headFraction: 0.46, maxZoom: 1.6 };

/** Chin-to-hairline covers roughly 77% of the head; the mesh's top vertex sits near the hairline. */
const HEAD_OVER_FACE = 1.3;

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

/** Largest crop of the given aspect that fits the source, in uv units. */
export function maxCrop(srcW: number, srcH: number, aspect: number): { w: number; h: number } {
  const srcAspect = srcW / srcH;
  return srcAspect > aspect ? { w: aspect / srcAspect, h: 1 } : { w: 1, h: srcAspect / aspect };
}

export function composeCrop(
  face: FaceGeometry | null, srcW: number, srcH: number, opt: CompositionOptions = DEFAULT_COMPOSITION,
): Crop {
  const max = maxCrop(srcW, srcH, opt.aspect);
  if (!face) {
    const w = max.w / 1.03, h = max.h / 1.03;
    return { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
  }
  const faceLen = Math.hypot(face.chin[0] - face.foreheadTop[0], face.chin[1] - face.foreheadTop[1]);
  const headPx = faceLen * HEAD_OVER_FACE;
  const current = headPx / (max.h * srcH); // head fraction with no zoom at all
  // We can only zoom in. A face that is already large gets just enough slack to be re-positioned.
  const minZoom = current > 0.55 ? 1.0 : 1.05;
  const zoom = clamp(opt.headFraction / current, minZoom, opt.maxZoom);
  const w = max.w / zoom, h = max.h / zoom;
  const frac = current * zoom;

  // Eye line: upper third for looser framings, drifting toward 43% for tight head shots so the
  // crown keeps some headroom.
  const eyeY = 0.36 + 0.07 * smoothstep(0.35, 0.65, frac);
  // Lead room: leave a little more space on the side the face is turned toward.
  const eyeX = 0.5 - 0.05 * clamp(face.yaw / 0.5, -1, 1);

  let x = face.eyeMid[0] / srcW - eyeX * w;
  let y = face.eyeMid[1] / srcH - eyeY * h;
  // Keep the chin (and a little neck) inside the frame even for extreme head sizes.
  const chinV = (face.chin[1] / srcH - y) / h;
  if (chinV > 0.9) y += (chinV - 0.9) * h;
  x = clamp(x, 0, 1 - w);
  y = clamp(y, 0, 1 - h);
  return { x, y, w, h };
}

/** Critically-damped follow so the live framing glides instead of jittering with the landmarks. */
export class CropSmoother {
  private cur: Crop | null = null;

  reset(): void {
    this.cur = null;
  }

  update(target: Crop, dt: number, timeConstant = 0.28): Crop {
    if (!this.cur) {
      this.cur = { ...target };
      return this.cur;
    }
    const c = this.cur;
    const a = 1 - Math.exp(-dt / timeConstant);
    // Dead-band: ignore sub-0.4% wobble entirely.
    const dead = 0.004;
    const step = (cur: number, tgt: number) => (Math.abs(tgt - cur) < dead ? cur : cur + (tgt - cur) * a);
    const w = step(c.w, target.w);
    const h = (w / target.w) * target.h; // keep the aspect exact
    this.cur = {
      w, h,
      x: clamp(step(c.x + c.w / 2, target.x + target.w / 2) - w / 2, 0, 1 - w),
      y: clamp(step(c.y + c.h / 2, target.y + target.h / 2) - h / 2, 0, 1 - h),
    };
    return this.cur;
  }
}
