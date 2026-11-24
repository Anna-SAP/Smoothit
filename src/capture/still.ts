// One full-quality render of a single frame: the shared tail of every capture.
import { analyze } from '../engine/analysis';
import { Director, type Direction, faceExposureStops } from '../engine/director';
import type { Engine } from '../engine/engine';
import type { FaceGeometry } from '../engine/faceModel';
import type { Look } from '../engine/looks';
import type { Vision } from '../vision/vision';

export interface StillOptions {
  look: Look;
  mirror: boolean;
  /** Cap on the working resolution's long side. */
  maxLongSide: number;
  /** Also render the untouched frame through the same crop, for before/after comparison. */
  withOriginal?: boolean;
  /** Reuse a live Director so the still matches the preview's smoothed exposure and framing intent. */
  director?: Director;
  /** Face already detected on exactly this frame (skips detection). */
  face?: FaceGeometry | null;
}

export interface StillResult {
  pixels: Uint8Array;
  original: Uint8Array | null;
  width: number;
  height: number;
  direction: Direction;
}

export function processStill(engine: Engine, vision: Vision, frame: ImageBitmap, opts: StillOptions): StillResult {
  const srcW = frame.width, srcH = frame.height;
  const now = performance.now();
  const face = opts.face !== undefined ? opts.face : vision.detectFace(frame, srcW, srcH, opts.mirror, now)?.face ?? null;

  engine.ingest(frame, srcW, srcH, opts.mirror, opts.maxLongSide);
  const pixels0 = engine.readAnalysis();
  // A first metering pass tells how dim the subject is, so the segmenter can be shown a
  // normally exposed version of the frame.
  const metered = analyze(pixels0, face, null);
  const skinLum = 0.2126 * metered.skinLin[0] + 0.7152 * metered.skinLin[1] + 0.0722 * metered.skinLin[2];
  const preGain = metered.hasFace
    ? 2 ** Math.max(0, faceExposureStops(skinLum, metered.scleraLin, metered.skinHiLin, metered.skinClipped))
    : Math.min(4, Math.max(1, 0.18 / Math.max(metered.sceneLin, 0.01)));
  const seg = vision.segment(frame, srcW, srcH, opts.mirror, now, preGain);
  engine.setSegmentation(seg);
  const analysis = seg ? analyze(pixels0, face, seg) : metered;

  const director = opts.director ?? new Director();
  director.look = opts.look;
  const direction = director.update({ face, analysis, srcW, srcH, dt: opts.director ? 0.05 : Infinity });

  // Output size: the crop at working resolution, rounded to even pixels at the exact aspect.
  const crop = direction.params.crop;
  const height = Math.round((crop.h * engine.workHeight) / 2) * 2;
  const width = Math.round((height * director.composition.aspect) / 2) * 2;
  const out = { kind: 'capture' as const, width, height };
  const original = opts.withOriginal ? engine.renderOriginal(crop, out) : null;
  const pixels = engine.render(direction.params, out)!;
  return { pixels, original, width, height, direction };
}
