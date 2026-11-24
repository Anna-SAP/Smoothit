// The Director is the photographer in the loop. Each frame it reads the scene analysis and the face
// geometry and decides exposure, white balance, lighting correction, framing and lens-geometry
// correction - the decisions a studio makes with lights, lens choice, a light meter and a retoucher.
import { type Analysis, emptyAnalysis } from './analysis';
import { type CompositionOptions, CropSmoother, DEFAULT_COMPOSITION, composeCrop } from './composition';
import type { FaceGeometry } from './faceModel';
import { DEFAULT_PERSPECTIVE, buildWarpField, perspectiveDisplacements } from './geometry';
import { SH_TERMS, chooseKeySide, describeLighting, relightGains, relightNeed, solveSPD } from './lighting';
import { DEFAULT_LOOK, type Look } from './looks';
import { MESH_STRIDE, MESH_VERTEX_COUNT, buildFaceRenderData, underEyeWeights } from './meshBuffer';
import type { Crop, RenderParams } from './params';

export type HintId = 'noFace' | 'tooClose' | 'backlit' | 'lowAngle' | 'lowLight';

export interface DirectorInput {
  face: FaceGeometry | null;
  /** Fresh analysis for this frame, or null to keep using the last one. */
  analysis: Analysis | null;
  srcW: number;
  srcH: number;
  /** Seconds since the previous update. Use Infinity to bypass all temporal smoothing. */
  dt: number;
}

export interface Direction {
  params: RenderParams;
  /** Conditions the pipeline cannot fully fix by itself, most important first. */
  hints: HintId[];
  /** True when the scene is dark enough that the screen should act as a fill light. */
  wantsScreenLight: boolean;
  diagnostics: {
    exposureStops: number;
    wb: [number, number, number];
    distanceMm: number;
    lightContrast: number;
    lightSide: number;
    lightElevation: number;
    relightNeed: number;
    keySide: number;
    yawDeg: number;
    pitchDeg: number;
  };
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const lumaOf = (c: [number, number, number]) => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

/** Log2 chromaticity (R/G, B/G) of average skin under a neutral illuminant. */
const SKIN_NEUTRAL: [number, number] = [0.8, -0.46];

/**
 * White balance, estimated from skin with the background as a second witness.
 *
 * In log-chromaticity space three things move a skin sample, each along its own direction:
 *   s  skin's own pigmentation (melanin / haemoglobin)   (1, -0.57)   - must be preserved
 *   t  illuminant colour temperature (amber <-> blue)     (1, -1.2)    - to be removed
 *   m  illuminant tint (magenta <-> green)                (1,  1)      - to be removed
 * Skin alone gives two equations for three unknowns, and pigmentation is nearly parallel to colour
 * temperature, so the problem is solved as a small MAP estimate: Gaussian priors on s, t, m (the
 * camera's own white balance has already removed most of t and m) plus, when available, a
 * grey-world reading of the background's mid-tones, which sees t and m but not s.
 * @returns linear RGB gains that leave skin luminance unchanged
 */
export function estimateWhiteBalance(
  skinLin: [number, number, number], sceneChroma: [number, number] | null = null, sceneWeight = 0,
): [number, number, number] {
  const [r, g, b] = skinLin;
  if (!(r > 0 && g > 0 && b > 0)) return [1, 1, 1];
  const dRg = Math.log2(r / g) - SKIN_NEUTRAL[0], dBg = Math.log2(b / g) - SKIN_NEUTRAL[1];
  // Rows: [coefficients of (s, t, m)], observation, sigma.
  const rows: [number[], number, number][] = [
    [[1, 1, 1], dRg, 0.05],
    [[-0.57, -1.2, 1], dBg, 0.05],
    // Pigmentation gets a wide prior and colour temperature a narrow one: mistaking a deep or
    // tanned skin tone for a warm cast (and bleaching it) is the worse error by far.
    [[1, 0, 0], 0, 0.35],  // prior on s
    [[0, 1, 0], 0, 0.12],  // prior on t
    [[0, 0, 1], 0, 0.1],   // prior on m
  ];
  if (sceneChroma && sceneWeight > 0) {
    // A weak witness on purpose: walls, clothes and foliage are often genuinely coloured.
    const sigma = 0.6 / Math.sqrt(sceneWeight);
    rows.push([[0, 1, 1], sceneChroma[0], sigma], [[0, -1.2, 1], sceneChroma[1], sigma]);
  }
  const A = new Float64Array(9), rhs = new Float64Array(3);
  for (const [c, y, sigma] of rows) {
    const w = 1 / (sigma * sigma);
    for (let i = 0; i < 3; i++) {
      rhs[i] += w * c[i] * y;
      for (let j = 0; j < 3; j++) A[i * 3 + j] += w * c[i] * c[j];
    }
  }
  const [, t, m] = solveSPD(A, rhs, 3);
  const STRENGTH = 0.85;
  const gr = 2 ** clamp(-STRENGTH * (t + m), -0.9, 0.9);
  const gb = 2 ** clamp(-STRENGTH * (-1.2 * t + m), -0.9, 0.9);
  const k = lumaOf(skinLin) / lumaOf([r * gr, g, b * gb]);
  return [gr * k, k, gb * k];
}

/**
 * Exposure for the face, as an incident-light meter would set it: skin is placed at a brightness
 * that depends on how light the skin itself is. The eye whites serve as a built-in grey card to
 * tell "dark skin, well exposed" from "light skin, under-exposed".
 * @param skinHi luminance of the lit side of the face: it, not the average, must stay below white
 * @param clipped fraction of skin already clipped in the source (nothing to recover by darkening)
 */
export function faceExposureStops(skinLum: number, scleraLum: number, skinHi = 0, clipped = 0): number {
  if (!(skinLum > 0)) return 0;
  // Eye whites sit in the shade of lids and lashes, so they can only read too dark: a high ratio
  // is weak evidence, a low one (skin much darker than the sclera) is strong evidence of dark skin.
  const tone = scleraLum > 0 ? smoothstep(0.3, 0.95, skinLum / scleraLum) : 0.6;
  const target = 0.21 + 0.14 * tone; // linear: 50% .. 63% in display terms
  let stops = 0.8 * Math.log2(target / skinLum);
  if (skinHi > 0) stops = Math.min(stops, Math.log2(0.6 / skinHi));
  if (clipped > 0.08) stops = Math.max(stops, -0.3);
  return clamp(stops, -0.8, 1.5);
}

export class Director {
  look: Look = DEFAULT_LOOK;
  composition: CompositionOptions = DEFAULT_COMPOSITION;

  private analysis: Analysis = emptyAnalysis();
  private sh = new Float64Array(SH_TERMS);
  private exposure = 0;
  private wb: [number, number, number] = [1, 1, 1];
  private skinYcc: [number, number, number] = [0.6, -0.08, 0.1];
  private chromaTol = 0.06;
  private keySide: 1 | -1 = -1;
  private keySideVotes = 0;
  private primed = false;

  /** Current auto-exposure decision in stops (before the look's bias). */
  get exposureStops(): number {
    return this.exposure;
  }

  private cropSmoother = new CropSmoother();
  private meshData = new Float32Array(MESH_VERTEX_COUNT * MESH_STRIDE);
  private warpField: Float32Array | undefined;

  constructor() {
    this.sh[0] = 1;
  }

  reset(): void {
    this.primed = false;
    this.cropSmoother.reset();
  }

  update(input: DirectorInput): Direction {
    const { face, srcW, srcH, dt } = input;
    const look = this.look;
    const instant = !Number.isFinite(dt) || !this.primed;
    const a = (tau: number) => (instant ? 1 : 1 - Math.exp(-dt / tau));

    // --- absorb the new measurements, smoothing over time ---
    if (input.analysis) this.analysis = input.analysis;
    const an = this.analysis;
    if (input.analysis && an.hasFace) {
      const skinLum = lumaOf(an.skinLin);
      const k = a(0.5);
      this.exposure += (faceExposureStops(skinLum, an.scleraLin, an.skinHiLin, an.skinClipped) - this.exposure) * k;
      const wb = estimateWhiteBalance(an.skinLitLin, an.sceneChroma, an.sceneChromaWeight);
      for (let i = 0; i < 3; i++) {
        this.wb[i] += (wb[i] - this.wb[i]) * k;
        this.skinYcc[i] += (an.skinYcc[i] - this.skinYcc[i]) * k;
      }
      this.chromaTol += (clamp(3 * an.skinChromaSigma, 0.045, 0.09) - this.chromaTol) * k;
      for (let i = 0; i < SH_TERMS; i++) this.sh[i] += (an.sh[i] - this.sh[i]) * a(0.6);
      this.primed = true;
    } else if (input.analysis && !an.hasFace) {
      // No face: fall back to a gentle global exposure toward a mid-toned frame.
      const target = clamp(0.6 * Math.log2(0.2 / Math.max(an.sceneLin, 0.01)), -0.5, 1.2);
      this.exposure += (target - this.exposure) * a(0.8);
      for (let i = 0; i < 3; i++) this.wb[i] += (1 - this.wb[i]) * a(0.8);
    }

    // --- framing ---
    const target = composeCrop(face, srcW, srcH, this.composition);
    const crop: Crop = instant ? { ...target } : { ...this.cropSmoother.update(target, dt) };
    if (instant) { this.cropSmoother.reset(); this.cropSmoother.update(target, 0); }

    // --- face-dependent corrections ---
    const light = describeLighting(this.sh);
    let faceData: RenderParams['face'] = null;
    let warp: Float32Array | null = null;
    let vigCenter: [number, number] = [0.5, 0.42];
    if (face) {
      // Key-light side with hysteresis so it never flickers between frames.
      const side = chooseKeySide(face, light);
      if (side !== this.keySide) {
        this.keySideVotes += instant ? 100 : 1;
        if (this.keySideVotes > 12) { this.keySide = side; this.keySideVotes = 0; }
      } else this.keySideVotes = 0;

      const gains = an.hasFace ? relightGains(face, this.sh, look.light, this.keySide) : null;
      faceData = buildFaceRenderData(face, gains, underEyeWeights(face), this.meshData);

      const disp = perspectiveDisplacements(face, DEFAULT_PERSPECTIVE);
      this.warpField = buildWarpField(face, disp, crop, this.warpField);
      warp = this.warpField;
      vigCenter = [
        clamp((face.center[0] / srcW - crop.x) / crop.w, 0.25, 0.75),
        clamp((face.center[1] / srcH - crop.y) / crop.h, 0.25, 0.6),
      ];
    }

    // --- hints: only what software cannot fix ---
    const hints: HintId[] = [];
    const skinLum = lumaOf(an.skinLin);
    const lowLight = an.sceneLin < 0.045 || (an.hasFace && skinLum < 0.05);
    if (!face) hints.push('noFace');
    else {
      if (face.distanceMm > 0 && face.distanceMm < 250) hints.push('tooClose');
      if (an.hasFace && an.backgroundLin > 4 * skinLum && an.clipped > 0.06) hints.push('backlit');
      if (face.pitch > 0.32) hints.push('lowAngle');
    }
    if (lowLight) hints.push('lowLight');

    const g = look.grade;
    const warm = 2 ** (0.1 * g.warmth);
    const exposure = this.exposure + g.exposureBias;
    const wb: [number, number, number] = [this.wb[0] * warm, this.wb[1], this.wb[2] / warm];
    // Noise guard: a strong digital push amplifies sensor noise, so ease off sharpening and core
    // the finest detail levels.
    const push = smoothstep(0.5, 1.5, exposure);
    const params: RenderParams = {
      face: faceData,
      skin: {
        ycc: [...this.skinYcc], chromaTol: this.chromaTol, ...look.skin,
        // Blown-out skin has no highlight detail left to compress; doing so only paints flat patches.
        shine: look.skin.shine * (1 - smoothstep(0.03, 0.15, an.skinClipped)),
      },
      local: {
        // A touch of shaping always; the full correction only when the light is actually bad.
        relight: look.relight * (0.3 + 0.7 * relightNeed(light)),
        ...look.local,
        sharpen: look.local.sharpen * (1 - 0.8 * push),
        denoise: 0.035 * push,
      },
      // In very dim frames segmentation is least reliable: keep the background treatment subtle.
      background: { ...look.background, blur: look.background.blur * (1 - 0.5 * push) },
      grade: {
        wb, exposure, shoulder: g.shoulder, contrast: g.contrast, shadows: g.shadows, sat: g.sat, vibrance: g.vibrance,
        fade: g.fade, shadowTint: g.shadowTint, highTint: g.highTint, glow: g.glow, vignette: g.vignette, vigCenter,
        grain: g.grain,
      },
      crop,
      warp,
    };
    return {
      params,
      hints,
      wantsScreenLight: lowLight,
      diagnostics: {
        exposureStops: exposure, wb, distanceMm: face?.distanceMm ?? 0, lightContrast: light.contrast,
        lightSide: light.side, lightElevation: light.elevation, relightNeed: relightNeed(light), keySide: this.keySide,
        yawDeg: ((face?.yaw ?? 0) * 180) / Math.PI, pitchDeg: ((face?.pitch ?? 0) * 180) / Math.PI,
      },
    };
  }
}
