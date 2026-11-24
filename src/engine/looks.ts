// A "look" is what a photographer decides before the shoot: lighting pattern, retouch philosophy
// and colour grade. The Director adapts every look to the actual scene, so none of them needs any
// user adjustment; they only change the intent.
import type { LightTarget } from './lighting';

export interface Look {
  id: string;
  name: { zh: string; en: string };
  light: LightTarget;
  /** 0..1 strength of the relighting correction. */
  relight: number;
  skin: { smooth: number; smoothChroma: number; texKeep: number; even: number; shine: number; dodge: number };
  local: { eye: number; teeth: number; lip: number; sharpen: number };
  background: { blur: number; dim: number; desat: number };
  grade: {
    exposureBias: number;
    /** Positive = warmer. */
    warmth: number;
    shoulder: number;
    contrast: number;
    shadows: number;
    sat: number;
    vibrance: number;
    fade: number;
    shadowTint: [number, number, number];
    highTint: [number, number, number];
    glow: number;
    vignette: number;
    grain: number;
  };
}

const deg = (d: number) => (d * Math.PI) / 180;

export const LOOKS: Look[] = [
  {
    // Large soft key 35 deg off-axis with generous fill: the classic flattering studio portrait.
    id: 'studio',
    name: { zh: '影楼', en: 'Studio' },
    light: { azimuth: deg(35), elevation: deg(28), ratio: 2.4, wrap: 0.55 },
    relight: 0.7,
    skin: { smooth: 0.48, smoothChroma: 0.85, texKeep: 1.0, even: 0.5, shine: 0.4, dodge: 0.65 },
    local: { eye: 1.0, teeth: 1.0, lip: 0.5, sharpen: 0.28 },
    background: { blur: 0.016, dim: 0.25, desat: 0.08 },
    grade: {
      exposureBias: 0.05, warmth: 0.15, shoulder: 0.6, contrast: 0.14, shadows: 0.1, sat: 0.0, vibrance: 0.08, fade: 0.008,
      shadowTint: [-0.004, 0.0, 0.006], highTint: [0.008, 0.003, -0.006], glow: 0.16, vignette: 0.16, grain: 0.02,
    },
  },
  {
    // On-axis beauty light, crisp and glossy, background falling away as with flash at night.
    id: 'redcarpet',
    name: { zh: '红毯', en: 'Red Carpet' },
    light: { azimuth: deg(8), elevation: deg(32), ratio: 1.6, wrap: 0.35 },
    relight: 0.75,
    skin: { smooth: 0.52, smoothChroma: 0.9, texKeep: 1.05, even: 0.55, shine: 0.25, dodge: 0.7 },
    local: { eye: 1.2, teeth: 1.2, lip: 0.9, sharpen: 0.4 },
    background: { blur: 0.02, dim: 0.75, desat: 0.15 },
    grade: {
      exposureBias: 0.2, warmth: 0.1, shoulder: 0.6, contrast: 0.3, shadows: 0.08, sat: 0.06, vibrance: 0.2, fade: 0.0,
      shadowTint: [-0.003, 0.0, 0.004], highTint: [0.004, 0.002, -0.002], glow: 0.14, vignette: 0.3, grain: 0.016,
    },
  },
  {
    // Sculpted 3:1 side light, restrained colour, more skin texture: fashion editorial.
    id: 'editorial',
    name: { zh: '杂志', en: 'Editorial' },
    light: { azimuth: deg(48), elevation: deg(30), ratio: 3.0, wrap: 0.3 },
    relight: 0.75,
    skin: { smooth: 0.36, smoothChroma: 0.8, texKeep: 1.1, even: 0.4, shine: 0.3, dodge: 0.5 },
    local: { eye: 1.1, teeth: 0.8, lip: 0.6, sharpen: 0.42 },
    background: { blur: 0.018, dim: 0.5, desat: 0.3 },
    grade: {
      exposureBias: 0.0, warmth: -0.1, shoulder: 0.55, contrast: 0.36, shadows: 0.02, sat: -0.14, vibrance: 0.05, fade: 0.03,
      shadowTint: [-0.008, 0.002, 0.012], highTint: [0.006, 0.004, -0.004], glow: 0.1, vignette: 0.26, grain: 0.03,
    },
  },
  {
    // Warm, low-contrast negative-film rendering with halation and visible grain.
    id: 'film',
    name: { zh: '胶片', en: 'Film' },
    light: { azimuth: deg(30), elevation: deg(25), ratio: 2.2, wrap: 0.6 },
    relight: 0.6,
    skin: { smooth: 0.4, smoothChroma: 0.8, texKeep: 1.0, even: 0.4, shine: 0.3, dodge: 0.5 },
    local: { eye: 0.8, teeth: 0.7, lip: 0.9, sharpen: 0.15 },
    background: { blur: 0.014, dim: 0.15, desat: 0.0 },
    grade: {
      exposureBias: 0.12, warmth: 0.55, shoulder: 0.5, contrast: 0.06, shadows: 0.12, sat: -0.04, vibrance: 0.1, fade: 0.05,
      shadowTint: [-0.006, 0.006, 0.004], highTint: [0.014, 0.004, -0.012], glow: 0.26, vignette: 0.2, grain: 0.045,
    },
  },
];

export const DEFAULT_LOOK = LOOKS[0];

export function lookById(id: string): Look {
  return LOOKS.find((l) => l.id === id) ?? DEFAULT_LOOK;
}
