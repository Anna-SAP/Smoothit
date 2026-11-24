// Everything the renderer needs for one frame. Produced by the Director, consumed by the Engine.

export interface Crop {
  /** Source-uv rectangle that becomes the output frame. */
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FaceRenderData {
  /** Face region of interest in source uv (x0, y0, w, h) that the mesh buffers cover. */
  roi: [number, number, number, number];
  /** Interleaved mesh vertices, MESH_STRIDE floats each (see meshBuffer.ts). */
  vertices: Float32Array;
  vertexCount: number;
  /** Inter-ocular distance as a fraction of the source width. */
  iodUv: number;
}

export interface SkinParams {
  /** Measured skin tone in gamma space: luma, B-Y, R-Y. */
  ycc: [number, number, number];
  chromaTol: number;
  /** 0..1 removal of the blemish band's luminance (also softens facial modelling: keep moderate). */
  smooth: number;
  /** 0..1 removal of the blemish band's colour (redness, blotches, spots). */
  smoothChroma: number;
  /** Multiplier on the fine texture band that is added back (1 = untouched pores). */
  texKeep: number;
  /** Chroma evening toward the regional skin tone. */
  even: number;
  /** Highlight (shine) compression on skin. */
  shine: number;
  /** Under-eye dodge. */
  dodge: number;
}

export interface LocalParams {
  relight: number;
  eye: number;
  teeth: number;
  lip: number;
  sharpen: number;
  /** Noise-coring threshold in gamma units (0 = off). */
  denoise: number;
}

export interface BackgroundParams {
  /** Bokeh disc radius as a fraction of the source width (0 disables). */
  blur: number;
  /** Background darkening in stops. */
  dim: number;
  desat: number;
}

export interface GradeParams {
  wb: [number, number, number];
  exposure: number;
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
  /** Vignette centre in output uv. */
  vigCenter: [number, number];
  grain: number;
}

export interface RenderParams {
  face: FaceRenderData | null;
  skin: SkinParams;
  local: LocalParams;
  background: BackgroundParams;
  grade: GradeParams;
  crop: Crop;
  /** Backward-warp displacement of each warp-grid vertex in source uv, or null for none. */
  warp: Float32Array | null;
}

export const WARP_COLS = 36;
export const WARP_ROWS = 48;

export const NEUTRAL_GRADE: GradeParams = {
  wb: [1, 1, 1], exposure: 0, shoulder: 1, contrast: 0, shadows: 0, sat: 0, vibrance: 0, fade: 0,
  shadowTint: [0, 0, 0], highTint: [0, 0, 0], glow: 0, vignette: 0, vigCenter: [0.5, 0.5], grain: 0,
};
