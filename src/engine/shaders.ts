// GLSL ES 3.00 sources for the imaging pipeline. They are plain GLSL on purpose: the same shaders
// run under WebGL2 (web / Capacitor) and port 1:1 to OpenGL ES 3 on Android and nearly 1:1 to Metal.

const HEAD = `#version 300 es
precision highp float;
`;

const COMMON = `
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 toLin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGam(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
float max3(vec3 c) { return max(c.r, max(c.g, c.b)); }
float min3(vec3 c) { return min(c.r, min(c.g, c.b)); }
`;

const ROI = `
uniform vec4 uRoi; // face region of interest in source uv: x0, y0, w, h
vec4 roiTex(sampler2D t, vec2 uv, vec4 outside) {
  vec2 p = (uv - uRoi.xy) / uRoi.zw;
  if (p.x <= 0.0 || p.y <= 0.0 || p.x >= 1.0 || p.y >= 1.0) return outside;
  return texture(t, p);
}
`;

export const VS_FULL = `${HEAD}
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

/** Copy / bilinear resample, with optional horizontal mirror (selfie view). */
export const FS_COPY = `${HEAD}
uniform sampler2D uTex;
uniform float uMirror;
in vec2 vUv;
out vec4 o;
void main() {
  vec2 uv = vUv;
  if (uMirror > 0.5) uv.x = 1.0 - uv.x;
  o = vec4(texture(uTex, uv).rgb, 1.0);
}`;

/** 9-tap Gaussian (5 bilinear fetches), separable; uDir is the per-sample step in uv. */
export const FS_BLUR = `${HEAD}
uniform sampler2D uTex;
uniform vec2 uDir;
in vec2 vUv;
out vec4 o;
void main() {
  vec4 s = texture(uTex, vUv) * 0.2270270270;
  s += (texture(uTex, vUv + uDir * 1.3846153846) + texture(uTex, vUv - uDir * 1.3846153846)) * 0.3162162162;
  s += (texture(uTex, vUv + uDir * 3.2307692308) + texture(uTex, vUv - uDir * 3.2307692308)) * 0.0702702703;
  o = s;
}`;

/** Face mesh -> two "G-buffer" attachments in face-ROI space. */
export const VS_MESH = `${HEAD}
in vec3 aPos;   // source uv + normalised depth
in vec2 aA;     // relight gain in stops, under-eye dodge weight
in vec4 aB;     // region one-hot: eye, lip, mouth interior, skin
uniform vec4 uRoi;
out vec2 vA;
out vec4 vB;
void main() {
  vec2 p = (aPos.xy - uRoi.xy) / uRoi.zw;
  gl_Position = vec4(p * 2.0 - 1.0, clamp(aPos.z, -0.95, 0.95), 1.0);
  vA = aA;
  vB = aB;
}`;

export const FS_MESH = `${HEAD}
in vec2 vA;
in vec4 vB;
layout(location = 0) out vec4 oA; // r: gain (stops/4 + 0.5), g: coverage, b: dodge
layout(location = 1) out vec4 oB; // r: eye, g: lip, b: mouth interior, a: facial skin
void main() {
  oA = vec4(clamp(vA.x * 0.25 + 0.5, 0.0, 1.0), 1.0, vA.y, 1.0);
  oB = vB;
}`;

/**
 * Skin weight at quarter resolution. Combines three independent cues so that no single failure
 * (mesh drift, segmentation blob, unusual lighting colour) smooths something that is not skin:
 *   mesh region  - geometric: inside the face, outside eyes / lips / brows
 *   segmentation - semantic: face-skin + body-skin classes (neck, shoulders, arms)
 *   colour       - statistical: chroma close to this person's measured skin tone
 */
export const FS_SKINMASK = `${HEAD}${COMMON}${ROI}
uniform sampler2D uColor, uGB, uSeg;
uniform float uHasFace, uHasSeg;
uniform vec3 uSkinYcc;  // measured skin: luma, Cb, Cr (gamma space)
uniform float uChromaTol;
in vec2 vUv;
out vec4 o;
void main() {
  vec3 c = texture(uColor, vUv).rgb;
  float y = luma(c);
  vec2 cc = vec2(c.b - y, c.r - y);
  float d = distance(cc, uSkinYcc.yz) / uChromaTol;
  float colour = exp(-0.5 * d * d) * smoothstep(0.22, 0.5, y / max(uSkinYcc.x, 1e-3));
  vec4 gb = uHasFace > 0.5 ? roiTex(uGB, vUv, vec4(0.0)) : vec4(0.0);
  vec4 seg = uHasSeg > 0.5 ? texture(uSeg, vUv) : vec4(0.0);
  float feature = max(gb.r, max(gb.g, gb.b));
  float meshSkin = gb.a * mix(0.3, 1.0, colour);
  float segSkin = seg.b * colour;
  float w = max(meshSkin, segSkin) * (1.0 - feature) * (1.0 - 0.9 * seg.g);
  o = vec4(w, colour, 0.0, 1.0);
}`;

/**
 * Separable masked bilateral filter: the low-frequency "base" layer of the skin. Non-skin pixels
 * get no weight, so eyes, lips, hair and background never bleed into the skin base.
 */
export const FS_BILATERAL = `${HEAD}${COMMON}
uniform sampler2D uColor, uMask;
uniform vec2 uDir;
uniform float uSigmaR;
in vec2 vUv;
out vec4 o;
void main() {
  vec3 c0 = texture(uColor, vUv).rgb;
  float y0 = luma(c0);
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int i = -7; i <= 7; i++) {
    vec2 uv = vUv + uDir * float(i);
    vec3 c = texture(uColor, uv).rgb;
    float m = texture(uMask, uv).r;
    float ws = exp(-0.5 * float(i * i) / 10.5);
    float dy = (luma(c) - y0) / uSigmaR;
    float w = ws * exp(-0.5 * dy * dy) * (m + 0.02);
    acc += c * w;
    wsum += w;
  }
  o = vec4(acc / wsum, 1.0);
}`;

/** Premultiply colour by a mask (for normalised convolution): rgb * m, m. */
export const FS_PREMULT = `${HEAD}
uniform sampler2D uColor, uMask;
in vec2 vUv;
out vec4 o;
void main() {
  float m = texture(uMask, vUv).r;
  o = vec4(texture(uColor, vUv).rgb * m, m);
}`;

// --- Guided-filter matte refinement (He et al. 2010) with luminance as the guide. -----------------
export const FS_GF_PACK = `${HEAD}${COMMON}
uniform sampler2D uColor, uSeg;
in vec2 vUv;
out vec4 o;
void main() {
  float I = luma(texture(uColor, vUv).rgb);
  float p = texture(uSeg, vUv).r;
  o = vec4(I, p, I * p, I * I);
}`;

export const FS_GF_AB = `${HEAD}
uniform sampler2D uMean;
uniform float uEps;
in vec2 vUv;
out vec4 o;
void main() {
  vec4 m = texture(uMean, vUv);
  float varI = m.a - m.r * m.r;
  float cov = m.b - m.r * m.g;
  float a = cov / (varI + uEps);
  o = vec4(a, m.g - a * m.r, 0.0, 1.0);
}`;

export const FS_GF_APPLY = `${HEAD}${COMMON}
uniform sampler2D uColor, uAB;
in vec2 vUv;
out vec4 o;
void main() {
  vec4 ab = texture(uAB, vUv);
  float q = ab.r * luma(texture(uColor, vUv).rgb) + ab.g;
  // Inclusive on purpose: a thin ring of un-blurred background next to the subject is far less
  // objectionable than fine hair or an earring dissolving into the blur.
  q = smoothstep(0.04, 0.5, q);
  o = vec4(q, q, q, 1.0);
}`;

/** Fallback matte when float render targets are unavailable. */
export const FS_MATTE_SIMPLE = `${HEAD}
uniform sampler2D uSeg;
in vec2 vUv;
out vec4 o;
void main() {
  float q = smoothstep(0.3, 0.7, texture(uSeg, vUv).r);
  o = vec4(q, q, q, 1.0);
}`;

/**
 * Lens-like background blur: a disc kernel in linear light where bright samples weigh more
 * (specular highlights bloom into bokeh discs) and foreground samples weigh nothing (no halo of
 * subject colour leaking into the background).
 */
export const FS_BOKEH = `${HEAD}${COMMON}
uniform sampler2D uColor, uMatte;
uniform vec2 uRadius;
in vec2 vUv;
out vec4 o;
const int N = 44;
void main() {
  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  for (int i = 0; i < N; i++) {
    float r = sqrt((float(i) + 0.5) / float(N));
    float a = float(i) * 2.39996323;
    vec2 uv = vUv + vec2(cos(a), sin(a)) * r * uRadius;
    vec3 lin = toLin(texture(uColor, uv).rgb);
    float fg = texture(uMatte, uv).r;
    float l = luma(lin);
    float w = (1.0 - fg) * (1.0 + 10.0 * l * l * l) + 1e-4;
    acc += lin * w;
    wsum += w;
  }
  o = vec4(toGam(acc / wsum), 1.0);
}`;

/**
 * Portrait retouch at full resolution, in source space.
 *   skin   : frequency separation - keep the base (shape, light) and the fine texture (pores),
 *            drop the band in between (blotches, blemishes); even out chroma; tame shine;
 *            dodge the under-eye area.
 *   detail : crisp eyes / brows / hair, whiter sclera and teeth, healthier lips.
 *   light  : per-pixel exposure map from the 3D face mesh (see lighting.ts).
 */
export const FS_COMPOSITE = `${HEAD}${COMMON}${ROI}
uniform sampler2D uSrc, uHalf, uQuarter, uEighth, uBase, uWide, uSkin, uGA, uGB, uMatteDbg;
uniform float uHasFace, uDebug;
uniform float uSmooth, uSmoothChroma, uTexKeep, uTexLevel, uEven, uShine, uDodge;
uniform float uRelight, uEye, uTeeth, uLip, uSharpen, uNoise;
uniform float uSkinY;
in vec2 vUv;
out vec4 o;
void main() {
  vec3 src = texture(uSrc, vUv).rgb;
  vec3 lowH = texture(uHalf, vUv).rgb;
  vec3 lowQ = texture(uQuarter, vUv).rgb;
  // Noise coring over two pyramid levels (only when the Director expects visible sensor noise):
  // small-amplitude differences are noise and are dropped, large ones are edges and are kept.
  if (uNoise > 0.0) {
    vec3 dH = src - lowH, dQ = lowH - lowQ;
    float kH = smoothstep(uNoise, 3.0 * uNoise, abs(luma(dH)));
    float kQ = smoothstep(0.5 * uNoise, 1.5 * uNoise, abs(luma(dQ)));
    lowH = lowQ + vec3(luma(dQ)) * kQ + (dQ - luma(dQ)) * kQ * 0.35;
    src = lowH + vec3(luma(dH)) * kH + (dH - luma(dH)) * kH * 0.25;
  }
  // Texture band = everything finer than a pore-sized blur; the pyramid level follows face size.
  vec3 lowT = uTexLevel < 1.0 ? mix(lowH, lowQ, uTexLevel) : mix(lowQ, texture(uEighth, vUv).rgb, uTexLevel - 1.0);
  vec3 detail = src - lowT;
  float skinW = texture(uSkin, vUv).r;
  vec3 base = texture(uBase, vUv).rgb;
  vec4 wideS = texture(uWide, vUv);
  vec3 wide = mix(base, wideS.rgb / max(wideS.a, 1e-3), smoothstep(0.12, 0.4, wideS.a));
  vec4 ga = vec4(0.5, 0.0, 0.0, 0.0);
  vec4 gb = vec4(0.0);
  if (uHasFace > 0.5) {
    ga = roiTex(uGA, vUv, ga);
    gb = roiTex(uGB, vUv, gb);
  }

  // --- skin ---
  // Colour corrections are made at constant brightness: 'tone' is the regional skin colour scaled
  // to this pixel's own luminance. It is only trusted where it is already close to the pixel
  // ('near'): across a shadow edge, next to hair or under coloured light the regional average is a
  // different colour for a good reason, and pulling toward it would paint the skin.
  vec3 b = base;
  float yb = max(luma(b), 1e-3);
  float yw = max(luma(wide), 1e-3);
  vec3 tone = wide * (yb / yw);
  vec3 dt = tone - b;
  float near = exp(-dot(dt, dt) / 0.006) * smoothstep(0.55, 0.85, min(yb, yw) / max(yb, yw));
  b = mix(b, tone, uEven * near);                     // even out blotchy colour
  float knee = uSkinY * 1.32;
  float over = max(yb - knee, 0.0);
  float ybn = (yb - over) + over / (1.0 + over * uShine * 5.0);  // identity below the knee
  b *= ybn / yb;                                      // tame greasy shine, keep the highlight's shape
  float dodge = ga.b * uDodge;
  float lift = clamp(yw / ybn, 1.0, 1.25);
  b = mix(b, mix(b, tone * (ybn / yb), 0.5 * near) * lift, dodge); // under-eye: toward the cheek's tone
  // Blemish band = what lies between the pore-scale blur and the base. Its COLOUR part is blotches,
  // redness and spots and is removed almost entirely; its LUMINANCE part also carries the modelling
  // of the face (smile lines, cheek and nose contours), so it is only softened - the difference
  // between retouched skin and a wax figure.
  vec3 mid = lowT - base;
  float midY = luma(mid);
  float sw = skinW;
  vec3 retouched = b + detail * uTexKeep + (1.0 - uSmooth) * vec3(midY) + (1.0 - uSmoothChroma) * (mid - vec3(midY));
  vec3 col = mix(src, retouched, sw);

  // --- crisp non-skin detail ---
  col += uSharpen * (src - lowH) * (1.0 - skinW);

  // --- eyes ---
  float y = luma(col);
  float sat = (max3(col) - min3(col)) / max(max3(col), 1e-3);
  float eye = gb.r;
  float sclera = eye * smoothstep(uSkinY * 0.72, uSkinY * 1.12, y) * (1.0 - smoothstep(0.16, 0.38, sat));
  col = mix(col, vec3(y), 0.22 * sclera * uEye);
  col *= 1.0 + 0.07 * sclera * uEye;
  col += (src - lowQ) * 0.55 * eye * (1.0 - sclera) * uEye;
  col *= 1.0 + 0.05 * eye * uEye;

  // --- teeth ---
  float teeth = gb.b * smoothstep(uSkinY * 0.72, uSkinY * 1.08, y) * (1.0 - smoothstep(0.22, 0.48, sat));
  col = mix(col, vec3(y) * vec3(0.99, 1.0, 1.03), 0.45 * teeth * uTeeth);
  col *= 1.0 + 0.09 * teeth * uTeeth;

  // --- lips ---
  y = luma(col);
  col = mix(vec3(y), col, 1.0 + 0.16 * gb.g * uLip);

  // --- relight (linear light) ---
  float stops = (ga.r - 0.5) * 4.0 * uRelight;
  col = toGam(toLin(clamp(col, 0.0, 1.0)) * exp2(stops));
  // Opened-up shadows are more saturated than lit skin; take a little colour back out.
  col = mix(vec3(luma(col)), col, 1.0 - 0.4 * clamp(stops, 0.0, 1.0));

  // Developer views of the intermediate maps (see Engine.debugView).
  if (uDebug > 0.5) {
    if (uDebug < 1.5) col = vec3(skinW);
    else if (uDebug < 2.5) col = vec3(ga.r, ga.g, ga.b);
    else if (uDebug < 3.5) col = gb.rgb + vec3(gb.a * 0.3);
    else if (uDebug < 4.5) col = base;
    else if (uDebug < 5.5) col = vec3(texture(uMatteDbg, vUv).r);
    else if (uDebug < 6.5) col = wide;
    else col = vec3(0.5) + detail * 4.0;
  }

  o = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

/** Final pass runs in OUTPUT space on a warp grid: reframing + lens-geometry correction + grade. */
export const VS_FINAL = `${HEAD}
in vec2 aPos;     // output uv
in vec2 aSrcUv;   // where this output point samples the source frame
uniform float uFlipY;
out vec2 vSrc;
out vec2 vOut;
void main() {
  vSrc = aSrcUv;
  vOut = aPos;
  gl_Position = vec4(aPos.x * 2.0 - 1.0, (aPos.y * 2.0 - 1.0) * uFlipY, 0.0, 1.0);
}`;

export const FS_FINAL = `${HEAD}${COMMON}
uniform sampler2D uBeauty, uBokeh, uMatte, uGlow;
uniform float uHasMatte, uBgDim, uBgDesat;
uniform vec3 uWb;
uniform float uExposure, uShoulder, uContrast, uShadows, uSat, uVibrance, uFade;
uniform vec3 uShadowTint, uHighTint;
uniform float uGlowAmt, uVignette, uGrain, uSeed;
uniform vec2 uVigCenter, uOutSize;
in vec2 vSrc;
in vec2 vOut;
out vec4 o;

float hash(vec2 p) {
  vec3 p3 = fract(vec3(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  vec3 lin = toLin(texture(uBeauty, vSrc).rgb);

  // Subject / background separation: defocus, slightly darker, slightly less saturated.
  if (uHasMatte > 0.5) {
    float fg = texture(uMatte, vSrc).r;
    vec3 bg = toLin(texture(uBokeh, vSrc).rgb);
    bg = mix(bg, vec3(luma(bg)), uBgDesat) * exp2(-uBgDim);
    lin = mix(bg, lin, fg);
  }

  // Diffusion glow: highlights bleed softly, like a mist filter in front of the lens.
  vec3 glow = toLin(texture(uGlow, vSrc).rgb);
  lin += uGlowAmt * glow * smoothstep(0.3, 1.0, luma(glow));
  lin /= 1.0 + 0.2 * uGlowAmt;

  // White balance and exposure in linear light.
  lin *= uWb * exp2(uExposure);

  // Highlight shoulder: identity below the knee, smooth roll-off above (no hard clipping on skin).
  float peak = mix(luma(lin), max3(lin), 0.6);
  float t = uShoulder;
  float mapped = peak <= t ? peak : t + (1.0 - t) * (1.0 - exp(-(peak - t) / (1.0 - t)));
  lin *= mapped / max(peak, 1e-4);

  vec3 g = toGam(clamp(lin, 0.0, 1.0));

  // Tone curve on luminance only, colour carried along at constant chroma ratio: per-channel
  // curves would shift lit and shadowed skin to different hues.
  float y = max(luma(g), 1e-4);
  float yt = y + uShadows * 2.6 * y * (1.0 - y) * (1.0 - y) * (1.0 - y);   // open the shadows (fill)
  yt = mix(yt, yt * yt * (3.0 - 2.0 * yt), uContrast);                      // S-curve
  yt = uFade + yt * (1.0 - uFade);
  g *= yt / y;
  // Scaling can push a saturated channel past white: fold the excess back toward neutral.
  float mx = max3(g);
  if (mx > 1.0) g = mix(vec3(yt), g, (1.0 - yt) / max(mx - yt, 1e-4));

  // Colour.
  y = luma(g);
  float sat = max3(g) - min3(g);
  g = mix(vec3(y), g, 1.0 + uSat + uVibrance * (1.0 - sat));
  g += uShadowTint * (1.0 - y) * (1.0 - y) + uHighTint * y * y;

  // Vignette centred on the subject, in an aspect-corrected circle.
  vec2 dv = (vOut - uVigCenter) * vec2(uOutSize.x / uOutSize.y, 1.0);
  g *= 1.0 - uVignette * smoothstep(0.30, 1.05, length(dv));

  // Fine luminance grain: restores a photographic surface and hides 8-bit banding.
  vec2 px = vOut * uOutSize;
  float n = hash(px + uSeed) + hash(px * 0.5 + uSeed * 1.7) - 1.0;
  g += uGrain * n * (0.35 + 1.3 * y * (1.0 - y));

  o = vec4(clamp(g, 0.0, 1.0), 1.0);
}`;
