// The GPU imaging pipeline. One instance owns one WebGL2 context and renders both the live preview
// and the full-resolution capture with the same passes, so what the viewfinder shows is what is saved.
//
//   source space                                             output space
//   ────────────                                             ────────────
//   ingest ─► pyramid (1/2, 1/4, 1/8) ─► analysis read-back
//   face mesh ─► G-buffers (relight gain, regions)
//   segmentation ─► guided-filter matte
//   skin mask ─► masked bilateral base ─► wide tone base
//   bokeh (masked disc blur)
//   composite (retouch + local + relight) ───────────────►  final: reframe + lens warp + grade
import { FullscreenTriangle, type GL, Program, Target, createTexture } from './gl';
import { MESH_STRIDE } from './meshBuffer';
import { type Crop, NEUTRAL_GRADE, type RenderParams, WARP_COLS, WARP_ROWS } from './params';
import * as S from './shaders';

export interface SegmentationMask {
  /** RGBA8: r = person, g = hair, b = skin (face + body), a = clothes. Row 0 is the top. */
  data: Uint8Array;
  width: number;
  height: number;
  /** Bumped whenever data changes, so the engine re-uploads only when needed. */
  version: number;
}

export interface AnalysisPixels {
  data: Uint8Array;
  width: number;
  height: number;
}

export type RenderOutput = { kind: 'screen' } | { kind: 'capture'; width: number; height: number };

export const DEBUG_VIEWS = ['none', 'skin', 'light', 'regions', 'base', 'matte', 'wide', 'texture'] as const;
export type DebugView = (typeof DEBUG_VIEWS)[number];

const ANALYSIS_H = 256;
/** Spatial sigma of the skin base layer as a fraction of the inter-ocular distance. */
const BASE_SIGMA_IOD = 0.055;
/** Range sigma (gamma luma) of the skin base layer: larger steps than this are treated as form, not blemish. */
const BASE_SIGMA_RANGE = 0.075;

export class Engine {
  readonly gl: GL;
  private tri: FullscreenTriangle;
  private floatTargets: boolean;

  private pCopy: Program;
  private pBlur: Program;
  private pMesh: Program;
  private pSkinMask: Program;
  private pBilateral: Program;
  private pPremult: Program;
  private pGfPack: Program;
  private pGfAB: Program;
  private pGfApply: Program;
  private pMatteSimple: Program;
  private pBokeh: Program;
  private pComposite: Program;
  private pFinal: Program;

  private rawTex: WebGLTexture;
  private segTex: WebGLTexture;
  private segVersion = -1;
  private hasSeg = false;

  private src: Target;
  private half: Target;
  private quarter: Target;
  private eighth: Target;
  private analysis: Target;
  private gbuf: Target;
  private gA: Target;
  private gB: Target;
  private gTmp: Target;
  private gfA: Target | null = null;
  private gfB: Target | null = null;
  private gfAB: Target | null = null;
  private matte: Target;
  private skinMask: Target;
  private bil: Target;
  private base: Target;
  private wideA: Target;
  private wideB: Target;
  private bokehA: Target;
  private bokehB: Target;
  private glowA: Target;
  private glowB: Target;
  private beauty: Target;
  private capture: Target;

  private meshVao: WebGLVertexArrayObject;
  private meshBuf: WebGLBuffer;
  private gridVao: WebGLVertexArrayObject;
  private gridUvBuf: WebGLBuffer;
  private gridUv: Float32Array;
  private gridIndexCount: number;

  /** Developer aid: show an intermediate map instead of the graded picture. */
  debugView: DebugView = 'none';

  private analysisPixels: Uint8Array = new Uint8Array(0);
  private seed = 0;

  constructor(readonly canvas: HTMLCanvasElement | OffscreenCanvas) {
    const gl = canvas.getContext('webgl2', {
      alpha: false, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: 'high-performance',
    }) as GL | null;
    if (!gl) throw new Error('WebGL2 is not available on this device');
    this.gl = gl;
    this.floatTargets = !!(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    gl.disable(gl.BLEND);

    this.tri = new FullscreenTriangle(gl);
    const full = (fs: string) => new Program(gl, S.VS_FULL, fs);
    this.pCopy = full(S.FS_COPY);
    this.pBlur = full(S.FS_BLUR);
    this.pMesh = new Program(gl, S.VS_MESH, S.FS_MESH, ['aPos', 'aA', 'aB']);
    this.pSkinMask = full(S.FS_SKINMASK);
    this.pBilateral = full(S.FS_BILATERAL);
    this.pPremult = full(S.FS_PREMULT);
    this.pGfPack = full(S.FS_GF_PACK);
    this.pGfAB = full(S.FS_GF_AB);
    this.pGfApply = full(S.FS_GF_APPLY);
    this.pMatteSimple = full(S.FS_MATTE_SIMPLE);
    this.pBokeh = full(S.FS_BOKEH);
    this.pComposite = full(S.FS_COMPOSITE);
    this.pFinal = new Program(gl, S.VS_FINAL, S.FS_FINAL, ['aPos', 'aSrcUv']);

    this.rawTex = createTexture(gl);
    this.segTex = createTexture(gl);
    const t = (o = {}) => new Target(gl, o);
    this.src = t(); this.half = t(); this.quarter = t(); this.eighth = t(); this.analysis = t();
    this.gbuf = t({ attachments: 2, depth: true }); this.gA = t(); this.gB = t(); this.gTmp = t();
    if (this.floatTargets) {
      this.gfA = t({ float: true }); this.gfB = t({ float: true }); this.gfAB = t({ float: true });
    }
    this.matte = t(); this.skinMask = t(); this.bil = t(); this.base = t();
    this.wideA = t({ float: this.floatTargets }); this.wideB = t({ float: this.floatTargets }); this.bokehA = t(); this.bokehB = t();
    this.glowA = t(); this.glowB = t(); this.beauty = t(); this.capture = t();

    // Face mesh geometry (dynamic).
    this.meshVao = gl.createVertexArray()!;
    this.meshBuf = gl.createBuffer()!;
    gl.bindVertexArray(this.meshVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuf);
    const stride = MESH_STRIDE * 4;
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, stride, 12);
    gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 4, gl.FLOAT, false, stride, 20);

    // Warp grid for the final pass: static output positions, dynamic source uvs.
    const cols = WARP_COLS, rows = WARP_ROWS;
    const pos = new Float32Array((cols + 1) * (rows + 1) * 2);
    for (let j = 0, k = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) { pos[k++] = i / cols; pos[k++] = j / rows; }
    const idx = new Uint16Array(cols * rows * 6);
    for (let j = 0, k = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const a = j * (cols + 1) + i, b = a + 1, c = a + cols + 1, d = c + 1;
        idx[k++] = a; idx[k++] = b; idx[k++] = c; idx[k++] = b; idx[k++] = d; idx[k++] = c;
      }
    }
    this.gridIndexCount = idx.length;
    this.gridUv = new Float32Array(pos.length);
    this.gridVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.gridVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.gridUvBuf = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridUvBuf);
    gl.bufferData(gl.ARRAY_BUFFER, this.gridUv, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, gl.createBuffer());
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
    gl.bindVertexArray(null);
  }

  get maxTextureSize(): number {
    return this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number;
  }

  /** Working (source-space) resolution of the frame currently loaded. */
  get workWidth(): number { return this.src.w; }
  get workHeight(): number { return this.src.h; }

  private pass(target: Target, prog: Program): Program {
    target.bind();
    return prog.use();
  }

  /**
   * Separable Gaussian blur of the given sigma (in pixels of `out`). The 9-tap kernel only stays
   * alias-free for steps up to ~1.5 texels, so wide blurs are built from several narrow passes
   * (variances add) instead of one sparse one.
   */
  private blur(input: WebGLTexture, tmp: Target, out: Target, sigmaPx: number): void {
    const BASE_SIGMA = 1.8; // sigma of the kernel at a step of one texel
    const passes = Math.min(4, Math.max(1, Math.ceil((sigmaPx / (BASE_SIGMA * 1.5)) ** 2)));
    const step = sigmaPx / (BASE_SIGMA * Math.sqrt(passes));
    let src = input;
    for (let i = 0; i < passes; i++) {
      this.pass(tmp, this.pBlur).tex('uTex', 0, src).f('uDir', step / tmp.w, 0);
      this.tri.draw();
      this.pass(out, this.pBlur).tex('uTex', 0, tmp.tex).f('uDir', 0, step / out.h);
      this.tri.draw();
      src = out.tex;
    }
  }

  /**
   * Load a frame: upload, optionally mirror, resample to the working size and build the pyramid.
   * @param workLong long side of the working resolution in pixels.
   */
  ingest(source: TexImageSource, srcW: number, srcH: number, mirror: boolean, workLong: number): void {
    const gl = this.gl;
    const scale = Math.min(1, workLong / Math.max(srcW, srcH), this.maxTextureSize / Math.max(srcW, srcH));
    const w = Math.max(16, Math.round((srcW * scale) / 8) * 8);
    const h = Math.max(16, Math.round((srcH * scale) / 8) * 8);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.rawTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);

    this.src.resize(w, h);
    this.half.resize(w / 2, h / 2);
    this.quarter.resize(w / 4, h / 4);
    this.eighth.resize(w / 8, h / 8);
    this.analysis.resize(Math.round((ANALYSIS_H * w) / h / 2) * 2, ANALYSIS_H);

    this.pass(this.src, this.pCopy).tex('uTex', 0, this.rawTex).f('uMirror', mirror ? 1 : 0);
    this.tri.draw();
    for (const [from, to] of [[this.src, this.half], [this.half, this.quarter], [this.quarter, this.eighth]] as const) {
      this.pass(to, this.pCopy).tex('uTex', 0, from.tex).f('uMirror', 0);
      this.tri.draw();
    }
    this.pass(this.analysis, this.pCopy).tex('uTex', 0, this.quarter.w >= this.analysis.w ? this.quarter.tex : this.half.tex).f('uMirror', 0);
    this.tri.draw();
  }

  /** Small RGBA copy of the current frame for CPU-side scene analysis (row 0 = top). */
  readAnalysis(): AnalysisPixels {
    const gl = this.gl, a = this.analysis;
    if (this.analysisPixels.length !== a.w * a.h * 4) this.analysisPixels = new Uint8Array(a.w * a.h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, a.fbo);
    gl.readPixels(0, 0, a.w, a.h, gl.RGBA, gl.UNSIGNED_BYTE, this.analysisPixels);
    return { data: this.analysisPixels, width: a.w, height: a.h };
  }

  setSegmentation(seg: SegmentationMask | null): void {
    this.hasSeg = !!seg;
    if (!seg || seg.version === this.segVersion) return;
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.segTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, seg.width, seg.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, seg.data);
    this.segVersion = seg.version;
  }

  /** Render the frame loaded by ingest(). Returns RGBA pixels (row 0 = top) for capture output. */
  render(p: RenderParams, out: RenderOutput): Uint8Array | null {
    const gl = this.gl;
    const face = p.face;
    const hasFace = face ? 1 : 0;
    const roi = face ? face.roi : [0, 0, 1, 1];
    const W = this.src.w, H = this.src.h;
    const hiRes = out.kind === 'capture';

    // 1. Face mesh -> G-buffers (face-ROI space), then soften.
    if (face) {
      const gsize = hiRes ? 512 : 256;
      this.gbuf.resize(gsize, gsize); this.gA.resize(gsize, gsize); this.gB.resize(gsize, gsize); this.gTmp.resize(gsize, gsize);
      this.gbuf.bind();
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.clearBufferfv(gl.COLOR, 0, [0.5, 0, 0, 0]);
      gl.clearBufferfv(gl.COLOR, 1, [0, 0, 0, 0]);
      gl.clearBufferfv(gl.DEPTH, 0, [1]);
      this.pMesh.use().f('uRoi', roi[0], roi[1], roi[2], roi[3]);
      gl.bindVertexArray(this.meshVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.meshBuf);
      gl.bufferData(gl.ARRAY_BUFFER, face.vertices, gl.DYNAMIC_DRAW);
      gl.drawArrays(gl.TRIANGLES, 0, face.vertexCount);
      gl.disable(gl.DEPTH_TEST);
      // Lighting gain varies slowly: wide blur. Region masks need tighter edges: narrow blur.
      this.blur(this.gbuf.texs[0], this.gTmp, this.gA, gsize * 0.03);
      this.blur(this.gbuf.texs[1], this.gTmp, this.gB, gsize * 0.006);
    }

    // 2. Segmentation -> edge-aware matte at half resolution.
    this.matte.resize(W / 2, H / 2);
    if (this.hasSeg) {
      if (this.gfA && this.gfB && this.gfAB) {
        const lw = this.quarter.w, lh = this.quarter.h;
        this.gfA.resize(lw, lh); this.gfB.resize(lw, lh); this.gfAB.resize(lw, lh);
        const r = Math.max(1.5, lw * 0.011);
        this.pass(this.gfA, this.pGfPack).tex('uColor', 0, this.quarter.tex).tex('uSeg', 1, this.segTex);
        this.tri.draw();
        this.blur(this.gfA.tex, this.gfB, this.gfA, r);
        this.pass(this.gfAB, this.pGfAB).tex('uMean', 0, this.gfA.tex).f('uEps', 0.0025);
        this.tri.draw();
        this.blur(this.gfAB.tex, this.gfB, this.gfAB, r);
        this.pass(this.matte, this.pGfApply).tex('uColor', 0, this.half.tex).tex('uAB', 1, this.gfAB.tex);
      } else {
        this.pass(this.matte, this.pMatteSimple).tex('uSeg', 0, this.segTex);
      }
      this.tri.draw();
    }

    // 3. Skin mask and frequency-separation base layers.
    const qw = this.quarter.w, qh = this.quarter.h;
    this.skinMask.resize(qw, qh); this.bil.resize(qw, qh); this.base.resize(qw, qh);
    this.pass(this.skinMask, this.pSkinMask)
      // Colour statistics come from the 1/8 level: robust against sensor noise in dim light.
      .tex('uColor', 0, this.eighth.tex).tex('uGB', 1, this.gB.tex).tex('uSeg', 2, this.segTex)
      .f('uRoi', roi[0], roi[1], roi[2], roi[3]).f('uHasFace', hasFace).f('uHasSeg', this.hasSeg ? 1 : 0)
      .f('uSkinYcc', p.skin.ycc[0], p.skin.ycc[1], p.skin.ycc[2]).f('uChromaTol', p.skin.chromaTol);
    this.tri.draw();

    // Blur radius follows the face: a fixed fraction of the inter-ocular distance, in quarter-res pixels.
    const iodQ = (face ? face.iodUv : 0.2) * qw;
    const stepQ = Math.max(0.5, (iodQ * BASE_SIGMA_IOD) / 3.24);
    this.pass(this.bil, this.pBilateral).tex('uColor', 0, this.quarter.tex).tex('uMask', 1, this.skinMask.tex)
      .f('uDir', stepQ / qw, 0).f('uSigmaR', BASE_SIGMA_RANGE);
    this.tri.draw();
    this.pass(this.base, this.pBilateral).tex('uColor', 0, this.bil.tex).tex('uMask', 1, this.skinMask.tex)
      .f('uDir', 0, stepQ / qh).f('uSigmaR', BASE_SIGMA_RANGE);
    this.tri.draw();

    // Wide regional skin tone (normalised convolution over skin only) at 1/8 resolution.
    const ew = this.eighth.w, eh = this.eighth.h;
    this.wideA.resize(ew, eh); this.wideB.resize(ew, eh);
    this.pass(this.wideA, this.pPremult).tex('uColor', 0, this.base.tex).tex('uMask', 1, this.skinMask.tex);
    this.tri.draw();
    this.blur(this.wideA.tex, this.wideB, this.wideA, Math.max(1.5, (iodQ / 2) * 0.22));

    // 4. Background bokeh (quarter res) and diffusion glow (eighth res).
    this.bokehA.resize(qw, qh); this.bokehB.resize(qw, qh);
    const doBokeh = this.hasSeg && p.background.blur > 0;
    if (doBokeh) {
      this.pass(this.bokehB, this.pBokeh).tex('uColor', 0, this.quarter.tex).tex('uMatte', 1, this.matte.tex)
        .f('uRadius', p.background.blur, (p.background.blur * W) / H);
      this.tri.draw();
      this.blur(this.bokehB.tex, this.bokehA, this.bokehB, 1.1);
    }
    this.glowA.resize(ew, eh); this.glowB.resize(ew, eh);
    if (p.grade.glow > 0) {
      this.blur(this.eighth.tex, this.glowB, this.glowA, ew * 0.045);
    }

    // 5. Retouch composite at full working resolution.
    this.beauty.resize(W, H);
    const iodFull = (face ? face.iodUv : 0.2) * W;
    // Pore-scale blur ~1.1% of the inter-ocular distance; pyramid level k blurs by ~0.8 * 2^k px.
    const texLevel = Math.min(2, Math.max(0, Math.log2(Math.max(1e-3, (0.011 * iodFull) / 0.8))));
    this.pass(this.beauty, this.pComposite)
      .tex('uSrc', 0, this.src.tex).tex('uHalf', 1, this.half.tex).tex('uQuarter', 2, this.quarter.tex)
      .tex('uBase', 3, this.base.tex).tex('uWide', 4, this.wideA.tex).tex('uSkin', 5, this.skinMask.tex)
      .tex('uGA', 6, this.gA.tex).tex('uGB', 7, this.gB.tex).tex('uMatteDbg', 8, this.matte.tex).tex('uEighth', 9, this.eighth.tex)
      .f('uDebug', DEBUG_VIEWS.indexOf(this.debugView))
      .f('uRoi', roi[0], roi[1], roi[2], roi[3]).f('uHasFace', hasFace)
      .f('uSmooth', p.skin.smooth).f('uSmoothChroma', p.skin.smoothChroma).f('uTexKeep', p.skin.texKeep).f('uTexLevel', texLevel)
      .f('uEven', p.skin.even).f('uShine', p.skin.shine).f('uDodge', p.skin.dodge)
      .f('uRelight', p.local.relight).f('uEye', p.local.eye).f('uTeeth', p.local.teeth)
      .f('uLip', p.local.lip).f('uSharpen', p.local.sharpen).f('uNoise', p.local.denoise).f('uSkinY', p.skin.ycc[0]);
    this.tri.draw();

    // 6. Final: reframe + lens-geometry warp + grade, in output space.
    if (this.debugView !== 'none') return this.finalPass(this.beauty.tex, p.crop, p.warp, null, false, out);
    return this.finalPass(this.beauty.tex, p.crop, p.warp, p, doBokeh, out);
  }

  /** The unprocessed frame through the same crop: the honest "before" for comparisons. */
  renderOriginal(crop: Crop, out: RenderOutput): Uint8Array | null {
    return this.finalPass(this.src.tex, crop, null, null, false, out);
  }

  private finalPass(
    colour: WebGLTexture, crop: Crop, warp: Float32Array | null, p: RenderParams | null, matte: boolean, out: RenderOutput,
  ): Uint8Array | null {
    const gl = this.gl;
    const uv = this.gridUv;
    for (let j = 0, k = 0; j <= WARP_ROWS; j++) {
      for (let i = 0; i <= WARP_COLS; i++, k += 2) {
        uv[k] = crop.x + (i / WARP_COLS) * crop.w - (warp ? warp[k] : 0);
        uv[k + 1] = crop.y + (j / WARP_ROWS) * crop.h - (warp ? warp[k + 1] : 0);
      }
    }
    let ow: number, oh: number;
    if (out.kind === 'capture') {
      this.capture.resize(out.width, out.height);
      this.capture.bind();
      ow = out.width; oh = out.height;
    } else {
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      ow = gl.drawingBufferWidth; oh = gl.drawingBufferHeight;
      gl.viewport(0, 0, ow, oh);
    }
    const g = p ? p.grade : NEUTRAL_GRADE;
    const bg = p ? p.background : { blur: 0, dim: 0, desat: 0 };
    this.seed = (this.seed + 17.13) % 1000;
    this.pFinal.use()
      .tex('uBeauty', 0, colour).tex('uBokeh', 1, this.bokehB.tex).tex('uMatte', 2, this.matte.tex).tex('uGlow', 3, this.glowA.tex)
      .f('uFlipY', out.kind === 'capture' ? 1 : -1)
      .f('uHasMatte', matte ? 1 : 0).f('uBgDim', bg.dim).f('uBgDesat', bg.desat)
      .f('uWb', g.wb[0], g.wb[1], g.wb[2]).f('uExposure', g.exposure).f('uShoulder', g.shoulder)
      .f('uContrast', g.contrast).f('uShadows', g.shadows).f('uSat', g.sat).f('uVibrance', g.vibrance).f('uFade', g.fade)
      .f('uShadowTint', g.shadowTint[0], g.shadowTint[1], g.shadowTint[2])
      .f('uHighTint', g.highTint[0], g.highTint[1], g.highTint[2])
      .f('uGlowAmt', g.glow).f('uVignette', g.vignette).f('uVigCenter', g.vigCenter[0], g.vigCenter[1])
      .f('uGrain', g.grain).f('uSeed', out.kind === 'capture' ? 7.7 : this.seed).f('uOutSize', ow, oh);
    gl.bindVertexArray(this.gridVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.gridUvBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, uv);
    gl.drawElements(gl.TRIANGLES, this.gridIndexCount, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);

    if (out.kind !== 'capture') return null;
    const pixels = new Uint8Array(ow * oh * 4);
    gl.readPixels(0, 0, ow, oh, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return pixels;
  }
}
