// On-device vision models (MediaPipe Tasks, WASM + WebGL): a 478-point 3D face mesh with expression
// blendshapes, and a multi-class selfie segmenter (hair / skin / clothes / background).
// Nothing leaves the device: the models and the runtime are bundled with the app.
import { FaceLandmarker, FilesetResolver, ImageSegmenter } from '@mediapipe/tasks-vision';
import type { SegmentationMask } from '../engine/engine';
import { type FaceGeometry, buildFaceGeometry } from '../engine/faceModel';
import { NUM_LANDMARKS } from './topology';

export interface VisionOptions {
  /** URL prefix of the app's static assets (contains wasm/ and models/). */
  baseUrl: string;
  /** VIDEO tracks across frames (live camera); IMAGE treats every call independently. */
  mode: 'VIDEO' | 'IMAGE';
  delegate?: 'GPU' | 'CPU';
}

export interface FaceObservation {
  face: FaceGeometry;
  /** Expression blendshape scores by name (eyeBlinkLeft, mouthSmileRight, ...). */
  blend: Record<string, number>;
}

const SEG_LONG_SIDE = 256;
// Class indices of selfie_multiclass_256x256.
const SEG_BACKGROUND = 0, SEG_HAIR = 1, SEG_BODY_SKIN = 2, SEG_FACE_SKIN = 3, SEG_CLOTHES = 4;

export class Vision {
  private segCanvas: HTMLCanvasElement | OffscreenCanvas;
  private segCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D;
  private segVersion = 0;
  private segPrev: Uint8Array | null = null;
  private lm = new Float32Array(NUM_LANDMARKS * 3);

  private constructor(
    private landmarker: FaceLandmarker,
    private segmenter: ImageSegmenter | null,
    private mode: 'VIDEO' | 'IMAGE',
    readonly delegate: 'GPU' | 'CPU',
  ) {
    this.segCanvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(4, 4) : document.createElement('canvas');
    this.segCtx = this.segCanvas.getContext('2d', { willReadFrequently: false }) as CanvasRenderingContext2D;
  }

  static async create(opts: VisionOptions): Promise<Vision> {
    const fileset = await FilesetResolver.forVisionTasks(`${opts.baseUrl}wasm`);
    const build = async (delegate: 'GPU' | 'CPU') => {
      const landmarker = await FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: `${opts.baseUrl}models/face_landmarker.task`, delegate },
        runningMode: opts.mode,
        numFaces: 1,
        outputFaceBlendshapes: true,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      let segmenter: ImageSegmenter | null = null;
      try {
        segmenter = await ImageSegmenter.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: `${opts.baseUrl}models/selfie_multiclass_256x256.tflite`, delegate },
          runningMode: opts.mode,
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        });
      } catch (err) {
        // The pipeline degrades gracefully without segmentation (no bokeh, mesh-only skin mask).
        console.warn('Segmentation unavailable:', err);
      }
      return new Vision(landmarker, segmenter, opts.mode, delegate);
    };
    if (opts.delegate) return build(opts.delegate);
    try {
      return await build('GPU');
    } catch (err) {
      console.warn('GPU delegate failed, falling back to CPU:', err);
      return build('CPU');
    }
  }

  get hasSegmentation(): boolean {
    return this.segmenter !== null;
  }

  /**
   * Detect the primary face.
   * @param mirror true when the engine shows this frame mirrored; landmarks are returned in that space.
   */
  detectFace(source: TexImageSource, width: number, height: number, mirror: boolean, timestampMs: number): FaceObservation | null {
    const res = this.mode === 'VIDEO'
      ? this.landmarker.detectForVideo(source as HTMLVideoElement, timestampMs)
      : this.landmarker.detect(source as HTMLImageElement);
    const pts = res.faceLandmarks?.[0];
    if (!pts || pts.length < 468) return null;
    const lm = this.lm;
    const n = Math.min(pts.length, NUM_LANDMARKS);
    for (let i = 0; i < n; i++) {
      lm[i * 3] = mirror ? 1 - pts[i].x : pts[i].x;
      lm[i * 3 + 1] = pts[i].y;
      lm[i * 3 + 2] = pts[i].z;
    }
    const blend: Record<string, number> = {};
    for (const c of res.faceBlendshapes?.[0]?.categories ?? []) blend[c.categoryName] = c.score;
    return { face: buildFaceGeometry(n === NUM_LANDMARKS ? lm : lm.subarray(0, n * 3), width, height), blend };
  }

  /**
   * Segment the frame into person / hair / skin / clothes. Returns null if unavailable.
   * @param gain brightness multiplier applied to the model's input only: the segmenter is far more
   *   reliable on a normally exposed image than on a dim or backlit one.
   */
  segment(
    source: CanvasImageSource, width: number, height: number, mirror: boolean, timestampMs: number, gain = 1,
  ): SegmentationMask | null {
    if (!this.segmenter) return null;
    const scale = SEG_LONG_SIDE / Math.max(width, height);
    const w = Math.max(8, Math.round(width * scale)), h = Math.max(8, Math.round(height * scale));
    if (this.segCanvas.width !== w || this.segCanvas.height !== h) {
      this.segCanvas.width = w;
      this.segCanvas.height = h;
      this.segPrev = null;
    }
    const ctx = this.segCtx;
    ctx.setTransform(mirror ? -1 : 1, 0, 0, 1, mirror ? w : 0, 0);
    ctx.globalCompositeOperation = 'copy';
    ctx.globalAlpha = 1;
    ctx.drawImage(source, 0, 0, w, h);
    // Additive re-draws multiply brightness (works on every canvas implementation, unlike filters).
    ctx.globalCompositeOperation = 'lighter';
    for (let g = Math.min(4, gain) - 1; g > 0.05; g -= 1) {
      ctx.globalAlpha = Math.min(1, g);
      ctx.drawImage(source, 0, 0, w, h);
    }
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    const res = this.mode === 'VIDEO'
      ? this.segmenter.segmentForVideo(this.segCanvas, timestampMs)
      : this.segmenter.segment(this.segCanvas);
    const masks = res.confidenceMasks;
    if (!masks || masks.length <= SEG_CLOTHES) {
      res.close();
      return null;
    }
    const mw = masks[0].width, mh = masks[0].height;
    const bg = masks[SEG_BACKGROUND].getAsFloat32Array();
    const hair = masks[SEG_HAIR].getAsFloat32Array();
    const body = masks[SEG_BODY_SKIN].getAsFloat32Array();
    const faceSkin = masks[SEG_FACE_SKIN].getAsFloat32Array();
    const clothes = masks[SEG_CLOTHES].getAsFloat32Array();
    const data = new Uint8Array(mw * mh * 4);
    for (let i = 0, k = 0; i < mw * mh; i++, k += 4) {
      data[k] = (1 - bg[i]) * 255;
      data[k + 1] = hair[i] * 255;
      data[k + 2] = Math.min(1, body[i] + faceSkin[i]) * 255;
      data[k + 3] = clothes[i] * 255;
    }
    res.close();
    // Light temporal smoothing against frame-to-frame mask flicker in live mode.
    const prev = this.segPrev;
    if (this.mode === 'VIDEO' && prev && prev.length === data.length) {
      for (let i = 0; i < data.length; i++) data[i] = (data[i] * 3 + prev[i]) >> 2;
    }
    this.segPrev = data;
    return { data, width: mw, height: mh, version: ++this.segVersion };
  }

  close(): void {
    this.landmarker.close();
    this.segmenter?.close();
  }
}
