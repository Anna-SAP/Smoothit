// Camera stream management. The web camera API is the portable baseline; where the platform exposes
// more (exposure compensation through MediaStream Image Capture), it is used as a progressive
// enhancement so that real photons, not digital gain, fix an under-exposed face.

export type Facing = 'user' | 'environment';

export class CameraError extends Error {
  constructor(readonly kind: 'denied' | 'unavailable' | 'insecure', message: string) {
    super(message);
  }
}

interface ExposureCaps {
  min: number;
  max: number;
  step: number;
}

export class Camera {
  readonly video: HTMLVideoElement;
  facing: Facing = 'user';
  private stream: MediaStream | null = null;
  private exposureCaps: ExposureCaps | null = null;
  private exposureBias = 0;

  constructor() {
    const v = document.createElement('video');
    v.playsInline = true;
    v.muted = true;
    v.autoplay = true;
    v.setAttribute('playsinline', '');
    v.setAttribute('muted', '');
    // WebKit only keeps decoding a camera stream for a video element that is in the document and
    // not display:none, so keep it attached but invisible; the picture is drawn by the engine.
    v.style.cssText = 'position:fixed;left:0;top:0;width:2px;height:2px;opacity:0;pointer-events:none;z-index:-1';
    v.setAttribute('aria-hidden', 'true');
    document.body.appendChild(v);
    this.video = v;
  }

  get width(): number {
    return this.video.videoWidth;
  }

  get height(): number {
    return this.video.videoHeight;
  }

  /** Front-camera frames are shown and saved mirrored, as people know themselves from mirrors. */
  get mirrored(): boolean {
    return this.facing === 'user';
  }

  get ready(): boolean {
    return this.video.readyState >= 2 && this.video.videoWidth > 0;
  }

  private async open(facing: Facing, w: number, h: number): Promise<MediaStream> {
    return navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: facing }, width: { ideal: w }, height: { ideal: h }, frameRate: { ideal: 30 } },
    });
  }

  async start(facing: Facing = this.facing): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new CameraError(window.isSecureContext ? 'unavailable' : 'insecure', 'Camera API not available');
    }
    this.stop();
    this.facing = facing;
    let stream: MediaStream;
    try {
      // Ask for the sensor's native 4:3 at high resolution. Browsers disagree on whether the
      // constraint is expressed in sensor or screen orientation, so if the first answer is small
      // try the transposed request and keep whichever stream has more pixels.
      stream = await this.open(facing, 2560, 1920);
      const s = stream.getVideoTracks()[0]?.getSettings() ?? {};
      if ((s.width ?? 0) * (s.height ?? 0) < 1.4e6) {
        try {
          const alt = await this.open(facing, 1920, 2560);
          const a = alt.getVideoTracks()[0]?.getSettings() ?? {};
          const better = (a.width ?? 0) * (a.height ?? 0) > (s.width ?? 0) * (s.height ?? 0);
          (better ? stream : alt).getTracks().forEach((t) => t.stop());
          if (better) stream = alt;
        } catch {
          /* keep the first stream */
        }
      }
    } catch (err) {
      const name = (err as DOMException)?.name;
      if (name === 'NotAllowedError' || name === 'SecurityError') throw new CameraError('denied', 'Camera permission denied');
      throw new CameraError('unavailable', `Camera unavailable: ${name ?? err}`);
    }
    this.stream = stream;
    this.video.srcObject = stream;
    await this.video.play().catch(() => undefined);
    if (!this.ready) {
      await new Promise<void>((resolve) => {
        const done = () => { this.video.removeEventListener('loadeddata', done); resolve(); };
        this.video.addEventListener('loadeddata', done);
        setTimeout(done, 3000);
      });
    }
    this.probeExposure();
  }

  stop(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.exposureCaps = null;
    this.exposureBias = 0;
  }

  private probeExposure(): void {
    const track = this.stream?.getVideoTracks()[0];
    const caps = track?.getCapabilities?.() as (MediaTrackCapabilities & { exposureCompensation?: ExposureCaps }) | undefined;
    const ec = caps?.exposureCompensation;
    this.exposureCaps = ec && ec.max > ec.min ? { min: ec.min, max: ec.max, step: ec.step || 0.1 } : null;
  }

  get supportsExposureBias(): boolean {
    return this.exposureCaps !== null;
  }

  /**
   * Nudge the sensor's exposure toward what the face needs (in stops, relative to the camera's own
   * metering). Slow, bounded steps: this is an outer loop around the pipeline's digital exposure.
   * @returns the bias now applied
   */
  nudgeExposure(deltaStops: number): number {
    const caps = this.exposureCaps;
    const track = this.stream?.getVideoTracks()[0];
    if (!caps || !track) return 0;
    const next = Math.min(caps.max, Math.max(caps.min, this.exposureBias + deltaStops));
    const stepped = Math.round(next / caps.step) * caps.step;
    if (Math.abs(stepped - this.exposureBias) >= caps.step * 0.5) {
      this.exposureBias = stepped;
      track
        .applyConstraints({ advanced: [{ exposureCompensation: stepped } as MediaTrackConstraintSet] })
        .catch(() => { this.exposureCaps = null; });
    }
    return this.exposureBias;
  }
}
