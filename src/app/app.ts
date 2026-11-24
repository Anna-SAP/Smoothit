// The camera app: live viewfinder loop, one-tap capture with burst selection, review and gallery.
// The user-facing surface is deliberately tiny - a shutter, a camera flip and four looks - because
// every photographic decision is made by the Director.
import { Camera, CameraError, type Facing } from '../camera/camera';
import { processStill } from '../capture/still';
import { analyze } from '../engine/analysis';
import { frameSignals, scoreFrame } from '../engine/bestshot';
import { Director, type Direction, type HintId } from '../engine/director';
import { Engine, type SegmentationMask } from '../engine/engine';
import type { FaceGeometry } from '../engine/faceModel';
import { DEFAULT_LOOK, LOOKS, type Look, lookById } from '../engine/looks';
import { Vision } from '../vision/vision';
import { Gallery, type Shot } from './gallery';
import { hintText, lang, t } from './i18n';
import { photoFilename, savePhoto, sharePhoto } from './save';

type SoftMode = 'auto' | 'on' | 'off';

/** Preview quality ladder: the loop steps down when a device cannot hold the frame rate. */
const QUALITY = [
  { workLong: 1024, segEvery: 3 },
  { workLong: 832, segEvery: 4 },
  { workLong: 640, segEvery: 6 },
];
const BURST_FRAMES = 5;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const nextPaint = () => new Promise<void>((r) => requestAnimationFrame(() => setTimeout(r, 0)));

async function pixelsToBlob(pixels: Uint8Array, width: number, height: number, quality: number, maxSide = 0): Promise<Blob> {
  const full = document.createElement('canvas');
  full.width = width;
  full.height = height;
  full.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(pixels), width, height), 0, 0);
  let canvas = full;
  if (maxSide && Math.max(width, height) > maxSide) {
    const k = maxSide / Math.max(width, height);
    canvas = document.createElement('canvas');
    canvas.width = Math.round(width * k);
    canvas.height = Math.round(height * k);
    const ctx = canvas.getContext('2d')!;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(full, 0, 0, canvas.width, canvas.height);
  }
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('JPEG encoding failed'))), 'image/jpeg', quality),
  );
}

export class App {
  private canvas = $<HTMLCanvasElement>('view');
  private engine!: Engine;
  private vision!: Vision;
  private camera = new Camera();
  private director = new Director();
  private gallery = new Gallery();

  private look: Look = DEFAULT_LOOK;
  private softMode: SoftMode = 'auto';
  private softLatched = false;
  private softSince = 0;
  private softActive = false;

  private running = false;
  private busy = false;
  private frame = 0;
  private lastTick = 0;
  private lastVideoTime = -1;
  private seg: SegmentationMask | null = null;
  private lastFace: FaceGeometry | null = null;
  private lastDirection: Direction | null = null;
  private quality = 0;
  private costEma = 0;
  private lastExposureNudge = 0;

  private hintShown: HintId | null = null;
  private hintCandidate: HintId | null = null;
  private hintSince = 0;
  private startedAt = 0;

  private reviewShot: Shot | null = null;
  private reviewUrls: string[] = [];
  private debug = new URLSearchParams(location.search).has('debug');

  async boot(): Promise<void> {
    this.bindUi();
    const bootText = $('bootText'), bar = $('bootBar'), retry = $<HTMLButtonElement>('bootRetry');
    retry.hidden = true;
    const step = (text: string, pct: number) => { bootText.textContent = text; bar.style.width = `${pct}%`; };
    try {
      step(t('loading'), 8);
      try {
        this.engine ??= new Engine(this.canvas);
      } catch {
        throw new CameraError('unavailable', t('errorWebgl'));
      }
      step(t('loadingModels'), 30);
      const delegate = new URLSearchParams(location.search).get('delegate') as 'GPU' | 'CPU' | null;
      this.vision ??= await Vision.create({ baseUrl: import.meta.env.BASE_URL, mode: 'VIDEO', delegate: delegate ?? undefined });
      step(t('loadingCamera'), 75);
      await this.camera.start('user');
      step('', 100);
      $('boot').classList.add('done');
      this.startedAt = performance.now();
      this.resize();
      this.start();
      void this.refreshThumb();
      void navigator.wakeLock?.request('screen').catch(() => undefined);
    } catch (err) {
      console.error(err);
      const kind = err instanceof CameraError ? err.kind : 'unavailable';
      const message = kind === 'denied' ? t('errorDenied') : kind === 'insecure' ? t('errorInsecure')
        : err instanceof CameraError && err.message === t('errorWebgl') ? t('errorWebgl') : t('errorUnavailable');
      bootText.textContent = `${t('errorTitle')} · ${message}`;
      bar.style.width = '0%';
      retry.hidden = false;
      retry.textContent = t('retry');
      retry.onclick = () => void this.boot();
    }
  }

  // ------------------------------------------------------------------ UI wiring

  private uiBound = false;

  private bindUi(): void {
    if (this.uiBound) return;
    this.uiBound = true;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

    const looks = $('looks');
    for (const l of LOOKS) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = l.name[lang];
      b.dataset.look = l.id;
      b.setAttribute('role', 'tab');
      b.setAttribute('aria-selected', String(l.id === this.look.id));
      b.onclick = () => this.setLook(l.id);
      looks.appendChild(b);
    }

    $('shutter').onclick = () => void this.capture();
    $('flip').onclick = () => void this.flip();
    $('softlight').onclick = () => this.cycleSoftMode();
    this.renderSoftButton();

    $('galleryBtn').onclick = () => void this.openGallery();
    $('galleryClose').onclick = () => { $('gallery').hidden = true; this.start(); };
    $('galleryTitle').textContent = t('gallery');

    $('reviewClose').onclick = () => this.closeReview();
    $('reviewDelete').onclick = () => void this.deleteReviewShot();
    const save = $<HTMLButtonElement>('reviewSave'), share = $('reviewShare'), compare = $('reviewCompare');
    save.textContent = t('save');
    share.textContent = t('share');
    compare.textContent = t('compare');
    save.onclick = async () => {
      if (!this.reviewShot) return;
      await savePhoto(this.reviewShot.image, photoFilename(new Date(this.reviewShot.createdAt)));
      save.textContent = t('saved');
      setTimeout(() => { save.textContent = t('save'); }, 1600);
    };
    share.onclick = () => {
      if (this.reviewShot) void sharePhoto(this.reviewShot.image, photoFilename(new Date(this.reviewShot.createdAt)));
    };
    // Press and hold (on the button or the photo itself) to see the unprocessed frame.
    const showOriginal = (on: boolean) => {
      const img = $<HTMLImageElement>('reviewImg');
      if (this.reviewUrls.length === 2) img.src = this.reviewUrls[on ? 1 : 0];
    };
    for (const el of [compare, $('reviewImg')]) {
      el.addEventListener('pointerdown', (e) => { e.preventDefault(); showOriginal(true); });
      for (const ev of ['pointerup', 'pointercancel', 'pointerleave']) el.addEventListener(ev, () => showOriginal(false));
    }

    window.addEventListener('resize', () => this.resize());
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) this.running = false;
      else if ($('review').hidden && $('gallery').hidden) this.start();
    });
  }

  private setLook(id: string): void {
    this.look = lookById(id);
    this.director.look = this.look;
    for (const b of Array.from($('looks').children) as HTMLElement[]) {
      b.setAttribute('aria-selected', String(b.dataset.look === id));
    }
  }

  private cycleSoftMode(): void {
    this.softMode = this.softMode === 'auto' ? 'on' : this.softMode === 'on' ? 'off' : 'auto';
    this.renderSoftButton();
    this.applySoftLight();
  }

  private renderSoftButton(): void {
    const b = $('softlight');
    b.textContent = t(this.softMode === 'auto' ? 'softAuto' : this.softMode === 'on' ? 'softOn' : 'softOff');
    b.dataset.on = String(this.softMode !== 'off');
  }

  private applySoftLight(): void {
    // A phone screen only lights a face it is pointed at: front camera only.
    const active = this.camera.facing === 'user' && (this.softMode === 'on' || (this.softMode === 'auto' && this.softLatched));
    if (active !== this.softActive) {
      this.softActive = active;
      document.body.classList.toggle('softlight', active);
      document.querySelector('meta[name="theme-color"]')?.setAttribute('content', active ? '#fff1df' : '#0c0b0b');
    }
  }

  private resize(): void {
    const rect = this.canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(rect.width * dpr)), h = Math.max(2, Math.round(rect.height * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  // ------------------------------------------------------------------ live loop

  private start(): void {
    if (this.running) return;
    this.running = true;
    this.lastTick = performance.now();
    const loop = () => {
      if (!this.running) return;
      try {
        if (!this.busy) this.tick();
      } catch (err) {
        console.error('frame failed', err);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private tick(): void {
    const cam = this.camera;
    if (!cam.ready) return;
    const video = cam.video;
    if (video.currentTime === this.lastVideoTime) return; // no new frame yet
    this.lastVideoTime = video.currentTime;

    const t0 = performance.now();
    const dt = Math.min(0.25, (t0 - this.lastTick) / 1000);
    this.lastTick = t0;
    const w = cam.width, h = cam.height, mirror = cam.mirrored;
    const q = QUALITY[this.quality];

    const obs = this.vision.detectFace(video, w, h, mirror, t0);
    const face = obs?.face ?? null;
    this.lastFace = face;
    if (this.frame % q.segEvery === 0) {
      const gain = 2 ** Math.min(2, Math.max(0, this.director.exposureStops));
      this.seg = this.vision.segment(video, w, h, mirror, t0, gain) ?? this.seg;
    }
    this.engine.ingest(video, w, h, mirror, q.workLong);
    this.engine.setSegmentation(this.seg);
    const analysis = this.frame % 2 === 0 ? analyze(this.engine.readAnalysis(), face, this.seg) : null;
    const direction = this.director.update({ face, analysis, srcW: w, srcH: h, dt });
    this.engine.render(direction.params, { kind: 'screen' });
    this.lastDirection = direction;
    this.frame++;

    this.updateSoftLight(direction, t0);
    this.updateHint(direction, t0);
    this.assistExposure(t0);
    this.adaptQuality(performance.now() - t0, dt);
  }

  private updateSoftLight(d: Direction, now: number): void {
    // Hysteresis in both level and time: the screen light itself brightens the scene it measures.
    const lum = d.diagnostics.sceneLin;
    const want = this.softLatched ? lum < 0.12 : d.wantsScreenLight;
    if (want === this.softLatched) this.softSince = now;
    else if (now - this.softSince > (want ? 600 : 1800)) {
      this.softLatched = want;
      this.softSince = now;
    }
    this.applySoftLight();
  }

  private updateHint(d: Direction, now: number): void {
    let top = d.hints.find((h) => h !== 'lowLight' || this.softActive) ?? null;
    if (top === 'noFace' && now - this.startedAt < 2500) top = null;
    if (top !== this.hintCandidate) {
      this.hintCandidate = top;
      this.hintSince = now;
    }
    if (this.hintCandidate !== this.hintShown && now - this.hintSince > 900) {
      this.hintShown = this.hintCandidate;
      const el = $('hint');
      if (this.hintShown) el.textContent = hintText(this.hintShown);
      el.classList.toggle('show', !!this.hintShown);
    }
  }

  /** Outer exposure loop: where the platform allows it, ask the sensor for the light we need. */
  private assistExposure(now: number): void {
    if (!this.camera.supportsExposureBias || now - this.lastExposureNudge < 700) return;
    const need = this.director.exposureStops;
    if (need > 0.5) this.camera.nudgeExposure(0.33);
    else if (need < -0.35) this.camera.nudgeExposure(-0.33);
    else return;
    this.lastExposureNudge = now;
  }

  private adaptQuality(costMs: number, dt: number): void {
    this.costEma = this.costEma ? this.costEma * 0.9 + costMs * 0.1 : costMs;
    if (this.frame % 45 === 0) {
      if (this.costEma > 52 && this.quality < QUALITY.length - 1) this.quality++;
      else if (this.costEma < 20 && this.quality > 0) this.quality--;
    }
    if (this.debug && this.frame % 6 === 0 && this.lastDirection) {
      const g = this.lastDirection.diagnostics;
      const el = $('debug');
      el.hidden = false;
      el.textContent =
        `${(1 / Math.max(dt, 1e-3)).toFixed(0)} fps · ${this.costEma.toFixed(0)} ms · q${this.quality} · ${this.vision.delegate}\n` +
        `${this.camera.width}x${this.camera.height} → work ${this.engine.workWidth}x${this.engine.workHeight}\n` +
        `exp ${g.exposureStops.toFixed(2)} · wb ${g.wb.map((v) => v.toFixed(2)).join(' ')}\n` +
        `dist ${g.distanceMm.toFixed(0)} mm · yaw ${g.yawDeg.toFixed(0)} pitch ${g.pitchDeg.toFixed(0)}\n` +
        `light ${g.lightContrast.toFixed(1)}:1 side ${g.lightSide.toFixed(2)} need ${g.relightNeed.toFixed(2)}`;
    }
  }

  private async flip(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    const next: Facing = this.camera.facing === 'user' ? 'environment' : 'user';
    try {
      await this.camera.start(next);
    } catch {
      await this.camera.start(next === 'user' ? 'environment' : 'user').catch(() => undefined);
    }
    this.director.reset();
    this.seg = null;
    this.lastVideoTime = -1;
    this.applySoftLight();
    this.busy = false;
  }

  // ------------------------------------------------------------------ capture

  /**
   * One tap. Shoots a short burst, keeps the frame with open eyes and the least motion, and
   * develops it at full resolution through the same pipeline the viewfinder was showing.
   */
  async capture(): Promise<Shot | null> {
    if (this.busy || !this.camera.ready) return null;
    this.busy = true;
    const shutter = $<HTMLButtonElement>('shutter'), veil = $('veil'), flash = $('flash');
    shutter.disabled = true;
    const cam = this.camera, mirror = cam.mirrored;
    let best: { bmp: ImageBitmap; face: FaceGeometry | null; score: number } | null = null;
    try {
      if (this.softActive) {
        // Screen flash: the whole display becomes the key light for the duration of the burst.
        flash.classList.add('on');
        await sleep(260);
      }
      let prev = this.lastFace;
      for (let i = 0; i < BURST_FRAMES; i++) {
        const bmp = await createImageBitmap(cam.video);
        const obs = this.vision.detectFace(bmp, bmp.width, bmp.height, mirror, performance.now());
        const face = obs?.face ?? null;
        const score = scoreFrame(frameSignals(face, prev, obs?.blend ?? null));
        prev = face ?? prev;
        if (!best || score > best.score) {
          best?.bmp.close();
          best = { bmp, face, score };
        } else bmp.close();
        if (i < BURST_FRAMES - 1) await sleep(60);
      }
      flash.classList.remove('on');
      veil.classList.remove('blink');
      void veil.offsetWidth;
      veil.classList.add('blink');
      veil.dataset.label = t('developing');
      veil.classList.add('developing');
      await nextPaint();

      const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
      const maxLongSide = Math.min(memory <= 4 ? 2048 : 3072, this.engine.maxTextureSize);
      const res = processStill(this.engine, this.vision, best!.bmp, {
        look: this.look, mirror, maxLongSide, withOriginal: true, director: this.director, face: best!.face,
      });
      const [image, original, thumb] = await Promise.all([
        pixelsToBlob(res.pixels, res.width, res.height, 0.95),
        res.original ? pixelsToBlob(res.original, res.width, res.height, 0.9) : Promise.resolve(null),
        pixelsToBlob(res.pixels, res.width, res.height, 0.8, 360),
      ]);
      const shot: Shot = {
        id: Date.now(), createdAt: Date.now(), look: this.look.id, width: res.width, height: res.height, image, original, thumb,
      };
      await this.gallery.add(shot);
      void this.refreshThumb(shot);
      this.showReview(shot);
      return shot;
    } catch (err) {
      console.error('capture failed', err);
      return null;
    } finally {
      best?.bmp.close();
      flash.classList.remove('on');
      veil.classList.remove('developing');
      shutter.disabled = false;
      this.seg = null; // the segmenter last saw the still, not the live frame
      this.busy = false;
    }
  }

  // ------------------------------------------------------------------ review & gallery

  private showReview(shot: Shot): void {
    this.running = false;
    this.reviewUrls.forEach((u) => URL.revokeObjectURL(u));
    this.reviewUrls = [URL.createObjectURL(shot.image)];
    if (shot.original) this.reviewUrls.push(URL.createObjectURL(shot.original));
    this.reviewShot = shot;
    $<HTMLImageElement>('reviewImg').src = this.reviewUrls[0];
    $('reviewTag').textContent = lookById(shot.look).name[lang];
    $('reviewCompare').style.visibility = shot.original ? 'visible' : 'hidden';
    $('gallery').hidden = true;
    $('review').hidden = false;
  }

  private closeReview(): void {
    $('review').hidden = true;
    this.reviewShot = null;
    this.start();
  }

  private async deleteReviewShot(): Promise<void> {
    if (!this.reviewShot) return;
    await this.gallery.remove(this.reviewShot.id);
    this.closeReview();
    void this.refreshThumb();
  }

  private thumbUrl: string | null = null;

  private async refreshThumb(latest?: Shot): Promise<void> {
    const shot = latest ?? (await this.gallery.list())[0];
    if (this.thumbUrl) URL.revokeObjectURL(this.thumbUrl);
    this.thumbUrl = shot ? URL.createObjectURL(shot.thumb) : null;
    $('galleryBtn').style.backgroundImage = this.thumbUrl ? `url(${this.thumbUrl})` : '';
  }

  private gridUrls: string[] = [];

  private async openGallery(): Promise<void> {
    if (this.busy) return;
    this.running = false;
    const grid = $('galleryGrid');
    this.gridUrls.forEach((u) => URL.revokeObjectURL(u));
    this.gridUrls = [];
    grid.textContent = '';
    const shots = await this.gallery.list();
    if (!shots.length) {
      const empty = document.createElement('div');
      empty.className = 'empty';
      empty.textContent = t('galleryEmpty');
      grid.appendChild(empty);
    }
    for (const shot of shots) {
      const url = URL.createObjectURL(shot.thumb);
      this.gridUrls.push(url);
      const b = document.createElement('button');
      b.type = 'button';
      b.style.backgroundImage = `url(${url})`;
      b.onclick = () => this.showReview(shot);
      grid.appendChild(b);
    }
    $('gallery').hidden = false;
  }

  /** Test and debugging surface (window.smoothit). */
  get state() {
    return {
      running: this.running, busy: this.busy, frame: this.frame, quality: this.quality, costMs: this.costEma,
      softActive: this.softActive, look: this.look.id, hint: this.hintShown,
      camera: { width: this.camera.width, height: this.camera.height, facing: this.camera.facing },
      diagnostics: this.lastDirection?.diagnostics ?? null, hasFace: !!this.lastFace, hasSegmentation: !!this.seg,
    };
  }
}
