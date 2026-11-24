// Minimal WebGL2 helpers. Convention for every offscreen texture in the engine: uv (0,0) is the
// TOP-left of the image, so landmark coordinates can be used as texture coordinates directly.
export type GL = WebGL2RenderingContext;

function compile(gl: GL, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) ?? '';
    const numbered = src.split('\n').map((l, i) => `${String(i + 1).padStart(3)}: ${l}`).join('\n');
    throw new Error(`Shader compile failed: ${log}\n${numbered}`);
  }
  return sh;
}

export class Program {
  readonly prog: WebGLProgram;
  private locs = new Map<string, WebGLUniformLocation | null>();

  /** @param attribs attribute names bound to locations 0..n-1 in order. */
  constructor(private gl: GL, vs: string, fs: string, attribs: string[] = ['aPos']) {
    const p = gl.createProgram()!;
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
    attribs.forEach((name, i) => gl.bindAttribLocation(p, i, name));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(`Program link failed: ${gl.getProgramInfoLog(p)}`);
    }
    this.prog = p;
  }

  use(): this {
    this.gl.useProgram(this.prog);
    return this;
  }

  private loc(name: string): WebGLUniformLocation | null {
    let l = this.locs.get(name);
    if (l === undefined) {
      l = this.gl.getUniformLocation(this.prog, name);
      this.locs.set(name, l);
    }
    return l;
  }

  f(name: string, ...v: number[]): this {
    const gl = this.gl, l = this.loc(name);
    if (l === null) return this;
    if (v.length === 1) gl.uniform1f(l, v[0]);
    else if (v.length === 2) gl.uniform2f(l, v[0], v[1]);
    else if (v.length === 3) gl.uniform3f(l, v[0], v[1], v[2]);
    else gl.uniform4f(l, v[0], v[1], v[2], v[3]);
    return this;
  }

  tex(name: string, unit: number, tex: WebGLTexture | null): this {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    const l = this.loc(name);
    if (l !== null) gl.uniform1i(l, unit);
    return this;
  }
}

export function createTexture(gl: GL, filter: number = gl.LINEAR): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

export interface TargetOptions {
  /** Half-float colour (needs EXT_color_buffer_float or EXT_color_buffer_half_float). */
  float?: boolean;
  /** Number of colour attachments (default 1). */
  attachments?: number;
  depth?: boolean;
}

/** A render target: one or more colour textures (plus optional depth) on a framebuffer. */
export class Target {
  readonly texs: WebGLTexture[] = [];
  readonly fbo: WebGLFramebuffer;
  private depthRb: WebGLRenderbuffer | null = null;
  w = 0;
  h = 0;

  constructor(private gl: GL, private opts: TargetOptions = {}) {
    this.fbo = gl.createFramebuffer()!;
    const n = opts.attachments ?? 1;
    for (let i = 0; i < n; i++) this.texs.push(createTexture(gl));
    if (opts.depth) this.depthRb = gl.createRenderbuffer();
  }

  get tex(): WebGLTexture {
    return this.texs[0];
  }

  /** Ensure the target has the given size. Returns true if it was (re)allocated. */
  resize(w: number, h: number): boolean {
    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));
    if (w === this.w && h === this.h) return false;
    const gl = this.gl;
    this.w = w;
    this.h = h;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    const bufs: number[] = [];
    this.texs.forEach((t, i) => {
      gl.bindTexture(gl.TEXTURE_2D, t);
      if (this.opts.float) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
      else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, t, 0);
      bufs.push(gl.COLOR_ATTACHMENT0 + i);
    });
    gl.drawBuffers(bufs);
    if (this.depthRb) {
      gl.bindRenderbuffer(gl.RENDERBUFFER, this.depthRb);
      gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, w, h);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.depthRb);
    }
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`Framebuffer incomplete: 0x${status.toString(16)}`);
    return true;
  }

  bind(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, this.w, this.h);
  }
}

/** A single triangle covering the viewport. */
export class FullscreenTriangle {
  private vao: WebGLVertexArrayObject;
  constructor(private gl: GL) {
    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }
  draw(): void {
    const gl = this.gl;
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
