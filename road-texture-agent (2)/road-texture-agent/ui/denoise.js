// Intel Open Image Denoise (its "RT" filter for path-traced pictures) ported to
// WebGL2, so Render photo can clear its noise the way Blender and Cycles do,
// in the browser, with nothing to install: the same neural network with the
// same trained weights, run by fragment shaders on the graphics card.
//
// Ported from Open Image Denoise 2.4.1 (Copyright 2018 Intel Corporation,
// Apache License 2.0): core/unet_filter.cpp (the network and its tiles),
// core/color.h (the PU transfer curve), devices/gpu/gpu_input_process.h and
// gpu_output_process.h (what goes in and comes out), gpu_autoexposure.h (the
// input scale). Weights: rt_hdr_calb_cnrm.tza from RenderKit/oidn-weights
// (Apache License 2.0). See vendor/oidn/LICENSE.txt.
// Changes: written in JavaScript and GLSL; tensors kept as half-float texture
// arrays, four channels per layer; pooling done with the convolution before it,
// upsampling and concatenation by the convolution after it.
//
// Inputs are textures in the same WebGL2 context: the noisy colour (linear,
// HDR), the albedo (0..1) and the normal (-1..1) of what each pixel sees, all
// without noise and filtered like the colour. Rows run bottom to top (GL's way).

// the PU transfer curve (perceptually uniform encoding of HDR values)
const PU = { a: 1.41283765e3, b: 1.64593172, c: 4.31384981e-1, d: -2.94139609e-3, e: 1.92653254e-1,
             f: 6.26026094e-3, g: 9.98620152e-1, y0: 1.5794576e-6, y1: 3.22087631e-2, x0: 2.23151711e-3, x1: 3.70974749e-1 };
function puForward(y){
  if(y <= PU.y0) return PU.a * y;
  if(y <= PU.y1) return PU.b * Math.pow(y, PU.c) + PU.d;
  return PU.e * Math.log(y + PU.f) + PU.g;
}
const X_MAX = puForward(65504);                       // the largest half float, encoded
const RECEPTIVE_FIELD = 174, ALIGN = 16;
const OVERLAP = Math.ceil(RECEPTIVE_FIELD / 2 / ALIGN) * ALIGN;
const ROWS_PER_LINE = 32;                             // weight rows side by side in the weight texture

const f = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
const GLSL_PU = `
float puForward(float y){
  if(y <= ${f(PU.y0)}) return ${f(PU.a)} * y;
  if(y <= ${f(PU.y1)}) return ${f(PU.b)} * pow(y, ${f(PU.c)}) + (${f(PU.d)});
  return ${f(PU.e)} * log(y + ${f(PU.f)}) + ${f(PU.g)};
}
float puInverse(float x){
  if(x <= ${f(PU.x0)}) return x / ${f(PU.a)};
  if(x <= ${f(PU.x1)}) return pow((x - (${f(PU.d)})) / ${f(PU.b)}, ${f(1 / PU.c)});
  return exp((x - ${f(PU.g)}) / ${f(PU.e)}) - ${f(PU.f)};
}
vec3 sane(vec3 v){ return vec3(isnan(v.x) ? 0.0 : v.x, isnan(v.y) ? 0.0 : v.y, isnan(v.z) ? 0.0 : v.z); }
`;

// ----------------------------------------------------------------- weights
function halfToFloat(h){
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  if(e === 0) return s * m * 2 ** -24;
  if(e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** Reads an OIDN weights file (.tza, version 2): its tensors by name, as floats. */
export function parseTZA(buffer){
  const view = new DataView(buffer), bytes = new Uint8Array(buffer);
  if(view.getUint16(0, true) !== 0x41d7 || view.getUint8(2) !== 2) throw new Error('not an Open Image Denoise weights file');
  let o = Number(view.getBigUint64(4, true));
  const n = view.getUint32(o, true); o += 4;
  const text = new TextDecoder(), out = new Map();
  for(let t = 0; t < n; t++){
    const len = view.getUint16(o, true); o += 2;
    const name = text.decode(bytes.subarray(o, o + len)); o += len;
    const nd = view.getUint8(o); o += 1;
    const dims = [];
    for(let i = 0; i < nd; i++){ dims.push(view.getUint32(o, true)); o += 4; }
    o += nd;                                                                    // layout: oihw or x
    const type = String.fromCharCode(view.getUint8(o)); o += 1;
    const at = Number(view.getBigUint64(o, true)); o += 8;
    const count = dims.reduce((a, b) => a * b, 1), data = new Float32Array(count);
    if(type === 'h') for(let i = 0; i < count; i++) data[i] = halfToFloat(view.getUint16(at + i * 2, true));
    else if(type === 'f') for(let i = 0; i < count; i++) data[i] = view.getFloat32(at + i * 4, true);
    else throw new Error('unknown tensor type ' + type);
    out.set(name, { dims, data });
  }
  return out;
}

// The U-Net of OIDN's base and small models: [name, inputs, resolution level, pool after]. An
// input written name^ is the level below, upsampled; 'input' is the colour, albedo and normal.
const UNET = [
  ['enc_conv0', ['input'], 0, false],
  ['enc_conv1', ['enc_conv0'], 0, true],
  ['enc_conv2', ['enc_conv1'], 1, true],
  ['enc_conv3', ['enc_conv2'], 2, true],
  ['enc_conv4', ['enc_conv3'], 3, true],
  ['enc_conv5a', ['enc_conv4'], 4, false],
  ['enc_conv5b', ['enc_conv5a'], 4, false],
  ['dec_conv4a', ['enc_conv5b^', 'enc_conv3'], 3, false],
  ['dec_conv4b', ['dec_conv4a'], 3, false],
  ['dec_conv3a', ['dec_conv4b^', 'enc_conv2'], 2, false],
  ['dec_conv3b', ['dec_conv3a'], 2, false],
  ['dec_conv2a', ['dec_conv3b^', 'enc_conv1'], 1, false],
  ['dec_conv2b', ['dec_conv2a'], 1, false],
  ['dec_conv1a', ['dec_conv2b^', 'input'], 0, false],
  ['dec_conv1b', ['dec_conv1a'], 0, false],
  ['dec_conv0', ['dec_conv1b'], 0, false],
];

// ----------------------------------------------------------------- shaders
const VERT = `#version 300 es
void main(){ vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;

const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp sampler2DArray;
`;

// weights: row r holds, for 4 output channels and 4 input channels, the 9 taps as
// 4 texels each (one per input channel, its 4 output channels in rgba)
const WEIGHT_FN = `
uniform sampler2D uW;
mat4 W(int r, int t){
  ivec2 c = ivec2((r & ${ROWS_PER_LINE - 1}) * 36 + t * 4, r >> ${Math.log2(ROWS_PER_LINE)});
  return mat4(texelFetch(uW, c, 0), texelFetch(uW, c + ivec2(1, 0), 0), texelFetch(uW, c + ivec2(2, 0), 0), texelFetch(uW, c + ivec2(3, 0), 0));
}`;

// a 3x3 convolution (ReLU after), g groups of 4 output channels at once; input A
// (na layers, upsampled from the level below when up) and B (nb layers, concatenated)
function convSource(na, nb, up, g){
  const cin = na + nb, L = [];
  L.push(HEAD, WEIGHT_FN, 'uniform sampler2DArray uA;', nb ? 'uniform sampler2DArray uB;' : '',
    'uniform int uRow0; uniform ivec2 uSize;', `uniform vec4 uBias[${g}];`);
  for(let i = 0; i < g; i++) L.push(`layout(location = ${i}) out vec4 o${i};`);
  L.push('void main(){', ' ivec2 p = ivec2(gl_FragCoord.xy);');
  for(let i = 0; i < g; i++) L.push(` vec4 a${i} = uBias[${i}];`);
  // which taps are inside the picture (outside: zero padding)
  L.push(' bool inside[9];');
  for(let t = 0; t < 9; t++){
    const dx = t % 3 - 1, dy = Math.floor(t / 3) - 1;
    L.push(` { ivec2 q = p + ivec2(${dx}, ${dy}); inside[${t}] = all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, uSize)); }`);
  }
  const block = (src, n, k0, upsample) => {
    L.push(` for(int k = 0; k < ${n}; k++){`, `  int r = uRow0 + ${k0} + k;`);
    for(let t = 0; t < 9; t++){
      const dx = t % 3 - 1, dy = Math.floor(t / 3) - 1;
      const q = `p + ivec2(${dx}, ${dy})`;
      L.push(`  if(inside[${t}]){ vec4 x = texelFetch(${src}, ivec3(${upsample ? `(${q}) >> 1` : q}, k), 0);`);
      for(let i = 0; i < g; i++) L.push(`   a${i} += W(r + ${i * cin}, ${t}) * x;`);
      L.push('  }');
    }
    L.push(' }');
  };
  block('uA', na, 0, up);
  if(nb) block('uB', nb, na, false);
  for(let i = 0; i < g; i++) L.push(` o${i} = max(a${i}, 0.0);`);
  L.push('}');
  return L.join('\n');
}

// the same followed by 2x2 max pooling: each output pixel works out the convolution at
// the four pixels below it (sharing a 4x4 patch of the input) and keeps the largest
function convPoolSource(na, g){
  const L = [];
  L.push(HEAD, WEIGHT_FN, 'uniform sampler2DArray uA; uniform int uRow0; uniform ivec2 uSize;', `uniform vec4 uBias[${g}];`);
  for(let i = 0; i < g; i++) L.push(`layout(location = ${i}) out vec4 o${i};`);
  L.push('void main(){', ' ivec2 b = ivec2(gl_FragCoord.xy) * 2 - 1;');
  for(let i = 0; i < g; i++) for(let s = 0; s < 4; s++) L.push(` vec4 a${i}_${s} = uBias[${i}];`);
  L.push(' bool inside[16];');
  for(let u = 0; u < 16; u++) L.push(` { ivec2 q = b + ivec2(${u % 4}, ${u >> 2}); inside[${u}] = all(greaterThanEqual(q, ivec2(0))) && all(lessThan(q, uSize)); }`);
  L.push(` for(int k = 0; k < ${na}; k++){`, '  int r = uRow0 + k;', '  vec4 x[16];');
  for(let u = 0; u < 16; u++) L.push(`  x[${u}] = inside[${u}] ? texelFetch(uA, ivec3(b + ivec2(${u % 4}, ${u >> 2}), k), 0) : vec4(0.0);`);
  for(let t = 0; t < 9; t++){
    const dx = t % 3, dy = Math.floor(t / 3);
    for(let i = 0; i < g; i++){
      L.push(`  { mat4 w = W(r + ${i * na}, ${t});`);
      for(let s = 0; s < 4; s++) L.push(`   a${i}_${s} += w * x[${(dy + (s >> 1)) * 4 + dx + (s & 1)}];`);
      L.push('  }');
    }
  }
  L.push(' }');
  for(let i = 0; i < g; i++) L.push(` o${i} = max(max(max(a${i}_0, a${i}_1), max(a${i}_2, a${i}_3)), 0.0);`);
  L.push('}');
  return L.join('\n');
}

// colour, albedo and normal into the network's input (OIDN's input process, HDR)
const INPUT = HEAD + GLSL_PU + `
uniform sampler2D uColor, uAlbedo, uNormal;
uniform ivec2 uOrigin, uLo, uHi; uniform float uScale;
layout(location = 0) out vec4 o0; layout(location = 1) out vec4 o1; layout(location = 2) out vec4 o2;
void main(){
  // the tile's part of the picture; zero round it
  ivec2 img = ivec2(gl_FragCoord.xy) + uOrigin;
  if(any(lessThan(img, uLo)) || any(greaterThanEqual(img, uHi))){ o0 = o1 = o2 = vec4(0.0); return; }
  vec3 c = clamp(sane(texelFetch(uColor, img, 0).rgb * uScale), 0.0, 3.4e38);
  o0 = vec4(puForward(c.r), puForward(c.g), puForward(c.b), 0.0) * ${f(1 / X_MAX)};
  o1 = vec4(clamp(sane(texelFetch(uAlbedo, img, 0).rgb), 0.0, 1.0), 0.0);
  o2 = vec4(clamp(sane(texelFetch(uNormal, img, 0).rgb), -1.0, 1.0) * 0.5 + 0.5, 0.0);
}`;

// the network's output back to HDR colour (OIDN's output process)
const OUTPUT = HEAD + GLSL_PU + `
uniform sampler2DArray uSrc; uniform ivec2 uOrigin; uniform float uInvScale;
out vec4 o;
void main(){
  ivec2 p = ivec2(gl_FragCoord.xy) - uOrigin;
  vec3 v = max(sane(texelFetch(uSrc, ivec3(p, 0), 0).rgb), 0.0) * ${f(X_MAX)};
  o = vec4(vec3(puInverse(v.r), puInverse(v.g), puInverse(v.b)) * uInvScale, 1.0);
}`;

// the mean brightness of each bin of up to 16 x 16 pixels (OIDN's autoexposure)
const BINS = HEAD + `
uniform sampler2D uColor; uniform ivec2 uImage, uBins;
out vec4 o;
void main(){
  ivec2 b = ivec2(gl_FragCoord.xy), lo = b * uImage / uBins, hi = (b + 1) * uImage / uBins;
  float s = 0.0;
  for(int y = 0; y < 16; y++){
    if(lo.y + y >= hi.y) break;
    for(int x = 0; x < 16; x++){
      if(lo.x + x >= hi.x) break;
      vec3 c = texelFetch(uColor, lo + ivec2(x, y), 0).rgb;
      c = clamp(vec3(isnan(c.x) ? 0.0 : c.x, isnan(c.y) ? 0.0 : c.y, isnan(c.z) ? 0.0 : c.z), 0.0, 3.4e38);
      s += dot(c, vec3(0.212671, 0.715160, 0.072169));
    }
  }
  o = vec4(s / float((hi.x - lo.x) * (hi.y - lo.y)), 0.0, 0.0, 1.0);
}`;

// ----------------------------------------------------------------- the denoiser
export class Denoiser {
  /** Loads the weights (an OIDN .tza file) and prepares the network in this WebGL2 context. */
  static async load(gl, url){
    const res = await fetch(url);
    if(!res.ok) throw new Error(`could not load the denoiser's weights (${res.status})`);
    return new Denoiser(gl, parseTZA(await res.arrayBuffer()));
  }

  constructor(gl, tensors, { maxTilePixels = 1.4e6 } = {}){
    if(!(typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext)) throw new Error('the denoiser needs WebGL2');
    if(!gl.getExtension('EXT_color_buffer_float')) throw new Error('this graphics card cannot draw into float pictures');
    if(tensors.has('enc_conv1b.weight')) throw new Error('the large OIDN model is not supported');
    this.gl = gl;
    this.maxTilePixels = maxTilePixels;
    this.maxG = Math.min(4, gl.getParameter(gl.MAX_DRAW_BUFFERS));
    this.programs = new Map();
    this.pool = [];                                                     // free tensors
    this.dims = null;
    // channels and layers of every value in the network, and the lanes' channel numbers
    const lanes = new Map(), channels = new Map();
    const inC = tensors.get('enc_conv0.weight').dims[1];
    if(inC !== 9) throw new Error('the denoiser needs a model with albedo and normal');
    channels.set('input', 9);
    lanes.set('input', [0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1]);
    this.convs = [];
    let row = 0;
    const weightRows = [];
    for(const [name, inputs, level, pool] of UNET){
      const w = tensors.get(name + '.weight'), b = tensors.get(name + '.bias');
      if(!w || !b) throw new Error('the weights lack ' + name);
      const [cout, cin] = w.dims;
      // the input lanes in order, each the channel of the weights it multiplies (-1: none)
      const srcs = inputs.map(s => s.replace('^', '')), inLanes = [];
      let offset = 0;
      for(const s of srcs){
        for(const c of lanes.get(s)) inLanes.push(c < 0 ? -1 : c + offset);
        offset += channels.get(s);
      }
      if(offset !== cin) throw new Error(`${name} expects ${cin} input channels, gets ${offset}`);
      const cinG = inLanes.length / 4, coutG = Math.ceil(cout / 4);
      const conv = { name, inputs: srcs, up: inputs[0].endsWith('^'), level, pool, cout, coutG, cinG,
                     na: lanes.get(srcs[0]).length / 4, nb: srcs[1] ? lanes.get(srcs[1]).length / 4 : 0,
                     row0: row, bias: new Float32Array(coutG * 4) };
      conv.bias.set(b.data);
      for(let og = 0; og < coutG; og++){
        for(let k = 0; k < cinG; k++){
          const line = new Float32Array(36 * 4);
          for(let t = 0; t < 9; t++){
            // GL rows run upward, OIDN's downward: tap t (dx, dy) is kernel (kh = 1 - dy, kw = 1 + dx)
            const kh = 2 - Math.floor(t / 3), kw = t % 3;
            for(let j = 0; j < 4; j++){
              const ci = inLanes[k * 4 + j];
              for(let l = 0; l < 4; l++){
                const co = og * 4 + l;
                line[(t * 4 + j) * 4 + l] = ci < 0 || co >= cout ? 0 : w.data[((co * cin + ci) * 3 + kh) * 3 + kw];
              }
            }
          }
          weightRows.push(line);
          row++;
        }
      }
      channels.set(name, cout);
      lanes.set(name, Array.from({ length: coutG * 4 }, (_, i) => (i < cout ? i : -1)));
      this.convs.push(conv);
    }
    if(this.convs[this.convs.length - 1].cout !== 3) throw new Error('the model does not give a colour');
    // the weights texture
    const lines = Math.ceil(row / ROWS_PER_LINE), width = ROWS_PER_LINE * 36;
    const data = new Float32Array(width * lines * 4);
    weightRows.forEach((r, i) => data.set(r, ((i >> Math.log2(ROWS_PER_LINE)) * width + (i & (ROWS_PER_LINE - 1)) * 36) * 4));
    this._unpackDefaults();
    this.weights = this._texture2D(width, lines, gl.RGBA32F, gl.RGBA, gl.FLOAT, data);
    this.vao = gl.createVertexArray();
    this.fbo = gl.createFramebuffer();
    // last use of every value, so its textures go back to the pool
    this.lastUse = new Map();
    this.convs.forEach((c, i) => c.inputs.forEach(s => this.lastUse.set(s, i)));
  }

  _unpackDefaults(){
    const gl = this.gl;
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_ROWS, 0);
    gl.pixelStorei(gl.UNPACK_SKIP_PIXELS, 0);
  }

  _texture2D(w, h, internal, format, type, data){
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data || null);
    for(const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  _program(key, source){
    if(this.programs.has(key)) return this.programs.get(key);
    const gl = this.gl;
    const shader = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('denoiser shader: ' + gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, shader(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, shader(gl.FRAGMENT_SHADER, source));
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error('denoiser program: ' + gl.getProgramInfoLog(p));
    const u = {};
    const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
    for(let i = 0; i < n; i++){ const a = gl.getActiveUniform(p, i), name = a.name.replace(/\[0\]$/, ''); u[name] = gl.getUniformLocation(p, a.name); }
    const prog = { p, u };
    this.programs.set(key, prog);
    return prog;
  }

  // a tensor: a half-float texture array, four channels per layer, at a level of the tile
  _tensor(level, layers){
    const [w, h] = this.levelSize(level);
    const i = this.pool.findIndex(t => t.w === w && t.h === h && t.layers === layers);
    if(i >= 0) return this.pool.splice(i, 1)[0];
    const gl = this.gl, tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, tex);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.RGBA16F, w, h, layers);
    for(const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D_ARRAY, p, gl.NEAREST);
    return { tex, w, h, layers };
  }

  levelSize(level){ return [this.dims[0] >> level, this.dims[1] >> level]; }

  _bindTargets(tex, layer0, n, w, h){
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    const bufs = [];
    for(let i = 0; i < 8; i++){
      if(i < n) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, tex, 0, layer0 + i);
      else if(i < this.attached) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, null, 0, 0);
      if(i < n) bufs.push(gl.COLOR_ATTACHMENT0 + i);
    }
    this.attached = Math.max(this.attached || 0, n);
    gl.drawBuffers(bufs);
    gl.viewport(0, 0, w, h);
  }

  _bindTexture(unit, target, tex, loc){
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
    gl.uniform1i(loc, unit);
  }

  _state(){
    const gl = this.gl;
    for(const cap of [gl.BLEND, gl.DEPTH_TEST, gl.CULL_FACE, gl.SCISSOR_TEST, gl.STENCIL_TEST, gl.POLYGON_OFFSET_FILL, gl.SAMPLE_ALPHA_TO_COVERAGE, gl.RASTERIZER_DISCARD]) gl.disable(cap);
    gl.colorMask(true, true, true, true);
    gl.bindVertexArray(this.vao);
  }

  /** OIDN's input scale for an HDR picture: 0.18 over the mean (in log2) brightness of 16 x 16 bins. */
  autoexposure(color, width, height){
    const gl = this.gl, bw = Math.ceil(width / 16), bh = Math.ceil(height / 16);
    const tex = this._texture2D(bw, bh, gl.RGBA32F, gl.RGBA, gl.FLOAT, null);
    const fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, bw, bh);
    const { p, u } = this._program('bins', BINS);
    gl.useProgram(p);
    this._bindTexture(0, gl.TEXTURE_2D, color, u.uColor);
    gl.uniform2i(u.uImage, width, height); gl.uniform2i(u.uBins, bw, bh);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(bw * bh * 4);
    gl.readPixels(0, 0, bw, bh, gl.RGBA, gl.FLOAT, px);
    gl.deleteFramebuffer(fbo); gl.deleteTexture(tex);
    let sum = 0, count = 0;
    for(let i = 0; i < px.length; i += 4) if(px[i] > 1e-8){ sum += Math.log2(px[i]); count++; }
    return count ? 0.18 / 2 ** (sum / count) : 1;
  }

  // the tiles: [x, y, w, h] of the input (with the overlap) and of the part kept, in the picture
  _tiles(W, H){
    const up = v => Math.ceil(v / ALIGN) * ALIGN;
    let nx = 1, ny = 1;
    const size = () => [up(Math.ceil((W - 2 * OVERLAP) / nx) + 2 * OVERLAP), up(Math.ceil((H - 2 * OVERLAP) / ny) + 2 * OVERLAP)];
    let tw = up(W), th = up(H);
    while(tw * th > this.maxTilePixels){
      if(th >= tw) ny++; else nx++;
      [tw, th] = size();
      tw = Math.min(tw, up(W)); th = Math.min(th, up(H));
    }
    const split = (n, total) => {
      const out = [], core = Math.ceil(total / n);
      for(let i = 0; i < n; i++){
        const a = i * core, b = Math.min(total, (i + 1) * core);
        if(b > a) out.push([a, b]);
      }
      return out;
    };
    const tiles = [];
    for(const [y0, y1] of split(ny, H)){
      for(const [x0, x1] of split(nx, W)){
        const ix0 = Math.max(0, x0 - OVERLAP), iy0 = Math.max(0, y0 - OVERLAP);
        const ix1 = Math.min(W, x1 + OVERLAP), iy1 = Math.min(H, y1 + OVERLAP);
        tiles.push({ input: [ix0, iy0, ix1 - ix0, iy1 - iy0], keep: [x0, y0, x1 - x0, y1 - y0] });
      }
    }
    const w = up(Math.max(...tiles.map(t => t.input[2]))), h = up(Math.max(...tiles.map(t => t.input[3])));
    return { tiles, w, h };
  }

  /**
   * Denoises color (with its albedo and normal) into output: WebGL textures of
   * width x height (output: a float texture it can draw into). inputScale: as
   * OIDN's (by default worked out from the picture). Leaves the GL state changed:
   * the caller resets it (three.js: renderer.resetState()).
   */
  denoise({ color, albedo, normal, output, width, height, inputScale }){
    const gl = this.gl;
    this._state();
    this._unpackDefaults();
    const scale = Number.isFinite(inputScale) ? inputScale : this.autoexposure(color, width, height);
    const { tiles, w, h } = this._tiles(width, height);
    if(!this.dims || this.dims[0] !== w || this.dims[1] !== h){ this.release(); this.dims = [w, h]; }
    for(const tile of tiles){
      const [ix, iy, iw, ih] = tile.input;
      // the input's top left at the tile's top left (as OIDN does); GL rows run upward
      const origin = [ix, iy - (h - ih)];
      const values = new Map();
      const input = this._tensor(0, 3);
      values.set('input', input);
      this._state();
      this._bindTargets(input.tex, 0, 3, w, h);
      let pr = this._program('input', INPUT);
      gl.useProgram(pr.p);
      this._bindTexture(0, gl.TEXTURE_2D, color, pr.u.uColor);
      this._bindTexture(1, gl.TEXTURE_2D, albedo, pr.u.uAlbedo);
      this._bindTexture(2, gl.TEXTURE_2D, normal, pr.u.uNormal);
      gl.uniform2i(pr.u.uOrigin, origin[0], origin[1]); gl.uniform2i(pr.u.uLo, ix, iy); gl.uniform2i(pr.u.uHi, ix + iw, iy + ih);
      gl.uniform1f(pr.u.uScale, scale);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      this.convs.forEach((c, i) => {
        const outLevel = c.pool ? c.level + 1 : c.level, out = this._tensor(outLevel, c.coutG);
        const [lw, lh] = this.levelSize(c.level), [ow, oh] = this.levelSize(outLevel);
        const A = values.get(c.inputs[0]), B = c.nb ? values.get(c.inputs[1]) : null;
        const maxG = c.pool ? Math.min(2, this.maxG) : this.maxG;
        for(let og = 0; og < c.coutG; og += maxG){
          const g = Math.min(maxG, c.coutG - og);
          const key = c.pool ? `pool/${c.na}/${g}` : `conv/${c.na}/${c.nb}/${c.up}/${g}`;
          const prog = this._program(key, c.pool ? convPoolSource(c.na, g) : convSource(c.na, c.nb, c.up, g));
          gl.useProgram(prog.p);
          this._bindTargets(out.tex, og, g, ow, oh);
          gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.weights); gl.uniform1i(prog.u.uW, 0);
          this._bindTexture(1, gl.TEXTURE_2D_ARRAY, A.tex, prog.u.uA);
          if(B) this._bindTexture(2, gl.TEXTURE_2D_ARRAY, B.tex, prog.u.uB);
          gl.uniform1i(prog.u.uRow0, c.row0 + og * c.cinG);
          gl.uniform2i(prog.u.uSize, lw, lh);
          gl.uniform4fv(prog.u.uBias, c.bias.subarray(og * 4, (og + g) * 4));
          gl.drawArrays(gl.TRIANGLES, 0, 3);
        }
        values.set(c.name, out);
        for(const s of c.inputs) if(this.lastUse.get(s) === i){ this.pool.push(values.get(s)); values.delete(s); }
      });
      // the part of the tile kept, into the output
      const last = values.get(this.convs[this.convs.length - 1].name);
      const [kx, ky, kw, kh] = tile.keep;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      for(let i = 0; i < (this.attached || 0); i++) gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, null, 0, 0);
      this.attached = 0;
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, output, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.viewport(kx, ky, kw, kh);
      pr = this._program('output', OUTPUT);
      gl.useProgram(pr.p);
      this._bindTexture(0, gl.TEXTURE_2D_ARRAY, last.tex, pr.u.uSrc);
      gl.uniform2i(pr.u.uOrigin, origin[0], origin[1]);
      gl.uniform1f(pr.u.uInvScale, scale ? 1 / scale : 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      for(const v of values.values()) this.pool.push(v);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { inputScale: scale, tiles: tiles.length, tileSize: [w, h] };
  }

  /** Frees the network's working textures (kept between denoises of the same size). */
  release(){
    for(const t of this.pool) this.gl.deleteTexture(t.tex);
    this.pool = [];
  }

  dispose(){
    const gl = this.gl;
    this.release();
    gl.deleteTexture(this.weights);
    gl.deleteFramebuffer(this.fbo);
    gl.deleteVertexArray(this.vao);
    for(const { p } of this.programs.values()) gl.deleteProgram(p);
    this.programs.clear();
  }
}
