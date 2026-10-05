// The Look panel of the 3D tab: what the picture looks like after it is
// rendered, as in a photo editor. The film response (how light becomes the
// picture's tones: AgX, as in Blender, or Khronos Neutral, or ACES), exposure,
// contrast, brightness, highlights and shadows, saturation, white balance
// (warmth, tint), split toning, fade, vignette, sharpening, film grain and the
// glow round bright lights, plus colour lookup tables (LUTs: .cube files or
// Hald CLUT pictures, as film emulations and grades are passed around).
// Built-in presets; your own looks are saved in the workspace, and a look can
// be downloaded as a file (its LUT inside) and imported elsewhere.
// One pass does it all, from the scene's light to the finished picture, for
// the live view and for Render photo alike, so Save image keeps the look.
import * as THREE from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

// ----------------------------------------------------------------- settings
// exposure in stops; glow and saturation as factors (1: as rendered); the rest
// from -1 to 1 (or 0 to 1), 0 leaving the picture as it is
export const NEUTRAL = {
  film: 'agx', exposure: 0, contrast: 0, brightness: 0, highlights: 0, shadows: 0, saturation: 1,
  warmth: 0, tint: 0, split: 0, split_shadows: '#2f6f86', split_highlights: '#f2a65a',
  fade: 0, vignette: 0, sharpen: 0, grain: 0, glow: 1, lut: null, lut_mix: 1,
};
export const PRESETS = [
  { id: 'neutral', name: 'Neutral', settings: {} },
  { id: 'natural', name: 'Natural', settings: { contrast: 0.1, highlights: -0.2, shadows: 0.15, saturation: 1.06, sharpen: 0.25 } },
  { id: 'warm', name: 'Warm evening', settings: { warmth: 0.35, tint: 0.05, contrast: 0.12, highlights: -0.15, saturation: 1.1,
      split: 0.18, split_shadows: '#5a4a6e', split_highlights: '#ffb066', vignette: 0.25, glow: 1.35 } },
  { id: 'cool', name: 'Cool morning', settings: { warmth: -0.3, tint: -0.04, contrast: -0.05, brightness: 0.08, saturation: 0.9,
      fade: 0.1, glow: 1.15, split: 0.12, split_shadows: '#3d5f8a', split_highlights: '#e8eef5' } },
  { id: 'cinematic', name: 'Cinematic (teal and orange)', settings: { contrast: 0.22, highlights: -0.15, shadows: -0.05, saturation: 0.95,
      split: 0.45, split_shadows: '#1f7a8c', split_highlights: '#ff9a4a', vignette: 0.35, sharpen: 0.15, grain: 0.12 } },
  { id: 'faded', name: 'Faded film', settings: { contrast: -0.2, fade: 0.4, saturation: 0.8, warmth: 0.1, grain: 0.35, vignette: 0.2,
      split: 0.2, split_shadows: '#3a5a6a', split_highlights: '#f0d9a0' } },
  { id: 'vivid', name: 'Vivid', settings: { film: 'neutral', contrast: 0.25, saturation: 1.35, sharpen: 0.3, highlights: -0.1 } },
  { id: 'bw', name: 'Black and white', settings: { saturation: 0, contrast: 0.25, sharpen: 0.15, grain: 0.2, vignette: 0.2 } },
];

// ----------------------------------------------------------------- the pass
const VERT = `
precision highp float;
uniform mat4 modelViewMatrix; uniform mat4 projectionMatrix;
in vec3 position; in vec2 uv;
out vec2 vUv;
void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;

const FRAG = `
precision highp float;
precision highp sampler3D;
uniform sampler2D tDiffuse;
uniform vec2 texel;
uniform vec3 wb;
uniform float sharpen, contrast, brightness, highlights, shadows, saturation, split, fade, vignette, grain, seed, aspect;
uniform vec3 splitShadows, splitHighlights;
#ifdef USE_LUT
uniform sampler3D tLut; uniform float lutSize, lutMix; uniform vec3 lutMin, lutMax;
#endif
#include <tonemapping_pars_fragment>
#include <colorspace_pars_fragment>
in vec2 vUv;
out vec4 outColor;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

// the scene's light (linear) to the picture's tones (sRGB, 0..1): white balance,
// exposure (in toneMappingExposure) and the film response
vec3 film(vec3 c){
  c = max(c * wb, 0.0);
  #if defined(LOOK_AGX)
    c = AgXToneMapping(c);
  #elif defined(LOOK_NEUTRAL)
    c = NeutralToneMapping(c);
  #else
    c = ACESFilmicToneMapping(c);
  #endif
  return clamp(sRGBTransferOETF(vec4(c, 1.0)).rgb, 0.0, 1.0);
}
// the scene's light at a point: anything not a number (a stray, too bright sample) as black
vec3 src(vec2 at){
  vec3 c = texture(tDiffuse, at).rgb;
  if(any(isnan(c)) || any(isinf(c)) || !all(lessThan(abs(c), vec3(1.0e30)))) c = vec3(0.0);
  return clamp(c, 0.0, 6.0e4);
}
float hash(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

void main(){
  vec3 c = film(src(vUv));
  if(sharpen > 0.0){
    vec3 n = film(src(vUv + vec2(texel.x, 0.0))) + film(src(vUv - vec2(texel.x, 0.0)))
           + film(src(vUv + vec2(0.0, texel.y))) + film(src(vUv - vec2(0.0, texel.y)));
    c = clamp(c + sharpen * (c - 0.25 * n), 0.0, 1.0);
  }
  #ifdef USE_LUT
    vec3 at = clamp((c - lutMin) / (lutMax - lutMin), 0.0, 1.0);
    c = mix(c, clamp(texture(tLut, at * ((lutSize - 1.0) / lutSize) + 0.5 / lutSize).rgb, 0.0, 1.0), lutMix);
  #endif
  // shadows and highlights: lifted or lowered, the ends (black, white) kept
  vec3 sh = c * (1.0 - c) * (1.0 - c) * 6.75, hi = c * c * (1.0 - c) * 6.75;
  c = clamp(c + 0.14 * shadows * sh + 0.14 * highlights * hi, 0.0, 1.0);
  // contrast: an S round the middle grey (or its reverse), black and white kept
  float k = exp2(contrast);
  c = mix(0.5 * pow(2.0 * c, vec3(k)), 1.0 - 0.5 * pow(2.0 - 2.0 * c, vec3(k)), step(0.5, c));
  // brightness: the middle tones brighter or darker
  c = pow(c, vec3(exp2(-brightness)));
  // saturation, then split toning: a colour in the shadows, another in the highlights
  float l = dot(c, LUMA);
  c = max(mix(vec3(l), c, saturation), 0.0);
  l = dot(c, LUMA);
  c += split * ((1.0 - l) * (1.0 - l) * (splitShadows - dot(splitShadows, LUMA))
              + l * l * (splitHighlights - dot(splitHighlights, LUMA)));
  // fade: black lifted to grey, as on old prints
  c = fade + clamp(c, 0.0, 1.0) * (1.0 - fade);
  // vignette: darker towards the corners
  vec2 d = (vUv - 0.5) * vec2(aspect, 1.0) / (0.5 * sqrt(aspect * aspect + 1.0));
  c *= 1.0 - vignette * pow(smoothstep(0.25, 1.0, length(d)), 1.6);
  // film grain (most in the middle tones), and a touch of dither against banding in the sky
  vec2 p = gl_FragCoord.xy;
  float t = hash(p + seed * vec2(37.1, 17.3)) + hash(p * 1.37 + seed * vec2(11.7, 59.2)) - 1.0;
  l = dot(c, LUMA);
  c += t * (grain * 0.16 * (0.25 + 3.0 * l * (1.0 - l)) + 0.6 / 255.0);
  outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}`;

const hex = h => new THREE.Color().setStyle(h, THREE.SRGBColorSpace).getRGB(new THREE.Color(), THREE.SRGBColorSpace);

export class GradePass extends Pass {
  constructor(){
    super();
    this.uniforms = {
      tDiffuse: { value: null }, toneMappingExposure: { value: 1 }, texel: { value: new THREE.Vector2() }, wb: { value: new THREE.Vector3(1, 1, 1) },
      sharpen: { value: 0 }, contrast: { value: 0 }, brightness: { value: 0 }, highlights: { value: 0 }, shadows: { value: 0 },
      saturation: { value: 1 }, split: { value: 0 }, splitShadows: { value: new THREE.Vector3() }, splitHighlights: { value: new THREE.Vector3() },
      fade: { value: 0 }, vignette: { value: 0 }, grain: { value: 0 }, seed: { value: 0 }, aspect: { value: 1 },
      tLut: { value: null }, lutSize: { value: 2 }, lutMix: { value: 1 }, lutMin: { value: new THREE.Vector3(0, 0, 0) }, lutMax: { value: new THREE.Vector3(1, 1, 1) },
    };
    this.material = new THREE.RawShaderMaterial({ name: 'LookShader', uniforms: this.uniforms, vertexShader: VERT, fragmentShader: FRAG,
                                                  glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false });
    this.quad = new FullScreenQuad(this.material);
    this.settings = { ...NEUTRAL };
    this.lut = null;                       // { texture, size, min, max }
    this.frame = 0;
    this._key = null;
    this.set(this.settings);
  }

  // the look's settings (any not given: neutral)
  set(s){
    const L = this.settings = { ...NEUTRAL, ...s };
    const u = this.uniforms;
    // white balance: warmer (more red, less blue) or cooler; tint: magenta or green;
    // kept at the same brightness
    const w = L.warmth * 0.22, t = L.tint * 0.16;
    const g = [1 + w, 1 - t, 1 - w];
    const y = 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2];
    u.wb.value.set(g[0] / y, g[1] / y, g[2] / y);
    u.sharpen.value = Math.max(0, L.sharpen) * 1.2;
    u.contrast.value = L.contrast * 0.8;
    u.brightness.value = L.brightness * 0.6;
    u.highlights.value = L.highlights; u.shadows.value = L.shadows;
    u.saturation.value = Math.max(0, L.saturation);
    u.split.value = Math.max(0, L.split) * 0.35;
    const a = hex(L.split_shadows), b = hex(L.split_highlights);
    u.splitShadows.value.set(a.r, a.g, a.b); u.splitHighlights.value.set(b.r, b.g, b.b);
    u.fade.value = Math.max(0, L.fade) * 0.22;
    u.vignette.value = Math.min(1, Math.max(0, L.vignette)) * 0.85;
    u.grain.value = Math.max(0, L.grain);
    u.lutMix.value = Math.min(1, Math.max(0, L.lut_mix));
    this._defines();
  }

  setLut(lut){
    if(this.lut && this.lut !== lut) this.lut.texture.dispose();
    this.lut = lut;
    const u = this.uniforms;
    if(lut){ u.tLut.value = lut.texture; u.lutSize.value = lut.size; u.lutMin.value.fromArray(lut.min); u.lutMax.value.fromArray(lut.max); }
    else u.tLut.value = null;
    this._defines();
  }

  _defines(){
    const film = { agx: 'LOOK_AGX', neutral: 'LOOK_NEUTRAL', aces: 'LOOK_ACES' }[this.settings.film] || 'LOOK_AGX';
    const key = film + (this.lut ? '+lut' : '');
    if(key === this._key) return;
    this._key = key;
    this.material.defines = { [film]: '' };
    if(this.lut) this.material.defines.USE_LUT = '';
    this.material.needsUpdate = true;
  }

  // the picture's brightness: the time of day's exposure, then the look's (in stops)
  exposureOf(base){ return base * Math.pow(2, this.settings.exposure); }

  render(renderer, writeBuffer, readBuffer){
    const u = this.uniforms;
    u.tDiffuse.value = readBuffer.texture;
    u.toneMappingExposure.value = this.exposureOf(renderer.toneMappingExposure);
    u.texel.value.set(1 / readBuffer.width, 1 / readBuffer.height);
    u.aspect.value = readBuffer.width / readBuffer.height;
    u.seed.value = (this.frame++ % 997) + 1;
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    if(!this.renderToScreen && this.clear) renderer.clear();
    this.quad.render(renderer);
  }

  dispose(){ this.material.dispose(); this.quad.dispose(); if(this.lut) this.lut.texture.dispose(); }
}

// ----------------------------------------------------------------- depth of field
// What a camera lens does: what is at the focus distance sharp, nearer and further
// blurred by the circle a point makes on the film, worked out as the lens would (thin
// lens: the focal length from the view's angle on a 35 mm frame, the aperture its f-stop),
// so the live view blurs as Render photo does with the same settings (its path tracer
// traces the lens itself). Each pixel gathers the pixels round it whose own blur reaches
// it, so a sharp subject keeps its edges against a blurred background, and a blurred
// foreground spreads over what is behind it.
const DOF_FRAG = `
precision highp float;
uniform sampler2D tDiffuse, tDepth;
uniform vec2 texel;
uniform float near, far, focus, lensF, aperture, filmH, maxCoc;
in vec2 vUv;
out vec4 outColor;
float viewZ(vec2 at){
  float d = texture(tDepth, at).x;
  return near * far / max(far - d * (far - near), 1e-6);
}
// the blur circle's radius in pixels for something z metres away (in front: negative)
float coc(float z){
  float c = aperture * lensF * (z - focus) / max(z * (focus - lensF), 1e-6);
  return clamp(0.5 * c / filmH / texel.y, -maxCoc, maxCoc);
}
vec3 src(vec2 at){
  vec3 c = texture(tDiffuse, at).rgb;
  if(any(isnan(c)) || any(isinf(c))) c = vec3(0.0);
  return clamp(c, 0.0, 6.0e4);
}
void main(){
  float zc = viewZ(vUv), cc = coc(zc);
  vec3 sum = src(vUv);
  float wsum = 1.0;
  // a spiral of samples out to the largest blur there can be
  const int N = 64;
  for(int i = 1; i < N; i++){
    float r = sqrt(float(i) / float(N)) * maxCoc;
    float a = float(i) * 2.39996323;
    vec2 at = vUv + vec2(cos(a), sin(a)) * r * texel;
    float zs = viewZ(at), cs = coc(zs);
    // a sample behind this pixel is seen only as far as this pixel's own blur lets it
    float reach = zs > zc ? min(abs(cs), abs(cc)) : abs(cs);
    float w = smoothstep(r - 1.0, r + 0.5, reach);
    sum += src(at) * w; wsum += w;
  }
  outColor = vec4(sum / wsum, 1.0);
}`;

export class DofPass extends Pass {
  constructor(){
    super();
    this.uniforms = { tDiffuse: { value: null }, tDepth: { value: null }, texel: { value: new THREE.Vector2() },
      near: { value: 0.5 }, far: { value: 8000 }, focus: { value: 20 }, lensF: { value: 0.035 }, aperture: { value: 0.0125 },
      filmH: { value: 0.024 }, maxCoc: { value: 16 } };
    this.material = new THREE.RawShaderMaterial({ name: 'DofShader', uniforms: this.uniforms, vertexShader: VERT, fragmentShader: DOF_FRAG,
                                                  glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false });
    this.quad = new FullScreenQuad(this.material);
    this.depth = null;                     // the scene's depth texture, from the contact shadows' pass
    this.camera = null;
    this.fStop = 2.8;
    this.focus = 20;
    this.enabled = false;
  }

  // with the camera as it is now (its near and far follow the zoom); the largest blur is
  // capped so the gathering stays quick
  render(renderer, writeBuffer, readBuffer){
    const u = this.uniforms, cam = this.camera;
    const f = cam.getFocalLength() / 1000, filmH = cam.getFilmHeight() / 1000;
    u.tDiffuse.value = readBuffer.texture; u.tDepth.value = this.depth;
    u.texel.value.set(1 / readBuffer.width, 1 / readBuffer.height);
    u.near.value = cam.near; u.far.value = cam.far;
    u.focus.value = Math.max(this.focus, f * 1.01); u.lensF.value = f; u.aperture.value = f / this.fStop; u.filmH.value = filmH;
    // the furthest blur: something at infinity (or very near), in pixels, within 3% of the height
    const inf = 0.5 * (f / this.fStop) * f / Math.max(u.focus.value - f, 1e-6) / filmH * readBuffer.height;
    u.maxCoc.value = Math.min(Math.max(inf * 1.5, 1), 0.03 * readBuffer.height, 40);
    renderer.setRenderTarget(this.renderToScreen ? null : writeBuffer);
    if(this.clear) renderer.clear();
    this.quad.render(renderer);
  }

  dispose(){ this.material.dispose(); this.quad.dispose(); }
}

// ----------------------------------------------------------------- LUTs
// a .cube file (as the server stores them: 3D, red changing fastest) as a 3D texture
export function parseCube(text){
  let size = 0, min = [0, 0, 0], max = [1, 1, 1], n = 0, data = null;
  for(const raw of text.split('\n')){
    const line = raw.trim();
    if(!line || line[0] === '#') continue;
    const head = line.split(/\s+/);
    if(head[0] === 'LUT_3D_SIZE'){ size = +head[1]; data = new Float32Array(size * size * size * 4); continue; }
    if(head[0] === 'DOMAIN_MIN'){ min = head.slice(1, 4).map(Number); continue; }
    if(head[0] === 'DOMAIN_MAX'){ max = head.slice(1, 4).map(Number); continue; }
    if(!/^[-+0-9.]/.test(head[0]) || !data) continue;
    data[n * 4] = +head[0]; data[n * 4 + 1] = +head[1]; data[n * 4 + 2] = +head[2]; data[n * 4 + 3] = 1; n++;
  }
  if(!size || n !== size * size * size) throw new Error(`not a 3D LUT (${n} rows for size ${size})`);
  // half floats: smooth (filtered) on every WebGL2 device, unlike full floats
  const half = new Uint16Array(data.length);
  for(let i = 0; i < data.length; i++) half[i] = THREE.DataUtils.toHalfFloat(data[i]);
  const texture = new THREE.Data3DTexture(half, size, size, size);
  texture.format = THREE.RGBAFormat; texture.type = THREE.HalfFloatType;
  texture.minFilter = texture.magFilter = THREE.LinearFilter;
  texture.wrapS = texture.wrapT = texture.wrapR = THREE.ClampToEdgeWrapping;
  texture.unpackAlignment = 1; texture.needsUpdate = true;
  return { texture, size, min, max };
}

// ----------------------------------------------------------------- the panel
// [setting, label, slider min, max, slider units per setting unit, shown as, what it does]
const SLIDERS = [
  ['exposure', 'Exposure', -300, 300, 100, v => (v > 0 ? '+' : '') + v.toFixed(1) + ' EV', 'Brighter or darker, in stops, as a camera\'s exposure: before the film response, so highlights roll off softly'],
  ['contrast', 'Contrast', -100, 100, 100, v => signed(v), 'More or less contrast round the middle grey; black and white stay put'],
  ['brightness', 'Brightness', -100, 100, 100, v => signed(v), 'The middle tones brighter or darker; black and white stay put'],
  ['highlights', 'Highlights', -100, 100, 100, v => signed(v), 'The bright tones lowered (bring back a bright sky) or raised'],
  ['shadows', 'Shadows', -100, 100, 100, v => signed(v), 'The dark tones lifted (see into the shade) or deepened'],
  ['saturation', 'Saturation', 0, 200, 100, v => Math.round(v * 100) + '%', 'How strong the colours are. 0: black and white'],
  ['warmth', 'Warmth', -100, 100, 100, v => signed(v), 'White balance: warmer (towards orange) or cooler (towards blue)'],
  ['tint', 'Tint', -100, 100, 100, v => signed(v), 'White balance: towards magenta or towards green'],
  ['split', 'Split toning', 0, 100, 100, v => Math.round(v * 100) + '%', 'One colour in the shadows and another in the highlights (pick them below), as in the teal and orange of films'],
  ['fade', 'Fade', 0, 100, 100, v => Math.round(v * 100) + '%', 'Blacks lifted to grey, as on an old print'],
  ['vignette', 'Vignette', 0, 100, 100, v => Math.round(v * 100) + '%', 'Darker towards the corners'],
  ['sharpen', 'Sharpen', 0, 100, 100, v => Math.round(v * 100) + '%', 'Crisper fine detail'],
  ['grain', 'Film grain', 0, 100, 100, v => Math.round(v * 100) + '%', 'Grain like film, mostly in the middle tones'],
  ['glow', 'Glow', 0, 300, 100, v => Math.round(v * 100) + '%', 'The glow round bright lights (lamps, the sun on glass and water), from the time of day\'s own: 100%'],
  ['lut_mix', 'LUT strength', 0, 100, 100, v => Math.round(v * 100) + '%', 'How much of the LUT is applied'],
];
const signed = v => (v > 0 ? '+' : '') + Math.round(v * 100);
const KEEP = 'rta.view3d.look';

export function lookPanel(pass, changed){
  const $ = s => document.querySelector(s);
  const host = $('#lookSliders');
  if(!host) return;
  let looks = [], luts = [], chosen = 'p:neutral', base = null;
  const lutCache = new Map();
  // the sliders, made from the list above
  for(const [key, label, min, max, , , title] of SLIDERS){
    const row = document.createElement('div');
    row.className = 'field';
    row.innerHTML = `<label for="lk_${key}">${label} <span id="lk_${key}Val" class="mono" style="color:var(--faint)"></span></label>`
      + `<input type="range" id="lk_${key}" min="${min}" max="${max}" step="1" title="${title}">`;
    (key === 'lut_mix' && $('#lookLutMix') ? $('#lookLutMix') : host).appendChild(row);
    if(key === 'split'){
      const c = document.createElement('div');
      c.className = 'field';
      c.innerHTML = '<label>Shadows / highlights colour</label><div style="display:flex;gap:6px">'
        + '<input type="color" id="lk_split_shadows" title="The colour put into the shadows" style="width:52px;height:26px;padding:0;border:1px solid var(--line);background:none">'
        + '<input type="color" id="lk_split_highlights" title="The colour put into the highlights" style="width:52px;height:26px;padding:0;border:1px solid var(--line);background:none"></div>';
      host.appendChild(c);
    }
  }
  let cur = { ...NEUTRAL };
  try{
    const kept = JSON.parse(localStorage.getItem(KEEP) || 'null');
    if(kept){ cur = { ...NEUTRAL, ...kept.settings }; chosen = kept.chosen || 'custom'; base = kept.base || null; }
  }catch(e){}
  const keep = () => { try{ localStorage.setItem(KEEP, JSON.stringify({ settings: cur, chosen, base })); }catch(e){} };
  const note = (t, bad) => { const n = $('#lookNote'); n.textContent = t; n.classList.toggle('bad', !!bad); };

  function show(){
    for(const [key, , , , k, fmt] of SLIDERS){
      $('#lk_' + key).value = Math.round(cur[key] * k);
      $('#lk_' + key + 'Val').textContent = fmt(cur[key]);
    }
    $('#lk_split_shadows').value = cur.split_shadows; $('#lk_split_highlights').value = cur.split_highlights;
    $('#lookFilm').value = cur.film;
    $('#lookLut').value = cur.lut || '';
    $('#lk_lut_mix').disabled = !cur.lut;
    $('#lookPreset').value = [...$('#lookPreset').options].some(o => o.value === chosen) ? chosen : 'custom';
    const mine = chosen.startsWith('u:') && looks.find(l => 'u:' + l.id === chosen);
    $('#btnLookDelete').disabled = !mine;
    if(mine && !$('#lookName').value) $('#lookName').value = mine.name;
    $('#btnLutDelete').disabled = !(cur.lut && luts.find(l => l.id === cur.lut && l.source === 'yours'));
  }

  async function lutFor(id){
    if(!id) return null;
    if(!lutCache.has(id)){
      const r = await fetch('/api/looks/lut/' + id);
      if(!r.ok) throw new Error('the LUT is no longer there');
      lutCache.set(id, parseCube(await r.text()));
    }
    return lutCache.get(id);
  }

  let lutToken = 0;
  async function apply(){
    pass.set(cur);
    const want = cur.lut, token = ++lutToken;
    try{
      const lut = await lutFor(want);
      if(token !== lutToken) return;
      // a LUT texture stays cached: the pass must not dispose of it when another is picked
      pass.lut = null;
      pass.setLut(lut);
    }catch(e){
      if(token !== lutToken) return;
      pass.lut = null; pass.setLut(null);
      cur.lut = null;
      note('The LUT could not be loaded: ' + e.message, true);
    }
    show(); keep();
    if(changed) changed(cur);
  }

  function pick(value){
    chosen = value;
    const [kind, id] = [value.slice(0, 1), value.slice(2)];
    let s = null, name = '';
    if(kind === 'p'){ const p = PRESETS.find(x => x.id === id); if(p){ s = p.settings; name = p.name; } }
    else { const l = looks.find(x => x.id === id); if(l){ s = l.settings; name = l.name; } }
    if(!s) return;
    cur = { ...NEUTRAL, ...s };
    base = name;
    $('#lookName').value = kind === 'u' ? name : '';
    note(kind === 'p' ? `${name}: a built-in look. Change anything, then Save to keep it as your own.`
                      : `${name}: ${kind === 'u' ? 'your saved look' : 'a look that came with the program'}.`);
    apply();
  }

  function fill(){
    const sel = $('#lookPreset');
    const group = (label, items) => items.length ? `<optgroup label="${label}">${items.join('')}</optgroup>` : '';
    const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
    sel.innerHTML = '<option value="custom" disabled>Custom</option>'
      + group('Presets', PRESETS.map(p => `<option value="p:${p.id}">${esc(p.name)}</option>`))
      + group('Came with the program', looks.filter(l => l.source === 'bundled').map(l => `<option value="b:${l.id}">${esc(l.name)}</option>`))
      + group('Your looks', looks.filter(l => l.source === 'yours').map(l => `<option value="u:${l.id}">${esc(l.name)}</option>`));
    $('#lookLut').innerHTML = '<option value="">None</option>'
      + group('Came with the program', luts.filter(l => l.source === 'bundled').map(l => `<option value="${l.id}">${esc(l.name)}</option>`))
      + group('Yours', luts.filter(l => l.source === 'yours').map(l => `<option value="${l.id}">${esc(l.name)}${l.size ? ` (${l.size}³)` : ''}</option>`));
  }

  async function load(){
    try{
      const r = await fetch('/api/looks');
      const j = await r.json();
      looks = j.looks || []; luts = j.luts || [];
    }catch(e){ looks = []; luts = []; }
    fill();
    if(cur.lut && !luts.find(l => l.id === cur.lut)) cur.lut = null;
  }

  // a slider or colour moved: the look is now custom (based on the last one picked)
  const edited = () => {
    if(chosen !== 'custom'){ base = base || 'Neutral'; chosen = 'custom'; note(`Custom, from ${base}. Save keeps it.`); }
    apply();
  };
  for(const [key, , , , k] of SLIDERS){
    $('#lk_' + key).addEventListener('input', () => { cur[key] = +$('#lk_' + key).value / k; edited(); });
    // double-click a slider: back to its neutral value
    $('#lk_' + key).addEventListener('dblclick', () => { cur[key] = NEUTRAL[key]; edited(); });
  }
  for(const key of ['split_shadows', 'split_highlights'])
    $('#lk_' + key).addEventListener('input', () => { cur[key] = $('#lk_' + key).value; edited(); });
  $('#lookFilm').addEventListener('change', () => { cur.film = $('#lookFilm').value; edited(); });
  $('#lookLut').addEventListener('change', () => { cur.lut = $('#lookLut').value || null; if(cur.lut) cur.lut_mix = cur.lut_mix || 1; edited(); });
  $('#lookPreset').addEventListener('change', () => pick($('#lookPreset').value));
  $('#btnLookReset').addEventListener('click', () => pick('p:neutral'));

  $('#btnLookSave').addEventListener('click', async () => {
    const name = $('#lookName').value.trim() || `My look ${looks.filter(l => l.source === 'yours').length + 1}`;
    // the same name as the saved look picked: that look is updated
    const same = chosen.startsWith('u:') ? looks.find(l => 'u:' + l.id === chosen && l.name === name) : null;
    const prior = looks.find(l => l.source === 'yours' && l.name === name);
    const r = await fetch('/api/looks', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, settings: cur, id: (same || prior || {}).id }) });
    if(!r.ok){ note('Could not save the look: ' + (await r.text()), true); return; }
    const lk = await r.json();
    await load();
    chosen = 'u:' + lk.id; base = lk.name; $('#lookName').value = lk.name;
    note(`Saved as "${lk.name}" in your workspace (looks folder).`);
    show(); keep();
  });

  $('#btnLookDelete').addEventListener('click', async () => {
    const l = looks.find(x => 'u:' + x.id === chosen);
    if(!l) return;
    await fetch('/api/looks/' + l.id, { method: 'DELETE' });
    await load();
    chosen = 'custom'; $('#lookName').value = '';
    note(`Deleted "${l.name}". The picture keeps its settings until you pick another look.`);
    show(); keep();
  });

  $('#btnLutDelete').addEventListener('click', async () => {
    const l = luts.find(x => x.id === cur.lut && x.source === 'yours');
    if(!l) return;
    await fetch('/api/looks/lut/' + l.id, { method: 'DELETE' });
    lutCache.delete(l.id);
    cur.lut = null;
    await load();
    note(`Removed the LUT "${l.name}".`);
    edited();
  });

  $('#btnLookImport').addEventListener('click', () => $('#lookFile').click());
  $('#lookFile').addEventListener('change', async () => {
    const files = [...$('#lookFile').files];
    $('#lookFile').value = '';
    let last = null;
    for(const f of files){
      const fd = new FormData();
      fd.append('file', f);
      const r = await fetch('/api/looks/import', { method: 'POST', body: fd });
      let j = null;
      try{ j = await r.json(); }catch(e){}
      if(!r.ok){ note(`${f.name}: ${(j && j.detail) || 'could not be imported'}`, true); continue; }
      last = j;
    }
    if(!last) return;
    await load();
    if(last.look){
      lutCache.clear();
      pick('u:' + last.look.id);
      note(`Imported the look "${last.look.name}"${last.lut ? ' with its LUT' : ''}.`);
    }else if(last.lut){
      cur.lut = last.lut.id; cur.lut_mix = 1;
      edited();
      note(`Imported the LUT "${last.lut.name}" (${last.lut.size}³) and applied it. Save keeps it in a look.`);
    }
  });

  $('#btnLookDownload').addEventListener('click', async () => {
    const name = $('#lookName').value.trim() || (chosen === 'custom' ? `${base || 'My'} look` : base) || 'Look';
    const { lut, ...settings } = cur;
    const file = { format: 'road-texture-agent look', version: 1, name, settings };
    if(lut){
      const r = await fetch('/api/looks/lut/' + lut);
      if(r.ok){ file.lut_cube = await r.text(); file.lut_name = (luts.find(l => l.id === lut) || {}).name || 'LUT'; }
    }
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(file, null, 1)], { type: 'application/json' }));
    a.download = name.replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '_') + '.look.json';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  });

  load().then(() => {
    if(chosen !== 'custom' && !chosen.startsWith('p:') && !looks.find(l => l.id === chosen.slice(2))) chosen = 'custom';
    apply();
    if(chosen === 'custom' && base) note(`Custom, from ${base}.`);
  });
  return { get: () => ({ ...cur }), set: s => { cur = { ...NEUTRAL, ...s }; chosen = 'custom'; return apply(); }, pick, reload: load };
}
