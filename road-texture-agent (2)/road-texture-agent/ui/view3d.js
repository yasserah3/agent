// The 3D tab: the whole place in 3D, to look around. Generate 3D scene builds the
// streets, inner streets, sidewalks, kerbs, blocks and objects from the last
// generated texture (the same model as the GLB export, with the 3D model
// settings of the Generate tab) and adds street lamps along the sidewalks.
// The live view has a sky, the sun with its shadows, soft contact shadows
// (ambient occlusion), filmic colour and a glow around bright lights. The sky
// is Blender's own physical sky (ui/sky_blender.js, ported from Blender), so
// the sky, the colour of the sunlight and the balance of sun and sky are
// those of Blender and Cycles. Render
// photo then traces the light properly (three-gpu-pathtracer), a still that
// sharpens for as long as the camera stays put, its noise cleared by Intel Open
// Image Denoise (ui/denoise.js, ported to WebGL2). Dawn, Day and Night set the
// sun, the sky and the street lamps. Metres, Y up: the map's x is +X, its y +Z,
// the top of the map north. Everything is bundled in ui/vendor, so it works offline.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildSkyMaps } from './sky_blender.js';
import { Denoiser } from './denoise.js';

const $ = s => document.querySelector(s);
const host = $('#view3d'), msg = $('#view3dMsg');

// ----------------------------------------------------------------- times of day
// sun (or moon) height and compass direction in degrees (0 north, 90 east). By
// day and at dawn Blender's sky gives the sun's colour and strength and the
// sky's light; the exposure is set from them like a camera's (bias: brighter or
// darker than that). At night: the moon's colour and strength, the exposure,
// and how much the night sky lights the scene. Then the street lamps (0 off,
// 1 full), the glow (strength, radius, and from how bright, on screen), the
// haze, and the land's colour
const TIMES = {
  dawn:  { elev: 4, azim: 100, sky: true, bias: 0.7, env: 1, lamps: 0.6, bloom: [0.25, 0.35, 3.0],
           fog: [0x8a7f7c, 0.00035], ground: 0x4a4d3c },
  day:   { elev: 52, azim: 215, sky: true, bias: 1.0, env: 1, lamps: 0, bloom: [0.12, 0.4, 6.0],
           fog: [0xc9d6e0, 0.00035], ground: 0x5c6648 },
  night: { elev: 38, azim: 300, color: 0x9fb6ff, sun: 0.12, sky: false, exposure: 0.9, env: 0.15, lamps: 1,
           bloom: [0.6, 0.5, 1.5], fog: [0x070b16, 0.0005], ground: 0x2a2e26 },
};
const LAMP = { height: 8.0, arm: 1.6, candela: 320, pool: 12 };
const PHOTO_LAMPS = 1000;

// the Light panel: the sun's (or moon's) strength and the street
// lamps' brightness and colour, kept in this browser for the next time
const LIGHT = { sun: 1, lamps: 1, colour: '#ffcf96' };
try{ Object.assign(LIGHT, JSON.parse(localStorage.getItem('rta.view3d.light') || '{}')); }catch(e){}
const saveLight = () => { try{ localStorage.setItem('rta.view3d.light', JSON.stringify(LIGHT)); }catch(e){} };
// Render photo's noise cleared by Open Image Denoise (on unless turned off here before)
let DENOISE = true;
try{ DENOISE = localStorage.getItem('rta.view3d.denoise') !== 'off'; }catch(e){}

let renderer = null, scene, camera, controls, composer, renderPass, gtao, bloom, output;
let envScene, pmrem, envRT = null, nightTex = null, stars = null, ground, skyNow = null, skyToken = 0, sunBase = 1;
let sun, hemi, world = null, lamps = null, time = 'day', radius = 300, centre = new THREE.Vector3();
let visible = false, running = false, poolAt = null, photo = null, busy = false;

// ----------------------------------------------------------------- set up
function init(){
  try{
    renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
  }catch(e){
    renderer = null;
    msg.textContent = 'This browser cannot show 3D here: WebGL is switched off or not available.';
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.style.display = 'block';
  host.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, 1, 0.5, 8000);
  // left drag: orbit, right drag (or Shift + drag): pan, wheel: zoom
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;                          // never below the ground
  controls.maxDistance = 2500;
  controls.addEventListener('start', () => { if(photo && photo.on) stopPhoto('The camera moved: back to the live view.'); });

  // the night sky's light from all round is worked out from a scene of its own
  envScene = new THREE.Scene();
  pmrem = new THREE.PMREMGenerator(renderer);

  sun = new THREE.DirectionalLight(0xffffff, 3);
  sun.castShadow = true;
  const size = renderer.capabilities.maxTextureSize >= 8192 ? 4096 : 2048;
  sun.shadow.mapSize.set(size, size);
  sun.shadow.bias = -0.0002;
  sun.shadow.normalBias = 0.04;
  scene.add(sun, sun.target);
  hemi = new THREE.HemisphereLight(0xbcd2ff, 0x40382c, 0.0);           // a touch of fill at night only
  scene.add(hemi);
  // the land round the place, fading into the haze at the horizon
  ground = new THREE.Mesh(new THREE.CircleGeometry(6000, 96).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x5c6648, roughness: 1 }));
  ground.position.y = -0.3;                                           // well under the streets, blocks and islands
  ground.receiveShadow = true;
  ground.name = 'ground';
  scene.add(ground);
  const grid = new THREE.GridHelper(500, 50, 0x6a7280, 0x353b45);    // until a scene is built: 10 m squares
  grid.name = 'grid';
  scene.add(grid);

  // the picture: the scene, contact shadows, the glow of bright lights, then filmic colour
  const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  composer = new EffectComposer(renderer, rt);
  renderPass = new RenderPass(scene, camera);
  gtao = new GTAOPass(scene, camera, 1, 1);
  gtao.updateGtaoMaterial({ radius: 1.6, distanceExponent: 1.5, thickness: 1.5, scale: 1.0, samples: 16 });
  gtao.blendIntensity = 0.9;
  bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.2, 0.4, 4.0);
  output = new OutputPass();
  composer.addPass(renderPass); composer.addPass(gtao); composer.addPass(bloom); composer.addPass(output);
  // the pools of lamp light are painted on, not surfaces: left out of the contact shadows
  const hide = gtao._overrideVisibility.bind(gtao);
  gtao._overrideVisibility = function(){
    hide();
    if(lamps && lamps.glows.visible){ lamps.glows.visible = false; this._visibilityCache.push(lamps.glows); }
  };

  setTime(time, false);
  // the other skies are worked out meanwhile, so changing the time is quick
  for(const t of Object.values(TIMES)) if(t.sky) skyFor(t);
  frame(new THREE.Vector3(), 250);
  new ResizeObserver(resize).observe(host);
  return true;
}

function resize(){
  const w = host.clientWidth, h = host.clientHeight;
  if(!renderer || !w || !h) return;
  const now = renderer.getSize(new THREE.Vector2());
  if(now.x === w && now.y === h) return;                               // the same size: nothing to do
  renderer.setSize(w, h);
  composer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if(photo && photo.on) stopPhoto('The view changed size: back to the live view.');
}

function frame(c, r){
  // looking at the middle from the south-east, the whole place in view
  centre.copy(c); radius = r;
  camera.position.set(c.x + 0.55 * r, Math.max(0.6 * r, 30), c.z + 0.85 * r);
  controls.target.copy(c);
  controls.maxDistance = Math.max(400, 3 * r);
  controls.update();
}

// ----------------------------------------------------------------- times of day
function sunDir(t){
  const e = THREE.MathUtils.degToRad(t.elev), a = THREE.MathUtils.degToRad(t.azim);
  return new THREE.Vector3(Math.cos(e) * Math.sin(a), Math.sin(e), -Math.cos(e) * Math.cos(a));
}

function nightSky(){
  // a deep blue sky, darker overhead, with stars and a little moonlit haze
  if(nightTex) return nightTex;
  const W = 2048, H = 1024, c = document.createElement('canvas');
  c.width = W; c.height = H;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0.0, '#020309'); grad.addColorStop(0.35, '#050a18'); grad.addColorStop(0.5, '#16213a');
  grad.addColorStop(0.53, '#0e1424'); grad.addColorStop(1.0, '#06080e');
  g.fillStyle = grad; g.fillRect(0, 0, W, H);
  nightTex = new THREE.CanvasTexture(c);
  nightTex.mapping = THREE.EquirectangularReflectionMapping;
  nightTex.colorSpace = THREE.SRGBColorSpace;
  return nightTex;
}

function starField(){
  // stars as points: sharp at any size of the view, the brighter ones fewer
  if(stars) return stars;
  let seed = 7;
  const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const n = 2400, pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
  for(let i = 0; i < n; i++){
    const y = Math.pow(rnd(), 0.7) * 0.98 + 0.02, a = rnd() * Math.PI * 2, r = Math.sqrt(1 - y * y);
    pos.set([Math.cos(a) * r * 5000, y * 5000, Math.sin(a) * r * 5000], i * 3);
    const b = 0.25 + 0.75 * Math.pow(rnd(), 4);
    col.set([b, b * (0.93 + 0.07 * rnd()), b * (0.85 + 0.15 * rnd())], i * 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  stars = new THREE.Points(geo, new THREE.PointsMaterial({ size: 1.6, sizeAttenuation: false, vertexColors: true, fog: false,
    depthWrite: false, toneMapped: false }));
  stars.name = 'stars';
  stars.frustumCulled = false;
  return stars;
}

// Blender's sky for a time of day, worked out once (a second or two, off the page
// in a worker where the browser allows) and kept
const skyCache = new Map();
let skyWorker, skyJobs = new Map(), skyJob = 0;
function skyFor(t){
  const key = t.elev + '/' + t.azim;
  if(skyCache.has(key)) return skyCache.get(key);
  const args = { elevation: THREE.MathUtils.degToRad(t.elev), azimuth: THREE.MathUtils.degToRad(t.azim),
                 width: 2048, height: 1024, discScale: 1e-3 };
  const job = new Promise(resolve => {
    try{
      if(skyWorker === undefined){
        skyWorker = new Worker(new URL('./sky_worker.js', import.meta.url), { type: 'module' });
        skyWorker.onmessage = e => { const done = skyJobs.get(e.data.id); skyJobs.delete(e.data.id); if(done) done(e.data); };
        skyWorker.onerror = () => { skyWorker = null; for(const [, done] of skyJobs) done(null); skyJobs.clear(); };
      }
      if(!skyWorker) throw new Error('no worker');
      const id = ++skyJob;
      skyJobs.set(id, resolve);
      skyWorker.postMessage({ id, ...args });
    }catch(e){ resolve(null); }
  }).then(d => d || buildSkyMaps(args))                               // no worker: here, on the page
    .then(d => {
      const tex = data => {
        const x = new THREE.DataTexture(data, d.width, d.height, THREE.RGBAFormat, THREE.HalfFloatType);
        x.mapping = THREE.EquirectangularReflectionMapping;
        x.colorSpace = THREE.LinearSRGBColorSpace;
        x.magFilter = x.minFilter = THREE.LinearFilter;
        x.generateMipmaps = false;
        x.needsUpdate = true;
        return x;
      };
      const plain = tex(d.plain);
      const Y = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
      return { plain, disc: tex(d.disc), env: pmrem.fromEquirectangular(plain),
               horizon: new THREE.Color().setRGB(d.horizon[0], d.horizon[1], d.horizon[2], THREE.LinearSRGBColorSpace),
               sunColor: new THREE.Color().setRGB(d.sun.color[0], d.sun.color[1], d.sun.color[2], THREE.LinearSRGBColorSpace),
               sunIrradiance: d.sun.irradiance,
               // the light on a level surface: the sun at its height, and the sky
               level: d.sun.irradiance * Math.max(Math.sin(args.elevation), 0) + Y(d.skyIrradiance) };
    });
  skyCache.set(key, job);
  return job;
}

function horizon(){
  // the colour of the night sky just above the horizon, all round. The middle of
  // eight directions, so a bright patch does not tint the whole haze
  try{
    const rt = new THREE.WebGLRenderTarget(32, 4, { type: THREE.FloatType });
    const cam = new THREE.PerspectiveCamera(4, 8, 1, 20000), px = new Float32Array(32 * 4 * 4), all = [];
    const keep = renderer.getRenderTarget(), tm = renderer.toneMapping;
    renderer.toneMapping = THREE.NoToneMapping;
    try{
      for(let k = 0; k < 8; k++){
        const a = k * Math.PI / 4;
        cam.position.set(0, 0, 0);
        cam.lookAt(Math.sin(a), Math.tan(THREE.MathUtils.degToRad(1.5)), -Math.cos(a));
        renderer.setRenderTarget(rt);
        renderer.render(envScene, cam);
        renderer.readRenderTargetPixels(rt, 0, 0, 32, 4, px);
        const c = [0, 0, 0];
        for(let i = 0; i < px.length; i += 4){ c[0] += px[i]; c[1] += px[i + 1]; c[2] += px[i + 2]; }
        all.push(c.map(v => v / 128));
      }
    }finally{
      renderer.setRenderTarget(keep); renderer.toneMapping = tm;
      rt.dispose();
    }
    const mid = ch => { const v = all.map(c => c[ch]).sort((a, b) => a - b); return (v[3] + v[4]) / 2; };
    const col = new THREE.Color().setRGB(mid(0), mid(1), mid(2), THREE.LinearSRGBColorSpace);
    return Number.isFinite(col.r) ? col : null;
  }catch(e){ return null; }
}

function setTime(name, render = true){
  if(photo && photo.on) stopPhoto();
  time = name;
  const t = TIMES[name], token = ++skyToken;
  document.querySelectorAll('[data-time3d]').forEach(b => b.setAttribute('aria-pressed', b.dataset.time3d === name ? 'true' : 'false'));
  ground.material.color.set(t.ground);
  const st = starField();
  if(t.sky) scene.remove(st); else scene.add(st);
  hemi.intensity = t.sky ? 0 : 0.06;
  if(lamps) lampLevel(t.lamps);
  if(render) poolAt = null;
  if(!t.sky){
    // night: the moon, and the night sky's light from all round
    skyNow = null;
    envScene.background = nightSky();
    if(envRT) envRT.dispose();
    envRT = pmrem.fromScene(envScene, 0, 1, 10000);
    scene.environment = envRT.texture;
    scene.environmentIntensity = t.env;
    scene.background = nightSky();
    scene.fog = new THREE.FogExp2(horizon() || new THREE.Color(t.fog[0]), t.fog[1]);
    sun.color.set(t.color);
    sunBase = t.sun;
    sun.intensity = sunBase * LIGHT.sun;
    renderer.toneMappingExposure = t.exposure;
    setBloom(t, t.exposure);
    return;
  }
  // dawn and day: Blender's sky, worked out the first time
  const job = skyFor(t);
  let ready = false;
  job.then(sk => {
    ready = true;
    if(token !== skyToken) return;                                      // another time was chosen meanwhile
    skyNow = sk;
    scene.background = sk.disc;
    scene.backgroundIntensity = 1;
    scene.environment = sk.env.texture;
    scene.environmentIntensity = t.env;
    // the haze takes the sky's own colour at the horizon, so the land fades into it
    scene.fog = new THREE.FogExp2(sk.horizon, t.fog[1]);
    sun.color.copy(sk.sunColor);
    sunBase = sk.sunIrradiance;
    sun.intensity = sunBase * LIGHT.sun;
    // exposed like a camera for the light falling on a level surface: a grey card shows
    // mid grey; bias makes the time brighter or darker than that
    const exposure = t.bias * Math.PI / Math.max(sk.level, 1e-6);
    renderer.toneMappingExposure = exposure;
    setBloom(t, exposure);
    msg.textContent = '';
    if(photo && photo.on) stopPhoto();
  });
  setTimeout(() => { if(!ready && token === skyToken) msg.textContent = 'Working out the sky…'; }, 150);
}

function setBloom(t, exposure){
  // the glow starts from a brightness on screen: in the scene's own units, that over the exposure
  bloom.strength = t.bloom[0]; bloom.radius = t.bloom[1]; bloom.threshold = t.bloom[2] / exposure;
}

// the sun's shadow covers what is in view: a small area close up (sharp
// shadows), the whole place from far away
function fitShadow(){
  const t = TIMES[time], d = sunDir(t);
  const dist = camera.position.distanceTo(controls.target);
  const r = THREE.MathUtils.clamp(dist * 0.9, 40, Math.max(radius * 1.2, 60));
  const sc = sun.shadow.camera;
  const texel = 2 * r / sun.shadow.mapSize.x;
  // whole shadow-map texels, so shadows do not shimmer as the view moves
  const tgt = controls.target.clone();
  tgt.x = Math.round(tgt.x / texel) * texel; tgt.z = Math.round(tgt.z / texel) * texel;
  sun.target.position.copy(tgt);
  sun.position.copy(tgt).addScaledVector(d, r * 3);
  sc.left = -r; sc.right = r; sc.top = r; sc.bottom = -r;
  sc.near = 1; sc.far = r * 6;
  sc.updateProjectionMatrix();
}

// ----------------------------------------------------------------- street lamps
function lampGeometry(){
  // a pole on a base, an arm out over the road, and a flat lamp head under its end
  const pole = new THREE.CylinderGeometry(0.07, 0.11, LAMP.height, 10).translate(0, LAMP.height / 2, 0);
  const base = new THREE.CylinderGeometry(0.17, 0.2, 0.5, 12).translate(0, 0.25, 0);
  const arm = new THREE.BoxGeometry(LAMP.arm, 0.07, 0.07).translate(LAMP.arm / 2, LAMP.height - 0.05, 0);
  const cap = new THREE.BoxGeometry(0.75, 0.1, 0.32).translate(LAMP.arm - 0.15, LAMP.height - 0.12, 0);
  const metal = mergeGeometries([pole, base, arm, cap].map(g => g.toNonIndexed()));
  const head = new THREE.BoxGeometry(0.62, 0.04, 0.26).translate(LAMP.arm - 0.15, LAMP.height - 0.19, 0);
  return { metal, head };
}

function glowTexture(){
  // a soft pool of light, brightest under the lamp
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d'), grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
  grad.addColorStop(0.6, 'rgba(255,255,255,0.12)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function buildLamps(list){
  if(lamps){ scene.remove(lamps.group); lamps.group.traverse(o => { if(o.geometry) o.geometry.dispose(); }); }
  lamps = null;
  if(!list || !list.length) return;
  const n = list.length, { metal, head } = lampGeometry();
  const metalMat = new THREE.MeshStandardMaterial({ color: 0x3b3f44, metalness: 0.7, roughness: 0.45 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0x222222, emissive: 0xffd49a, emissiveIntensity: 0, roughness: 0.3 });
  const poles = new THREE.InstancedMesh(metal, metalMat, n), heads = new THREE.InstancedMesh(head, headMat, n);
  poles.castShadow = true; poles.receiveShadow = true;
  const glowMat = new THREE.MeshBasicMaterial({ map: glowTexture(), color: 0xffc98a, transparent: true, opacity: 0,
    blending: THREE.AdditiveBlending, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -2, polygonOffsetUnits: -4, fog: true });
  const glows = new THREE.InstancedMesh(new THREE.PlaneGeometry(18, 18).rotateX(-Math.PI / 2), glowMat, n);
  glows.renderOrder = 2;
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), up = new THREE.Vector3(0, 1, 0);
  const items = list.map(([x, y, z, dx, dz], i) => {
    q.setFromAxisAngle(up, Math.atan2(-dz, dx));                    // the arm (local +X) reaches over the road
    m.compose(new THREE.Vector3(x, y, z), q, new THREE.Vector3(1, 1, 1));
    poles.setMatrixAt(i, m); heads.setMatrixAt(i, m);
    const hx = x + dx * (LAMP.arm - 0.15), hz = z + dz * (LAMP.arm - 0.15);
    return { foot: new THREE.Vector3(x, y, z), head: new THREE.Vector3(hx, y + LAMP.height - 0.2, hz),
             ground: new THREE.Vector3(hx + dx * 1.5, y + 0.02, hz + dz * 1.5), dir: new THREE.Vector3(dx, 0, dz) };
  });
  items.forEach((it, i) => glows.setMatrixAt(i, m.makeTranslation(it.ground.x, it.ground.y, it.ground.z)));
  // the nearest lamps light the scene for real; the others show a soft pool of light
  const pool = [];
  for(let i = 0; i < LAMP.pool; i++){
    const s = new THREE.SpotLight(LIGHT.colour, 0, 60, 1.15, 0.85, 2);
    s.castShadow = false;
    pool.push(s);
  }
  const group = new THREE.Group();
  group.name = 'Street lamps';
  group.add(poles, heads, glows, ...pool, ...pool.map(s => s.target));
  scene.add(group);
  lamps = { group, poles, heads, glows, pool, items, level: 0, metalMat, headMat };
  lampLevel(TIMES[time].lamps);
  poolAt = null;
}

function lampLevel(level){
  // the time's level (off by day) times the Light panel's brightness, in its colour
  level *= LIGHT.lamps;
  lamps.level = level;
  const c = new THREE.Color(LIGHT.colour);
  lamps.headMat.emissive.copy(c);
  lamps.headMat.emissiveIntensity = 30 * Math.min(level, 1.5);
  lamps.glows.material.color.copy(c);
  lamps.glows.material.opacity = Math.min(0.3 * level, 0.9);
  lamps.glows.visible = level > 0;
  for(const s of lamps.pool){ s.visible = level > 0; s.color.copy(c); s.intensity = LAMP.candela * level; }
}

function assignPool(){
  // the lamps nearest to where the camera looks get the real lights
  if(!lamps || !lamps.level) return;
  const t = controls.target;
  if(poolAt && poolAt.distanceTo(t) < 4) return;
  poolAt = t.clone();
  const order = lamps.items.map((it, i) => [(it.foot.x - t.x) ** 2 + (it.foot.z - t.z) ** 2, i]).sort((a, b) => a[0] - b[0]);
  const lit = new Set(), m = new THREE.Matrix4();
  lamps.pool.forEach((s, k) => {
    const o = order[k];
    if(!o){ s.intensity = 0; return; }
    const it = lamps.items[o[1]];
    lit.add(o[1]);
    s.intensity = LAMP.candela * lamps.level;
    s.position.copy(it.head);
    s.target.position.copy(it.head).addScaledVector(it.dir, 2.0).setY(it.foot.y - 1);
    s.target.updateMatrixWorld();
  });
  lamps.items.forEach((it, i) => {
    if(lit.has(i)) m.makeScale(0, 0, 0); else m.makeTranslation(it.ground.x, it.ground.y, it.ground.z);
    lamps.glows.setMatrixAt(i, m);
  });
  lamps.glows.instanceMatrix.needsUpdate = true;
}

// ----------------------------------------------------------------- the scene
async function generate(){
  if(busy) return;
  const gid = window.lastGeneration && window.lastGeneration();
  if(!gid){ note('Generate the texture first (Generate tab), then come back and press Generate 3D scene.', true); return; }
  if(!renderer && !init()) return;
  busy = true; $('#btnScene3d').disabled = true;
  stopPhoto();
  note('Building the 3D model: streets, sidewalks, kerbs, blocks and objects…');
  try{
    const r = await fetch('/api/export3d', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...window.exportSettings(), generation: gid, mesh: 'tiled' }) });
    const res = await r.json();
    if(!r.ok) throw new Error(res.detail || 'the 3D model could not be built');
    note('Loading the model into the view…');
    const gltf = await new GLTFLoader().loadAsync(res.url);
    if(world){ scene.remove(world); world.traverse(o => { if(o.geometry) o.geometry.dispose(); }); }
    world = gltf.scene;
    prepare(world);
    scene.add(world);
    const g = scene.getObjectByName('grid'); if(g) scene.remove(g);
    buildLamps(res.lamps || []);
    const box = new THREE.Box3().setFromObject(world);
    frame(box.getCenter(new THREE.Vector3()).setY(0), Math.max(60, box.getSize(new THREE.Vector3()).length() / 2));
    let tris = 0;
    world.traverse(o => { if(o.isMesh) tris += (o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3; });
    note(`${res.size_m[0]} × ${res.size_m[1]} m, ${Math.round(tris).toLocaleString()} triangles`
      + (res.scatter ? `, ${res.scatter.copies} object(s)` : '') + `, ${(res.lamps || []).length} street lamps.`
      + (res.streets_warning ? ' Note: ' + res.streets_warning + '.' : ''));
    $('#btnPhoto').disabled = false; $('#btnSaveImg').disabled = false;
    window.dispatchEvent(new CustomEvent('scene3d', { detail: { lamps: (res.lamps || []).length, triangles: Math.round(tris) } }));
  }catch(e){
    note('Could not build the 3D scene: ' + e.message, true);
  }
  busy = false; $('#btnScene3d').disabled = false;
}

function prepare(root){
  // roads close to each other in height are kept apart in depth, the textures kept
  // sharp at grazing angles, and the objects, sidewalks and kerbs cast shadows
  const aniso = renderer.capabilities.getMaxAnisotropy();
  root.traverse(o => {
    if(!o.isMesh) return;
    o.receiveShadow = true;
    const name = (o.material && o.material.name) || '';
    const part = (o.parent && o.parent.name) || o.name || '';
    o.castShadow = !/^Road|Markings/.test(part) && !/^Road|Block_paving|RoadMarkings/.test(name);
    for(const m of [].concat(o.material)){
      if(m.map) m.map.anisotropy = aniso;
      if(/Road_fill|Road_interchange/.test(m.name)){
        // laid just under the road strips: kept behind them, but never pushed as far as the land
        m.polygonOffset = true; m.polygonOffsetFactor = 0.5; m.polygonOffsetUnits = 2;
      }
      if(/RoadMarkings/.test(m.name)){
        m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -2;
      }
      m.needsUpdate = true;
    }
  });
}

// ----------------------------------------------------------------- photo render
// Path traced: light bounces between the surfaces, soft shadows from the sun
// and the sky, and the lamps near the view as real lights. It sharpens for as
// long as nothing moves; moving the camera returns to the live view.
async function startPhoto(){
  if(!world || busy) return;
  if(photo && photo.on){ stopPhoto(); return; }
  busy = true;
  note('Preparing the photo render: building the ray-tracing structure of the scene…');
  await new Promise(r => setTimeout(r, 30));
  try{
    const { WebGLPathTracer } = await import('three-gpu-pathtracer');
    if(!photo){
      const pt = new WebGLPathTracer(renderer);
      pt.renderDelay = 0; pt.fadeDuration = 0; pt.minSamples = 0;
      pt.rasterizeScene = false; pt.dynamicLowRes = false;
      pt.bounces = 5; pt.filterGlossyFactor = 0.5;
      pt.tiles.set(2, 2);
      photo = { pt, on: false };
      // on screen: the denoised picture once there is one, else the samples so far
      pt.renderToCanvasCallback = (target, r, quad) => {
        const auto = r.autoClear;
        r.autoClear = false;
        if(DENOISE && photo.clean && photo.clean.shown) quad.material.map = photo.clean.out.texture;
        quad.render(r);
        r.autoClear = auto;
      };
    }
    photo.clean = null;                                                 // nothing denoised yet
    photo.saved = { env: scene.environment, bg: scene.background, envI: scene.environmentIntensity };
    if(skyNow){
      // Blender's sky as the tracer's light from all round, without the sun's disc (the sun is
      // its own light), and with the disc as what the camera sees
      scene.environment = skyNow.plain; scene.background = skyNow.disc;
    }else{
      // the night sky, as a cube the tracer turns into its own all-round picture
      const cube = new THREE.WebGLCubeRenderTarget(512, { type: THREE.HalfFloatType });
      new THREE.CubeCamera(1, 10000, cube).update(renderer, envScene);
      photo.cube = cube;
      scene.environment = cube.texture; scene.background = cube.texture;
    }
    hemi.visible = false;
    if(stars) stars.visible = false;                                    // points are not traced: the sky has its own
    // lamps: whole meshes (instances are not traced), real lights for those around the view
    const extra = new THREE.Group();
    if(lamps){
      lamps.group.visible = false;
      const merged = (geo, mat) => {
        const parts = [], m = new THREE.Matrix4();
        for(let i = 0; i < lamps.items.length; i++){ lamps.poles.getMatrixAt(i, m); parts.push(geo.clone().applyMatrix4(m)); }
        return new THREE.Mesh(mergeGeometries(parts), mat);
      };
      extra.add(merged(lamps.poles.geometry, lamps.metalMat), merged(lamps.heads.geometry, lamps.headMat));
      if(lamps.level > 0){
        const tg = controls.target;
        // every lamp a real light (up to the nearest thousand): the tracer's light tree
        // picks, at each point, the lamps likely to light it, so many lamps stay cheap
        const near = lamps.items.map(it => [it.foot.distanceTo(tg), it]).sort((a, b) => a[0] - b[0]).slice(0, PHOTO_LAMPS);
        for(const [, it] of near){
          const s = new THREE.SpotLight(LIGHT.colour, LAMP.candela * lamps.level, 0, 1.15, 0.85, 2);
          s.position.copy(it.head);
          s.target.position.copy(it.head).addScaledVector(it.dir, 2.0).setY(it.foot.y - 1);
          extra.add(s, s.target);
        }
      }
    }
    scene.add(extra);
    photo.extra = extra;
    const textures = new Set();
    world.traverse(o => { if(o.isMesh) for(const m of [].concat(o.material)) if(m.map) textures.add(m.map); });
    photo.pt.textureSize.set(textures.size > 24 ? 512 : 1024, textures.size > 24 ? 512 : 1024);
    const t0 = performance.now();
    photo.pt.setScene(scene, camera);
    photo.on = true; photo.started = performance.now();
    note(`Rendering the photo (set up in ${((performance.now() - t0) / 1000).toFixed(1)} s). It gets sharper while the camera stays put; Save image keeps it.`);
    $('#btnPhoto').textContent = 'Back to live view';
  }catch(e){
    stopPhoto();
    note('The photo render is not available here: ' + e.message, true);
  }
  busy = false;
}

function stopPhoto(why){
  if(!photo || !photo.on) return;
  photo.on = false;
  dropClean();
  scene.environment = photo.saved.env; scene.background = photo.saved.bg; scene.environmentIntensity = photo.saved.envI;
  hemi.visible = true;
  if(stars) stars.visible = true;
  if(lamps) lamps.group.visible = true;
  if(photo.extra){ scene.remove(photo.extra); photo.extra.traverse(o => { if(o.geometry) o.geometry.dispose(); }); photo.extra = null; }
  if(photo.cube){ photo.cube.dispose(); photo.cube = null; }
  $('#btnPhoto').textContent = 'Render photo';
  if(why) note(why);
}

function dropClean(){
  if(!photo || !photo.clean) return;
  for(const rt of [photo.clean.albedo, photo.clean.normal, photo.clean.out]) if(rt) rt.dispose();
  photo.clean = null;
}

async function saveImage(){
  if(!renderer) return;
  // the photo as sharp as it is now: denoised again if it has had more samples since
  if(photo && photo.on && DENOISE && photo.pt.samples >= 1){
    const n = Math.floor(photo.pt.samples);
    if(!photo.clean || photo.clean.at < n) await denoisePhoto(n);
  }
  draw();
  renderer.domElement.toBlob(b => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = `scene_${time}${photo && photo.on ? '_photo' : ''}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/png');
}

// ----------------------------------------------------------------- drawing
function draw(){
  if(photo && photo.on){
    photo.pt.renderSample();
    const n = photo.pt.samples;
    // denoised after 4 samples, then each time the samples have grown four times over
    if(DENOISE && Number.isInteger(n) && n >= (photo.clean ? photo.clean.next : 4) && !photo.denoising) denoisePhoto(n);
    const k = Math.floor(n);
    if(k && k % 4 === 0 && !photo.denoising) photoNote(k);
    return;
  }
  // near and far planes follow the zoom, for depth precision both close up and far away
  const dist = camera.position.distanceTo(controls.target);
  camera.near = THREE.MathUtils.clamp(dist / 1500, 0.05, 2); camera.far = Math.max(9000, dist * 8);
  camera.updateProjectionMatrix();
  fitShadow();
  assignPool();
  if(stars) stars.position.copy(camera.position);                      // as far away as the sky
  composer.render();
}

function photoNote(n){
  const c = DENOISE && photo.clean && photo.clean.shown ? ` Showing it denoised at ${photo.clean.at} samples (Open Image Denoise).` : '';
  $('#photoNote').textContent = `${n} samples, ${((performance.now() - photo.started) / 1000).toFixed(0)} s.${c} Sharper with every sample; move the camera to go back to the live view.`;
}

// ----------------------------------------------------------------- denoising
// What the denoiser needs besides the noisy picture: the colour (albedo) and the
// direction (normal) of the surface each pixel sees, drawn without noise and
// filtered over the pixel the way the tracer's samples are (a tent, a pixel each
// way), from 16 slightly shifted views
const AUX_VIEWS = 16;
let auxQuad = null, denoiser = null, denoiserLoad = null;
const auxCache = new WeakMap();
function auxMaterial(m, kind){
  let c = auxCache.get(m);
  if(!c) auxCache.set(m, c = {});
  if(c[kind]) return c[kind];
  const a = m.clone();
  a.fog = false; a.toneMapped = false; a.transparent = false; a.blending = THREE.NoBlending;
  a.onBeforeCompile = sh => {
    const normal = sh.fragmentShader.includes('#include <normal_fragment_begin>') ? 'normal' : 'vec3(0.0)';
    sh.fragmentShader = sh.fragmentShader.replace('#include <dithering_fragment>', '#include <dithering_fragment>\n'
      + (kind === 'albedo' ? 'gl_FragColor = vec4(clamp(diffuseColor.rgb, 0.0, 1.0), 1.0);' : `gl_FragColor = vec4(${normal}, 1.0);`));
  };
  a.customProgramCacheKey = () => 'aux-' + kind;
  return c[kind] = a;
}

function renderAux(w, h){
  const make = type => new THREE.WebGLRenderTarget(w, h, { type, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter,
                                                           magFilter: THREE.NearestFilter, depthBuffer: type === THREE.HalfFloatType });
  const shot = make(THREE.HalfFloatType), out = { albedo: make(THREE.HalfFloatType), normal: make(THREE.HalfFloatType) };
  if(!auxQuad) auxQuad = new FullScreenQuad(new THREE.ShaderMaterial({
    uniforms: { map: { value: null }, weight: { value: 1 } },
    vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'uniform sampler2D map; uniform float weight; varying vec2 vUv; void main(){ gl_FragColor = texture2D(map, vUv) * weight; }',
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendEquation: THREE.AddEquation,
    depthTest: false, depthWrite: false, toneMapped: false }));
  const keep = { target: renderer.getRenderTarget(), colour: renderer.getClearColor(new THREE.Color()), alpha: renderer.getClearAlpha(),
                 auto: renderer.autoClear, shadows: renderer.shadowMap.autoUpdate, bg: scene.background, bgI: scene.backgroundIntensity };
  const swapped = [], lights = [];
  scene.traverseVisible(o => { if(o.isMesh && o.material) swapped.push([o, o.material]); });
  // the photo's lamp lights do not matter here (and are far too many for drawing)
  if(photo && photo.extra) photo.extra.traverse(o => { if(o.isLight && o.visible){ o.visible = false; lights.push(o); } });
  const tent = u => (u < 0.5 ? Math.sqrt(2 * u) - 1 : 1 - Math.sqrt(2 - 2 * u));
  renderer.shadowMap.autoUpdate = false;
  try{
    for(const kind of ['albedo', 'normal']){
      for(const [o, m] of swapped) o.material = Array.isArray(m) ? m.map(x => auxMaterial(x, kind)) : auxMaterial(m, kind);
      // the sky: as bright as it shows (albedo); no surface (normal)
      scene.background = kind === 'albedo' ? keep.bg : null;
      scene.backgroundIntensity = keep.bgI * renderer.toneMappingExposure;
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(out[kind]); renderer.clear();
      for(let i = 0; i < AUX_VIEWS; i++){
        const jx = tent(((i % 4) + 0.5) / 4), jy = tent((Math.floor(i / 4) + 0.5) / 4);
        camera.setViewOffset(w, h, jx, jy, w, h);
        renderer.autoClear = true;
        renderer.setRenderTarget(shot); renderer.render(scene, camera);
        renderer.autoClear = false;
        auxQuad.material.uniforms.map.value = shot.texture; auxQuad.material.uniforms.weight.value = 1 / AUX_VIEWS;
        renderer.setRenderTarget(out[kind]); auxQuad.render(renderer);
      }
    }
  }finally{
    for(const [o, m] of swapped) o.material = m;
    for(const l of lights) l.visible = true;
    camera.clearViewOffset();
    scene.background = keep.bg; scene.backgroundIntensity = keep.bgI;
    renderer.setRenderTarget(keep.target); renderer.setClearColor(keep.colour, keep.alpha);
    renderer.autoClear = keep.auto; renderer.shadowMap.autoUpdate = keep.shadows;
    shot.dispose();
  }
  return out;
}

async function denoisePhoto(n){
  if(!photo || !photo.on || photo.denoising) return;
  photo.denoising = true;
  const ph = photo;
  try{
    if(!denoiser){
      $('#photoNote').textContent = 'Loading the denoiser (Open Image Denoise)…';
      denoiserLoad = denoiserLoad || Denoiser.load(renderer.getContext(), new URL('./vendor/oidn/rt_hdr_calb_cnrm.tza', import.meta.url).href);
      denoiser = await denoiserLoad;
      if(!ph.on) return;
    }
    const target = ph.pt.target, w = target.width, h = target.height;
    if(!ph.clean || ph.clean.w !== w || ph.clean.h !== h){
      if(ph.clean) for(const rt of [ph.clean.albedo, ph.clean.normal, ph.clean.out]) rt.dispose();
      const aux = renderAux(w, h);
      const out = new THREE.WebGLRenderTarget(w, h, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: false,
                                                      minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
      renderer.initRenderTarget(out);
      ph.clean = { w, h, albedo: aux.albedo, normal: aux.normal, out, at: 0, next: 4, shown: false };
    }
    const tex = rt => renderer.properties.get(rt.texture).__webglTexture;
    const t0 = performance.now();
    try{
      ph.clean.info = denoiser.denoise({ color: tex(target), albedo: tex(ph.clean.albedo), normal: tex(ph.clean.normal),
                                         output: tex(ph.clean.out), width: w, height: h });
    }finally{
      renderer.resetState();
    }
    ph.clean.ms = performance.now() - t0;
    ph.clean.at = n; ph.clean.next = Math.max(4, n * 4); ph.clean.shown = true;
    photoNote(n);
  }catch(e){
    DENOISE = false;
    $('#photoDenoise').checked = false;
    $('#photoNote').textContent = 'The denoiser is not available here: ' + e.message;
    console.error(e);
  }finally{
    ph.denoising = false;
  }
}

function loop(){
  if(!visible){ running = false; return; }
  if(!(photo && photo.on)) controls.update();                        // the photo's camera stays exactly put
  draw();
  requestAnimationFrame(loop);
}

function note(text, bad){
  msg.textContent = '';
  const n = $('#scene3dNote');
  if(n){ n.textContent = text; n.classList.toggle('bad', !!bad); }
}

// drawn only while the 3D tab is showing
window.addEventListener('view3d', e => {
  visible = !!e.detail;
  if(!visible || (!renderer && !init())) return;
  resize();
  if(!running){ running = true; requestAnimationFrame(loop); }
});
$('#btnScene3d').addEventListener('click', generate);
// the Light panel
const showLight = () => {
  $('#sunStrength').value = Math.round(LIGHT.sun * 100); $('#sunStrengthVal').textContent = Math.round(LIGHT.sun * 100) + '%';
  $('#lampStrength').value = Math.round(LIGHT.lamps * 100); $('#lampStrengthVal').textContent = Math.round(LIGHT.lamps * 100) + '%';
  $('#lampColour').value = LIGHT.colour;
};
showLight();
$('#sunStrength').addEventListener('input', () => {
  LIGHT.sun = +$('#sunStrength').value / 100; showLight(); saveLight();
  if(!renderer) return;
  if(photo && photo.on) stopPhoto();
  // the sun (or moon) brighter or dimmer; the exposure stays, so the picture follows
  sun.intensity = sunBase * LIGHT.sun;
});
$('#lampStrength').addEventListener('input', () => {
  LIGHT.lamps = +$('#lampStrength').value / 100; showLight(); saveLight();
  if(photo && photo.on) stopPhoto();
  if(lamps) { lampLevel(TIMES[time].lamps); poolAt = null; }
});
$('#lampColour').addEventListener('input', () => {
  LIGHT.colour = $('#lampColour').value; saveLight();
  if(photo && photo.on) stopPhoto();
  if(lamps) lampLevel(TIMES[time].lamps);
});
$('#btnPhoto').addEventListener('click', startPhoto);
$('#btnSaveImg').addEventListener('click', saveImage);
$('#photoDenoise').checked = DENOISE;
$('#photoDenoise').addEventListener('change', () => {
  DENOISE = $('#photoDenoise').checked;
  try{ localStorage.setItem('rta.view3d.denoise', DENOISE ? 'on' : 'off'); }catch(e){}
  if(photo && photo.on && DENOISE && photo.clean) photo.clean.next = 0;    // denoise what there is now
});
document.querySelectorAll('[data-time3d]').forEach(b => b.addEventListener('click', () => { if(renderer) setTime(b.dataset.time3d); else time = b.dataset.time3d; }));
$('#btnView3dReset').addEventListener('click', () => { if(renderer) frame(centre, radius); });
window.view3dState = () => renderer ? { children: scene.children.length, width: host.clientWidth, height: host.clientHeight,
  camera: camera.position.toArray().map(v => Math.round(v)), time, lamps: lamps ? lamps.items.length : 0,
  world: !!world, photo: !!(photo && photo.on), samples: photo && photo.on ? photo.pt.samples : 0,
  denoised: photo && photo.on && photo.clean ? { at: photo.clean.at, ms: Math.round(photo.clean.ms || 0), ...photo.clean.info } : null } : null;
window.view3dControl = { setTime: n => setTime(n), tune: (n, patch) => { Object.assign(TIMES[n], patch); if(patch.candela) LAMP.candela = patch.candela; setTime(n); }, view: (pos, tgt) => { camera.position.set(...pos); controls.target.set(...tgt); controls.update(); poolAt = null; },
  photo: startPhoto, stop: stopPhoto, draw: () => draw(), denoise: () => photo && photo.on ? denoisePhoto(Math.floor(photo.pt.samples)) : null,
  // for comparisons: the path tracer's light tree on or off (off: each light as likely)
  lightTree: on => { if(photo && photo.on){ photo.pt._pathTracer.material.lightTree.enabled = on ? 1 : 0; photo.pt.reset(); dropClean(); } },
  capture: () => { draw(); return renderer.domElement.toDataURL('image/png'); } };
