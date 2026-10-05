// The 3D tab: the whole place in 3D, to look around. Generate 3D scene builds the
// streets, inner streets, sidewalks, kerbs, blocks and objects from the last
// generated texture (the same model as the GLB export, with the 3D model
// settings of the Generate tab) and adds street lamps along the sidewalks.
// The live view has a sky, the sun with its shadows, soft contact shadows
// (ambient occlusion), filmic colour and a glow around bright lights, graded by
// the Look panel (ui/look.js) as a photo editor would. The sky
// is Blender's own physical sky (ui/sky_blender.js, ported from Blender), so
// the sky, the colour of the sunlight and the balance of sun and sky are
// those of Blender and Cycles. Render
// photo then traces the light properly (three-gpu-pathtracer), a still that
// sharpens for as long as the camera stays put, its noise cleared by Intel Open
// Image Denoise (ui/denoise.js, ported to WebGL2). Bake light traces the light on
// the ground the same way, from above, and keeps it for the live view. Dawn, Day and Night set the
// sun, the sky and the street lamps. Metres, Y up: the map's x is +X, its y +Z,
// the top of the map north. Everything is bundled in ui/vendor, so it works offline.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { FullScreenQuad } from 'three/addons/postprocessing/Pass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { buildSkyMaps } from './sky_blender.js';
import { Denoiser } from './denoise.js';
import { GradePass, lookPanel, DofPass } from './look.js';

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
const TEXTURE_SLOTS = ['map', 'normalMap', 'roughnessMap', 'metalnessMap', 'emissiveMap', 'alphaMap', 'aoMap', 'bumpMap'];

// the Light panel: the sun's (or moon's) strength, the street lamps' brightness
// and colour, and how wet the streets are, kept in this browser for the next time
const LIGHT = { sun: 1, lamps: 1, colour: '#ffcf96', wet: 0, puddles: 0.4, mirror: true };
try{ Object.assign(LIGHT, JSON.parse(localStorage.getItem('rta.view3d.light') || '{}')); }catch(e){}
const saveLight = () => { try{ localStorage.setItem('rta.view3d.light', JSON.stringify(LIGHT)); }catch(e){} };
// the Ground plane panel: the land round the place, moved, turned and sized (metres);
// colour '' follows the time of day
const GROUND_DEFAULT = { show: true, x: 0, y: -0.3, z: 0, turn: 0, width: 12000, length: 12000, colour: '' };
const GROUND = { ...GROUND_DEFAULT };
try{ Object.assign(GROUND, JSON.parse(localStorage.getItem('rta.view3d.ground') || '{}')); }catch(e){}
const saveGround = () => { try{ localStorage.setItem('rta.view3d.ground', JSON.stringify(GROUND)); }catch(e){} };
// Render photo's noise cleared by Open Image Denoise (on unless turned off here before)
let DENOISE = true;
try{ DENOISE = localStorage.getItem('rta.view3d.denoise') !== 'off'; }catch(e){}
// the light tree compiled out, when a photo on this graphics card came out empty with it
// and right without it (see photoHealth)
let NO_TREE = false;
try{ NO_TREE = localStorage.getItem('rta.view3d.notree') === '1'; }catch(e){}

let renderer = null, scene, camera, controls, composer, renderPass, gtao, bloom, bloomFrom = null, photoHDR = null;
// the Look panel's pass: from the scene's light to the finished picture (film response, then the
// adjustments), for the live view and the photo alike
const look = new GradePass();
// the Look panel's depth of field (see the section on it below)
const dof = new DofPass();
const DOF = { on: false, fstop: 2.8, focus: 'auto', distance: 25, lens: 0 };
try{ Object.assign(DOF, JSON.parse(localStorage.getItem('rta.view3d.dof') || '{}')); }catch(e){}
const saveDof = () => { try{ localStorage.setItem('rta.view3d.dof', JSON.stringify(DOF)); }catch(e){} };
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
  applyLens();
  // left drag: orbit, right drag (or Shift + drag): pan, wheel: zoom
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;                          // never below the ground
  controls.maxDistance = 2500;
  controls.addEventListener('start', () => {
    if(photo && photo.on) stopPhoto('The camera moved: back to the live view.');
    centrePivot();                                                   // turn round what is in the middle of the view
  });
  // the wheel is ours (zooming towards the mouse); dragging stays the controls'
  controls.enableZoom = false;
  renderer.domElement.addEventListener('wheel', onWheel, { passive: false });
  renderer.domElement.addEventListener('dblclick', onDoubleClick);

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
  ground = new THREE.Mesh(new THREE.CircleGeometry(1, 128).rotateX(-Math.PI / 2),
    new THREE.MeshStandardMaterial({ color: 0x5c6648, roughness: 1 }));
  placeGround();                                                      // by default 30 cm under the streets, blocks and islands
  ground.receiveShadow = true;
  ground.name = 'ground';
  scene.add(ground);
  const grid = new THREE.GridHelper(500, 50, 0x6a7280, 0x353b45);    // until a scene is built: 10 m squares
  grid.name = 'grid';
  scene.add(grid);

  // the picture: the scene, contact shadows, the glow of bright lights, then the look (the
  // film response and the Look panel's adjustments)
  const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
  composer = new EffectComposer(renderer, rt);
  renderPass = new RenderPass(scene, camera);
  gtao = new GTAOPass(scene, camera, 1, 1);
  gtao.updateGtaoMaterial({ radius: 1.6, distanceExponent: 1.5, thickness: 1.5, scale: 1.0, samples: 16 });
  gtao.blendIntensity = 0.9;
  bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.2, 0.4, 4.0);
  // the glow is blurred over the whole picture: one invalid pixel (not a number, or
  // infinite) would spread through it and blacken everything, so it never gets in
  const hp = bloom.materialHighPassFilter;
  hp.fragmentShader = hp.fragmentShader.replace('vec4 texel = texture2D( tDiffuse, vUv );',
    'vec4 texel = texture2D( tDiffuse, vUv );\n' + SANE_GLSL.replace(/c\b/g, 'texel.rgb'));
  hp.needsUpdate = true;
  composer.addPass(renderPass); composer.addPass(gtao); composer.addPass(dof); composer.addPass(bloom); composer.addPass(look);
  dof.depth = gtao.depthTexture; dof.camera = camera; dof.enabled = DOF.on;
  // the pools of lamp light are painted on, not surfaces: left out of the contact shadows
  const hide = gtao._overrideVisibility.bind(gtao);
  gtao._overrideVisibility = function(){
    hide();
    if(lamps && lamps.glows.visible){ lamps.glows.visible = false; this._visibilityCache.push(lamps.glows); }
  };

  setTime(time, false);
  // the other skies are worked out meanwhile, so changing the time is quick
  for(const t of Object.values(TIMES)) if(t.sky && !t.hdri) skyFor(t);
  // the HDRI sky in use last time, once the list is here
  if(SKY.id !== 'physical') skiesReady.then(() => useSky(SKY.id, false));
  frame(new THREE.Vector3(), 250);
  new ResizeObserver(resize).observe(host);
  return true;
}

// the land where the Ground plane panel puts it: a disc (an ellipse when width and
// length differ), turned round the up axis
function placeGround(){
  if(!ground) return;
  ground.visible = !!GROUND.show;
  ground.position.set(+GROUND.x || 0, Number.isFinite(+GROUND.y) ? +GROUND.y : -0.3, +GROUND.z || 0);
  ground.rotation.y = THREE.MathUtils.degToRad(+GROUND.turn || 0);
  ground.scale.set(Math.max(1, +GROUND.width || 1) / 2, 1, Math.max(1, +GROUND.length || 1) / 2);
  ground.material.color.set(GROUND.colour || TIMES[time].ground);
}

// the Look panel's Lens: the camera's focal length on a 35 mm frame (0: the 45 degree view it
// always had, about 25 mm)
function applyLens(){
  if(!camera) return;
  if(DOF.lens) camera.setFocalLength(DOF.lens); else camera.fov = 45;
  camera.updateProjectionMatrix();
}

function resize(){
  const w = host.clientWidth, h = host.clientHeight;
  if(!renderer || !w || !h) return;
  const now = renderer.getSize(new THREE.Vector2());
  if(now.x === w && now.y === h) return;                               // the same size: nothing to do
  renderer.setSize(w, h);
  composer.setSize(w, h);
  camera.aspect = w / h;
  applyLens();                                                         // the same lens at the new shape
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

// ----------------------------------------------------------------- navigation
// The wheel zooms towards the point under the mouse, a share of the way each notch,
// so it never stalls short of something (the controls' own zoom closes in on one
// fixed point by ever smaller steps, and that point can be far from where you
// look). Grabbing the view puts the point it turns round on the ground in the
// middle of the view: orbiting turns round what you look at, and panning keeps pace
// with that ground. W A S D or the arrows walk, Q and E go down and up, Shift is
// faster; a double click brings that spot to the middle
const STREET_Y = 0;                                  // the streets' level, to aim at
const ZOOM_STEP = 0.18;                              // how much closer one wheel notch gets
const EYE_M = 1.6;                                   // the lowest the wheel brings the camera: eye height
const aim = new THREE.Raycaster(), aimNdc = new THREE.Vector2();
const keys = new Set();
let walkedAt = 0;

function onGround(r, y = STREET_Y){
  if(Math.abs(r.direction.y) < 1e-6) return null;
  const t = (y - r.origin.y) / r.direction.y;
  return t > 0 ? r.origin.clone().addScaledVector(r.direction, t) : null;
}

function rayAt(clientX, clientY){
  camera.updateMatrixWorld();                                          // as it is now, not as last drawn
  const b = renderer.domElement.getBoundingClientRect();
  aimNdc.set(((clientX - b.left) / b.width) * 2 - 1, -((clientY - b.top) / b.height) * 2 + 1);
  aim.setFromCamera(aimNdc, camera);
  return aim.ray;
}

// the pivot moved along the line of sight onto the ground: the view does not move,
// only the point it turns round (and pans by)
function centrePivot(){
  camera.updateMatrixWorld();
  const dir = camera.getWorldDirection(new THREE.Vector3());
  const p = onGround(new THREE.Ray(camera.position.clone(), dir));
  const far = Math.max(60, Math.min(radius * 1.5, 3 * Math.max(camera.position.y - STREET_Y, 20)));
  const d = THREE.MathUtils.clamp(p ? camera.position.distanceTo(p) : far, 1, far);
  controls.target.copy(camera.position).addScaledVector(dir, d);
}

function onWheel(e){
  e.preventDefault();
  if(!renderer) return;
  if(photo && photo.on) stopPhoto('The camera moved: back to the live view.');
  const dy = e.deltaY * (e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1);
  const notches = THREE.MathUtils.clamp(-dy / 100, -4, 4);              // wheel up (forward): closer
  if(!notches) return;
  const r = rayAt(e.clientX, e.clientY);
  // towards the ground under the mouse, or a point straight ahead of it (over the sky)
  const p = onGround(r) || r.origin.clone().addScaledVector(r.direction, camera.position.distanceTo(controls.target));
  let k = Math.pow(1 - ZOOM_STEP, notches);                         // under 1 closer, over 1 further
  const h = camera.position.y - STREET_Y;
  if(k > 1) k = Math.min(k, controls.maxDistance / Math.max(camera.position.distanceTo(controls.target), 1e-6));
  else if(h * k < EYE_M){
    // down at eye height: no lower, but on along the street towards the point instead
    const kh = Math.min(1, EYE_M / Math.max(h, 1e-6));
    if(kh < 1){ camera.position.sub(p).multiplyScalar(kh).add(p); controls.target.sub(p).multiplyScalar(kh).add(p); }
    // a stride towards the point (or straight on), at least a metre and a half a notch
    const ahead = new THREE.Vector3(p.x - camera.position.x, 0, p.z - camera.position.z);
    if(ahead.lengthSq() < 0.01) ahead.copy(camera.getWorldDirection(new THREE.Vector3())).setY(0);
    if(ahead.lengthSq() > 1e-8){
      ahead.setLength(Math.max(ahead.length(), 8) * (1 - k));
      camera.position.add(ahead); controls.target.add(ahead);
    }
    camera.position.y = Math.max(camera.position.y, STREET_Y + EYE_M);
    controls.update();
    return;
  }
  // scaled round p: the point under the mouse stays under it, the direction of view stays
  camera.position.sub(p).multiplyScalar(k).add(p);
  controls.target.sub(p).multiplyScalar(k).add(p);
  controls.update();
}

function onDoubleClick(e){
  const p = onGround(rayAt(e.clientX, e.clientY));
  if(!p) return;
  if(photo && photo.on) stopPhoto('The camera moved: back to the live view.');
  const move = p.clone().sub(controls.target);
  camera.position.add(move); controls.target.add(move);
  controls.update();
}

const WALK_KEYS = ['w', 'a', 's', 'd', 'q', 'e', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright'];
addEventListener('keydown', e => {
  if(!visible || !renderer || e.ctrlKey || e.metaKey || e.altKey) return;
  const t = e.target;
  if(t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(t.tagName))) return;
  const k = e.key.toLowerCase();
  if(k === 'shift'){ keys.add('shift'); return; }
  if(WALK_KEYS.includes(k)){ keys.add(k); e.preventDefault(); }
  if(k === 'b' && BAKE.list.length && !e.repeat){ BAKE.on = !BAKE.on; applyBake(); }   // compare baked and live
});
addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));
addEventListener('blur', () => keys.clear());

// a step of walking, at a pace that suits the height: slow in the street, fast from above
function walk(){
  const now = performance.now(), dt = Math.min(0.1, (now - (walkedAt || now)) / 1000);
  walkedAt = now;
  if(![...keys].some(k => k !== 'shift')) return;
  if(photo && photo.on) stopPhoto('The camera moved: back to the live view.');
  const fwd = camera.getWorldDirection(new THREE.Vector3()).setY(0);
  if(fwd.lengthSq() < 1e-6) fwd.copy(new THREE.Vector3(0, 1, 0).applyQuaternion(camera.quaternion)).setY(0);   // looking straight down
  fwd.normalize();
  const right = new THREE.Vector3().crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
  const speed = THREE.MathUtils.clamp((camera.position.y - STREET_Y) * 1.2, 4, 600) * (keys.has('shift') ? 4 : 1);
  const m = new THREE.Vector3();
  if(keys.has('w') || keys.has('arrowup')) m.add(fwd);
  if(keys.has('s') || keys.has('arrowdown')) m.sub(fwd);
  if(keys.has('d') || keys.has('arrowright')) m.add(right);
  if(keys.has('a') || keys.has('arrowleft')) m.sub(right);
  if(m.lengthSq()) m.normalize().multiplyScalar(speed * dt);
  if(keys.has('e')) m.y += 0.6 * speed * dt;
  if(keys.has('q')) m.y -= 0.6 * speed * dt;
  if(camera.position.y + m.y < STREET_Y + 0.6) m.y = STREET_Y + 0.6 - camera.position.y;   // never into the street (Q goes below eye height)
  camera.position.add(m); controls.target.add(m);
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
  ground.material.color.set(GROUND.colour || t.ground);
  const st = starField();
  if(t.sky) scene.remove(st); else scene.add(st);
  hemi.intensity = t.sky ? 0 : 0.06;
  if(lamps) lampLevel(t.lamps);
  applyBake();                                                        // this time's bake, if there is one
  if(render) poolAt = null;
  if(t.hdri && hdri){
    // an HDRI sky: its picture, its light from all round, its sun, turned and as strong as set
    const h = hdri, s = h.scale * Math.pow(2, SKY.strength), r = THREE.MathUtils.degToRad(SKY.rotation);
    skyNow = { plain: h.env, disc: h.bg.texture };                     // the photo and the bake trace these
    scene.background = h.bg.texture; scene.backgroundIntensity = s;
    scene.environment = h.pmrem.texture; scene.environmentIntensity = s;
    scene.backgroundRotation.set(0, r, 0); scene.environmentRotation.set(0, r, 0);
    scene.fog = new THREE.FogExp2(new THREE.Color().setRGB(h.horizon[0] * s, h.horizon[1] * s, h.horizon[2] * s, THREE.LinearSRGBColorSpace), t.fog[1]);
    const y = h.sun.Y || 1;
    sun.color.setRGB(h.sun.rgb[0] / y, h.sun.rgb[1] / y, h.sun.rgb[2] / y, THREE.LinearSRGBColorSpace);
    sunBase = t.elev > -1 ? h.sun.Y * s : 0;                            // a sun under the horizon gives no light
    sun.intensity = sunBase * LIGHT.sun;
    // exposed as the time of day it stands for (Strength then brightens or darkens it)
    const kind = SKY_KINDS[h.kind] || SKY_KINDS.day;
    const exposure = kind.exposure || kind.bias * Math.PI / Math.max(h.level * h.scale, 1e-6);
    renderer.toneMappingExposure = exposure;
    setBloom(t, exposure);
    msg.textContent = '';
    return;
  }
  scene.backgroundRotation.set(0, 0, 0); scene.environmentRotation.set(0, 0, 0);
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
  // (the look's included); the look's Glow scales its strength
  bloomFrom = [t, exposure];
  bloom.strength = t.bloom[0] * look.settings.glow; bloom.radius = t.bloom[1];
  bloom.threshold = t.bloom[2] / look.exposureOf(exposure);
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

// ----------------------------------------------------------------- HDRI skies
// Instead of the physical sky, a photographed sky (an HDRI): the ones that come with the
// program (ui/skies, CC0 from Poly Haven) or your own (.hdr, .exr, or a .jpg or .png
// panorama). What the camera sees is the sharp picture (4k), its light the HDR: the sun
// in it is found (the brightest spot, less the sky round it) and becomes the sun's light
// with its shadows, the rest lights the scene from all round. HDR pictures do not say how
// bright they really are, so each kind of sky is brought to the light of the time of day
// it stands for (a clear day, dusk), and night skies kept dim under the street lamps.
// Rotation turns the sky (and its sun) round; Strength makes it brighter or darker.
const SKY_KINDS = {
  day:      { label: 'Day', level: 139.7, bias: 0.9, lamps: 0, bloom: TIMES.day.bloom, fog: TIMES.day.fog[1], ground: TIMES.day.ground },
  overcast: { label: 'Overcast', level: 45, bias: 0.5, lamps: 0, bloom: [0.1, 0.4, 6.0], fog: 0.0005, ground: 0x56604a },
  sunset:   { label: 'Sunset or dawn', level: 8.28, bias: 0.4, lamps: 0.6, bloom: TIMES.dawn.bloom, fog: TIMES.dawn.fog[1], ground: TIMES.dawn.ground },
  night:    { label: 'Night', skyY: 0.02, exposure: 0.9, lamps: 1, bloom: TIMES.night.bloom, fog: TIMES.night.fog[1], ground: TIMES.night.ground },
};
// the sky in use ('physical': Blender's, by the time of day), how far it is turned and how strong
const SKY = { id: 'physical', rotation: 0, strength: 0 };
try{ Object.assign(SKY, JSON.parse(localStorage.getItem('rta.view3d.sky') || '{}')); }catch(e){}
const saveSky = () => { try{ localStorage.setItem('rta.view3d.sky', JSON.stringify(SKY)); }catch(e){} };
let skyList = [], hdri = null, hdriLoading = null, skyQuad = null;

// a sky picture into a picture of the light (linear, half floats), w x h: the sharp picture
// unsqueezed (mode 0, its k), an HDR as it is (1), or an ordinary photo from sRGB (2); each
// pixel the average of 4 x 4 samples, so a larger picture made smaller keeps its light
function skyTarget(src, mode, k, w, h, type = THREE.HalfFloatType){
  if(!skyQuad) skyQuad = new FullScreenQuad(new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false,
    uniforms: { map: { value: null }, mode: { value: 0 }, k: { value: 1 }, texel: { value: new THREE.Vector2() } },
    vertexShader: 'precision highp float; uniform mat4 modelViewMatrix; uniform mat4 projectionMatrix; in vec3 position; in vec2 uv; out vec2 vUv;\n'
      + 'void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: `precision highp float; uniform sampler2D map; uniform int mode; uniform float k; uniform vec2 texel; in vec2 vUv; out vec4 outColor;
      vec3 light(vec2 at){
        vec3 c = texture(map, vec2(fract(at.x), clamp(at.y, 0.0, 1.0))).rgb;
        if(mode == 0){ vec3 t = min(pow(c, vec3(2.2)), vec3(0.9995)); return t / (1.0 - t) / k; }
        if(mode == 2) return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c));
        return c;
      }
      void main(){
        vec3 s = vec3(0.0);
        for(int y = 0; y < 4; y++) for(int x = 0; x < 4; x++) s += light(vUv + (vec2(x, y) - 1.5) / 4.0 * texel);
        s /= 16.0;
        if(any(isnan(s)) || any(isinf(s))) s = vec3(0.0);
        outColor = vec4(clamp(s, 0.0, 6.0e4), 1.0);
      }` }));
  const rt = new THREE.WebGLRenderTarget(w, h, { type, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter,
                                                 generateMipmaps: false, wrapS: THREE.RepeatWrapping });
  rt.texture.mapping = THREE.EquirectangularReflectionMapping;
  rt.texture.colorSpace = THREE.LinearSRGBColorSpace;
  const u = skyQuad.material.uniforms;
  u.map.value = src; u.mode.value = mode; u.k.value = k; u.texel.value.set(1 / w, 1 / h);
  const keep = renderer.getRenderTarget();
  renderer.setRenderTarget(rt); skyQuad.render(renderer); renderer.setRenderTarget(keep);
  return rt;
}

// what the sky's light is: the sun (its direction, colour and the light it gives a surface
// facing it), the rest of the sky without it (to light from all round), the light on a level
// surface, the colour just above the horizon. data: RGBA floats, the bottom row first
function analyseSky(data, W, H){
  const lat = j => ((j + 0.5) / H - 0.5) * Math.PI;
  const dir = (i, j) => { const a = ((i + 0.5) / W - 0.5) * 2 * Math.PI, l = lat(j);
                          return [Math.cos(a) * Math.cos(l), Math.sin(l), Math.sin(a) * Math.cos(l)]; };
  const dOmega = j => (2 * Math.PI / W) * (Math.PI / H) * Math.cos(lat(j));
  const Y = k => 0.2126 * data[k] + 0.7152 * data[k + 1] + 0.0722 * data[k + 2];
  // the brightest spot, from a little under the horizon (a setting sun) up
  let best = -1, bi = 0, bj = 0;
  for(let j = Math.floor(H * 0.48); j < H; j++) for(let i = 0; i < W; i++){
    const y = Y((j * W + i) * 4);
    if(y > best){ best = y; bi = i; bj = j; }
  }
  const d0 = dir(bi, bj), band = Math.ceil(20 / 180 * H) + 1;
  const near = [];                                           // [k, cos of the angle to the spot, j, i]
  for(let j = Math.max(0, bj - band); j < Math.min(H, bj + band); j++) for(let i = 0; i < W; i++){
    const d = dir(i, j), c = d[0] * d0[0] + d[1] * d0[1] + d[2] * d0[2];
    if(c > Math.cos(THREE.MathUtils.degToRad(20))) near.push([(j * W + i) * 4, c, j, i]);
  }
  // the sky round it (12 to 20 degrees off), the darker 60%: what is left without the sun
  const ring = near.filter(p => p[1] < Math.cos(THREE.MathUtils.degToRad(12))).map(p => p[0]).sort((a, b) => Y(a) - Y(b));
  const keepN = Math.max(1, Math.floor(ring.length * 0.6)), bg = [0, 0, 0];
  for(let n = 0; n < keepN; n++) for(let c = 0; c < 3; c++) bg[c] += data[ring[n] + c] / keepN;
  const bgY = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2];
  const env = data.slice();
  const sunRGB = [0, 0, 0], sd = [0, 0, 0];
  for(const [k, c, j, i] of near){
    if(c < Math.cos(THREE.MathUtils.degToRad(12)) || Y(k) < 1.5 * bgY) continue;
    const w = dOmega(j), d = dir(i, j);
    let ey = 0;
    for(let ch = 0; ch < 3; ch++){
      const ex = Math.max(data[k + ch] - bg[ch], 0);
      sunRGB[ch] += ex * w; env[k + ch] = data[k + ch] - ex;
      ey += [0.2126, 0.7152, 0.0722][ch] * ex;
    }
    for(let a = 0; a < 3; a++) sd[a] += ey * w * d[a];
  }
  const sunY = 0.2126 * sunRGB[0] + 0.7152 * sunRGB[1] + 0.0722 * sunRGB[2];
  const sl = Math.hypot(...sd) || 1, sdir = sunY > 0 ? sd.map(v => v / sl) : d0;
  // the rest of the sky's light on a level surface, its middle brightness, the horizon's colour
  let skyLevel = 0;
  const ups = [], hz = Array.from({ length: 16 }, () => [0, 0, 0, 0]);
  for(let j = Math.floor(H / 2); j < H; j++){
    const l = lat(j), w = dOmega(j) * Math.sin(l);
    for(let i = 0; i < W; i++){
      const k = (j * W + i) * 4, y = 0.2126 * env[k] + 0.7152 * env[k + 1] + 0.0722 * env[k + 2];
      skyLevel += y * w;
      if((i & 3) === 0) ups.push(y);
      if(l > THREE.MathUtils.degToRad(0.5) && l < THREE.MathUtils.degToRad(3)){
        const h = hz[Math.floor(i / W * 16)];
        for(let c = 0; c < 3; c++) h[c] += env[k + c];
        h[3]++;
      }
    }
  }
  ups.sort((a, b) => a - b);
  const elev = Math.asin(THREE.MathUtils.clamp(sdir[1], -1, 1));
  return { env, sun: { dir: sdir, rgb: sunRGB, Y: sunY, sharp: best / Math.max(bgY, 1e-6) },
           level: skyLevel + sunY * Math.max(Math.sin(elev), 0), skyY: ups[ups.length >> 1] || 0,
           horizon: [0, 1, 2].map(c => { const v = hz.map(h => h[c] / Math.max(h[3], 1)).sort((a, b) => a - b); return (v[7] + v[8]) / 2; }) };
}

const loadTexture = url => new Promise((ok, bad) => new THREE.TextureLoader().load(url, ok, undefined, () => bad(new Error('could not load ' + url))));
const loadHDR = (url, type) => new HDRLoader().setDataType(type).loadAsync(url);
const loadEXR = (url, type) => new EXRLoader().setDataType(type).loadAsync(url);

// a sky of the list, ready to use: its picture, its light (with and without the sun) and what
// the light is (see analyseSky)
async function buildHdri(e){
  let bgSrc, mode = 1, k = 1, light;
  const toFree = [];
  if(e.builtin){
    const base = new URL('./', import.meta.url).href;
    [bgSrc, light] = await Promise.all([loadTexture(base + e.background_url), loadHDR(base + e.light_url, THREE.FloatType)]);
    bgSrc.colorSpace = THREE.NoColorSpace; mode = 0; k = e.background_k;
    toFree.push(bgSrc, light);
  }else{
    const url = e.file_url, f = (e.format || '').toLowerCase();
    bgSrc = f === 'exr' ? await loadEXR(url, THREE.HalfFloatType) : f === 'hdr' ? await loadHDR(url, THREE.HalfFloatType) : await loadTexture(url);
    if(!(f === 'exr' || f === 'hdr')){ bgSrc.colorSpace = THREE.NoColorSpace; mode = 2; }
    toFree.push(bgSrc);
  }
  bgSrc.wrapS = THREE.RepeatWrapping; bgSrc.needsUpdate = true;
  const max = Math.min(4096, renderer.capabilities.maxTextureSize);
  const bw = Math.min(max, (bgSrc.image && bgSrc.image.width) || 4096);
  const bg = skyTarget(bgSrc, mode, k, bw, bw / 2);
  // the light at 1024 x 512, as floats here to look at (the bottom row first)
  let data, W = 1024, H = 512;
  if(light){
    W = light.image.width; H = light.image.height;
    const src = light.image.data;
    data = new Float32Array(W * H * 4);
    // the loader's rows run from the top (it flips them when drawing): turned round here
    for(let j = 0; j < H; j++) data.set(src.subarray((H - 1 - j) * W * 4, (H - j) * W * 4), j * W * 4);
  }else{
    const rt = skyTarget(bgSrc, mode, k, W, H, THREE.FloatType);
    data = new Float32Array(W * H * 4);
    renderer.readRenderTargetPixels(rt, 0, 0, W, H, data);
    rt.dispose();
  }
  for(const t of toFree) t.dispose();
  const a = analyseSky(data, W, H);
  const half = new Uint16Array(W * H * 4);
  for(let i = 0; i < half.length; i++) half[i] = (i & 3) === 3 ? 0x3c00 : THREE.DataUtils.toHalfFloat(Math.min(a.env[i], 6.0e4));
  const env = new THREE.DataTexture(half, W, H, THREE.RGBAFormat, THREE.HalfFloatType);
  env.mapping = THREE.EquirectangularReflectionMapping; env.colorSpace = THREE.LinearSRGBColorSpace;
  env.magFilter = env.minFilter = THREE.LinearFilter; env.generateMipmaps = false; env.wrapS = THREE.RepeatWrapping;
  env.needsUpdate = true;
  const kind = SKY_KINDS[e.kind] || SKY_KINDS.day;
  // brought to its kind's light: a level surface lit as the time of day's, or (night) the
  // sky's middle brightness that of a night sky
  const scale = e.kind === 'night' ? kind.skyY / Math.max(a.skyY, 1e-6) : kind.level / Math.max(a.level, 1e-6);
  return { id: e.id, entry: e, kind: e.kind || 'day', bg, env, pmrem: pmrem.fromEquirectangular(env),
           sun: a.sun, level: a.level, skyY: a.skyY, horizon: a.horizon, scale };
}

function dropHdri(h){
  if(!h) return;
  h.bg.dispose(); h.env.dispose(); h.pmrem.dispose();
}

// the sun of the sky in use, turned with it: its height and bearing (as the times of day give them)
function hdriSun(h){
  const r = THREE.MathUtils.degToRad(SKY.rotation);
  const v = new THREE.Vector3(...h.sun.dir).applyAxisAngle(new THREE.Vector3(0, 1, 0), r);
  return { elev: THREE.MathUtils.radToDeg(Math.asin(THREE.MathUtils.clamp(v.y, -1, 1))),
           azim: (THREE.MathUtils.radToDeg(Math.atan2(v.x, -v.z)) + 360) % 360 };
}

// the HDRI sky as the time of day 'hdri': what setTime needs of it
function hdriTime(h){
  const kind = SKY_KINDS[h.kind] || SKY_KINDS.day, s = hdriSun(h);
  return { elev: s.elev, azim: s.azim, sky: true, hdri: true, bias: kind.bias || 1, env: 1, lamps: kind.lamps,
           bloom: kind.bloom, fog: [0, kind.fog], ground: kind.ground };
}

// use a sky: 'physical' (the time of day's), or one of the list, loaded the first time
async function useSky(id, render = true){
  SKY.id = id; saveSky(); showSky();
  if(!renderer) return;
  if(id === 'physical'){
    if(time === 'hdri') setTime('day', render);
    return;
  }
  const e = skyList.find(x => x.id === id);
  if(!e){ skyNoteText('That sky is not in the list any more: back to the physical sky.', true); SKY.id = 'physical'; saveSky(); showSky(); return; }
  if(hdri && hdri.id === id){ TIMES.hdri = hdriTime(hdri); setTime('hdri', render); return; }
  const token = {};
  hdriLoading = token;
  skyNoteText(`Loading ${e.name}…`);
  try{
    const h = await buildHdri(e);
    if(hdriLoading !== token){ dropHdri(h); return; }               // another sky was picked meanwhile
    const old = hdri;
    hdri = h; hdriLoading = null;
    TIMES.hdri = hdriTime(h);
    setTime('hdri', render);
    dropHdri(old);
    showSky();
  }catch(err){
    console.error('HDRI sky:', err);
    if(hdriLoading === token) hdriLoading = null;
    skyNoteText(`${e.name} could not be loaded here: ${err.message}`, true);
  }
}

// a change of rotation or strength: the same sky, turned or brighter
function hdriChanged(){
  if(time !== 'hdri' || !hdri) return;
  TIMES.hdri = hdriTime(hdri);
  setTime('hdri');
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
  bakePatch(metalMat); bakePatch(headMat);                            // the walls' bake lights them too
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
// what the scene was built from (the texture and the 3D model settings, materials
// included), to tell when it is out of date
let builtFrom = null;
const buildKey = () => JSON.stringify([window.lastGeneration && window.lastGeneration(), window.exportSettings ? window.exportSettings() : null]);

// keep: Update view, the same camera (and time of day and look); else the whole place framed
async function generate(keep = false){
  if(busy) return;
  const gid = window.lastGeneration && window.lastGeneration();
  if(!gid){ note('Generate the texture first (Generate tab), then come back and press Generate 3D scene.', true); return; }
  if(!renderer && !init()) return;
  busy = true; $('#btnScene3d').disabled = true; $('#btnScene3dUpdate').disabled = true;
  stopPhoto();
  note(keep ? 'Updating the scene with the current materials and settings…' : 'Building the 3D model: streets, sidewalks, kerbs, blocks and objects…');
  const key = buildKey();
  // its line in the console: the server's stages with the time left, then the loading
  const job = window.trackJob ? window.trackJob(keep ? 'Updating the 3D scene' : 'Building the 3D scene') : null;
  try{
    const r = await fetch('/api/export3d', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...window.exportSettings(), generation: gid, mesh: 'tiled', progress: job ? job.id : undefined }) });
    const res = await r.json();
    if(!r.ok) throw new Error(res.detail || 'the 3D model could not be built');
    note('Loading the model into the view…');
    const t0 = performance.now(), mb = (res.file_bytes || 0) / 1048576;
    if(job) job.set(`loading the model into the view${mb ? ` (${mb.toFixed(0)} MB)` : ''}`, 0);
    const gltf = await new GLTFLoader().loadAsync(res.url, ev => {
      if(!job || !ev.total) return;
      const f = ev.loaded / ev.total, s = (performance.now() - t0) / 1000;
      if(f < 1) job.set(`loading the model into the view (${mb.toFixed(0)} MB)`, f, f > 0.05 && s > 1 ? s * (1 - f) / f : null);
      else job.set('reading the model', null);
    });
    if(job) job.set('putting the model in the scene', null);
    if(world){ scene.remove(world); world.traverse(o => { if(o.geometry) o.geometry.dispose(); }); }
    dropBakes();
    world = gltf.scene;
    prepare(world);
    scene.add(world);
    const g = scene.getObjectByName('grid'); if(g) scene.remove(g);
    buildLamps(res.lamps || []);
    const box = new THREE.Box3().setFromObject(world);
    const c = box.getCenter(new THREE.Vector3()).setY(0), rad = Math.max(60, box.getSize(new THREE.Vector3()).length() / 2);
    if(keep && builtFrom){ centre.copy(c); radius = rad; controls.maxDistance = Math.max(400, 3 * rad); poolAt = null; }
    else frame(c, rad);
    builtFrom = key;
    staleCheck();
    applyBake();
    let tris = 0;
    world.traverse(o => { if(o.isMesh) tris += (o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3; });
    note(`${res.size_m[0]} × ${res.size_m[1]} m, ${Math.round(tris).toLocaleString()} triangles`
      + (res.scatter ? `, ${res.scatter.copies} object(s)` : '') + `, ${(res.lamps || []).length} street lamps.`
      + (res.streets_warning ? ' Note: ' + res.streets_warning + '.' : '')
      + (res.materials_used && Object.keys(res.materials_used).length
         ? ' Materials: ' + Object.entries(res.materials_used).map(([k, v]) => `${k} ${v}`).join(', ') + '.' : ''));
    $('#btnPhoto').disabled = false; $('#btnSaveImg').disabled = false; $('#btnBake').disabled = false;
    if(job) job.done(true, `${Math.round(tris).toLocaleString()} triangles, ${(res.lamps || []).length} street lamps`);
    window.dispatchEvent(new CustomEvent('scene3d', { detail: { lamps: (res.lamps || []).length, triangles: Math.round(tris) } }));
  }catch(e){
    if(job) job.done(false, e.message);
    note('Could not build the 3D scene: ' + e.message, true);
  }
  busy = false; $('#btnScene3d').disabled = false; $('#btnScene3dUpdate').disabled = !world;
}

// a scene built from other settings (materials, the 3D model settings, a newer
// texture) than the current ones: Update view says so
function staleCheck(){
  const b = $('#btnScene3dUpdate');
  if(!b) return;
  const stale = !!(world && builtFrom && buildKey() !== builtFrom);
  b.classList.toggle('attention', stale);
  b.title = stale ? 'The materials or settings changed since this scene was built: update it, keeping the camera'
                  : 'Build the scene again with the current materials and settings, keeping the camera';
  const n = $('#scene3dStale');
  if(n) n.hidden = !stale;
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
      for(const t of [m.map, m.normalMap, m.roughnessMap]) if(t) t.anisotropy = aniso;
      // the street surfaces, as built: what Wet roads starts from
      if(/^(Road|Sidewalk|Kerb|Block|Bridge)/.test(m.name) && !m.userData.dry)
        m.userData.dry = { colour: m.color.clone(), roughness: m.roughness, normalScale: m.normalScale ? m.normalScale.clone() : null,
                           paint: /Markings/.test(m.name) };
      if(/Road_fill|Road_interchange/.test(m.name)){
        // laid just under the road strips: kept behind them, but never pushed as far as the land
        m.polygonOffset = true; m.polygonOffsetFactor = 0.5; m.polygonOffsetUnits = 2;
      }
      if(/RoadMarkings/.test(m.name)){
        m.polygonOffset = true; m.polygonOffsetFactor = -1; m.polygonOffsetUnits = -2;
      }
      bakePatch(m);                                                     // can take Bake light's light
      m.needsUpdate = true;
    }
  });
  applyWet(root);
}

// Wet roads: water darkens the surfaces (paint less, it soaks up little), makes them
// glossy so the lamps and the sky shine in them, and fills the fine grain. Applied to
// the materials themselves, so the photo render has it too
function applyWet(root = world){
  if(!root) return;
  const w = LIGHT.wet;
  root.traverse(o => {
    if(!o.isMesh) return;
    for(const m of [].concat(o.material)){
      const d = m.userData.dry;
      if(!d) continue;
      m.color.copy(d.colour).multiplyScalar(1 - (d.paint ? 0.15 : 0.35) * w);
      m.roughness = d.roughness * (1 - 0.82 * w);
      if(d.normalScale) m.normalScale.copy(d.normalScale).multiplyScalar(1 - 0.6 * w);
    }
  });
}

// ----------------------------------------------------------------- depth of field
// Where the lens is focused: on what is in the middle of the view (found once the camera
// has stopped for a moment, then eased to, as a camera's autofocus does), or at a fixed
// distance, picked by clicking a spot. Rays are cast against the scene's surfaces, indexed
// the first time (three-mesh-bvh), so it is quick on a whole city too
let focusNow = null, focusAim = null, focusMoved = 0, focusBusy = false, bvhFor = null, bvhLib = null;
const focusSeen = new THREE.Matrix4(), focusRay = new THREE.Raycaster();

async function indexScene(){
  if(!world) return false;
  if(!bvhLib) bvhLib = await import('three-mesh-bvh');
  if(bvhFor !== world){
    world.traverse(o => {
      if(!o.isMesh || o.isInstancedMesh || !o.geometry || o.geometry.boundsTree) return;
      o.geometry.computeBoundsTree = bvhLib.computeBoundsTree; o.geometry.computeBoundsTree();
      o.raycast = bvhLib.acceleratedRaycast;
    });
    bvhFor = world;
  }
  return true;
}

// the distance to the first surface along the view through ndc (x, y from -1 to 1), or null
function surfaceAt(x, y){
  if(!world || bvhFor !== world) return null;
  camera.updateMatrixWorld();
  focusRay.setFromCamera(new THREE.Vector2(x, y), camera);
  focusRay.firstHitOnly = true;
  const hit = focusRay.intersectObject(world, true)[0];
  return hit ? hit.distance : null;
}

// the focus distance now (metres); now: worked out at once (for the photo), not eased
function focusDistance(now = false){
  if(DOF.focus === 'fixed') return Math.max(0.3, +DOF.distance || 25);
  const t = performance.now();
  if(!focusSeen.equals(camera.matrixWorld)){ focusSeen.copy(camera.matrixWorld); focusMoved = t; focusAim = null; }
  if((now || (focusAim === null && t - focusMoved > 150)) && !focusBusy){
    if(bvhFor === world){
      const d = surfaceAt(0, 0);
      focusAim = d === null ? 2000 : d;                                 // the sky: far away
    }else if(world){
      focusBusy = true;
      indexScene().then(() => { focusBusy = false; focusAim = null; focusMoved = 0; });
    }
  }
  if(focusAim !== null){
    if(focusNow === null || now) focusNow = focusAim;
    else focusNow *= Math.pow(focusAim / focusNow, 0.2);              // eased, as an autofocus pulls
  }
  if(focusNow === null) focusNow = camera.position.distanceTo(controls.target);
  const v = $('#dofFocusVal');
  if(v){ const txt = focusNow >= 1000 ? 'now: far' : `now ${focusNow < 10 ? focusNow.toFixed(1) : Math.round(focusNow)} m`; if(v.textContent !== txt) v.textContent = txt; }
  return focusNow;
}

// the camera the photo traces: with depth of field, a physical camera (the same view, its
// lens open to the f-stop, focused as the live view is)
async function photoCamera(){
  if(!DOF.on) return camera;
  const { PhysicalCamera } = await import('three-gpu-pathtracer');
  if(DOF.focus === 'auto' && await indexScene()) focusAim = null;   // focused on what is in the middle now
  const c = new PhysicalCamera(camera.fov, camera.aspect, camera.near, camera.far);
  c.position.copy(camera.position); c.quaternion.copy(camera.quaternion);
  c.filmGauge = camera.filmGauge; c.zoom = camera.zoom;
  c.updateProjectionMatrix(); c.updateMatrixWorld();
  c.fStop = DOF.fstop; c.focusDistance = focusDistance(true); c.apertureBlades = 0;
  return c;
}

// ----------------------------------------------------------------- photo render
// Path traced: light bounces between the surfaces, soft shadows from the sun
// and the sky, and the lamps near the view as real lights. It sharpens for as
// long as nothing moves; moving the camera returns to the live view.
// the path tracer, made once and shared by the photo and the bake
async function pathTracer(){
  const { WebGLPathTracer } = await import('three-gpu-pathtracer');
  if(!photo){
    const pt = new WebGLPathTracer(renderer);
    pt.renderDelay = 0; pt.fadeDuration = 0; pt.minSamples = 0;
    pt.rasterizeScene = false; pt.dynamicLowRes = false;
    pt.bounces = 5; pt.filterGlossyFactor = 0.5;
    pt.tiles.set(2, 2);
    photo = { pt, on: false };
    if(NO_TREE) pt._pathTracer.material.setDefine('LIGHT_TREE', 0);
    // on screen: the denoised picture once there is one, else the samples so far, with the
    // glow and the look as the live view has them
    pt.renderToCanvasCallback = (target, r, quad) => {
      if(DENOISE && photo.clean && photo.clean.shown) quad.material.map = photo.clean.out.texture;
      if(photo.plain){
        // as the path tracer draws it itself (no glow, no look): see photoHealth
        const auto = r.autoClear;
        r.autoClear = false; quad.render(r); r.autoClear = auto;
      }else presentPhoto(r, quad);
    };
  }
  return photo.pt;
}

// the scene as the tracer sees it: Blender's sky (or the night sky as a cube), and the
// lamps as real lights, every one up to the nearest lampCount to `around` (the tracer's
// light tree picks those likely to light each point); with lampMeshes, the lamps as
// whole meshes too (instances are not traced). Undone by unstage
function stage({ lampMeshes = true, around = controls.target, lampCount = PHOTO_LAMPS } = {}){
  const st = { env: scene.environment, bg: scene.background, envI: scene.environmentIntensity, cube: null };
  if(skyNow){
    // Blender's sky as the tracer's light from all round, without the sun's disc (the sun is
    // its own light), and with the disc as what the camera sees
    scene.environment = skyNow.plain; scene.background = skyNow.disc;
  }else{
    const cube = new THREE.WebGLCubeRenderTarget(512, { type: THREE.HalfFloatType });
    new THREE.CubeCamera(1, 10000, cube).update(renderer, envScene);
    st.cube = cube;
    scene.environment = cube.texture; scene.background = cube.texture;
  }
  hemi.visible = false;
  if(stars) stars.visible = false;                                    // points are not traced: the sky has its own
  const extra = new THREE.Group();
  if(lamps){
    lamps.group.visible = false;
    if(lampMeshes){
      const merged = (geo, mat) => {
        const parts = [], m = new THREE.Matrix4();
        for(let i = 0; i < lamps.items.length; i++){ lamps.poles.getMatrixAt(i, m); parts.push(geo.clone().applyMatrix4(m)); }
        return new THREE.Mesh(mergeGeometries(parts), mat);
      };
      extra.add(merged(lamps.poles.geometry, lamps.metalMat), merged(lamps.heads.geometry, lamps.headMat));
    }
    if(lamps.level > 0){
      const near = lamps.items.map(it => [it.foot.distanceTo(around), it]).sort((a, b) => a[0] - b[0]).slice(0, lampCount);
      for(const [, it] of near){
        const s = new THREE.SpotLight(LIGHT.colour, LAMP.candela * lamps.level, 0, 1.15, 0.85, 2);
        s.position.copy(it.head);
        s.target.position.copy(it.head).addScaledVector(it.dir, 2.0).setY(it.foot.y - 1);
        extra.add(s, s.target);
      }
    }
  }
  scene.add(extra);
  st.extra = extra;
  // every picture the tracer holds (colour, bump, roughness…) at one size: full while they fit
  const textures = new Set();
  world.traverse(o => { if(o.isMesh) for(const m of [].concat(o.material)) for(const k of TEXTURE_SLOTS) if(m[k]) textures.add(m[k]); });
  const tsize = textures.size > 48 ? 512 : 1024;
  photo.pt.textureSize.set(tsize, tsize);
  return st;
}

function unstage(st){
  if(!st) return;
  scene.environment = st.env; scene.background = st.bg; scene.environmentIntensity = st.envI;
  hemi.visible = true;
  if(stars) stars.visible = true;
  if(lamps) lamps.group.visible = true;
  if(st.extra){ scene.remove(st.extra); st.extra.traverse(o => { if(o.geometry) o.geometry.dispose(); }); st.extra = null; }
  if(st.cube){ st.cube.dispose(); st.cube = null; }
}

async function startPhoto(){
  if(!world || busy || BAKE.busy) return;
  if(photo && photo.on){ stopPhoto(); return; }
  busy = true;
  note('Preparing the photo render: building the ray-tracing structure of the scene…');
  await new Promise(r => setTimeout(r, 30));
  try{
    await pathTracer();
    photo.clean = null;                                                 // nothing denoised yet
    photo.checked = false; photo.retried = false;
    photo.saved = stage();
    photo.extra = photo.saved.extra;
    const t0 = performance.now();
    photo.cam = await photoCamera();
    photo.lens = DOF.on ? { radius: camera.getFocalLength() / DOF.fstop / 2000, focus: focusDistance(true) } : null;
    photo.pt.setScene(scene, photo.cam);
    photo.on = true; photo.started = performance.now();
    note(`Rendering the photo (set up in ${((performance.now() - t0) / 1000).toFixed(1)} s). It gets sharper while the camera stays put; Save image keeps it.`);
    $('#btnPhoto').textContent = 'Back to live view';
  }catch(e){
    stopPhoto();
    note('The photo render is not available here: ' + e.message, true);
  }
  busy = false;
}

// a pixel that is not a number or is infinite (a rare slip in the path tracer) becomes
// black, so it stays one dark dot instead of spreading; values are kept in half-float range
const SANE_GLSL = 'if(any(isnan(c)) || any(isinf(c)) || !all(lessThan(abs(c), vec3(1.0e30)))) c = vec3(0.0);\n'
                + 'c = clamp(c, 0.0, 6.0e4);';
let cleanQuad = null;
function cleanCopy(r, map){
  if(!cleanQuad) cleanQuad = new FullScreenQuad(new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, uniforms: { map: { value: null } }, depthTest: false, depthWrite: false,
    vertexShader: 'precision highp float; uniform mat4 modelViewMatrix; uniform mat4 projectionMatrix; in vec3 position; in vec2 uv; out vec2 vUv;\n'
      + 'void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'precision highp float; uniform sampler2D map; in vec2 vUv; out vec4 outColor;\n'
      + 'void main(){ ivec2 n = textureSize(map, 0); vec3 c = texelFetch(map, clamp(ivec2(vUv * vec2(n)), ivec2(0), n - 1), 0).rgb;\n'
      + SANE_GLSL + '\noutColor = vec4(c, 1.0); }' }));
  cleanQuad.material.uniforms.map.value = map;
  cleanQuad.render(r);
}

// the photo's light (linear, as traced) into a picture of its own (cleaned of invalid
// pixels), the glow added, then the look to the screen
function presentPhoto(r, quad){
  const size = r.getDrawingBufferSize(new THREE.Vector2());
  if(!photoHDR || photoHDR.width !== size.x || photoHDR.height !== size.y){
    if(photoHDR) photoHDR.dispose();
    photoHDR = new THREE.WebGLRenderTarget(size.x, size.y, { type: THREE.HalfFloatType, depthBuffer: false });
  }
  const auto = r.autoClear;
  r.autoClear = false;
  r.setRenderTarget(photoHDR); r.clear();
  cleanCopy(r, quad.material.map);
  if(bloom.strength > 0){ bloom.renderToScreen = false; bloom.render(r, null, photoHDR, 0, false); }
  look.renderToScreen = true;
  look.render(r, null, photoHDR);
  r.setRenderTarget(null);
  r.autoClear = auto;
}

function stopPhoto(why){
  if(!photo || !photo.on) return;
  photo.on = false;
  dropClean();
  unstage(photo.saved);
  photo.saved = null; photo.extra = null;
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
    const pt = photo.pt;
    // nothing to show until the first sample: the live view stays (not a black picture) while
    // the tracer's shader compiles, which on Windows (DirectX) can take a minute or more
    pt.renderToCanvas = pt.samples > 0;
    pt.renderSample();
    if(pt.samples === 0){
      const s = Math.round((performance.now() - photo.started) / 1000);
      const msg = pt.isCompiling ? `Compiling the path tracer's shader for this graphics card… ${s} s. `
        + 'The first time in a session this can take a minute or more (on Windows especially); then the samples start.'
        : `Starting… ${s} s.`;
      if($('#photoNote').textContent !== msg) $('#photoNote').textContent = msg;
      return;
    }
    if(!photo.checked && photo.pt.samples >= 2) photoHealth();
    if(!photo.on) return;
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
  if(dof.enabled){ dof.fStop = DOF.fstop; dof.focus = focusDistance(); }
  mirrorPass();
  composer.render();
}

// ----------------------------------------------------------------- photo health
// Is there any light in the photo? On some graphics cards the path tracer can come out
// empty: a shader the card's driver does not compile, or a feature it lacks. Then say
// so with the reason, try once without the light tree, and if still empty go back to
// the live view with the details, instead of showing a black picture
function gpuInfo(){
  const gl = renderer.getContext();
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const has = n => !!gl.getExtension(n);
  return { name: String(dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)),
           textures: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS),
           floatTarget: has('EXT_color_buffer_float'), floatLinear: has('OES_texture_float_linear'), floatBlend: has('EXT_float_blend') };
}

// what went wrong with a program, from all its logs: the errors first (the link step's,
// on DirectX the real one), warnings only if there is nothing else
function programError(d){
  const logs = [d.programLog, d.fragmentShader && d.fragmentShader.log, d.vertexShader && d.vertexShader.log]
    .map(l => String(l || '').replace(/\x00/g, '')).join('\n').split(/\n+/).map(l => l.trim()).filter(Boolean);
  const errors = logs.filter(l => /error/i.test(l) && !/^warning/i.test(l));
  const rest = logs.filter(l => !/^warning/i.test(l));
  return (errors.length ? errors : rest.length ? rest : logs).slice(0, 4).join(' ') || 'no log';
}

// a grid of pixels of a float picture: how many have light, how many are invalid,
// where the brightest is, and null if this browser cannot read float pictures back
const GRID = [9, 7];
function lightIn(rt){
  const px = new Float32Array(4);
  let lit = 0, bad = 0, read = 0, top = 0;
  const values = [];
  for(let j = 0; j < GRID[1]; j++) for(let i = 0; i < GRID[0]; i++){
    px.fill(-1);
    renderer.readRenderTargetPixels(rt, Math.floor((i + 0.5) / GRID[0] * rt.width), Math.floor((j + 0.5) / GRID[1] * rt.height), 1, 1, px);
    if(px[3] === -1){ values.push(0); continue; }
    read++;
    const v = px[0] + px[1] + px[2];
    if(!Number.isFinite(v)){ bad++; values.push(0); }
    else{ if(v > 1e-7) lit++; values.push(v); top = Math.max(top, v); }
  }
  return read ? { lit, bad, read, top, values } : null;
}

// the same grid of what the screen shows now (0..765 each), read straight after drawing
function shownOnScreen(){
  const gl = renderer.getContext(), c = renderer.domElement, buf = new Uint8Array(4), out = [];
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  for(let j = 0; j < GRID[1]; j++) for(let i = 0; i < GRID[0]; i++){
    gl.readPixels(Math.floor((i + 0.5) / GRID[0] * c.width), Math.floor((j + 0.5) / GRID[1] * c.height), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf);
    out.push(buf[0] + buf[1] + buf[2]);
  }
  renderer.resetState();
  return out;
}

// the first error of the photo's own drawing steps (the cleaning copy, the glow, the look)
function displayErrors(){
  for(const m of [cleanQuad && cleanQuad.material, bloom && bloom.materialHighPassFilter, look.material]){
    const prog = m && renderer.properties.get(m).currentProgram;
    const d = prog && prog.diagnostics;
    if(d && !d.runnable) return `${m.name || m.type}: ` + programError(d);
  }
  return null;
}

function photoHealth(){
  photo.checked = true;
  const mat = photo.pt._pathTracer.material;
  const prog = renderer.properties.get(mat).currentProgram;
  const d = prog && prog.diagnostics;
  const shader = d && !d.runnable ? programError(d) : null;
  let c = null;
  try{ c = lightIn(photo.pt._pathTracer.target); }catch(e){}
  const tree = mat.defines.LIGHT_TREE !== 0;
  if(!shader && (!c || c.lit > 0)){
    // the photo has light: does it reach the screen through the glow and the look?
    if(c && !photo.plain){
      const screen = shownOnScreen(), k = look.exposureOf(renderer.toneMappingExposure);
      // points bright enough that they cannot show black, yet all of them black
      const bright = c.values.map((v, i) => [v * k, screen[i]]).filter(([v]) => v > 0.05);
      if(bright.length && bright.every(([, sv]) => sv <= 3)){
        photo.plain = true;
        const g = gpuInfo(), err = displayErrors();
        const msg = `The photo has light, but its glow and look came out black on this graphics card (${g.name})`
                  + (err ? `: ${err.slice(0, 300)}` : '') + '. Showing it without them. Please send this message.';
        console.error('Render photo:', msg);
        note(msg, true);
      }
    }
    if(photo.retried){
      // empty with the light tree, right without it: keep it off on this card
      NO_TREE = true;
      try{ localStorage.setItem('rta.view3d.notree', '1'); }catch(e){}
      note('This graphics card renders photos without the light tree: night photos with many lamps sharpen more slowly.');
    }
    return;
  }
  const g = gpuInfo();
  const why = shader ? 'the path tracer\'s shader did not compile on this graphics card: ' + shader
            : c.bad === c.read ? 'every pixel came out invalid (not a number)' : 'the picture came out black, with no light at all';
  const facts = `Graphics: ${g.name}, ${g.textures} textures per shader. Float pictures ${g.floatTarget ? 'yes' : 'NO'}, float filtering ${g.floatLinear ? 'yes' : 'NO'}, `
              + `float blending ${g.floatBlend ? 'yes' : 'NO'}. Light tree ${tree ? 'on' : 'off'}.`;
  console.error('Render photo came out empty:', why, facts, shader || '');
  if(tree && !photo.retried){
    photo.retried = true; photo.checked = false;
    mat.setDefine('LIGHT_TREE', 0);
    photo.pt.reset(); dropClean();
    $('#photoNote').textContent = `The photo came out empty (${why}). Trying again without the light tree…`;
    return;
  }
  stopPhoto();
  const msg = `Render photo does not work on this graphics card yet: ${why}. ${facts} Please send this message (also in the browser console, F12).`;
  note(msg, true);
  $('#photoNote').textContent = msg;
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
  if(c[kind]){
    // as the material is now (Wet roads changes its colour and bump)
    c[kind].color.copy(m.color);
    if(m.normalScale) c[kind].normalScale.copy(m.normalScale);
    return c[kind];
  }
  const a = m.clone();
  a.fog = false; a.toneMapped = false; a.transparent = false; a.blending = THREE.NoBlending;
  a.onBeforeCompile = sh => {
    const normal = sh.fragmentShader.includes('#include <normal_fragment_begin>') ? 'normal' : 'vec3(0.0)';
    let out = kind === 'albedo' ? 'gl_FragColor = vec4(clamp(diffuseColor.rgb, 0.0, 1.0), 1.0);' : `gl_FragColor = vec4(${normal}, 1.0);`;
    if(kind === 'dist'){
      // how far the surface is from the bake's camera (Bake light, walls and objects)
      sh.uniforms.auxCam = AUX_CAM;
      sh.vertexShader = 'varying vec3 vBakeWorld;\n' + sh.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\n' + BAKE_WORLD_VERT);
      sh.fragmentShader = 'uniform vec3 auxCam;\nvarying vec3 vBakeWorld;\n' + sh.fragmentShader;
      out = 'gl_FragColor = vec4(distance(vBakeWorld, auxCam), 0.0, 0.0, 1.0);';
    }
    if(kind === 'height'){
      // the height of the surface over the bake's lowest point (Bake light)
      sh.uniforms.auxBase = AUX_BASE;
      sh.vertexShader = 'varying vec3 vBakeWorld;\n' + sh.vertexShader.replace('#include <project_vertex>', '#include <project_vertex>\n' + BAKE_WORLD_VERT);
      sh.fragmentShader = 'uniform float auxBase;\nvarying vec3 vBakeWorld;\n' + sh.fragmentShader;
      out = 'gl_FragColor = vec4(vBakeWorld.y - auxBase, 0.0, 0.0, 1.0);';
    }
    sh.fragmentShader = sh.fragmentShader.replace('#include <dithering_fragment>', '#include <dithering_fragment>\n' + out);
  };
  a.customProgramCacheKey = () => 'aux-' + kind;
  return c[kind] = a;
}

// cam: the photo's camera, or the bake's; kinds: albedo, normal (and height for the bake);
// extra: the lamps' lights added for the tracer
function renderAux(w, h, { cam = camera, kinds = ['albedo', 'normal'], extra = photo && photo.extra, lens = null } = {}){
  const make = type => new THREE.WebGLRenderTarget(w, h, { type, format: THREE.RGBAFormat, minFilter: THREE.NearestFilter,
                                                           magFilter: THREE.NearestFilter, depthBuffer: type === THREE.HalfFloatType });
  const shot = make(THREE.HalfFloatType), out = {};
  for(const kind of kinds) out[kind] = make(THREE.HalfFloatType);
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
  if(extra) extra.traverse(o => { if(o.isLight && o.visible){ o.visible = false; lights.push(o); } });
  const tent = u => (u < 0.5 ? Math.sqrt(2 * u) - 1 : 1 - Math.sqrt(2 - 2 * u));
  // with depth of field, each view from another spot on the lens, aimed so the focus stays put
  const p0 = cam.position.clone(), right = new THREE.Vector3(), up = new THREE.Vector3();
  if(lens){ cam.updateMatrixWorld(); right.setFromMatrixColumn(cam.matrixWorld, 0); up.setFromMatrixColumn(cam.matrixWorld, 1); }
  const fpx = h / 2 / Math.tan(THREE.MathUtils.degToRad(cam.fov || 45) / 2);
  renderer.shadowMap.autoUpdate = false;
  try{
    for(const kind of kinds){
      for(const [o, m] of swapped) o.material = Array.isArray(m) ? m.map(x => auxMaterial(x, kind)) : auxMaterial(m, kind);
      // the sky: as bright as it shows (albedo); no surface (normal)
      scene.background = kind === 'albedo' ? keep.bg : null;
      scene.backgroundIntensity = keep.bgI * renderer.toneMappingExposure;
      renderer.setClearColor(0x000000, 0);
      renderer.setRenderTarget(out[kind]); renderer.clear();
      for(let i = 0; i < AUX_VIEWS; i++){
        let jx = tent(((i % 4) + 0.5) / 4), jy = tent((Math.floor(i / 4) + 0.5) / 4);
        if(lens){
          const r = lens.radius * Math.sqrt((i + 0.5) / AUX_VIEWS), a = i * 2.39996323;
          const lx = r * Math.cos(a), ly = r * Math.sin(a);
          cam.position.copy(p0).addScaledVector(right, lx).addScaledVector(up, ly);
          cam.updateMatrixWorld();
          jx -= lx * fpx / lens.focus; jy += ly * fpx / lens.focus;
        }
        cam.setViewOffset(w, h, jx, jy, w, h);
        renderer.autoClear = true;
        renderer.setRenderTarget(shot); renderer.render(scene, cam);
        renderer.autoClear = false;
        auxQuad.material.uniforms.map.value = shot.texture; auxQuad.material.uniforms.weight.value = 1 / AUX_VIEWS;
        renderer.setRenderTarget(out[kind]); auxQuad.render(renderer);
      }
    }
  }finally{
    for(const [o, m] of swapped) o.material = m;
    for(const l of lights) l.visible = true;
    cam.clearViewOffset();
    if(lens){ cam.position.copy(p0); cam.updateMatrixWorld(); }
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
      const aux = renderAux(w, h, { cam: ph.cam || camera, lens: ph.lens });
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
    if(!ph.clean.at){
      const c = (() => { try{ return lightIn(ph.clean.out); }catch(e){ return null; } })();
      if(c && c.lit === 0){
        const g = gpuInfo();
        throw new Error(`its picture came out ${c.bad ? 'invalid' : 'black'} on this graphics card (${g.name}); showing the samples without it`);
      }
    }
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

// ----------------------------------------------------------------- baked light
// Bake light traces the light falling on the streets, sidewalks, islands and flat roofs
// as the photo does (the sun's soft shadows, the sky, light bounced off the walls, the
// lamps' pools), but from high above, for the whole area at once, and keeps it as a
// light map: how much light reaches each spot, without the surface's own colour (the
// traced picture divided by the colour of what it saw). With Baked light on, the live
// view takes the matt light of those surfaces from the map instead of working it out,
// while the shine (the sky and the lamps in wet roads, highlights) stays live, so the
// camera moves freely with the photo's light. Only what the bake saw from above takes
// it: walls, kerb faces and the ground under trees or bridges keep the live light. A
// bake belongs to its light (time of day, sun, lamps): change those and the live light
// shows until the next bake, and the last three bakes are kept for going back
const BAKE_PX = 2048;                                // the light map's longest side, at most
const BAKE_KEEP = 3;
const BAKE = { busy: false, cancel: false, on: false, list: [], shown: null };
const AUX_BASE = { value: 0 };
// shared by every street material: the map, where a world point falls in it, and the
// heights that count as the same surface
const BAKE_U = { bakeMap: { value: null }, bakeMatrix: { value: new THREE.Matrix4() }, bakeOn: { value: 0 },
                 bakeTol: { value: new THREE.Vector4(0.05, 0.1, 0, 0) },
                 // walls and objects: the light seen from the bake's camera, up to 4 views side by side in one map
                 vbMap: { value: null }, vbOn: { value: 0 }, vbN: { value: 0 }, vbPos: { value: new THREE.Vector3() },
                 vbMat: { value: [0, 1, 2, 3].map(() => new THREE.Matrix4()) }, vbRect: { value: [0, 1, 2, 3].map(() => new THREE.Vector4()) } };
const AUX_CAM = { value: new THREE.Vector3() };
const BAKE_WORLD_VERT = `{
  vec4 bp = vec4( transformed, 1.0 );
  #ifdef USE_BATCHING
    bp = batchingMatrix * bp;
  #endif
  #ifdef USE_INSTANCING
    bp = instanceMatrix * bp;
  #endif
  vBakeWorld = ( modelMatrix * bp ).xyz;
}`;
const BAKE_FRAG = `
if( bakeOn > 0.5 ){
  vec3 bE = vec3( 0.0 );
  float bw = 0.0;
  // the four texels round this point, each counting only if it was traced on this
  // surface (the same height), not on a roof or a tree above it or the road below
  vec4 bq = bakeMatrix * vec4( vBakeWorld, 1.0 );
  vec2 buv = bq.xy / bq.w;
  ivec2 bn = textureSize( bakeMap, 0 );
  vec2 bp = buv * vec2( bn ) - 0.5, bf = fract( bp );
  ivec2 b0 = ivec2( floor( bp ) );
  vec3 bsum = vec3( 0.0 );
  float bwsum = 0.0;
  for( int k = 0; k < 4; k ++ ){
    ivec2 o = ivec2( k & 1, k >> 1 );
    vec4 s = texelFetch( bakeMap, clamp( b0 + o, ivec2( 0 ), bn - 1 ), 0 );
    float w = ( o.x == 1 ? bf.x : 1.0 - bf.x ) * ( o.y == 1 ? bf.y : 1.0 - bf.y );
    w *= 1.0 - smoothstep( bakeTol.x, bakeTol.y, abs( s.a + bakeTol.z - vBakeWorld.y ) );
    bsum += s.rgb * w; bwsum += w;
  }
  vec2 bedge = min( buv, 1.0 - buv ) * vec2( bn );
  // facing up (not walls or kerb faces), traced here, and fading out over the last
  // 6% towards the map's edge, so the baked area blends into the live light
  #ifdef DOUBLE_SIDED
    float bup = vBakeUp * faceDirection;
  #else
    float bup = vBakeUp;
  #endif
  bw = smoothstep( 0.55, 0.8, bup ) * smoothstep( 0.05, 0.35, bwsum )
     * smoothstep( 1.0, max( 8.0, 0.06 * float( min( bn.x, bn.y ) ) ), min( bedge.x, bedge.y ) );
  bE = bsum / max( bwsum, 1e-4 );
  // walls, kerb faces, poles and objects (and the ground near by): the light the bake's camera
  // saw on this very surface (the same distance from it, facing it), from the first of its
  // views that saw it
  vec3 vE = vec3( 0.0 );
  float vw = 0.0;
  if( vbOn > 0.5 ){
    float dz = distance( vBakeWorld, vbPos );
    #ifdef DOUBLE_SIDED
      vec3 vn = normalize( vBakeN ) * faceDirection;
    #else
      vec3 vn = normalize( vBakeN );
    #endif
    float facing = dot( vn, ( vbPos - vBakeWorld ) / max( dz, 1e-4 ) );
    if( facing > 0.05 ){
      for( int i = 0; i < 4; i ++ ){
        if( i >= vbN ) break;
        vec4 q = vbMat[ i ] * vec4( vBakeWorld, 1.0 );
        if( q.w <= 0.0 ) continue;
        vec2 uv = q.xy / q.w;
        if( any( lessThan( uv, vec2( 0.0 ) ) ) || any( greaterThan( uv, vec2( 1.0 ) ) ) ) continue;
        vec4 r = vbRect[ i ];
        vec2 p = r.xy + uv * r.zw - 0.5, f = fract( p );
        ivec2 p0 = ivec2( floor( p ) ), lo = ivec2( r.xy ), hi = ivec2( r.xy + r.zw ) - 1;
        float tol = 0.02 + 0.004 * dz / max( facing, 0.2 );
        vec3 s3 = vec3( 0.0 );
        float ws = 0.0;
        for( int k = 0; k < 4; k ++ ){
          ivec2 o = ivec2( k & 1, k >> 1 );
          vec4 s = texelFetch( vbMap, clamp( p0 + o, lo, hi ), 0 );
          float w = ( o.x == 1 ? f.x : 1.0 - f.x ) * ( o.y == 1 ? f.y : 1.0 - f.y );
          w *= 1.0 - smoothstep( tol, 2.0 * tol, abs( s.a - dz ) );
          s3 += s.rgb * w; ws += w;
        }
        if( ws > 0.05 ){
          vec2 e = min( uv, 1.0 - uv ) * r.zw;
          vE = s3 / ws;
          vw = smoothstep( 0.05, 0.35, ws ) * smoothstep( 0.05, 0.25, facing ) * smoothstep( 0.5, 6.0, min( e.x, e.y ) );
          break;
        }
      }
    }
  }
  float tw = vw + bw * ( 1.0 - vw );
  if( tw > 0.0 ){
    vec3 E = ( vE * vw + bE * bw * ( 1.0 - vw ) ) / tw;
    reflectedLight.directDiffuse *= 1.0 - tw;
    reflectedLight.indirectDiffuse = mix( reflectedLight.indirectDiffuse, E * BRDF_Lambert( material.diffuseContribution ), tw );
  }
}`;

// a material that can take the baked light (prepare); a street surface (wet: the street
// materials Wet roads works on) also the wet roads' reflections and puddles
function bakePatch(m){
  if(!m.isMeshStandardMaterial || m.userData.bakeable) return;
  m.userData.bakeable = true;
  const wet = !!m.userData.dry;
  m.onBeforeCompile = sh => {
    Object.assign(sh.uniforms, BAKE_U);
    if(wet) Object.assign(sh.uniforms, REFL_U);
    sh.vertexShader = 'varying vec3 vBakeWorld;\nvarying float vBakeUp;\nvarying vec3 vBakeN;\n' + sh.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\n' + BAKE_WORLD_VERT + '\nvBakeN = normalize( ( vec4( transformedNormal, 0.0 ) * viewMatrix ).xyz ); vBakeUp = vBakeN.y;');
    let f = 'uniform highp sampler2D bakeMap;\nuniform mat4 bakeMatrix;\nuniform float bakeOn;\nuniform vec4 bakeTol;\n'
      + 'uniform highp sampler2D vbMap;\nuniform float vbOn;\nuniform int vbN;\nuniform vec3 vbPos;\nuniform mat4 vbMat[ 4 ];\nuniform vec4 vbRect[ 4 ];\n'
      + 'varying vec3 vBakeWorld;\nvarying float vBakeUp;\nvarying vec3 vBakeN;\n'
      + sh.fragmentShader.replace('#include <lights_fragment_end>', '#include <lights_fragment_end>\n' + BAKE_FRAG);
    if(wet) f = REFL_PARS + f.replace('#include <color_fragment>', '#include <color_fragment>\n' + REFL_PUDDLE)
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = mix( roughnessFactor, 0.03, rPud );')
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nnormal = normalize( mix( normal, nonPerturbedNormal, 0.9 * rPud ) );')
      .replace('#include <lights_fragment_maps>', '#include <lights_fragment_maps>\n' + REFL_FRAG);
    sh.fragmentShader = f;
  };
  m.customProgramCacheKey = () => wet ? 'bakeable-wet' : 'bakeable';
}

// ----------------------------------------------------------------- wet roads: reflections
// With Wet roads, the live view mirrors the scene in the water: the scene drawn once more
// from under the street (the camera reflected in it, as in a mirror), half as sharp, and
// the street surfaces show it where the sky would shine in them, blurred as much as they
// are rough, with Fresnel (strong at a glancing view, faint looking down) as any shine.
// Puddles: standing water in patches, smooth as a mirror and a little darker. The photo
// render traces its own reflections (wet roads evenly, without the puddles)
const REFL_U = { reflMap: { value: null }, reflMatrix: { value: new THREE.Matrix4() }, reflOn: { value: 0 },
                 reflPlane: { value: 0 }, reflLod: { value: 6 }, puddles: { value: 0 } };
const REFL_PARS = `
uniform sampler2D reflMap;
uniform mat4 reflMatrix;
uniform float reflOn, reflPlane, reflLod, puddles;
float rPud = 0.0;
float rHash( vec2 p ){ return fract( sin( dot( p, vec2( 127.1, 311.7 ) ) ) * 43758.5453 ); }
float rNoise( vec2 p ){
  vec2 i = floor( p ), f = fract( p ); f = f * f * ( 3.0 - 2.0 * f );
  return mix( mix( rHash( i ), rHash( i + vec2( 1.0, 0.0 ) ), f.x ), mix( rHash( i + vec2( 0.0, 1.0 ) ), rHash( i + vec2( 1.0, 1.0 ) ), f.x ), f.y );
}
`;
// the puddles: where a broad pattern (metres across, with finer edges) is above the level
// the amount sets; only on the level street surfaces
const REFL_PUDDLE = `
if( puddles > 0.0 ){
  vec2 pq = vBakeWorld.xz;
  float pn = 0.55 * rNoise( pq / 6.5 ) + 0.3 * rNoise( pq / 2.1 + 17.0 ) + 0.15 * rNoise( pq / 0.7 + 3.0 );
  float lev = 1.0 - 0.62 * puddles;
  rPud = smoothstep( lev, lev + 0.05, pn ) * smoothstep( 0.85, 0.97, vBakeUp )
       * ( 1.0 - smoothstep( 0.25, 0.6, abs( vBakeWorld.y - reflPlane ) ) );
  diffuseColor.rgb *= 1.0 - 0.3 * rPud;
}
`;
const REFL_FRAG = `
#if defined( RE_IndirectSpecular )
if( reflOn > 0.5 ){
  vec4 rq = reflMatrix * vec4( vBakeWorld, 1.0 );
  // a little rippled by the surface's bump
  vec2 ruv = rq.xy / rq.w + ( normal.xy - nonPerturbedNormal.xy ) * 0.06;
  float rIn = step( 0.0, rq.w ) * step( 0.0, ruv.x ) * step( ruv.x, 1.0 ) * step( 0.0, ruv.y ) * step( ruv.y, 1.0 );
  float rw = rIn * smoothstep( 0.8, 0.95, vBakeUp ) * ( 1.0 - smoothstep( 0.25, 0.6, abs( vBakeWorld.y - reflPlane ) ) );
  if( rw > 0.0 ){
    vec3 rc = textureLod( reflMap, clamp( ruv, 0.0, 1.0 ), clamp( material.roughness * 12.0, 0.0, reflLod ) ).rgb;
    radiance = mix( radiance, rc, rw );
  }
}
#endif
`;
let reflRT = null;
const reflCam = new THREE.PerspectiveCamera(), reflV = {
  pos: new THREE.Vector3(), cam: new THREE.Vector3(), view: new THREE.Vector3(), look: new THREE.Vector3(), target: new THREE.Vector3(),
  rot: new THREE.Matrix4(), normal: new THREE.Vector3(0, 1, 0), plane: new THREE.Plane(), clip: new THREE.Vector4(), q: new THREE.Vector4() };

// the mirror picture for this frame (or none: dry, or reflections off)
function mirrorPass(){
  const wanted = world && LIGHT.wet > 0 && LIGHT.mirror && !(photo && photo.on);
  REFL_U.puddles.value = world && LIGHT.wet > 0 ? LIGHT.puddles * Math.min(1, LIGHT.wet * 1.5) : 0;
  if(!wanted){ REFL_U.reflOn.value = 0; REFL_U.reflMap.value = null; return; }
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  const w = Math.max(16, Math.floor(size.x / 2)), h = Math.max(16, Math.floor(size.y / 2));
  if(!reflRT || reflRT.width !== w || reflRT.height !== h){
    if(reflRT) reflRT.dispose();
    reflRT = new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, generateMipmaps: true,
                                                minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter });
    REFL_U.reflLod.value = Math.max(0, Math.log2(Math.min(w, h)) - 3);
  }
  // the camera reflected in the street's level (three's Reflector, for a level plane)
  const V = reflV, y = STREET_Y;
  camera.updateMatrixWorld();
  V.pos.set(0, y, 0); V.cam.setFromMatrixPosition(camera.matrixWorld);
  V.pos.x = V.cam.x; V.pos.z = V.cam.z;
  if(V.cam.y <= y + 0.01){ REFL_U.reflOn.value = 0; REFL_U.reflMap.value = null; return; }
  V.view.subVectors(V.pos, V.cam).reflect(V.normal).negate().add(V.pos);
  V.rot.extractRotation(camera.matrixWorld);
  V.look.set(0, 0, -1).applyMatrix4(V.rot).add(V.cam);
  V.target.subVectors(V.pos, V.look).reflect(V.normal).negate().add(V.pos);
  reflCam.position.copy(V.view);
  reflCam.up.set(0, 1, 0).applyMatrix4(V.rot).reflect(V.normal);
  reflCam.lookAt(V.target);
  reflCam.far = camera.far; reflCam.near = camera.near;
  reflCam.updateMatrixWorld();
  reflCam.projectionMatrix.copy(camera.projectionMatrix);
  REFL_U.reflMatrix.value.set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 0.5, 0.5, 0, 0, 0, 1)
    .multiply(reflCam.projectionMatrix).multiply(reflCam.matrixWorldInverse);
  // nothing under the street drawn: the near plane tilted onto it (oblique clipping), 15 cm
  // above, so the street, the sidewalks and the blocks (10 cm up) are left out of their own
  // reflection, which would otherwise show their undersides
  V.plane.setFromNormalAndCoplanarPoint(V.normal, new THREE.Vector3(0, y + 0.15, 0)).applyMatrix4(reflCam.matrixWorldInverse);
  V.clip.set(V.plane.normal.x, V.plane.normal.y, V.plane.normal.z, V.plane.constant);
  const P = reflCam.projectionMatrix.elements;
  V.q.set((Math.sign(V.clip.x) + P[8]) / P[0], (Math.sign(V.clip.y) + P[9]) / P[5], -1, (1 + P[10]) / P[14]);
  V.clip.multiplyScalar(2 / V.clip.dot(V.q));
  P[2] = V.clip.x; P[6] = V.clip.y; P[10] = V.clip.z + 1; P[14] = V.clip.w;
  reflCam.projectionMatrixInverse.copy(reflCam.projectionMatrix).invert();
  // drawn without the mirror itself (it cannot show in its own picture), the painted lamp
  // pools on the ground, or new shadows
  REFL_U.reflOn.value = 0; REFL_U.reflMap.value = null;
  const glows = lamps && lamps.glows.visible, keep = renderer.getRenderTarget(), shadows = renderer.shadowMap.autoUpdate;
  if(glows) lamps.glows.visible = false;
  renderer.shadowMap.autoUpdate = false;
  renderer.setRenderTarget(reflRT);
  renderer.state.buffers.depth.setMask(true);
  renderer.clear();
  renderer.render(scene, reflCam);
  renderer.setRenderTarget(keep);
  renderer.shadowMap.autoUpdate = shadows;
  if(glows) lamps.glows.visible = true;
  REFL_U.reflMap.value = reflRT.texture; REFL_U.reflOn.value = 1; REFL_U.reflPlane.value = y;
}

// the light the bakes are for: the time of day, the sun, the lamps
const bakeKey = () => JSON.stringify([time, LIGHT.sun, LIGHT.lamps, LIGHT.colour].concat(time === 'hdri' ? [SKY.id, SKY.rotation, SKY.strength] : []));
const TIME_NAMES = { dawn: 'Dawn', day: 'Day', night: 'Night' };
const lightName = (t, sky) => t === 'hdri' ? (sky || (hdri ? hdri.entry.name : 'HDRI sky')) : TIME_NAMES[t];

// shows the bake for the light now (if Baked light is on and there is one), else the live light
function applyBake(){
  const b = BAKE.on ? BAKE.list.find(x => x.key === bakeKey()) || null : null;
  BAKE.shown = b;
  BAKE_U.bakeOn.value = b ? 1 : 0;
  BAKE_U.bakeMap.value = b ? b.map.texture : null;
  if(b){ BAKE_U.bakeMatrix.value.copy(b.matrix); BAKE_U.bakeTol.value.set(b.tol[0], b.tol[1], b.base, 0); }
  // walls and objects, from where that bake's camera stood
  const v = b && b.walls;
  BAKE_U.vbOn.value = v ? 1 : 0; BAKE_U.vbMap.value = v ? v.map.texture : null; BAKE_U.vbN.value = v ? v.mats.length : 0;
  if(v){
    BAKE_U.vbPos.value.copy(v.pos);
    v.mats.forEach((m, i) => BAKE_U.vbMat.value[i].copy(m));
    v.rects.forEach((r, i) => BAKE_U.vbRect.value[i].copy(r));
  }
  // the lamps' painted pools and the contact shadows are in the bake already
  if(lamps) lamps.glows.visible = lamps.level > 0 && !b;
  if(gtao) gtao.blendIntensity = b ? 0.4 : 0.9;
  bakeNote();
}

function bakeNote(text, bad){
  const n = $('#bakeNote');
  if(!n) return;
  const box = $('#bakeOn');
  box.disabled = !BAKE.list.length; box.checked = BAKE.on;
  if(text === undefined){
    const any = BAKE.list.find(x => x.key === bakeKey());
    const about = b => `${lightName(b.time, b.sky)}, ${b.area === 'view' ? 'what you saw' : 'the whole place'} at ${Math.round(b.mpp * 100)} cm a pixel`
      + (b.walls ? `, walls and objects ${b.walls.mode === 'around' ? 'all round' : 'in view'}` : '') + `, ${b.samples} samples`;
    if(BAKE.busy) return;
    if(BAKE.shown) text = `Baked light on (${about(BAKE.shown)}). Move round freely: the streets, sidewalks and islands have the photo's light`
      + (BAKE.shown.walls ? `, and so do the walls, kerb faces, poles and objects the bake saw from where it was made; what it did not see keeps the live light.`
                          : `; walls, kerb faces and the ground under trees keep the live light.`);
    else if(BAKE.on && BAKE.list.length) text = `No bake for this light yet (${lightName(time)}${LIGHT.sun !== 1 || LIGHT.lamps !== 1 ? ', with these sun and lamp settings' : ''}): showing the live light. Bake light bakes it; going back to a baked light shows its bake again.`;
    else if(any) text = `Baked (${about(any)}). Tick Baked light to see it in the live view.`;
    else text = 'Bake light traces the light on the ground as Render photo does (it does not need a photo first) and keeps it, so the live view shows the photo\'s light while you move round. What you see: the ground in view now, as sharp as its size allows; the whole place: all of it, coarser. Walls and objects: the light on them too, from where you stand.';
    bad = false;
  }
  n.textContent = text; n.classList.toggle('bad', !!bad);
}

// the bakes are of this scene: a new scene starts without them
function dropBakes(){
  BAKE.cancel = true;
  for(const b of BAKE.list) dropBake(b);
  BAKE.list = [];
  applyBake();
}

// the area to bake: a square round where the camera looks, or the whole place, in
// square pixels, and a camera high above it (looking almost straight down, so the
// tracer's own camera, the same as the photo's, is used)
function bakeArea(kind){
  const box = new THREE.Box3().setFromObject(world);
  let x0 = box.min.x - 2, x1 = box.max.x + 2, z0 = box.min.z - 2, z1 = box.max.z + 2;
  const most = Math.min(BAKE_PX, renderer.capabilities.maxTextureSize);
  let mpp = Math.max(0.2, Math.max(x1 - x0, z1 - z0) / most);
  if(kind === 'view'){
    camera.updateMatrixWorld();
    // what the camera sees of the ground: where the edges and corners of the view meet
    // it (a ray over the horizon stops at a distance that grows with the height), within
    // the place; as sharp as the map's size allows, never finer than 10 cm a pixel
    const far = Math.max(150, 4 * Math.max(camera.position.y - box.min.y, 5)), seen = [];
    for(const [u, v] of [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]){
      aim.setFromCamera(new THREE.Vector2(u, v), camera);
      let p = onGround(aim.ray, box.min.y);
      if(!p || p.distanceTo(aim.ray.origin) > far) p = aim.ray.origin.clone().addScaledVector(aim.ray.direction, far);
      seen.push(p);
    }
    const a = [Math.max(x0, Math.min(...seen.map(p => p.x))), Math.min(x1, Math.max(...seen.map(p => p.x))),
               Math.max(z0, Math.min(...seen.map(p => p.z))), Math.min(z1, Math.max(...seen.map(p => p.z)))];
    if(a[1] - a[0] > 10 && a[3] - a[2] > 10){
      [x0, x1, z0, z1] = a;
      mpp = Math.max(0.1, Math.max(x1 - x0, z1 - z0) / most);
    }else kind = 'whole';
  }
  const w = Math.max(16, Math.ceil((x1 - x0) / mpp)), h = Math.max(16, Math.ceil((z1 - z0) / mpp));
  const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2, sx = w * mpp, sz = h * mpp;
  const base = box.min.y, top = box.max.y, D = Math.max(400, 6 * Math.max(sx, sz));
  const cam = new THREE.PerspectiveCamera(THREE.MathUtils.radToDeg(2 * Math.atan(sz / 2 / D)), sx / sz,
                                          Math.max(1, D - (top - base) - 5), D + 10);
  cam.position.set(cx, base + D, cz);
  cam.up.set(0, 0, -1);                                                // the top of the map: north
  cam.lookAt(cx, base, cz);
  cam.updateMatrixWorld(); cam.updateProjectionMatrix();
  // a world point to its place in the map (0..1), as the tracer saw it
  const matrix = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 1)
    .multiply(cam.projectionMatrix).multiply(cam.matrixWorldInverse);
  return { kind, w, h, mpp, base, cam, matrix, centre: new THREE.Vector3(cx, base, cz) };
}

// the walls' bake, made from where the camera stands: the view itself (as sharp as the bake's
// size allows), or all round (four views a little wider than a quarter turn each, side by side
// in one map); each with where a world point falls in it
function wallViews(mode){
  if(mode !== 'view' && mode !== 'around') return { mode: 'none', list: [] };
  camera.updateMatrixWorld();
  const most = Math.min(BAKE_PX, renderer.capabilities.maxTextureSize), pos = camera.position.clone();
  const make = (cam, w, h, rect) => {
    cam.updateProjectionMatrix(); cam.updateMatrixWorld();
    const matrix = new THREE.Matrix4().set(0.5, 0, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 1)
      .multiply(cam.projectionMatrix).multiply(cam.matrixWorldInverse);
    return { cam, w, h, rect, matrix };
  };
  if(mode === 'view'){
    const asp = camera.aspect, side = Math.min(most, 1600);
    const w = asp >= 1 ? side : Math.max(16, Math.round(side * asp)), h = asp >= 1 ? Math.max(16, Math.round(side / asp)) : side;
    const cam = new THREE.PerspectiveCamera(camera.fov, asp, Math.max(0.05, camera.near), camera.far);
    cam.position.copy(camera.position); cam.quaternion.copy(camera.quaternion);
    return { mode, pos, W: w, H: h, list: [make(cam, w, h, [0, 0, w, h])] };
  }
  const S = Math.min(1024, Math.floor(most / 2));
  const list = [[1, 0], [-1, 0], [0, 1], [0, -1]].map(([dx, dz], i) => {
    const cam = new THREE.PerspectiveCamera(95, 1, 0.05, camera.far);
    cam.position.copy(pos); cam.lookAt(pos.x + dx, pos.y, pos.z + dz);
    return make(cam, S, S, [(i % 2) * S, Math.floor(i / 2) * S, S, S]);
  });
  return { mode, pos, W: 2 * S, H: 2 * S, list };
}

function dropBake(b){
  b.map.dispose();
  if(b.walls) b.walls.map.dispose();
}

let bakeQuad = null;
// the light reaching each spot: the traced (denoised) picture over the colour of what it
// saw, and the height of that surface; nothing where the surface is too dark to tell
function lightMap(light, aux, w, h, into = null, at = null){
  if(!bakeQuad) bakeQuad = new FullScreenQuad(new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3, depthTest: false, depthWrite: false,
    uniforms: { light: { value: null }, albedo: { value: null }, height: { value: null }, offset: { value: new THREE.Vector2() } },
    vertexShader: 'precision highp float; uniform mat4 modelViewMatrix; uniform mat4 projectionMatrix; in vec3 position;\n'
      + 'void main(){ gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
    fragmentShader: 'precision highp float; uniform sampler2D light, albedo, height; uniform vec2 offset; out vec4 outColor;\n'
      + 'void main(){ ivec2 p = ivec2(gl_FragCoord.xy - offset); vec3 c = texelFetch(light, p, 0).rgb;\n' + SANE_GLSL + '\n'
      + 'vec3 a = texelFetch(albedo, p, 0).rgb; float hgt = texelFetch(height, p, 0).r;\n'
      + 'bool ok = max(max(a.r, a.g), a.b) > 0.01 && !isnan(hgt);\n'
      + 'outColor = ok ? vec4(min(3.14159265 * c / max(a, vec3(0.02)), vec3(6.0e4)), hgt) : vec4(0.0, 0.0, 0.0, 1.0e4); }' }));
  const map = into || new THREE.WebGLRenderTarget(w, h, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false,
                                                          minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false });
  const u = bakeQuad.material.uniforms;
  u.light.value = light; u.albedo.value = aux.albedo.texture; u.height.value = (aux.height || aux.dist).texture;
  u.offset.value.set(at ? at[0] : 0, at ? at[1] : 0);
  const keep = renderer.getRenderTarget(), auto = renderer.autoClear;
  if(at) map.viewport.set(at[0], at[1], w, h);
  renderer.autoClear = !at;                                            // into its part only: the other views stay
  renderer.setRenderTarget(map);
  bakeQuad.render(renderer);
  if(at) map.viewport.set(0, 0, map.width, map.height);
  renderer.autoClear = auto;
  renderer.setRenderTarget(keep);
  return map;
}

// samples until there are `samples`, a few tiles at a time, waiting for the graphics card
// between them, so the live view keeps moving meanwhile
async function traceBake(pt, samples, progress){
  const gl = renderer.getContext();
  const pause = ms => new Promise(r => setTimeout(r, ms));
  let k = 1;
  while(pt.samples < samples){
    if(BAKE.cancel) return false;
    if(pt.isCompiling){ progress(); await pause(250); pt.renderSample(); continue; }
    const t = performance.now();
    for(let i = 0; i < k && pt.samples < samples; i++) pt.renderSample();
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    while(gl.getSyncParameter(sync, gl.SYNC_STATUS) !== gl.SIGNALED) await pause(4);
    gl.deleteSync(sync);
    const dt = performance.now() - t;
    k = dt < 30 ? Math.min(k * 2, 256) : dt > 90 ? Math.max(1, k >> 1) : k;
    progress();
    await pause(0);
  }
  return true;
}

async function bakeLight(){
  if(BAKE.busy){ BAKE.cancel = true; return; }                       // the button stops it
  if(!world || busy) return;
  if(photo && photo.on) stopPhoto();
  const kind = $('#bakeArea').value, samples = Math.max(1, Math.round(+$('#bakeQuality').value) || 128), key = bakeKey(), bakeTime = time;
  BAKE.busy = true; BAKE.cancel = false;
  $('#btnBake').textContent = 'Stop baking'; $('#btnPhoto').disabled = true;
  const started = performance.now();
  const secs = () => Math.round((performance.now() - started) / 1000);
  const job = window.trackJob ? window.trackJob('Baking the light') : null;
  if(job) job.set('building the ray-tracing structure of the scene');
  bakeNote('Preparing the bake: building the ray-tracing structure of the scene…');
  await new Promise(r => setTimeout(r, 30));
  let pt;
  try{ pt = await pathTracer(); }
  catch(e){
    BAKE.busy = false;
    $('#btnBake').textContent = 'Bake light'; $('#btnPhoto').disabled = !world;
    bakeNote('Bake light is not available here: ' + e.message, true);
    if(job) job.done(false, e.message);
    return;
  }
  const a = bakeArea(kind);
  const faces = wallViews($('#bakeWalls') ? $('#bakeWalls').value : 'none');
  const keepPt = { sync: pt.synchronizeRenderSize, tiles: pt.tiles.clone() };
  let aux = null, faux = [], atlas = null, done = false;
  try{
    // traced as the photo, but matt: what comes back is the light the surfaces take in and
    // give back evenly, not their shine (that stays live)
    // the lamps as in the photo, poles and heads too: their shadows fall on the ground (from
    // above they cover a few spots, which keep the live light)
    const st = stage({ lampMeshes: true, around: a.centre, lampCount: 4000 });
    const matt = new Map();
    try{
      scene.traverseVisible(o => {
        if(!o.isMesh) return;
        for(const m of [].concat(o.material)) if(m.isMeshStandardMaterial && !matt.has(m)){
          matt.set(m, { metalness: m.metalness, metalnessMap: m.metalnessMap, own: Object.prototype.hasOwnProperty.call(m, 'specularIntensity'), spec: m.specularIntensity });
          m.metalness = 0; m.metalnessMap = null; m.specularIntensity = 0;
        }
      });
      pt.synchronizeRenderSize = false;
      pt._pathTracer.setSize(a.w, a.h);
      pt.tiles.set(Math.ceil(a.w / 512), Math.ceil(a.h / 512));
      pt.setScene(scene, a.cam);
    }finally{
      for(const [m, v] of matt){
        m.metalness = v.metalness; m.metalnessMap = v.metalnessMap;
        if(v.own) m.specularIntensity = v.spec; else delete m.specularIntensity;
      }
    }
    // what the tracer sees, drawn without noise: colour, direction and height (from above), or
    // distance (from the walls' views)
    AUX_BASE.value = a.base;
    try{
      aux = renderAux(a.w, a.h, { cam: a.cam, kinds: ['albedo', 'normal', 'height'], extra: st.extra });
      for(const f of faces.list){
        AUX_CAM.value.copy(f.cam.position);
        faux.push(renderAux(f.w, f.h, { cam: f.cam, kinds: ['albedo', 'normal', 'dist'], extra: st.extra }));
      }
    }finally{ unstage(st); }
    // the time left from the samples' own pace (the shader compiling first is not part of it):
    // the work is every view's pixels times the samples
    const views = [{ w: a.w, h: a.h, cam: null, aux, label: a.kind === 'view' ? 'the ground you see' : 'the whole place' }]
      .concat(faces.list.map((f, i) => ({ ...f, aux: faux[i], label: faces.list.length > 1 ? `walls and objects, view ${i + 1} of ${faces.list.length}` : 'walls and objects you see' })));
    const work = views.reduce((k, v) => k + v.w * v.h * samples, 0);
    let workDone = 0, paced = null;
    const lights = [];
    for(let vi = 0; vi < views.length; vi++){
      const v = views[vi];
      if(vi > 0){
        pt._pathTracer.setSize(v.w, v.h);
        pt.tiles.set(Math.ceil(v.w / 512), Math.ceil(v.h / 512));
        pt.setCamera(v.cam);
      }
      pt.reset();
      const ok = await traceBake(pt, samples, () => {
        const n = Math.floor(pt.samples), s = secs(), now = workDone + n * v.w * v.h;
        if(n >= 1 && paced === null) paced = { work: now, t: performance.now() };
        const rate = paced && now - paced.work > 0 ? (now - paced.work) / ((performance.now() - paced.t) / 1000) : 0;
        const left = rate > 0 ? (work - now) / rate : null;
        bakeNote(pt.isCompiling && !n ? `Compiling the path tracer's shader for this graphics card… ${s} s (the first time in a session it can take a minute or more).`
          : `Baking the light of ${v.label} (${v.w} × ${v.h} px): ${n} of ${samples} samples, ${s} s`
            + (left != null ? `, about ${Math.max(1, Math.round(left))} s left.` : '.') + ' The view stays live meanwhile.');
        if(job) job.set(pt.isCompiling && !n ? "compiling the path tracer's shader (once a session)" : `${v.label}: ${n} of ${samples} samples`,
                        pt.isCompiling && !n ? null : now / work, left);
      });
      if(!ok) throw new Error('stopped');
      workDone += samples * v.w * v.h;
      const c = (() => { try{ return lightIn(pt.target); }catch(e){ return null; } })();
      if(c && c.lit === 0) throw new Error(`the traced light came out ${c.bad ? 'invalid' : 'black'} on this graphics card (${gpuInfo().name})`);
      // the noise cleared as the photo's is (if the denoiser is there), then into the light map
      let light = pt.target.texture, out = null;
      if(DENOISE){
        try{
          bakeNote('Clearing the noise (Open Image Denoise)…');
          if(job) job.set('clearing the noise (Open Image Denoise)', workDone / work);
          if(!denoiser){
            denoiserLoad = denoiserLoad || Denoiser.load(renderer.getContext(), new URL('./vendor/oidn/rt_hdr_calb_cnrm.tza', import.meta.url).href);
            denoiser = await denoiserLoad;
          }
          if(BAKE.cancel) throw new Error('stopped');
          out = new THREE.WebGLRenderTarget(v.w, v.h, { type: THREE.FloatType, format: THREE.RGBAFormat, depthBuffer: false,
                                                        minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter });
          renderer.initRenderTarget(out);
          const tex = rt => renderer.properties.get(rt.texture).__webglTexture;
          try{
            denoiser.denoise({ color: tex(pt.target), albedo: tex(v.aux.albedo), normal: tex(v.aux.normal), output: tex(out), width: v.w, height: v.h });
          }finally{ renderer.resetState(); }
          light = out.texture;
        }catch(e){
          if(out){ out.dispose(); out = null; }
          if(e.message === 'stopped') throw e;
          console.error('Bake light: denoiser', e);
        }
      }
      if(vi === 0) lights.push(lightMap(light, v.aux, v.w, v.h));
      else{
        if(!atlas){
          atlas = new THREE.WebGLRenderTarget(faces.W, faces.H, { type: THREE.HalfFloatType, format: THREE.RGBAFormat, depthBuffer: false,
                                                                  minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter, generateMipmaps: false });
          const keep = renderer.getRenderTarget(), cc = renderer.getClearColor(new THREE.Color()), ca = renderer.getClearAlpha();
          renderer.setRenderTarget(atlas); renderer.setClearColor(0x000000, 1e4); renderer.clear();
          renderer.setClearColor(cc, ca); renderer.setRenderTarget(keep);
        }
        lightMap(light, v.aux, v.w, v.h, atlas, v.rect);
      }
      if(out) out.dispose();
    }
    const map = lights[0];
    const walls = atlas ? { map: atlas, mode: faces.mode, pos: faces.pos.clone(), mats: faces.list.map(f => f.matrix),
                            rects: faces.list.map(f => new THREE.Vector4(...f.rect)) } : null;
    BAKE.list = BAKE.list.filter(b => { if(b.key !== key) return true; dropBake(b); return false; });
    BAKE.list.unshift({ key, time: bakeTime, sky: bakeTime === 'hdri' && hdri ? hdri.entry.name : null, area: a.kind, mpp: a.mpp, samples, map, matrix: a.matrix,
                        base: a.base, tol: [0.03 + 0.1 * a.mpp, 0.06 + 0.2 * a.mpp], seconds: secs(), walls });
    atlas = null;
    while(BAKE.list.length > BAKE_KEEP) dropBake(BAKE.list.pop());
    BAKE.on = true;
    done = true;
  }catch(e){
    if(e.message !== 'stopped'){ console.error('Bake light:', e); bakeNote('The bake did not work here: ' + e.message, true); }
  }finally{
    if(aux) for(const rt of Object.values(aux)) rt.dispose();
    for(const fa of faux) for(const rt of Object.values(fa)) rt.dispose();
    if(atlas) atlas.dispose();
    // the tracer back as the photo has it, its big pictures let go
    pt.synchronizeRenderSize = keepPt.sync; pt.tiles.copy(keepPt.tiles);
    pt._pathTracer.setSize(16, 16);
    pt.reset();
    BAKE.busy = false;
    $('#btnBake').textContent = 'Bake light'; $('#btnPhoto').disabled = !world;
  }
  if(job) job.done(done, done ? `${samples} samples` : (BAKE.cancel ? 'by you' : 'see the 3D tab'));
  if(done){
    applyBake(); bakeNote(`Baked in ${secs()} s. ` + $('#bakeNote').textContent);
    msg.textContent = 'Baked light on: move round freely. B switches it off and on to compare with the live light.';
    setTimeout(() => { if(msg.textContent.startsWith('Baked light on')) msg.textContent = ''; }, 8000);
  }
  else if(BAKE.cancel){ applyBake(); bakeNote('The bake was stopped. ' + $('#bakeNote').textContent); }
}

function loop(){
  if(!visible){ running = false; return; }
  walk();
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
  staleCheck();
  if(!running){ running = true; requestAnimationFrame(loop); }
});
$('#btnScene3d').addEventListener('click', () => generate(false));
$('#btnScene3dUpdate').addEventListener('click', () => generate(true));
window.addEventListener('materials', staleCheck);
// settings changed in the Generate tab show when coming back here
document.addEventListener('change', e => { if(!e.target.closest('#pane-view3d')) staleCheck(); });
// the Light panel
const showLight = () => {
  $('#sunStrength').value = Math.round(LIGHT.sun * 100); $('#sunStrengthVal').textContent = Math.round(LIGHT.sun * 100) + '%';
  $('#lampStrength').value = Math.round(LIGHT.lamps * 100); $('#lampStrengthVal').textContent = Math.round(LIGHT.lamps * 100) + '%';
  $('#lampColour').value = LIGHT.colour;
  $('#wetRoads').value = Math.round(LIGHT.wet * 100); $('#wetRoadsVal').textContent = Math.round(LIGHT.wet * 100) + '%';
  $('#puddles').value = Math.round(LIGHT.puddles * 100); $('#puddlesVal').textContent = Math.round(LIGHT.puddles * 100) + '%';
  $('#liveMirror').checked = !!LIGHT.mirror;
  $('#puddles').disabled = !(LIGHT.wet > 0); $('#liveMirror').disabled = !(LIGHT.wet > 0);
};
showLight();
$('#sunStrength').addEventListener('input', () => {
  LIGHT.sun = +$('#sunStrength').value / 100; showLight(); saveLight();
  if(!renderer) return;
  if(photo && photo.on) stopPhoto();
  // the sun (or moon) brighter or dimmer; the exposure stays, so the picture follows
  sun.intensity = sunBase * LIGHT.sun;
  applyBake();
});
$('#lampStrength').addEventListener('input', () => {
  LIGHT.lamps = +$('#lampStrength').value / 100; showLight(); saveLight();
  if(photo && photo.on) stopPhoto();
  if(lamps) { lampLevel(TIMES[time].lamps); poolAt = null; }
  applyBake();
});
$('#wetRoads').addEventListener('input', () => {
  LIGHT.wet = +$('#wetRoads').value / 100; showLight(); saveLight();
  if(photo && photo.on) stopPhoto();
  applyWet();
});
$('#puddles').addEventListener('input', () => { LIGHT.puddles = +$('#puddles').value / 100; showLight(); saveLight(); });
$('#liveMirror').addEventListener('change', () => { LIGHT.mirror = $('#liveMirror').checked; saveLight(); });
$('#lampColour').addEventListener('input', () => {
  LIGHT.colour = $('#lampColour').value; saveLight();
  if(photo && photo.on) stopPhoto();
  if(lamps) lampLevel(TIMES[time].lamps);
  applyBake();
});
// the Sky panel: the physical sky or an HDRI, its rotation and strength, your own skies
const PHYSICAL_THUMB = 'linear-gradient(#3f6fb0, #8fb5df 70%, #d8e3ea)';
function skyNoteText(text, bad){
  const n = $('#skyNote');
  if(n){ n.textContent = text; n.classList.toggle('bad', !!bad); }
}
function showSky(){
  const box = $('#skyList');
  if(!box) return;
  const base = new URL('./', import.meta.url).href;
  const cards = [{ id: 'physical', name: 'Physical sky', sub: 'by the time of day', bg: PHYSICAL_THUMB }].concat(skyList.map(e => ({
    id: e.id, name: e.name, sub: (SKY_KINDS[e.kind] || SKY_KINDS.day).label + (e.builtin ? '' : ', yours'),
    bg: e.thumb_url ? `url("${base + e.thumb_url}")` : 'linear-gradient(#56606e, #9aa4b0)' })));
  box.innerHTML = '';
  for(const c of cards){
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'sky-card'; b.setAttribute('aria-pressed', c.id === SKY.id ? 'true' : 'false');
    b.innerHTML = '<div class="sky-thumb"></div><b></b><small></small>';
    b.querySelector('.sky-thumb').style.backgroundImage = c.bg;
    b.querySelector('b').textContent = c.name; b.querySelector('small').textContent = c.sub;
    b.title = c.id === 'physical' ? "Blender's physical sky, set by the time of day below" : `Use the sky "${c.name}"`;
    b.addEventListener('click', () => { if(c.id !== SKY.id || (c.id !== 'physical' && time !== 'hdri')) useSky(c.id); });
    box.appendChild(b);
  }
  const e = skyList.find(x => x.id === SKY.id), on = !!e;
  $('#skyRotation').disabled = !on; $('#skyStrength').disabled = !on;
  $('#skyRotation').value = SKY.rotation; $('#skyRotationVal').textContent = Math.round(SKY.rotation) + '°';
  $('#skyStrength').value = Math.round(SKY.strength * 10);
  $('#skyStrengthVal').textContent = (SKY.strength > 0 ? '+' : '') + SKY.strength.toFixed(1) + ' EV';
  $('#skyKindRow').hidden = !(e && !e.builtin);
  if(e) $('#skyKind').value = e.kind || 'day';
  $('#btnSkyDelete').disabled = !(e && !e.builtin);
  if(hdriLoading) return;
  if(!e) skyNoteText(SKY.id === 'physical' ? "Physical sky: Blender's sky for the time of day below. Pick a photographed sky (HDRI) for clouds, "
    + 'its own sun and light; Import HDRI uses your own.' : 'Loading the list of skies…');
  else{
    const sunNote = hdri && hdri.id === e.id ? (hdri.sun.sharp > 300 && hdriSun(hdri).elev > -1
      ? ` Its sun is at ${Math.round(hdriSun(hdri).elev)}° and casts the shadows.` : ' No sharp sun in it: soft light from the whole sky.') : '';
    skyNoteText((e.builtin ? `${e.title}, by ${(e.authors || []).join(', ')}, Poly Haven, ${e.licence}.` : `Your sky, from ${e.original || e.file}.`)
      + sunNote + ' Render photo and Bake light use it too.');
  }
}
async function refreshSkies(){
  try{
    const r = await fetch('/api/skies');
    if(!r.ok) throw new Error(r.statusText);
    skyList = (await r.json()).skies || [];
  }catch(e){ skyList = []; }
  showSky();
}
const skiesReady = refreshSkies();
$('#skyRotation').addEventListener('input', () => {
  SKY.rotation = +$('#skyRotation').value; saveSky(); showSky();
  if(photo && photo.on) stopPhoto();
  hdriChanged();
});
$('#skyStrength').addEventListener('input', () => {
  SKY.strength = +$('#skyStrength').value / 10; saveSky(); showSky();
  if(photo && photo.on) stopPhoto();
  hdriChanged();
});
$('#skyRotation').addEventListener('dblclick', () => { SKY.rotation = 0; saveSky(); showSky(); hdriChanged(); });
$('#skyStrength').addEventListener('dblclick', () => { SKY.strength = 0; saveSky(); showSky(); hdriChanged(); });
$('#skyKind').addEventListener('change', async () => {
  const e = skyList.find(x => x.id === SKY.id);
  if(!e || e.builtin) return;
  try{
    const r = await fetch(`/api/skies/${e.id}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ kind: $('#skyKind').value }) });
    if(!r.ok) throw new Error((await r.json()).detail || r.statusText);
    Object.assign(e, await r.json());
  }catch(err){ skyNoteText('Could not change it: ' + err.message, true); return; }
  if(hdri && hdri.id === e.id){
    // the same picture, brought to its new kind of light
    const kind = SKY_KINDS[e.kind] || SKY_KINDS.day;
    hdri.kind = e.kind;
    hdri.scale = e.kind === 'night' ? kind.skyY / Math.max(hdri.skyY, 1e-6) : kind.level / Math.max(hdri.level, 1e-6);
    if(photo && photo.on) stopPhoto();
    hdriChanged();
  }
  showSky();
});
$('#btnSkyImport').addEventListener('click', () => $('#skyFile').click());
$('#skyFile').addEventListener('change', async () => {
  const f = $('#skyFile').files[0];
  $('#skyFile').value = '';
  if(!f) return;
  const fd = new FormData();
  fd.append('file', f); fd.append('kind', /night/i.test(f.name) ? 'night' : /sunset|sunrise|dusk|dawn/i.test(f.name) ? 'sunset' : /overcast|cloudy/i.test(f.name) ? 'overcast' : 'day');
  skyNoteText(`Importing ${f.name} (${(f.size / 1048576).toFixed(1)} MB)…`);
  try{
    const r = await fetch('/api/skies/import', { method: 'POST', body: fd });
    const res = await r.json();
    if(!r.ok) throw new Error(res.detail || r.statusText);
    await refreshSkies();
    window.dispatchEvent(new CustomEvent('rta-log', { detail: `Sky imported: ${res.name} (light like: ${(SKY_KINDS[res.kind] || SKY_KINDS.day).label}; change it under the sky list if it is not).` }));
    useSky(res.id);
  }catch(err){ skyNoteText(`${f.name} could not be imported: ${err.message}`, true); }
});
$('#btnSkyDelete').addEventListener('click', async () => {
  const e = skyList.find(x => x.id === SKY.id);
  if(!e || e.builtin || !confirm(`Delete the sky "${e.name}" from the workspace?`)) return;
  try{ await fetch(`/api/skies/${e.id}`, { method: 'DELETE' }); }catch(_){}
  if(hdri && hdri.id === e.id){ const h = hdri; hdri = null; useSky('physical'); dropHdri(h); }
  else useSky('physical');
  await refreshSkies();
});
// the Ground plane panel
const GROUND_FIELDS ={ gndX: 'x', gndY: 'y', gndZ: 'z', gndTurn: 'turn', gndW: 'width', gndL: 'length' };
const showGround = () => {
  $('#gndShow').checked = !!GROUND.show;
  for(const [id, k] of Object.entries(GROUND_FIELDS)) $('#' + id).value = GROUND[k];
  $('#gndAuto').checked = !GROUND.colour;
  $('#gndColour').disabled = !GROUND.colour;
  if(GROUND.colour) $('#gndColour').value = GROUND.colour;
};
showGround();
const groundChanged = () => {
  saveGround();
  if(!renderer) return;
  if(photo && photo.on) stopPhoto('The ground changed: back to the live view.');
  placeGround();
};
$('#gndShow').addEventListener('change', () => { GROUND.show = $('#gndShow').checked; groundChanged(); });
for(const [id, k] of Object.entries(GROUND_FIELDS)) $('#' + id).addEventListener('input', () => {
  const v = parseFloat($('#' + id).value);
  if(Number.isFinite(v)) { GROUND[k] = v; groundChanged(); }
});
$('#gndAuto').addEventListener('change', () => {
  GROUND.colour = $('#gndAuto').checked ? '' : $('#gndColour').value;
  $('#gndColour').disabled = !GROUND.colour; groundChanged();
});
$('#gndColour').addEventListener('input', () => { GROUND.colour = $('#gndColour').value; groundChanged(); });
$('#btnGndReset').addEventListener('click', () => { Object.assign(GROUND, GROUND_DEFAULT); showGround(); groundChanged(); });
// the Look panel: a change shows at once, in the live view and the photo alike (no new render)
const lookApi = lookPanel(look, () => { if(bloomFrom) setBloom(...bloomFrom); });
// its depth of field: kept in this browser; the photo is traced again with a new setting
// the slider: f/0.005 to f/22 evenly in stops (below f/1, more than a real lens: a whole street
// looks like a model, the miniature look)
const FSTOP = v => 0.005 * Math.pow(22 / 0.005, v / 100), FSTOP_V = n => Math.round(100 * Math.log(n / 0.005) / Math.log(22 / 0.005));
const showDof = () => {
  $('#dofOn').checked = DOF.on;
  $('#dofFstop').value = FSTOP_V(DOF.fstop);
  $('#dofFstopVal').textContent = 'f/' + (DOF.fstop < 0.1 ? DOF.fstop.toFixed(3) : DOF.fstop < 1 ? DOF.fstop.toFixed(2) : DOF.fstop < 10 ? DOF.fstop.toFixed(1) : Math.round(DOF.fstop));
  $('#dofFocus').value = DOF.focus; $('#dofDist').value = +(+DOF.distance).toFixed(1);
  $('#camLens').value = String(DOF.lens || 0);
  $('#dofDist').disabled = DOF.focus !== 'fixed';
  if(DOF.focus === 'fixed') $('#dofFocusVal').textContent = '';
  for(const id of ['dofFstop', 'dofFocus', 'btnDofPick']) $('#' + id).disabled = !DOF.on;
  if(!DOF.on) $('#dofDist').disabled = true;
};
const dofChanged = () => {
  saveDof(); showDof();
  dof.enabled = DOF.on;
  if(photo && photo.on) stopPhoto('Depth of field changed: back to the live view (Render photo traces it again).');
};
showDof();
$('#dofOn').addEventListener('change', () => { DOF.on = $('#dofOn').checked; dofChanged(); if(DOF.on && renderer) indexScene(); });
$('#camLens').addEventListener('change', () => { DOF.lens = +$('#camLens').value; saveDof(); applyLens(); if(photo && photo.on) stopPhoto('The lens changed: back to the live view.'); });
$('#dofFstop').addEventListener('input', () => { const n = FSTOP(+$('#dofFstop').value); DOF.fstop = +n.toFixed(n < 0.1 ? 3 : n < 1 ? 2 : 1); dofChanged(); });
$('#dofFocus').addEventListener('change', () => {
  if($('#dofFocus').value === 'fixed' && focusNow) DOF.distance = +focusNow.toFixed(1);   // from where it is focused now
  DOF.focus = $('#dofFocus').value; dofChanged();
});
$('#dofDist').addEventListener('input', () => { const v = parseFloat($('#dofDist').value); if(v > 0){ DOF.distance = v; saveDof(); if(photo && photo.on) stopPhoto(); } });
// Pick focus: the next click in the view focuses at that spot
let dofPicking = false;
$('#btnDofPick').addEventListener('click', async () => {
  if(!renderer || !world) return;
  await indexScene();
  dofPicking = true;
  host.style.cursor = 'crosshair';
  $('#btnDofPick').textContent = 'Click a spot in the view…';
});
host.addEventListener('click', e => {
  if(!dofPicking) return;
  dofPicking = false; host.style.cursor = ''; $('#btnDofPick').textContent = 'Pick focus in the view';
  const b = renderer.domElement.getBoundingClientRect();
  const d = surfaceAt(((e.clientX - b.left) / b.width) * 2 - 1, -((e.clientY - b.top) / b.height) * 2 + 1);
  if(d === null) return;
  DOF.focus = 'fixed'; DOF.distance = +d.toFixed(2); dofChanged();
}, true);
$('#btnPhoto').addEventListener('click', startPhoto);
$('#btnBake').addEventListener('click', bakeLight);
$('#bakeOn').addEventListener('change', () => { BAKE.on = $('#bakeOn').checked; if(renderer) applyBake(); });
for(const id of ['bakeArea', 'bakeQuality', 'bakeWalls']){
  const el = $('#' + id);
  try{ const v = localStorage.getItem('rta.view3d.' + id); if(v && [...el.options].some(o => o.value === v)) el.value = v; }catch(e){}
  el.addEventListener('change', () => { try{ localStorage.setItem('rta.view3d.' + id, el.value); }catch(e){} });
}
bakeNote();
$('#btnSaveImg').addEventListener('click', saveImage);
$('#photoDenoise').checked = DENOISE;
$('#photoDenoise').addEventListener('change', () => {
  DENOISE = $('#photoDenoise').checked;
  try{ localStorage.setItem('rta.view3d.denoise', DENOISE ? 'on' : 'off'); }catch(e){}
  if(photo && photo.on && DENOISE && photo.clean) photo.clean.next = 0;    // denoise what there is now
});
// a time of day is the physical sky's: picking one leaves an HDRI sky
document.querySelectorAll('[data-time3d]').forEach(b => b.addEventListener('click', () => {
  if(SKY.id !== 'physical'){ SKY.id = 'physical'; saveSky(); hdriLoading = null; showSky(); }
  if(renderer) setTime(b.dataset.time3d); else time = b.dataset.time3d;
}));
$('#btnView3dReset').addEventListener('click', () => { if(renderer) frame(centre, radius); });
window.view3dState = () => renderer ? { children: scene.children.length, width: host.clientWidth, height: host.clientHeight,
  camera: camera.position.toArray().map(v => Math.round(v)), time, lamps: lamps ? lamps.items.length : 0,
  world: !!world, photo: !!(photo && photo.on), samples: photo && photo.on ? photo.pt.samples : 0,
  denoised: photo && photo.on && photo.clean ? { at: photo.clean.at, ms: Math.round(photo.clean.ms || 0), ...photo.clean.info } : null,
  bake: { busy: BAKE.busy, on: BAKE.on, shown: !!BAKE.shown, list: BAKE.list.map(b => ({ time: b.time, area: b.area, w: b.map.width, h: b.map.height,
          mpp: b.mpp, samples: b.samples, seconds: b.seconds })) } } : null;
window.view3dControl = { setTime: n => setTime(n), tune: (n, patch) => { Object.assign(TIMES[n], patch); if(patch.candela) LAMP.candela = patch.candela; setTime(n); }, view: (pos, tgt) => { camera.position.set(...pos); controls.target.set(...tgt); controls.update(); poolAt = null; },
  photo: startPhoto, stop: stopPhoto,
  bake: (area, samples, walls) => {
    if(area) $('#bakeArea').value = area;
    if(walls) $('#bakeWalls').value = walls;
    if(samples){
      const q = $('#bakeQuality');
      if(![...q.options].some(o => o.value === String(samples))) q.add(new Option(`${samples} samples`, String(samples)));
      q.value = String(samples);
    }
    return bakeLight();
  },
  baked: on => { BAKE.on = !!on; applyBake(); },
  wet: (wet, puddles, mirror = true) => { LIGHT.wet = wet; if(puddles != null) LIGHT.puddles = puddles; LIGHT.mirror = mirror; showLight(); applyWet(); }, draw: () => draw(), denoise: () => photo && photo.on ? denoisePhoto(Math.floor(photo.pt.samples)) : null,
  // for comparisons: the path tracer's light tree on or off (off: each light as likely)
  lightTree: on => { if(photo && photo.on){ photo.pt._pathTracer.material.lightTree.enabled = on ? 1 : 0; photo.pt.reset(); dropClean(); } },
  capture: () => { draw(); return renderer.domElement.toDataURL('image/png'); }, look: lookApi,
  dof: async patch => { Object.assign(DOF, patch); dofChanged(); applyLens(); if(DOF.on) await indexScene(); focusAim = null; focusMoved = 0;
                        return { ...DOF, now: DOF.on ? focusDistance(true) : null }; },
  // the sky: an id of the list (or 'physical'), turned and as strong as given; resolves when shown
  sky: async (id, rotation, strength) => {
    await skiesReady;
    if(rotation != null) SKY.rotation = rotation;
    if(strength != null) SKY.strength = strength;
    await useSky(id);
    return { time, id: SKY.id, sun: hdri && time === 'hdri' ? { ...hdriSun(hdri), sharp: hdri.sun.sharp, Y: hdri.sun.Y } : null,
             scale: hdri ? hdri.scale : null, exposure: renderer.toneMappingExposure, skies: skyList.map(e => e.id) };
  } };
