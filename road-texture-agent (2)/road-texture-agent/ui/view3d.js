// The 3D tab: the whole place in 3D, to look around. Generate 3D scene builds the
// streets, inner streets, sidewalks, kerbs, blocks and objects from the last
// generated texture (the same model as the GLB export, with the 3D model
// settings of the Generate tab) and adds street lamps along the sidewalks.
// The live view has a sky, the sun with its shadows, soft contact shadows
// (ambient occlusion), filmic colour and a glow around bright lights; Render
// photo then traces the light properly (three-gpu-pathtracer), a still that
// sharpens for as long as the camera stays put. Dawn, Day and Night set the
// sun, the sky and the street lamps. Metres, Y up: the map's x is +X, its y +Z,
// the top of the map north. Everything is bundled in ui/vendor, so it works offline.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { Sky } from 'three/addons/objects/Sky.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const $ = s => document.querySelector(s);
const host = $('#view3d'), msg = $('#view3dMsg');

// ----------------------------------------------------------------- times of day
// sun (or moon) height and compass direction in degrees (0 north, 90 east), its
// colour and strength, the sky, exposure, how much the sky lights the scene,
// the street lamps (0 off, 1 full), the glow, and the haze
const TIMES = {
  dawn:  { elev: 4, azim: 100, color: 0xffb47a, sun: 2.6, sky: true, turbidity: 7, rayleigh: 2.6, mie: 0.006, mieG: 0.9,
           clouds: 0.35, exposure: 0.8, env: 0.45, lamps: 0.6, bloom: [0.45, 0.5, 2.5], fog: [0x8a7f7c, 0.00035],
           ground: 0x4a4d3c },
  day:   { elev: 52, azim: 215, color: 0xfff3e2, sun: 3.2, sky: true, turbidity: 2.2, rayleigh: 1.0, mie: 0.004, mieG: 0.8,
           clouds: 0.3, exposure: 0.8, env: 0.4, lamps: 0, bloom: [0.1, 0.4, 6.0], fog: [0xc9d6e0, 0.00035],
           ground: 0x5c6648 },
  night: { elev: 38, azim: 300, color: 0x9fb6ff, sun: 0.12, sky: false, exposure: 0.9, env: 0.15, lamps: 1,
           bloom: [0.6, 0.5, 1.5], fog: [0x070b16, 0.0005], ground: 0x2a2e26 },
};
const LAMP = { height: 8.0, arm: 1.6, candela: 320, pool: 12 };

let renderer = null, scene, camera, controls, composer, renderPass, gtao, bloom, output;
let sky, skyEnv, envScene, pmrem, envRT = null, nightTex = null, stars = null, ground;
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

  // the sky, and the same sky without the sun's disc for the light it gives
  sky = new Sky();
  sky.scale.setScalar(6000);
  scene.add(sky);
  envScene = new THREE.Scene();
  skyEnv = new Sky();
  skyEnv.scale.setScalar(6000);
  skyEnv.material.uniforms.showSunDisc.value = 0;
  envScene.add(skyEnv);
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
  ground.position.y = -0.05;
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
  frame(new THREE.Vector3(), 250);
  new ResizeObserver(resize).observe(host);
  return true;
}

function resize(){
  const w = host.clientWidth, h = host.clientHeight;
  if(!renderer || !w || !h) return;
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

function horizon(){
  // the colour of the sky just above the horizon, all round (as the sky lights the
  // scene: no sun disc). The middle of eight directions, so the glow round a low
  // sun does not tint the whole haze
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
  const t = TIMES[name];
  const d = sunDir(t);
  for(const s of [sky, skyEnv]){
    const u = s.material.uniforms;
    u.sunPosition.value.copy(d);
    if(t.sky){
      u.turbidity.value = t.turbidity; u.rayleigh.value = t.rayleigh;
      u.mieCoefficient.value = t.mie; u.mieDirectionalG.value = t.mieG;
      u.cloudCoverage.value = t.clouds;
    }
  }
  sky.visible = t.sky;
  // what lights the scene from all round: the sky (without the sun's disc), or the night sky
  envScene.background = t.sky ? null : nightSky();
  skyEnv.visible = t.sky;
  if(envRT) envRT.dispose();
  envRT = pmrem.fromScene(envScene, 0, 1, 10000);
  scene.environment = envRT.texture;
  scene.environmentIntensity = t.env;
  scene.background = t.sky ? null : nightSky();
  scene.backgroundIntensity = 1;
  // the haze takes the sky's own colour at the horizon, so the land fades into it
  scene.fog = new THREE.FogExp2(horizon() || new THREE.Color(t.fog[0]), t.fog[1]);
  ground.material.color.set(t.ground);
  const st = starField();
  if(t.sky) scene.remove(st); else scene.add(st);
  sun.color.set(t.color);
  sun.intensity = t.sun;
  hemi.intensity = t.sky ? 0 : 0.06;
  renderer.toneMappingExposure = t.exposure;
  bloom.strength = t.bloom[0]; bloom.radius = t.bloom[1]; bloom.threshold = t.bloom[2];
  if(lamps) lampLevel(t.lamps);
  document.querySelectorAll('[data-time3d]').forEach(b => b.setAttribute('aria-pressed', b.dataset.time3d === name ? 'true' : 'false'));
  if(render) poolAt = null;
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
    const s = new THREE.SpotLight(0xffcf96, 0, 60, 1.15, 0.85, 2);
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
  lamps.level = level;
  lamps.headMat.emissiveIntensity = 30 * level;
  lamps.glows.material.opacity = 0.3 * level;
  lamps.glows.visible = level > 0;
  for(const s of lamps.pool){ s.visible = level > 0; s.intensity = LAMP.candela * level; }
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
      if(/Road_fill|Road_interchange|Block_paving/.test(m.name)){
        m.polygonOffset = true; m.polygonOffsetFactor = 1; m.polygonOffsetUnits = 2;
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
    }
    const t = TIMES[time];
    // the sky as an environment the tracer can sample, without the sun's disc (the sun is its own light)
    const cube = new THREE.WebGLCubeRenderTarget(512, { type: THREE.HalfFloatType });
    new THREE.CubeCamera(1, 10000, cube).update(renderer, envScene);
    photo.cube = cube;
    photo.saved = { env: scene.environment, bg: scene.background, envI: scene.environmentIntensity };
    scene.environment = cube.texture; scene.background = cube.texture;
    sky.visible = false; hemi.visible = false;
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
        // the lamps that light what is in view; the far ones show only their heads, so
        // the picture sharpens sooner
        const near = lamps.items.map(it => [it.foot.distanceTo(tg), it]).filter(a => a[0] < 120).sort((a, b) => a[0] - b[0]).slice(0, 24);
        for(const [, it] of near){
          const s = new THREE.SpotLight(0xffcf96, LAMP.candela * lamps.level, 0, 1.15, 0.85, 2);
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
  scene.environment = photo.saved.env; scene.background = photo.saved.bg; scene.environmentIntensity = photo.saved.envI;
  sky.visible = TIMES[time].sky; hemi.visible = true;
  if(stars) stars.visible = true;
  if(lamps) lamps.group.visible = true;
  if(photo.extra){ scene.remove(photo.extra); photo.extra.traverse(o => { if(o.geometry) o.geometry.dispose(); }); photo.extra = null; }
  if(photo.cube){ photo.cube.dispose(); photo.cube = null; }
  $('#btnPhoto').textContent = 'Render photo';
  if(why) note(why);
}

function saveImage(){
  if(!renderer) return;
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
    const n = Math.floor(photo.pt.samples);
    if(n && n % 4 === 0) $('#photoNote').textContent = `${n} samples, ${((performance.now() - photo.started) / 1000).toFixed(0)} s. Sharper with every sample; move the camera to go back to the live view.`;
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
$('#btnPhoto').addEventListener('click', startPhoto);
$('#btnSaveImg').addEventListener('click', saveImage);
document.querySelectorAll('[data-time3d]').forEach(b => b.addEventListener('click', () => { if(renderer) setTime(b.dataset.time3d); else time = b.dataset.time3d; }));
$('#btnView3dReset').addEventListener('click', () => { if(renderer) frame(centre, radius); });
window.view3dState = () => renderer ? { children: scene.children.length, width: host.clientWidth, height: host.clientHeight,
  camera: camera.position.toArray().map(v => Math.round(v)), time, lamps: lamps ? lamps.items.length : 0,
  world: !!world, photo: !!(photo && photo.on), samples: photo && photo.on ? photo.pt.samples : 0 } : null;
window.view3dControl = { setTime: n => setTime(n), tune: (n, patch) => { Object.assign(TIMES[n], patch); if(patch.candela) LAMP.candela = patch.candela; setTime(n); }, view: (pos, tgt) => { camera.position.set(...pos); controls.target.set(...tgt); controls.update(); poolAt = null; },
  photo: startPhoto, stop: stopPhoto, draw: () => draw(),
  capture: () => { draw(); return renderer.domElement.toDataURL('image/png'); } };
