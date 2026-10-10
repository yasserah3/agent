// The Dynamic creation tab: buildings made level by level (app/buildings.py keeps them).
// A building is a stack of levels from the ground up, each a box for now: the ground
// floor first, then the first floor, the second, and so on, and the roof always on top.
// A level is drawn as a rectangle in the viewport, on the ground for the ground floor and
// on the top of the level below for the others; its height, width, length and position
// are set on the right, where every level also has its floor plan. Levels are named by
// where they stand, so deleting or moving one names the others again. Metres, Y up; the
// building's front is towards -Z, the top of the plans: the side an object faces the
// street with. Saved as it changes; Use as object makes it an object layer for the map.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = s => document.querySelector(s);
const host = $('#bdView'), msgEl = $('#bdMsg');
const say = (text, kind) => window.dispatchEvent(new CustomEvent('rta-log', { detail: Object.assign(new String(text), { kind }) }));

// ----------------------------------------------------------------- levels (as app/buildings.py)
const ORDINALS = ['Ground', 'First', 'Second', 'Third', 'Fourth', 'Fifth', 'Sixth', 'Seventh', 'Eighth', 'Ninth',
                  'Tenth', 'Eleventh', 'Twelfth', 'Thirteenth', 'Fourteenth', 'Fifteenth', 'Sixteenth',
                  'Seventeenth', 'Eighteenth', 'Nineteenth', 'Twentieth'];
const levelName = (lv, i) => lv[i].kind === 'roof' ? 'Roof' : i < ORDINALS.length ? `${ORDINALS[i]} floor` : `Floor ${i}`;
const COLOURS = { ground: 0xccb08c, floors: [0xdbd9d1, 0xc2c9d4], roof: 0x737880 };
const colourOf = (lv, i) => lv[i].kind === 'roof' ? COLOURS.roof : i === 0 ? COLOURS.ground : COLOURS.floors[(i - 1) % 2];
const hex = c => '#' + c.toString(16).padStart(6, '0');
const HEIGHT = { ground: 4, floor: 3, roof: 1 };            // a new level's height (m): shops on the ground floor
const SNAP_M = 0.5, EDGE_SNAP_M = 0.6, MIN_M = 1;
const hasRoof = lv => lv.length > 0 && lv[lv.length - 1].kind === 'roof';
const baseOf = (lv, i) => lv.slice(0, i).reduce((s, l) => s + l.h, 0);
const m = v => `${+(Math.round(v * 100) / 100).toFixed(2)}`;
const rectBox = r => ({ x0: r.x - r.w / 2, x1: r.x + r.w / 2, z0: r.z - r.l / 2, z1: r.z + r.l / 2 });
// the parts of a level's rectangle past the one below it: rectangles (x0, x1, z0, z1), and how far it sticks out
function overhang(r, below){
  if(!r || !below) return { parts: [], most: 0 };
  const a = rectBox(r), b = rectBox(below);
  const ix0 = Math.max(a.x0, b.x0), ix1 = Math.min(a.x1, b.x1), iz0 = Math.max(a.z0, b.z0), iz1 = Math.min(a.z1, b.z1);
  const most = Math.max(b.x0 - a.x0, a.x1 - b.x1, b.z0 - a.z0, a.z1 - b.z1, 0);
  if(ix0 >= ix1 || iz0 >= iz1) return { parts: [a], most, all: true };
  const parts = [[a.x0, ix0, a.z0, a.z1], [ix1, a.x1, a.z0, a.z1], [ix0, ix1, a.z0, iz0], [ix0, ix1, iz1, a.z1]]
    .filter(([x0, x1, z0, z1]) => x1 - x0 > 0.005 && z1 - z0 > 0.005).map(([x0, x1, z0, z1]) => ({ x0, x1, z0, z1 }));
  return { parts, most: parts.length ? most : 0 };
}
// the drawn level below (the next one down with a rectangle), if any
function belowOf(lv, i){
  for(let k = i - 1; k >= 0; k--) if(lv[k].rect) return { k, rect: lv[k].rect };
  return null;
}

// ----------------------------------------------------------------- state
const B = { list: [], cur: null, sel: -1, draw: null, undo: [], timer: null, saving: null, visible: false, sent: null };

// ----------------------------------------------------------------- the viewport
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.appendChild(renderer.domElement);
renderer.domElement.style.display = 'block';
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x2b3137);
scene.fog = new THREE.Fog(0x2b3137, 60, 220);                 // the grid fades away, never a shimmer at the horizon
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 6000);
camera.position.set(32, 26, -38);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 3, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;                     // never under the ground
scene.add(new THREE.HemisphereLight(0xe3ebf4, 0x3b352c, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.7);
sun.position.set(45, 70, 30);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0005;
scene.add(sun, sun.target);
const ground = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), new THREE.MeshStandardMaterial({ color: 0x485057, roughness: 1 }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);
const grid = new THREE.Group();
const fine = new THREE.GridHelper(400, 400, 0x58616a, 0x515960), coarse = new THREE.GridHelper(400, 40, 0x7a8590, 0x7a8590);
fine.position.y = 0.01; coarse.position.y = 0.015;
grid.add(fine, coarse);
scene.add(grid);
// the front: an arrow on the ground towards -Z, past the building
const front = (() => {
  const c = document.createElement('canvas'); c.width = 256; c.height = 320;
  const g = c.getContext('2d');
  g.fillStyle = 'rgba(143,182,242,.95)';
  g.beginPath(); g.moveTo(128, 10); g.lineTo(230, 130); g.lineTo(165, 130); g.lineTo(165, 210); g.lineTo(91, 210); g.lineTo(91, 130); g.lineTo(26, 130); g.closePath(); g.fill();
  g.font = 'bold 56px system-ui, sans-serif'; g.textAlign = 'center'; g.fillText('FRONT', 128, 290);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(4, 5), new THREE.MeshBasicMaterial({ map: t, transparent: true, depthWrite: false }));
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.03;
  scene.add(mesh);
  return mesh;
})();
// the drawing plane: a grid at the height being drawn on, and the outline of the level below
let drawGrid = null;
function showDrawGrid(y, around){
  if(drawGrid){ scene.remove(drawGrid); drawGrid.geometry.dispose(); drawGrid.material.dispose(); drawGrid = null; }
  if(y === null) return;
  // half-metre lines over the building and some way round it, not out to the horizon
  const c = around.isEmpty() ? new THREE.Vector3() : around.getCenter(new THREE.Vector3());
  const span = Math.ceil((around.isEmpty() ? 20 : Math.max(around.max.x - around.min.x, around.max.z - around.min.z)) * 1.6 / 10 + 2) * 10;
  drawGrid = new THREE.GridHelper(span, span * 2, 0x3d7be0, 0x2f5fae);
  drawGrid.material.transparent = true; drawGrid.material.opacity = 0.4; drawGrid.material.depthWrite = false;
  drawGrid.position.set(Math.round(c.x), y + 0.02, Math.round(c.z));
  scene.add(drawGrid);
}
const belowLine = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xd9a441 }));
belowLine.visible = false;
scene.add(belowLine);
const levelsGroup = new THREE.Group();
scene.add(levelsGroup);

function resize(){
  const w = Math.max(1, host.clientWidth), h = Math.max(1, host.clientHeight);
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
resize();
function tick(){ controls.update(); renderer.render(scene, camera); }
window.addEventListener('create-tab', e => {
  B.visible = !!e.detail;
  renderer.setAnimationLoop(B.visible ? tick : null);
  if(B.visible){ resize(); if(!B.list.length) loadList(); }
});

// every drawn level as a box, on the levels below it; the selected one tinted blue
function build(){
  levelsGroup.children.forEach(o => { o.geometry.dispose(); o.material.dispose(); });
  levelsGroup.clear();
  const lv = B.cur ? B.cur.levels : [];
  let base = 0;
  const all = new THREE.Box3();
  lv.forEach((l, i) => {
    if(!l.rect){
      // not drawn yet: a ghost of its height on the footprint of the level below, so what stands
      // above it (the roof) does not seem to float
      const fb = belowOf(lv, i), r = fb ? fb.rect : { x: 0, z: 0, w: 6, l: 6 }, sel = i === B.sel;
      const g = new THREE.BoxGeometry(r.w, l.h, r.l);
      const ghost = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: sel ? 0x3d7be0 : 0xd9a441, transparent: true, opacity: 0.1, depthWrite: false }));
      ghost.position.set(r.x, base + l.h / 2, r.z);
      ghost.userData.level = i;
      levelsGroup.add(ghost);
      const ed = new THREE.LineSegments(new THREE.EdgesGeometry(g), new THREE.LineDashedMaterial({ color: sel ? 0x8fb6f2 : 0xd9a441, dashSize: 0.5, gapSize: 0.35 }));
      ed.computeLineDistances(); ed.position.copy(ghost.position);
      levelsGroup.add(ed);
    }
    if(l.rect){
      const r = l.rect, sel = i === B.sel;
      const mat = new THREE.MeshStandardMaterial({ color: colourOf(lv, i), roughness: 0.85,
        emissive: sel ? 0x1d4f9a : 0x000000, emissiveIntensity: sel ? 0.45 : 0 });
      const box = new THREE.Mesh(new THREE.BoxGeometry(r.w, l.h, r.l), mat);
      box.position.set(r.x, base + l.h / 2, r.z);
      box.castShadow = box.receiveShadow = true;
      box.userData.level = i;
      levelsGroup.add(box);
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry),
        new THREE.LineBasicMaterial({ color: sel ? 0x8fb6f2 : 0x22272c }));
      edges.position.copy(box.position);
      levelsGroup.add(edges);
      all.expandByObject(box);
    }
    base += l.h;
  });
  // the front arrow just past the building; the sun's shadows cover it
  const c = all.isEmpty() ? new THREE.Vector3() : all.getCenter(new THREE.Vector3());
  const size = all.isEmpty() ? new THREE.Vector3(10, 3, 10) : all.getSize(new THREE.Vector3());
  front.position.set(c.x, 0.03, (all.isEmpty() ? -5 : all.min.z) - 4);
  const reach = Math.max(size.x, size.y, size.z) * 0.8 + 20;
  Object.assign(sun.shadow.camera, { left: -reach, right: reach, top: reach, bottom: -reach, near: 1, far: reach * 4 + 100 });
  sun.shadow.camera.updateProjectionMatrix();
  sun.target.position.copy(c); sun.position.set(c.x + 45, 70 + size.y, c.z + 30);
  return all;
}
function fit(){
  const all = build();
  const c = all.isEmpty() ? new THREE.Vector3(0, 2, 0) : all.getCenter(new THREE.Vector3());
  const r = all.isEmpty() ? 12 : Math.max(8, all.getSize(new THREE.Vector3()).length() * 0.75);
  controls.target.copy(c);
  camera.position.set(c.x + r * 1.05, c.y + r * 0.85, c.z - r * 1.25);       // its front, and its right side
  camera.near = Math.max(0.05, r / 500); camera.far = r * 60 + 1000; camera.updateProjectionMatrix();
  scene.fog.near = r * 4; scene.fog.far = r * 4 + 160;
}
$('#btnBdFit').addEventListener('click', fit);

// ----------------------------------------------------------------- drawing a level's rectangle
const ray = new THREE.Raycaster();
function ndc(e){
  const r = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
}
// where the pointer meets the plane at height y
function onPlane(e, y){
  camera.updateMatrixWorld();                                    // where it is now, even between frames
  ray.setFromCamera(ndc(e), camera);
  const p = new THREE.Vector3();
  return ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), -y), p) ? p : null;
}
// to the half metre, or to an edge of the level below within reach (to draw the same rectangle)
function snap(v, edges){
  for(const e of edges) if(Math.abs(v - e) <= EDGE_SNAP_M) return e;
  return Math.round(v / SNAP_M) * SNAP_M;
}
function startDraw(i){
  const lv = B.cur.levels;
  B.draw = { i, start: null, before: null };
  const y = baseOf(lv, i), below = belowOf(lv, i);
  showDrawGrid(y, build());
  const bc = below && rectBox(below.rect);
  if(bc){
    belowLine.geometry.dispose();
    belowLine.geometry = new THREE.BufferGeometry().setFromPoints([[bc.x0, bc.z0], [bc.x1, bc.z0], [bc.x1, bc.z1], [bc.x0, bc.z1]]
      .map(([x, z]) => new THREE.Vector3(x, y + 0.05, z)));
  }
  belowLine.visible = !!bc;
  renderAll();
}
function endDraw(){ B.draw = null; showDrawGrid(null); belowLine.visible = false; renderAll(); }
function drawMessage(){
  if(!B.draw) return null;
  const lv = B.cur.levels, i = B.draw.i, name = levelName(lv, i), below = i > 0 ? levelName(lv, i - 1) : null;
  const r = lv[i].rect;
  if(B.draw.start && r) return `${name}: ${m(r.w)} × ${m(r.l)} m. Let go to keep it.`;
  return i === 0 ? `Draw the ${name}: press and drag on the ground. Esc to stop.`
    : `Draw the ${name}: press and drag on the top of the ${below}${belowOf(lv, i) ? ' (its outline in orange; corners snap to it)' : ''}. Esc to stop.`;
}
host.addEventListener('pointerdown', e => {
  if(!B.cur || e.button !== 0 || e.target.closest('.dc-modes, .vp-tools')) return;
  if(B.draw){
    const lv = B.cur.levels, y = baseOf(lv, B.draw.i), p = onPlane(e, y);
    if(!p) return;
    e.stopPropagation(); e.preventDefault();                      // not a turn of the view
    const below = belowOf(lv, B.draw.i), bc = below && rectBox(below.rect);
    B.draw.edges = bc ? { x: [bc.x0, bc.x1, below.rect.x], z: [bc.z0, bc.z1, below.rect.z] } : { x: [], z: [] };
    B.draw.start = [snap(p.x, B.draw.edges.x), snap(p.z, B.draw.edges.z)];
    B.draw.before = lv[B.draw.i].rect ? { ...lv[B.draw.i].rect } : null;
    pushUndo();
    return;
  }
  B.press = { x: e.clientX, y: e.clientY };                       // a click (not a drag) picks a level
}, true);
window.addEventListener('pointermove', e => {
  const d = B.draw;
  if(!d || !d.start) return;
  const lv = B.cur.levels, p = onPlane(e, baseOf(lv, d.i));
  if(!p) return;
  const x = snap(p.x, d.edges.x), z = snap(p.z, d.edges.z);
  const [x0, x1] = [Math.min(d.start[0], x), Math.max(d.start[0], x)], [z0, z1] = [Math.min(d.start[1], z), Math.max(d.start[1], z)];
  lv[d.i].rect = { x: (x0 + x1) / 2, z: (z0 + z1) / 2, w: Math.max(0.01, x1 - x0), l: Math.max(0.01, z1 - z0) };
  build(); renderMsg();
});
window.addEventListener('pointerup', e => {
  const d = B.draw;
  if(d && d.start){
    const lv = B.cur.levels, r = lv[d.i].rect, name = levelName(lv, d.i);
    d.start = null;
    if(!r || r.w < MIN_M || r.l < MIN_M){
      lv[d.i].rect = d.before; B.undo.pop();
      say(`Too small: drag a rectangle at least ${MIN_M} m each way.`, 'bad');
      build(); renderMsg();
      return;
    }
    const first = !d.before;
    endDraw(); changed();
    say(`${name}: ${m(r.w)} × ${m(r.l)} m, ${m(lv[d.i].h)} m high${first && d.i === 0 ? '. Create floor level adds the First floor on top of it' : ''}.`, 'ok');
    if(first && B.cur.levels.filter(l => l.rect).length === 1) fit();
    return;
  }
  if(B.press && B.cur && !B.draw){
    const moved = Math.hypot(e.clientX - B.press.x, e.clientY - B.press.y);
    B.press = null;
    if(moved > 4 || e.target !== renderer.domElement) return;
    ray.setFromCamera(ndc(e), camera);
    const hit = ray.intersectObjects(levelsGroup.children.filter(o => o.isMesh), false)[0];
    if(hit){ B.sel = hit.object.userData.level; renderAll(); }
  }
});
window.addEventListener('keydown', e => {
  if(!B.visible) return;
  if(e.key === 'Escape' && B.draw){
    if(B.draw.start){ B.cur.levels[B.draw.i].rect = B.draw.before; B.undo.pop(); }
    endDraw(); return;
  }
  if((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)){
    e.preventDefault(); undo();
  }
});

// ----------------------------------------------------------------- changes, undo, saving
function pushUndo(){
  if(!B.cur) return;
  B.undo.push(JSON.stringify({ levels: B.cur.levels, sel: B.sel }));
  if(B.undo.length > 60) B.undo.shift();
}
function undo(){
  if(!B.cur || !B.undo.length) return;
  if(B.draw) endDraw();
  const u = JSON.parse(B.undo.pop());
  B.cur.levels = u.levels; B.sel = Math.min(u.sel, u.levels.length - 1);
  say('Undone.');
  changed();
}
$('#btnBdUndo').addEventListener('click', undo);
function changed(){
  renderAll();
  $('#bdSaved').textContent = 'saving…';
  clearTimeout(B.timer);
  B.timer = setTimeout(save, 400);
}
async function save(){
  clearTimeout(B.timer); B.timer = null;
  const b = B.cur;
  if(!b) return;
  const body = { id: b.id, name: b.name, preset: b.preset, levels: b.levels, object: b.object };
  const job = (B.saving || Promise.resolve()).then(async () => {
    try{
      const r = await fetch('/api/buildings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const res = await r.json();
      if(!r.ok) throw new Error(res.detail || r.statusText);
      if(B.cur === b && !b.id) b.id = res.id;
      if(B.cur === b) $('#bdSaved').textContent = 'saved';
      await loadList(true);
    }catch(err){ say('Building not saved: ' + err.message, 'bad'); $('#bdSaved').textContent = 'not saved'; }
  });
  B.saving = job;
  await job;
}

// ----------------------------------------------------------------- buildings: list, new, open, delete
async function loadList(quiet){
  try{
    const r = await fetch('/api/buildings');
    B.list = (await r.json()).buildings || [];
  }catch(_){ B.list = []; }
  renderList();
  if(!quiet && !B.cur && B.list.length) open(B.list[0].id);
}
async function open(id){
  if(B.timer) await save();
  try{
    const r = await fetch('/api/buildings/' + id);
    if(!r.ok) throw new Error((await r.json()).detail || r.statusText);
    B.cur = await r.json();
  }catch(err){ say('Building not opened: ' + err.message, 'bad'); return; }
  B.sel = B.cur.levels.length ? B.cur.levels.length - 1 : -1; B.undo = []; B.sent = null;
  if(B.draw) endDraw();
  $('#bdSaved').textContent = 'saved';
  renderAll(); fit();
}
function renderList(){
  const box = $('#bdList'); box.innerHTML = '';
  B.list.forEach(b => {
    const it = document.createElement('button');
    it.type = 'button'; it.className = 'bd-item';
    it.setAttribute('aria-pressed', B.cur && B.cur.id === b.id ? 'true' : 'false');
    it.innerHTML = `<span></span><small>${b.levels} level${b.levels === 1 ? '' : 's'}${b.object ? ' · object' : ''}</small>`;
    it.firstChild.textContent = b.name;
    it.addEventListener('click', () => { if(!B.cur || B.cur.id !== b.id) open(b.id); });
    box.appendChild(it);
  });
}
$('#btnBdNew').addEventListener('click', async () => {
  if(B.timer) await save();
  if(B.draw) endDraw();
  B.cur = null; B.sel = -1; B.undo = []; B.sent = null;
  $('#bdSaved').textContent = '';
  renderAll(); fit();
  say('New building: pick a preset on the left to start it.');
});
$('#btnBdDelete').addEventListener('click', async () => {
  const b = B.cur; if(!b) return;
  if(!confirm(`Delete the building "${b.name}"?` + (b.object ? ' The object made from it stays in the Objects panel.' : ''))) return;
  clearTimeout(B.timer); B.timer = null;
  if(b.id) try{ await fetch('/api/buildings/' + b.id, { method: 'DELETE' }); }catch(_){}
  if(B.draw) endDraw();
  B.cur = null; B.sel = -1; B.undo = [];
  say(`Deleted the building ${b.name}.`);
  $('#bdSaved').textContent = '';
  await loadList(true); renderAll(); fit();
});
$('#bdName').addEventListener('input', () => {
  if(!B.cur) return;
  B.cur.name = $('#bdName').value.trim() || 'Building';
  changed();
});

// ----------------------------------------------------------------- presets
const PRESETS = [{ id: 'block', name: 'Simple block', title: 'Every level a box: draw its rectangle, set its height' }];
const presetIcon = '<svg viewBox="0 0 54 44" aria-hidden="true"><path d="M8 38 L8 26 L46 26 L46 38 Z" fill="#ccb08c"/><path d="M12 26 L12 15 L42 15 L42 26 Z" fill="#dbd9d1"/>'
  + '<path d="M12 15 L12 7 L42 7 L42 15 Z" fill="#c2c9d4"/><path d="M10 7 L10 4 L44 4 L44 7 Z" fill="#737880"/></svg>';
function renderPresets(){
  const box = $('#bdPresets'); box.innerHTML = '';
  PRESETS.forEach(p => {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'bd-preset'; b.title = p.title;
    b.setAttribute('aria-pressed', B.cur && B.cur.preset === p.id ? 'true' : 'false');
    b.innerHTML = presetIcon + `<span>${p.name}</span>`;
    b.addEventListener('click', () => usePreset(p));
    box.appendChild(b);
  });
  const later = document.createElement('button');
  later.type = 'button'; later.className = 'bd-preset'; later.disabled = true; later.title = 'Presets you import later';
  later.innerHTML = '<svg viewBox="0 0 54 44" aria-hidden="true"><path d="M27 12 V32 M17 22 H37" stroke="#6C747C" stroke-width="3" fill="none"/></svg><span>Imported (later)</span>';
  box.appendChild(later);
}
function usePreset(p){
  if(!B.cur){
    const n = B.list.length + 1;
    B.cur = { name: `Building ${n}`, preset: p.id, levels: [] };
    B.sel = -1; B.undo = [];
    say(`${B.cur.name}: ${p.name}. Press Create floor level: the Ground floor comes first, then draw its rectangle.`, 'ok');
    changed();
    return;
  }
  if(B.cur.preset !== p.id){ B.cur.preset = p.id; changed(); }
}

// ----------------------------------------------------------------- levels: create, delete, move, roof
$('#btnLvAdd').addEventListener('click', () => {
  if(!B.cur) return;
  if(B.draw) endDraw();
  pushUndo();
  const lv = B.cur.levels, at = hasRoof(lv) ? lv.length - 1 : lv.length;
  lv.splice(at, 0, { kind: 'floor', h: at === 0 ? HEIGHT.ground : HEIGHT.floor, rect: null });
  B.sel = at;
  say(`${levelName(lv, at)} created${hasRoof(lv) ? ', under the roof' : ''}: draw its rectangle in the viewport.`);
  changed(); startDraw(at);
});
$('#btnRoof').addEventListener('click', () => {
  const lv = B.cur && B.cur.levels;
  if(!lv || hasRoof(lv) || !lv.length) return;
  if(B.draw) endDraw();
  pushUndo();
  lv.push({ kind: 'roof', h: HEIGHT.roof, rect: null });
  B.sel = lv.length - 1;
  say('Roof created: always the top level. Draw its rectangle on the top floor.');
  changed(); startDraw(B.sel);
});
$('#btnLvDelete').addEventListener('click', () => {
  const lv = B.cur && B.cur.levels, i = B.sel;
  if(!lv || i < 0 || i >= lv.length) return;
  if(B.draw) endDraw();
  pushUndo();
  const name = levelName(lv, i);
  lv.splice(i, 1);
  B.sel = lv.length ? Math.min(i, lv.length - 1) : -1;
  const moved = lv.slice(i).filter(l => l.kind !== 'roof').length;
  say(`Deleted the ${name}.` + (moved ? ` The ${moved} level${moved > 1 ? 's' : ''} above moved down and ${moved > 1 ? 'were' : 'was'} named again.` : ''));
  changed();
});
// up (+1) or down (-1) the building: the levels swap places, names and heights follow
function moveLevel(i, d){
  const lv = B.cur.levels, j = i + d;
  if(j < 0 || j >= lv.length || lv[i].kind === 'roof' || lv[j].kind === 'roof') return;
  if(B.draw) endDraw();
  pushUndo();
  const was = levelName(lv, i), other = levelName(lv, j);
  [lv[i], lv[j]] = [lv[j], lv[i]];
  B.sel = j;
  say(`The ${was} moved ${d > 0 ? 'up' : 'down'}: it is now the ${levelName(lv, j)}, and the ${other} is now the ${levelName(lv, i)}.`);
  changed();
}

// ----------------------------------------------------------------- the selected level's options
const FIELDS = [['#lvH', 'h'], ['#lvW', 'w'], ['#lvL', 'l'], ['#lvX', 'x'], ['#lvZ', 'z']];
FIELDS.forEach(([id, key]) => {
  const el = $(id);
  el.addEventListener('focus', () => pushUndo());
  el.addEventListener('input', () => {
    const l = B.cur && B.cur.levels[B.sel]; if(!l) return;
    const v = parseFloat(el.value); if(!Number.isFinite(v)) return;
    if(key === 'h') l.h = Math.min(100, Math.max(0.1, v));
    else if(l.rect) l.rect[key] = key === 'w' || key === 'l' ? Math.min(2000, Math.max(0.5, v)) : Math.min(5000, Math.max(-5000, v));
    changed();
  });
});
$('#btnLvDraw').addEventListener('click', () => { if(B.cur && B.sel >= 0) startDraw(B.sel); });
$('#btnLvSame').addEventListener('click', () => {
  const lv = B.cur && B.cur.levels, below = lv && belowOf(lv, B.sel);
  if(!below) return;
  pushUndo();
  lv[B.sel].rect = { ...below.rect };
  if(B.draw) endDraw();
  say(`${levelName(lv, B.sel)}: the same rectangle as the ${levelName(lv, below.k)}, ${m(below.rect.w)} × ${m(below.rect.l)} m.`, 'ok');
  changed();
});
$('#btnBdObject').addEventListener('click', async () => {
  const b = B.cur; if(!b) return;
  await save();
  if(!b.id) return;
  $('#btnBdObject').disabled = true;
  try{
    const r = await fetch(`/api/buildings/${b.id}/object`, { method: 'POST' });
    const res = await r.json();
    if(!r.ok) throw new Error(res.detail || r.statusText);
    b.object = res.id; B.sent = JSON.stringify(b.levels);
    say(res.updated ? `${res.name}: its object updated, ${m(res.width_m)} × ${m(res.depth_m)} × ${m(res.height_m)} m; its placements follow.`
      : `${res.name} is now an object in the Generate tab (Objects): ${m(res.width_m)} × ${m(res.depth_m)} × ${m(res.height_m)} m. `
        + 'Place it, put it in a package, or press Place automatically on islands.', 'ok');
    window.dispatchEvent(new Event('objects-changed'));
    await loadList(true);
  }catch(err){ say('Not made an object: ' + err.message, 'bad'); }
  renderAll();
});

// ----------------------------------------------------------------- drawing the panels
function renderMsg(){
  const t = !B.cur ? 'Pick a preset on the left to start a building.'
    : B.draw ? drawMessage()
    : !B.cur.levels.length ? 'Press Create floor level: the Ground floor comes first.' : null;
  msgEl.hidden = !t; if(t) msgEl.textContent = t;
}
function renderLevels(){
  const lv = B.cur ? B.cur.levels : [], box = $('#bdLevels');
  box.innerHTML = '';
  $('#bdLevelCount').textContent = lv.length ? `${lv.length}, ${m(baseOf(lv, lv.length))} m high` : '';
  for(let i = lv.length - 1; i >= 0; i--){
    const l = lv[i], below = belowOf(lv, i), oh = overhang(l.rect, below && below.rect);
    const row = document.createElement('div');
    row.className = 'bd-lv'; row.setAttribute('role', 'option'); row.setAttribute('aria-selected', i === B.sel ? 'true' : 'false');
    row.dataset.level = i;
    const sub = !l.rect ? '<small class="warn">not drawn yet</small>'
      : `<small${oh.most > 0 ? ' class="warn"' : ''}>${m(l.rect.w)} × ${m(l.rect.l)} m · ${m(l.h)} m high${oh.most > 0 ? ' · sticks out' : ''}</small>`;
    const roof = l.kind === 'roof';
    const up = !roof && i + 1 < lv.length && lv[i + 1].kind !== 'roof', down = !roof && i > 0;
    row.innerHTML = `<span class="sw" style="background:${hex(colourOf(lv, i))}"></span><span><b></b>${sub}</span>`
      + `<button type="button" class="up" title="Move up the building" aria-label="Move ${levelName(lv, i)} up" ${up ? '' : 'disabled'}>↑</button>`
      + `<button type="button" class="down" title="Move down the building" aria-label="Move ${levelName(lv, i)} down" ${down ? '' : 'disabled'}>↓</button>`;
    row.querySelector('b').textContent = levelName(lv, i);
    row.addEventListener('click', ev => {
      if(ev.target.closest('button')) return;
      B.sel = i; renderAll();
    });
    row.querySelector('.up').addEventListener('click', () => moveLevel(i, +1));
    row.querySelector('.down').addEventListener('click', () => moveLevel(i, -1));
    box.appendChild(row);
  }
  const has = !!B.cur;
  $('#btnLvAdd').disabled = !has;
  $('#btnLvDelete').disabled = !has || B.sel < 0;
  $('#btnLvDelete').textContent = B.sel >= 0 && lv[B.sel] && lv[B.sel].kind === 'roof' ? 'Delete roof' : 'Delete floor level';
  $('#btnRoof').disabled = !has || !lv.length || hasRoof(lv);
  $('#btnRoof').title = !lv.length ? 'Create a floor level first' : hasRoof(lv) ? 'This building has its roof (always the top level)' : 'The roof: always the top level';
  $('#btnBdUndo').disabled = !B.undo.length;
}
function renderLevelBox(){
  const lv = B.cur ? B.cur.levels : [], l = lv[B.sel];
  FIELDS.forEach(([id, key]) => {
    const el = $(id);
    el.disabled = !l || (key !== 'h' && !l.rect);
    if(document.activeElement !== el) el.value = !l ? '' : key === 'h' ? m(l.h) : l.rect ? m(l.rect[key]) : '';
  });
  $('#btnLvDraw').disabled = !l;
  $('#btnLvDraw').textContent = !l || !l.rect ? 'Draw rectangle' : 'Draw again';
  const below = l && belowOf(lv, B.sel);
  $('#btnLvSame').disabled = !below;
  if(!l){ $('#bdLvTitle').textContent = 'No level selected'; $('#bdLvRange').textContent = ''; $('#bdLvNote').textContent = ''; return; }
  const y0 = baseOf(lv, B.sel);
  $('#bdLvTitle').textContent = levelName(lv, B.sel);
  $('#bdLvRange').textContent = `${m(y0)} to ${m(y0 + l.h)} m`;
  const oh = overhang(l.rect, below && below.rect);
  $('#bdLvNote').textContent = !l.rect ? 'Not drawn yet: press Draw rectangle, then press and drag in the viewport.'
    + (below ? ` Same as below copies the ${levelName(lv, below.k)}.` : '')
    : oh.all ? `It stands wholly off the ${levelName(lv, below.k)}: nothing holds it up (orange on its plan).`
    : oh.most > 0 ? `It sticks out past the ${levelName(lv, below.k)} by up to ${m(oh.most)} m (orange on its plan).`
    : B.sel > 0 && !below ? 'The levels below it are not drawn yet.' : '';
}
function renderPlans(){
  const lv = B.cur ? B.cur.levels : [], box = $('#bdPlans');
  box.innerHTML = '';
  const drawn = lv.map(l => l.rect).filter(Boolean);
  if(!drawn.length){ box.innerHTML = '<div class="note">Each level\'s plan shows here once it is drawn.</div>'; return; }
  // one scale for every plan, so sizes compare; the top of each plan is the front
  const bs = drawn.map(rectBox);
  let x0 = Math.min(...bs.map(b => b.x0)), x1 = Math.max(...bs.map(b => b.x1)), z0 = Math.min(...bs.map(b => b.z0)), z1 = Math.max(...bs.map(b => b.z1));
  const pad = Math.max(x1 - x0, z1 - z0) * 0.08 + 0.5;
  x0 -= pad; x1 += pad; z0 -= pad; z1 += pad;
  const aspect = Math.min(1.2, Math.max(0.35, (z1 - z0) / (x1 - x0)));
  for(let i = lv.length - 1; i >= 0; i--){
    const l = lv[i], below = belowOf(lv, i), oh = overhang(l.rect, below && below.rect);
    const card = document.createElement('div');
    card.className = 'bd-plan'; card.setAttribute('aria-selected', i === B.sel ? 'true' : 'false');
    const R = (b, attrs) => `<rect x="${b.x0}" y="${b.z0}" width="${b.x1 - b.x0}" height="${b.z1 - b.z0}" vector-effect="non-scaling-stroke" ${attrs}/>`;
    let svg = `<svg viewBox="${x0} ${z0} ${x1 - x0} ${z1 - z0}" preserveAspectRatio="xMidYMid meet" style="aspect-ratio:${1 / aspect}">`;
    if(below) svg += R(rectBox(below.rect), 'fill="none" stroke="#98A0A8" stroke-width="1.2" stroke-dasharray="4 3"');
    if(l.rect){
      svg += R(rectBox(l.rect), `fill="${hex(colourOf(lv, i))}" fill-opacity=".85" stroke="${i === B.sel ? '#8FB6F2' : '#E4E7EA'}" stroke-width="${i === B.sel ? 2 : 1.2}"`);
      oh.parts.forEach(p => { svg += R(p, 'fill="#D9A441" fill-opacity=".75" stroke="none"'); });
    }
    svg += '</svg>';
    const head = l.rect ? `${m(l.rect.w)} × ${m(l.rect.l)} m · ${m(l.h)} m high` : 'not drawn yet';
    card.innerHTML = `<div class="h"><b></b><small>${head}</small></div>${svg}`
      + (oh.most > 0 ? `<div class="w">Sticks out ${m(oh.most)} m past the ${levelName(lv, below.k)}${oh.all ? ', wholly off it' : ''}.</div>` : '');
    card.querySelector('b').textContent = levelName(lv, i);
    card.addEventListener('click', () => { B.sel = i; renderAll(); });
    box.appendChild(card);
  }
}
function renderAll(){
  const has = !!B.cur;
  $('#bdName').disabled = !has;
  if(document.activeElement !== $('#bdName')) $('#bdName').value = has ? B.cur.name : '';
  $('#btnBdDelete').disabled = !has;
  const drawnAny = has && B.cur.levels.some(l => l.rect);
  $('#btnBdObject').disabled = !drawnAny;
  $('#btnBdObject').textContent = has && B.cur.object ? 'Update its object' : 'Use as object';
  $('#bdObjNote').textContent = !has ? 'Its front, towards the top of the plans, is the object\'s front (+Y): the side that faces the street.'
    : B.cur.object ? (B.sent && B.sent !== JSON.stringify(B.cur.levels) ? 'Changed since it was made an object: press Update its object; its placements follow.'
                       : 'An object in the Generate tab (Objects). Update its object after changes; its placements follow.')
    : 'Its front, towards the top of the plans, is the object\'s front (+Y): the side that faces the street.';
  renderPresets(); renderList(); renderLevels(); renderLevelBox(); renderPlans(); renderMsg();
  build();
}
renderAll();
loadList();
// for the console and tests: the state, Fit, and where a point (metres) shows on the page
window.buildingEditor = { state: B, fit, toScreen: (x, y, z) => {
  camera.updateMatrixWorld();
  const v = new THREE.Vector3(x, y, z).project(camera), r = renderer.domElement.getBoundingClientRect();
  return [r.left + (v.x + 1) / 2 * r.width, r.top + (1 - v.y) / 2 * r.height];
} };
