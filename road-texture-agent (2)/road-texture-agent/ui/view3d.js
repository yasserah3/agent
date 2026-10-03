// The 3D tab's viewport: a scene to look around, in metres with Y up, as in the
// GLB export (the map's x is +X, its y +Z). three.js is bundled with the program
// in ui/vendor/three, so it works offline. Generate 3D scene is still to come:
// for now the scene holds the ground grid and the axes.
import * as THREE from './vendor/three/three.module.min.js';
import { OrbitControls } from './vendor/three/OrbitControls.js';

const host = document.getElementById('view3d'), msg = document.getElementById('view3dMsg');
let renderer = null, scene, camera, controls, visible = false, running = false;

function init(){
  try{
    renderer = new THREE.WebGLRenderer({ antialias: true });
  }catch(e){
    renderer = null;
    msg.textContent = 'This browser cannot show 3D here: WebGL is switched off or not available.';
    return false;
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.style.display = 'block';
  host.appendChild(renderer.domElement);
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1b1e24);
  scene.fog = new THREE.Fog(0x1b1e24, 600, 1600);
  camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
  scene.add(new THREE.GridHelper(500, 50, 0x6a7280, 0x353b45));     // 10 m squares, 500 m across
  scene.add(new THREE.AxesHelper(25));                               // X red, Y (up) green, Z blue
  scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x3a3f35, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(80, 120, 50);
  scene.add(sun);
  // left drag: orbit, right drag (or Shift + drag): pan, wheel: zoom
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495;                          // never below the ground
  controls.maxDistance = 2500;
  reset();
  new ResizeObserver(resize).observe(host);
  return true;
}
function reset(){
  camera.position.set(150, 120, 150);
  controls.target.set(0, 0, 0);
  controls.update();
}
function resize(){
  const w = host.clientWidth, h = host.clientHeight;
  if(!renderer || !w || !h) return;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
function loop(){
  if(!visible){ running = false; return; }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}
// drawn only while the 3D tab is showing
window.addEventListener('view3d', e => {
  visible = !!e.detail;
  if(!visible || (!renderer && !init())) return;
  resize();
  if(!running){ running = true; requestAnimationFrame(loop); }
});
document.getElementById('btnView3dReset').addEventListener('click', () => { if(renderer) reset(); });
window.view3dState = () => renderer ? { children: scene.children.length, width: host.clientWidth, height: host.clientHeight,
  camera: camera.position.toArray().map(v => Math.round(v)) } : null;
