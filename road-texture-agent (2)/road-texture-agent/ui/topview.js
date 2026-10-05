// The Generate tab's Top view: the 3D model (streets, sidewalks, kerbs, blocks and
// islands, markings and objects, with their materials) drawn from straight above,
// as a flat map picture covering exactly the generated texture's ground. Drawn in
// the browser (three.js), so it takes a moment once the model is built.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// url: the model (GLB); size_m: [width, depth] of the ground it covers; px: the
// longest side of the picture. Returns a canvas of that picture.
async function renderTopView(url, size_m, px = 4096){
  const canvas = document.createElement('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  try{
    const most = Math.min(px, renderer.capabilities.maxTextureSize, 8192);
    const [wm, hm] = size_m;
    const w = wm >= hm ? most : Math.max(1, Math.round(most * wm / hm));
    const h = wm >= hm ? Math.max(1, Math.round(most * hm / wm)) : most;
    renderer.setPixelRatio(1);
    renderer.setSize(w, h, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.setClearColor(0x121214, 1);
    const gltf = await new GLTFLoader().loadAsync(url);
    const scene = new THREE.Scene();
    const aniso = renderer.capabilities.getMaxAnisotropy();
    // flat colour, as a map: each surface's own colour picture, tone and weathering, no lighting
    gltf.scene.traverse(o => {
      if(!o.isMesh) return;
      o.material = [].concat(o.material).map(m => {
        if(m.map) m.map.anisotropy = aniso;
        return new THREE.MeshBasicMaterial({ map: m.map || null, color: m.color ? m.color.clone() : 0xffffff,
          vertexColors: !!o.geometry.attributes.color, transparent: m.transparent, opacity: m.opacity,
          alphaTest: m.alphaTest, side: THREE.DoubleSide });
      });
      if(o.material.length === 1) o.material = o.material[0];
    });
    scene.add(gltf.scene);
    // looking straight down, the top of the picture north (-z), the ground exactly in frame
    const cam = new THREE.OrthographicCamera(-wm / 2, wm / 2, hm / 2, -hm / 2, 0.1, 5000);
    cam.position.set(0, 2000, 0);
    cam.up.set(0, 0, -1);
    cam.lookAt(0, 0, 0);
    renderer.render(scene, cam);
    const out = document.createElement('canvas');
    out.width = w; out.height = h;
    out.getContext('2d').drawImage(canvas, 0, 0);
    gltf.scene.traverse(o => { if(o.geometry) o.geometry.dispose(); [].concat(o.material || []).forEach(m => { if(m.map) m.map.dispose(); m.dispose(); }); });
    return out;
  }finally{
    renderer.dispose();
    renderer.forceContextLoss();
  }
}
window.renderTopView = renderTopView;
