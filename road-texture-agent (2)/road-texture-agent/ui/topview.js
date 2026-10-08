// The Generate tab's Top view: the 3D model (streets, sidewalks, kerbs, blocks and
// islands, markings and objects, with their materials) drawn from straight above,
// as a flat map picture covering exactly the generated texture's ground. Drawn in
// the browser (three.js), so it takes a moment once the model is built. Ground keeps its
// variety: an island slot's Mix in patches and the colour's drift, read from the model's
// ground pattern as the 3D tab, the photo and the islands texture read it.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// A ground material's pattern, drift and mix (its extras), its textures loaded; null if none.
async function groundLook(parser, m){
  const x = m.userData || {};
  if(x.pattern == null) return null;
  const tex = async (i, srgb) => { const t = await parser.getDependency('texture', i); t.colorSpace = srgb ? THREE.SRGBColorSpace : THREE.NoColorSpace; return t; };
  const look = { pattern: await tex(x.pattern, false), drift: +x.drift || 0, mix: null };
  look.pattern.wrapS = look.pattern.wrapT = THREE.RepeatWrapping;
  const mx = x.mix, r = x.relief;
  if(mx && mx.map != null && mx.orh != null && mx.thresh != null && mx.scale > 0 && r && r.index != null){
    const off = Array.isArray(mx.offset) ? mx.offset : [0, 0];
    look.mix = { map: await tex(mx.map, true), orh: await tex(mx.orh, false), orhA: await tex(r.index, false),
                 thresh: +mx.thresh, scale: +mx.scale, offset: new THREE.Vector2(+off[0] || 0, +off[1] || 0),
                 depthA: +r.depth_m || 0, depthB: +mx.depth_m || 0 };
  }
  return look;
}
// the flat colour with the look: the second material where the pattern's red is above the
// slot's threshold (the higher stones of either at the edges), then the drift
function groundPatch(mat, look){
  mat.onBeforeCompile = sh => {
    let defs = '';
    sh.uniforms.groundPattern = { value: look.pattern };
    if(look.mix){
      defs += '#define USE_MIX\n';
      const g = look.mix;
      Object.assign(sh.uniforms, { mixMap: { value: g.map }, mixOrh: { value: g.orh }, pomMap: { value: g.orhA },
        mixThresh: { value: g.thresh }, mixScale: { value: g.scale }, mixOffset: { value: g.offset },
        pomDepth: { value: g.depthA }, mixDepth: { value: g.depthB } });
    }
    if(look.drift){ defs += '#define USE_DRIFT\n'; sh.uniforms.driftAmt = { value: look.drift }; }
    sh.vertexShader = 'varying vec2 vGroundW;\n' + sh.vertexShader.replace('#include <project_vertex>',
      '#include <project_vertex>\nvGroundW = ( modelMatrix * vec4( transformed, 1.0 ) ).xz;');
    sh.fragmentShader = defs + `varying vec2 vGroundW;
uniform sampler2D groundPattern;
#ifdef USE_MIX
uniform sampler2D mixMap, mixOrh, pomMap;
uniform float mixThresh, mixScale, pomDepth, mixDepth;
uniform vec2 mixOffset;
#endif
#ifdef USE_DRIFT
uniform float driftAmt;
#endif
` + sh.fragmentShader.replace('#include <map_fragment>', `
#ifdef USE_MAP
  vec4 sampledDiffuseColor = texture2D( map, vMapUv );
#ifdef USE_MIX
  float mixN = textureLod( groundPattern, vGroundW * mixScale + mixOffset, 0.0 ).r;
  if( mixN >= mixThresh - 0.12 ){
    vec2 uvB = vMapUv + vec2( 0.37, 0.61 );
    float hA = texture2D( pomMap, vMapUv ).b, hB = texture2D( mixOrh, uvB ).b;
    float s = ( mixN - mixThresh ) / 0.035 + 2.5 * ( hB * mixDepth - hA * pomDepth ) / max( max( pomDepth, mixDepth ), 1e-4 );
    sampledDiffuseColor = mix( sampledDiffuseColor, texture2D( mixMap, uvB ), smoothstep( 0.0, 1.0, clamp( 0.5 + s, 0.0, 1.0 ) ) );
  }
#endif
  diffuseColor *= sampledDiffuseColor;
#endif
#ifdef USE_DRIFT
  float gDriftA = texture2D( groundPattern, vGroundW / 54.4 + vec2( 0.11, 0.23 ) ).g - 0.5;
  float gDriftB = texture2D( groundPattern, vGroundW / 169.6 + vec2( 0.71, 0.47 ) ).b - 0.5;
  diffuseColor.rgb *= ( 1.0 + driftAmt * ( 2.0 * gDriftA + 1.6 * gDriftB ) ) * vec3( 1.0 + driftAmt * gDriftB, 1.0, 1.0 - driftAmt * gDriftB );
#endif
`);
  };
  mat.customProgramCacheKey = () => 'top-ground' + (look.mix ? '-mix' : '') + (look.drift ? '-drift' : '');
}

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
    const looks = new Map(), meshes = [], extra = new Set();
    gltf.scene.traverse(o => { if(o.isMesh) meshes.push(o); });
    for(const o of meshes) for(const m of [].concat(o.material))
      if(!looks.has(m)) looks.set(m, await groundLook(gltf.parser, m).catch(() => null));
    for(const o of meshes){
      o.material = [].concat(o.material).map(m => {
        if(m.map) m.map.anisotropy = aniso;
        const b = new THREE.MeshBasicMaterial({ map: m.map || null, color: m.color ? m.color.clone() : 0xffffff,
          vertexColors: !!o.geometry.attributes.color, transparent: m.transparent, opacity: m.opacity,
          alphaTest: m.alphaTest, side: THREE.DoubleSide });
        const look = looks.get(m);
        if(look){
          groundPatch(b, look);
          for(const t of [look.pattern, ...(look.mix ? [look.mix.map, look.mix.orh, look.mix.orhA] : [])]) extra.add(t);
          if(look.mix) look.mix.map.anisotropy = aniso;
        }
        return b;
      });
      if(o.material.length === 1) o.material = o.material[0];
    }
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
    for(const t of extra) t.dispose();
    return out;
  }finally{
    renderer.dispose();
    renderer.forceContextLoss();
  }
}
window.renderTopView = renderTopView;
