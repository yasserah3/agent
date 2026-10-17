const UI_VERSION = '2026.10.17-coords15';   // must match VERSION in server.py
(function(){
  const $ = (s,r=document)=>r.querySelector(s);
  const $$ = (s,r=document)=>[...r.querySelectorAll(s)];
  const API = '';                       // served by the Python server, same origin

  const S = {
    tab:'train',
    prime:{ material:null, line:null, noisy:null },
    noiseArt:null, primeResults:null, trainResults:null,
    pairs:[], pairUid:0, selPair:null,
    gen:{ mask:null, junctions:null, result:null, layer:'result', alignLines:false }, review:null,
    junctions:null,
    memory:null,
    online:false
  };

  /* ------------------------------------------------ console + status */
  function log(msg, kind){
    const li = document.createElement('li');
    li.textContent = msg; if(kind) li.className = kind;
    const ul = $('#log'); ul.appendChild(li); ul.scrollTop = ul.scrollHeight;
  }
  const status = t => $('#status').textContent = t;

  /* A long job's line in the console: what it is doing, the time so far and the time
     left, asked from the server about once a second while the job runs. */
  function fmtTime(s){
    s = Math.max(0, Math.round(s));
    if(s < 60) return `${s} s`;
    const m = Math.floor(s / 60);
    if(m < 60) return `${m} min ${String(s % 60).padStart(2, '0')} s`;
    return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, '0')} min`;
  }
  function trackJob(name){
    const id = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    const t0 = performance.now();
    const li = document.createElement('li'); li.className = 'hi job';
    const ul = $('#log'); ul.appendChild(li); ul.scrollTop = ul.scrollHeight;
    let snap = null, ended = false, busy = false, timer = null;
    const draw = () => {
      if(ended) return;
      const secs = (performance.now() - t0) / 1000;
      let s = `${name}: `;
      if(snap && !snap.unknown){
        s += snap.stage + (snap.done != null ? ` · ${Math.floor(snap.done * 100)}%` : '') + ` · ${fmtTime(secs)} so far`;
        if(snap.left != null) s += ` · about ${fmtTime(snap.left)} left`;
        else if(snap.done != null && !snap.finished) s += ' · working out the time left';
      } else s += `${fmtTime(secs)} so far`;
      li.textContent = s;
    };
    const ask = async () => {
      if(ended || busy) return;
      busy = true;
      try{
        const r = await fetch(API + '/api/progress/' + id);
        const j = r.ok ? await r.json() : null;
        if(j && timer) snap = j;                           // not once the page has taken over
        if(j && j.detail && window.onJobDetail) window.onJobDetail(j.detail, j);
      }catch(_){}
      busy = false; draw();
    };
    draw();
    timer = setInterval(ask, 1000);
    const tick = setInterval(draw, 1000);                  // the time so far moves between answers
    return {
      id,
      // the page takes over the line (its own work, after the server's part): no more asking
      set(stage, done = null, left = null){
        if(timer){ clearInterval(timer); timer = null; }
        snap = { stage, done, left }; draw();
      },
      done(ok, extra){
        if(ended) return; ended = true; clearInterval(timer); clearInterval(tick);
        const secs = (performance.now() - t0) / 1000;
        li.className = ok ? 'ok' : 'bad';
        li.textContent = `${name}: ${ok ? 'done' : 'stopped'} in ${fmtTime(secs)}` + (extra ? ` · ${extra}` : '');
      }
    };
  }
  window.trackJob = trackJob;
  // a line from the 3D tab (ui/view3d.js) for the console
  window.addEventListener('rta-log', e => log(String(e.detail), e.detail && e.detail.kind));

  async function api(path, opts){
    let job = null;
    if(opts && opts.job){
      job = trackJob(opts.job);
      let payload = {};
      try { payload = opts.body ? JSON.parse(opts.body) : {}; } catch(_) {}
      payload.progress = job.id;
      opts = Object.assign({}, opts, { body: JSON.stringify(payload) });
      delete opts.job;
    }
    try{
      const res = await fetch(API + path, opts);
      let body = null;
      try { body = await res.json(); } catch(e) { body = null; }
      if(!res.ok){
        const msg = (body && (body.error || body.detail)) || `${res.status} ${res.statusText}`;
        const err = new Error(msg); err.status = res.status; err.body = body; throw err;
      }
      if(job) job.done(true);
      return body;
    }catch(e){
      if(job) job.done(false);
      throw e;
    }
  }

  async function boot(){
    try{
      const h = await api('/api/health');
      S.online = true; S.memory = h.memory;
      $('.engine').innerHTML = '<i style="background:var(--green)"></i>Server connected';
      log('Connected to the local server.', 'ok');
      log(`Memory: ${h.memory.steps} steps, ${h.memory.images} images, ${h.memory.situations} situations.`);
      if(h.workspace) log(`Workspace: ${h.workspace}`);
      if(h.version !== UI_VERSION)
        log(`Version mismatch: this page is ${UI_VERSION} but the server is ${h.version || 'an older version'}. `
          + 'Copy the app, ui and server.py folders from the same zip, restart the server, and press Ctrl+F5.', 'bad');
      else log(`Version ${UI_VERSION}.`);
      if(h.fresh && !h.memory.trained_pairs)
        log('This workspace is new and empty. If you expected your earlier training, stop the server and copy your old workspace folder here, or put its path in workspace.txt next to server.py.', 'bad');
      renderMemory();
      try{ S.corrections = (await api('/api/corrections')).corrections; }catch(_){}
      loadMaterials();
      loadDecals(); loadMarks();
    }catch(e){
      $('.engine').innerHTML = '<i style="background:var(--red)"></i>Server not running';
      log('No server. Start it with: python server.py', 'bad');
    }
  }

  $('#uiVersion').textContent = 'v' + UI_VERSION;

  // the materials of streets, sidewalks and kerbs: your trained tiles, or one of the
  // scanned library's (app/scans). Each part has a chip; clicking it opens a choice of
  // balls with their names. The choice is kept in this browser for the next time.
  // squares: the blocks and islands between the roads, paved as the sidewalks unless chosen
  const MAT = { lib: null, parts: { street: '#matStreet', sidewalk: '#matSidewalk', kerb: '#matKerb', square: '#matSquare' },
                titles: { street: 'Street surface', sidewalk: 'Sidewalk surface', kerb: 'Kerb surface', square: 'Squares, blocks and islands' },
                open: null };
  const KIND_NAME = { asphalt: 'Asphalt', paving: 'Paving', concrete: 'Concrete', ground: 'Desert and ground' };
  // the island material slots (Blocks and islands): each a material and the islands picked
  // for it, remembered by a point inside each (mask pixels), kept with the mask on the server.
  // data: the numbered islands of the last texture; pick: the slot being picked for, or -1
  const ISLS = { slots: [], data: null, pick: -1, show: false, changed: () => {} };
  // the Lanes tool (lanes below): the streets of the last texture, the choices kept for the mask
  const LN = { TOL: 0.4, LANE: 3.0, HIGHWAY: 22.0, ISLAND: 1.0, EDGE: 0.3 };
  const LANES = { picks: [], streets: null, mpp: 1, on: false, sel: -1, mask: null };
  // the markings area (see markings area below)
  const MAREA = { area: null, on: false, mask: null, painted: 'null', busy: false, again: false, timer: null, drag: null };
  // project areas: rectangles where the project goes (flat road in the street material, no sidewalks, no markings)
  const PAREA = { picks: [], sel: new Set(), data: null, on: false, mask: null, scale: null, built: '[]', busy: false };
  const AUTO = { pick: -1 };          // automatic placement: the placement picking its islands (-1: none)
  try{
    const kept = JSON.parse(localStorage.getItem('rta.materials') || 'null');
    if(kept){
      for(const [part, id] of Object.entries(MAT.parts)) if(kept[part]) $(id).value = kept[part];
      if(typeof kept.tone === 'boolean') $('#matTone').checked = kept.tone;
    }
  }catch(_){}
  const keepMaterials = () => { try{ localStorage.setItem('rta.materials', JSON.stringify({
    street: $('#matStreet').value, sidewalk: $('#matSidewalk').value, kerb: $('#matKerb').value, square: $('#matSquare').value,
    tone: $('#matTone').checked })); }catch(_){} };
  $('#matTone').addEventListener('change', () => { keepMaterials(); window.dispatchEvent(new Event('materials')); });
  const yourBall = part => MAT.lib && MAT.lib.yours && MAT.lib.yours[part] ? API + `/api/materials/tiles/${part}/ball?v=${MAT.lib.yours[part]}` : null;
  function materialOf(part){
    const v = $(MAT.parts[part]).value;
    if(part === 'square' && (v === 'same' || v === 'tiles')){
      const sw = materialOf('sidewalk');
      return { id: 'same', name: 'Same as sidewalks', ball: sw.ball, same: sw };
    }
    const m = MAT.lib && MAT.lib.materials.find(x => x.id === v);
    return m ? { id: m.id, name: m.name, ball: API + `/api/materials/${m.id}/ball`, m } : { id: 'tiles', name: 'Your tiles', ball: yourBall(part) };
  }
  function slotMaterial(i){
    const v = (ISLS.slots[i] || {}).material;
    const m = MAT.lib && MAT.lib.materials.find(x => x.id === v);
    if(m) return { id: m.id, name: m.name, ball: API + `/api/materials/${m.id}/ball`, m };
    const sq = materialOf('square');
    return { id: 'same', name: 'As the other islands', ball: sq.ball, same: sq };
  }
  window.slotMaterial = slotMaterial;
  function showChips(){
    for(const chip of $$('.matChips .matChip')){
      const cur = materialOf(chip.dataset.part);
      $('b', chip).textContent = cur.name;
      const img = $('img', chip);
      if(cur.ball){ img.src = cur.ball; img.style.visibility = ''; } else { img.removeAttribute('src'); }
      chip.title = cur.m ? `${cur.m.title} by ${(cur.m.authors || []).join(', ')}, Poly Haven, ${cur.m.licence}; ${cur.m.size_m[0]} m across. Click to change`
                 : cur.same ? `Paved as the sidewalks (${cur.same.name}). Click to give the squares, blocks and islands their own, such as cobblestone fans`
                 : 'Your material tiles, built from your training. Click to change';
    }
  }
  async function loadMaterials(){
    try{ MAT.lib = await api('/api/materials'); }catch(_){ return; }
    // a kept choice that is no longer in the library goes back to your tiles
    for(const [part, id] of Object.entries(MAT.parts))
      if(!['tiles', 'same'].includes($(id).value) && !MAT.lib.materials.some(m => m.id === $(id).value && (MAT.lib.parts[part] || []).includes(m.kind)))
        $(id).value = part === 'square' ? 'same' : 'tiles';
    showChips();
    ISLS.changed(true);
  }
  function closeMatPop(){ $('#matPop').hidden = true; MAT.open = null; }
  // a slot's mix: the second ground material in patches through its islands ('auto': the
  // library's partner for its material, 'none'), how much of it and how big its patches
  const MIX_DEFAULT = { material: 'auto', amount: 0.3, size_m: 10 };
  const slotMix = i => Object.assign({}, MIX_DEFAULT, (ISLS.slots[i] || {}).mix || {});
  function mixPartner(i){
    const mix = slotMix(i), cur = slotMaterial(i);
    if(!cur.m || cur.m.kind !== 'ground') return null;
    const id = mix.material === 'auto' ? cur.m.mix_with : mix.material;
    const m = id && id !== 'none' && MAT.lib ? MAT.lib.materials.find(x => x.id === id) : null;
    return { mix, m, auto: mix.material === 'auto' };
  }
  function openMatPop(chip){
    if(chip.dataset.mix != null){ openMixPop(chip, +chip.dataset.slot); return; }
    const part = chip.dataset.part, slot = chip.dataset.slot != null ? +chip.dataset.slot : null;
    const key = slot != null ? 'island:' + slot : part;
    if(MAT.open === key){ closeMatPop(); return; }
    MAT.open = key;
    const pop = $('#matPop'), grid = $('#matPopGrid');
    const cur = slot != null ? (ISLS.slots[slot] || {}).material : $(MAT.parts[part]).value;
    $('#matPopTitle').textContent = slot != null ? `Island material, slot ${slot + 1}` : MAT.titles[part];
    grid.innerHTML = '';
    const card = (value, name, sub, ball, title) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'mp-card'; b.setAttribute('aria-pressed', value === cur); b.title = title || name;
      b.innerHTML = (ball ? `<img loading="lazy" alt="">` : '<div class="mp-none">no tiles yet</div>') + '<span></span><small></small>';
      if(ball) $('img', b).src = ball;
      $('span', b).textContent = name; $('small', b).textContent = sub;
      b.addEventListener('click', () => {
        if(slot != null){
          if(ISLS.slots[slot]) ISLS.slots[slot].material = value;
          closeMatPop(); log(`Island material slot ${slot + 1}: ${name}.`); ISLS.changed();
          return;
        }
        $(MAT.parts[part]).value = value; keepMaterials(); showChips(); closeMatPop();
        log(`${MAT.titles[part]}: ${name}.`);
        window.dispatchEvent(new Event('materials'));
      });
      grid.appendChild(b);
    };
    const group = text => { const g = document.createElement('div'); g.className = 'mp-group'; g.textContent = text; grid.appendChild(g); };
    group('Yours');
    if(slot != null){
      const sq = materialOf('square');
      card('same', 'As the other islands', sq.name, sq.ball, 'This slot\'s islands laid as the islands in no slot (the Squares, blocks and islands material)');
    }else if(part === 'square'){
      const sw = materialOf('sidewalk');
      card('same', 'Same as sidewalks', sw.name, sw.ball, 'The blocks and islands paved as the sidewalks, their paving running on from them');
    }else card('tiles', 'Your tiles', 'from your training', yourBall(part),
         yourBall(part) ? 'Your material tiles, built from your training' : 'No material tiles yet: train, or build them in the Memory tab');
    for(const kind of (MAT.lib ? MAT.lib.parts[slot != null ? 'island' : part] || [] : [])){
      const list = MAT.lib.materials.filter(m => m.kind === kind);
      if(!list.length) continue;
      group(`Scanned ${(KIND_NAME[kind] || kind).toLowerCase()}`);
      list.sort((a, b) => (b.pattern === 'fan') - (a.pattern === 'fan'));      // cobblestone fans first
      for(const m of list) card(m.id, m.name, `${m.size_m[0]} m across`, API + `/api/materials/${m.id}/ball`,
                                `${m.title} by ${(m.authors || []).join(', ')}, Poly Haven, ${m.licence}`);
    }
    if(!MAT.lib) group('The scanned library is not available: is the server running?');
    pop.hidden = false;
    // beside the chip, inside the window
    const r = chip.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
    let left = r.right + 10, top = r.top - 8;
    if(left + pw > innerWidth - 12) left = Math.max(12, r.left);
    if(left + pw > innerWidth - 12) left = Math.max(12, innerWidth - pw - 12);
    if(left < r.right && left + pw > r.left) top = r.bottom + 8;          // no room beside it: below
    top = Math.max(12, Math.min(top, innerHeight - ph - 12));
    pop.style.left = left + 'px'; pop.style.top = top + 'px';
  }
  // the picker for a slot's mix partner: the library's suggestion, none, or any ground material
  function openMixPop(chip, slot){
    const key = 'mix:' + slot;
    if(MAT.open === key){ closeMatPop(); return; }
    MAT.open = key;
    const pop = $('#matPop'), grid = $('#matPopGrid'), mix = slotMix(slot), own = slotMaterial(slot);
    $('#matPopTitle').textContent = `Mixed into slot ${slot + 1} (${own.name}), in patches`;
    grid.innerHTML = '';
    const card = (value, name, sub, ball, title) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'mp-card'; b.setAttribute('aria-pressed', value === mix.material); b.title = title || name;
      b.innerHTML = (ball ? `<img loading="lazy" alt="">` : '<div class="mp-none">none</div>') + '<span></span><small></small>';
      if(ball) $('img', b).src = ball;
      $('span', b).textContent = name; $('small', b).textContent = sub;
      b.addEventListener('click', () => {
        if(ISLS.slots[slot]) ISLS.slots[slot].mix = Object.assign(slotMix(slot), { material: value });
        closeMatPop(); log(`Slot ${slot + 1}: mixed with ${name}.`); ISLS.changed();
      });
      grid.appendChild(b);
    };
    const group = text => { const g = document.createElement('div'); g.className = 'mp-group'; g.textContent = text; grid.appendChild(g); };
    const sug = own.m && MAT.lib ? MAT.lib.materials.find(x => x.id === own.m.mix_with) : null;
    group('Mix');
    card('auto', sug ? `Suggested: ${sug.name}` : 'Suggested', 'a coarser ground of a like colour', sug ? API + `/api/materials/${sug.id}/ball` : null,
         'The library\'s partner for this material: a coarser (or finer) ground of about its colour');
    card('none', 'None', 'this material alone', null, 'No patches: this material alone');
    group('Scanned desert and ground');
    for(const m of (MAT.lib ? MAT.lib.materials : []).filter(m => m.kind === 'ground' && (!own.m || m.id !== own.m.id)))
      card(m.id, m.name, `${m.size_m[0]} m across`, API + `/api/materials/${m.id}/ball`, `${m.title} by ${(m.authors || []).join(', ')}, Poly Haven, ${m.licence}`);
    pop.hidden = false;
    const r = chip.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
    let left = r.right + 10, top = r.top - 8;
    if(left + pw > innerWidth - 12) left = Math.max(12, Math.min(r.left, innerWidth - pw - 12));
    if(left < r.right && left + pw > r.left) top = r.bottom + 8;
    top = Math.max(12, Math.min(top, innerHeight - ph - 12));
    pop.style.left = left + 'px'; pop.style.top = top + 'px';
  }
  $$('.matChip').forEach(chip => chip.addEventListener('click', e => { e.stopPropagation(); openMatPop(chip); }));
  window.openMatPop = openMatPop;
  $('#matPop .mp-x').addEventListener('click', closeMatPop);
  addEventListener('keydown', e => { if(e.key === 'Escape' && MAT.open) closeMatPop(); });
  addEventListener('pointerdown', e => { if(MAT.open && !e.target.closest('#matPop') && !e.target.closest('.matChip')) closeMatPop(); });
  addEventListener('resize', () => { if(MAT.open) closeMatPop(); });
  window.materialChoice = () => ({ street: $('#matStreet').value, sidewalk: $('#matSidewalk').value, kerb: $('#matKerb').value,
                                   square: $('#matSquare').value,
                                   match_tone: $('#matTone').checked, names: { street: materialOf('street').name,
                                   sidewalk: materialOf('sidewalk').name, kerb: materialOf('kerb').name, square: materialOf('square').name } });

  /* ------------------------------------------------ tabs */
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    S.tab = t.dataset.tab;
    $$('.tab').forEach(x => x.setAttribute('aria-selected', x === t));
    $('#pane-train').hidden = S.tab !== 'train';
    $('#pane-generate').hidden = S.tab !== 'generate';
    $('#pane-memory').hidden = S.tab !== 'memory';
    $('#pane-view3d').hidden = S.tab !== 'view3d';
    $('#pane-create').hidden = S.tab !== 'create';
    window.dispatchEvent(new CustomEvent('view3d', { detail: S.tab === 'view3d' }));   // ui/view3d.js draws only while showing
    window.dispatchEvent(new CustomEvent('create-tab', { detail: S.tab === 'create' })); // and ui/building.js
    if(S.tab === 'memory') renderPairs2();
    $('#clickHint').hidden = true;
    renderInfo();
    if(S.tab === 'train' || S.tab === 'generate') requestAnimationFrame(() => (S.tab === 'train' ? trainVp : genVp).fit());
  }));

  /* ------------------------------------------------ files */
  const picker = $('#picker');
  let onPick = null;
  function pick(cb){ onPick = cb; picker.value = ''; picker.click(); }
  picker.addEventListener('change', () => { const f = picker.files[0]; if(f && onPick) onPick(f); });

  function loadImage(url){
    return new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res(img);
      img.onerror = () => rej(new Error('could not display image'));
      img.src = url;
    });
  }

  async function upload(file, role, pair){
    const fd = new FormData();
    fd.append('file', file); fd.append('role', role); fd.append('pair', pair || '');
    const rec = await api('/api/upload', { method:'POST', body:fd });
    rec.img = await loadImage(API + rec.url);
    return rec;
  }

  function drawInto(el, src){
    el.innerHTML = '';
    const c = document.createElement('canvas');
    const k = Math.min(1, 360 / Math.max(src.width, src.height));
    c.width = Math.max(1, Math.round(src.width*k)); c.height = Math.max(1, Math.round(src.height*k));
    c.getContext('2d').drawImage(src, 0, 0, c.width, c.height);
    el.appendChild(c);
  }
  const fmt = n => (n ?? 0).toLocaleString('en-US');

  /* ------------------------------------------------ viewport */
  function makeViewport(root){
    const stage = $('.stage', root), canvas = $('canvas', stage), zoomEl = $('.vp-zoom', root);
    const enlargedEl = $('.vp-enlarged', root);
    let s = 1, tx = 0, ty = 0, drag = null, moved = false, has = false;
    let zoomWas = 1;
    const apply = () => {
      stage.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
      if(s !== zoomWas){ zoomWas = s; root.dispatchEvent(new Event('vpzoom')); }   // for handles kept at screen size
      root.dispatchEvent(new Event('vpview'));                                      // for what is drawn at screen size
      // smooth scaling when zoomed in; hard pixel squares only at extreme zoom, for inspecting pixels
      stage.classList.toggle('pixelated', s > 8);
      const pct = Math.round(s*100);
      zoomEl.textContent = pct + '%';
      // past 100% the image is only being enlarged; say so, since it reads as poor quality
      const over = has && pct > 100;
      zoomEl.classList.toggle('enlarged', over);
      zoomEl.title = over ? 'Zoomed past real size: enlarging existing pixels' : 'Current zoom';
      if(enlargedEl) enlargedEl.hidden = !over;
    };
    function fit(){
      if(!has) return;
      const r = root.getBoundingClientRect();
      if(!r.width || !r.height) return;
      s = Math.min(r.width/canvas.width, r.height/canvas.height) * 0.9;
      tx = (r.width - canvas.width*s)/2; ty = (r.height - canvas.height*s)/2; apply();
    }
    function one(){
      const r = root.getBoundingClientRect();
      s = 1; tx = (r.width - canvas.width)/2; ty = (r.height - canvas.height)/2; apply();
    }
    function show(src, label, keep){
      const same = canvas.width === src.width && canvas.height === src.height;
      canvas.width = src.width; canvas.height = src.height;
      canvas.getContext('2d').drawImage(src, 0, 0);
      has = true;
      $('.vp-empty', root).hidden = true;
      const l = $('.vp-label', root); l.hidden = false; l.textContent = label;
      if(!(keep && same)) fit();
    }
    root.addEventListener('wheel', e => {
      if(!has) return;
      e.preventDefault();
      const r = root.getBoundingClientRect();
      const mx = e.clientX - r.left, my = e.clientY - r.top;
      const ns = Math.min(24, Math.max(0.03, s * (e.deltaY < 0 ? 1.12 : 1/1.12)));
      tx = mx - (mx - tx)*(ns/s); ty = my - (my - ty)*(ns/s); s = ns; apply();
    }, { passive:false });
    root.addEventListener('pointerdown', e => {
      if(!has || e.target.closest('.vp-tools')) return;
      drag = { x:e.clientX, y:e.clientY, tx, ty }; moved = false;
    });
    addEventListener('pointermove', e => {
      if(!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if(Math.abs(dx)+Math.abs(dy) > 4){ moved = true; root.classList.add('dragging'); }
      if(moved){ tx = drag.tx + dx; ty = drag.ty + dy; apply(); }
    });
    addEventListener('pointerup', () => { drag = null; root.classList.remove('dragging'); });
    function zoomBy(k){
      if(!has) return;
      const r = root.getBoundingClientRect();
      const mx = r.width / 2, my = r.height / 2;
      const ns = Math.min(24, Math.max(0.03, s * k));
      tx = mx - (mx - tx) * (ns / s); ty = my - (my - ty) * (ns / s); s = ns; apply();
    }
    $('[data-in]', root).addEventListener('click', () => zoomBy(1.25));
    $('[data-out]', root).addEventListener('click', () => zoomBy(1 / 1.25));
    $('[data-fit]', root).addEventListener('click', fit);
    $('[data-one]', root).addEventListener('click', one);
    addEventListener('resize', fit);
    return { canvas, show, fit, wasDrag: () => moved, view: () => ({ s, tx, ty }) };
  }
  const trainVp = makeViewport($('#trainVp'));
  const genVp = makeViewport($('#genVp'));

  /* ------------------------------------------------ prime */
  const LABEL = { material:'Material alone', line:'Line alone', noisy:'Material with noise' };
  const selectRow = row => { $$('.imgrow, .pslot').forEach(r => r.classList.remove('sel')); if(row) row.classList.add('sel'); };

  $$('#primeList .imgrow').forEach(row => {
    const key = row.dataset.prime;
    $('.load', row).addEventListener('click', e => { e.stopPropagation(); pick(f => setPrime(key, f)); });
    row.addEventListener('click', () => {
      const rec = S.prime[key];
      if(!rec) return pick(f => setPrime(key, f));
      selectRow(row); trainVp.show(rec.img, `${LABEL[key]}, ${rec.width} × ${rec.height}`);
    });
  });

  async function setPrime(key, file){
    status('Uploading…');
    try{
      const rec = await upload(file, key);
      S.prime[key] = rec;
      const row = $(`#primeList [data-prime="${key}"]`);
      drawInto($('.thumb', row), rec.img);
      const sub = $('.imgrow-s', row);
      sub.textContent = `${rec.name}, ${rec.width} × ${rec.height}`; sub.className = 'imgrow-s';
      selectRow(row);
      trainVp.show(rec.img, `${LABEL[key]}, ${rec.width} × ${rec.height}`);
      log(`Loaded ${LABEL[key].toLowerCase()}: ${rec.name} (${rec.width} × ${rec.height}). Recorded as step ${rec.step}.`);
      status('Ready.');
      await refreshNoise(); updatePrime(); renderMemory();
    }catch(e){ log('Upload failed: ' + e.message, 'bad'); status('Upload failed.'); }
  }

  const swRow = $('#swRow');
  const loadPaving = () => pick(async f => {
    status('Uploading…');
    try{
      const rec = await upload(f, 'sidewalk');
      drawInto($('.thumb', swRow), rec.img);
      $('.imgrow-s', swRow).textContent = `${rec.name}, ${rec.width} × ${rec.height}`;
      log(`Loaded sidewalk paving: ${rec.name}. Rebuilding the tiles to use it…`);
      const res = await api('/api/tiles', { method:'POST', headers:{'Content-Type':'application/json'}, body: '{}', job:'Rebuilding the tiles' });
      const sw = (res.tiles.sidewalk || [])[0] || {};
      log(sw.method === 'pattern'
        ? `  Repeating pattern found (every ${sw.period_px.join(' × ')} px): tile cut to ${sw.repeats.join(' × ')} whole repeats, so the joints stay aligned.`
        : `  No repeating pattern found: tile built from mixed patches, like the asphalt.`, 'ok');
      renderTiles && renderTiles();
    }catch(e){ log(e.message, 'bad'); }
    status('Ready.');
  });
  $('.load', swRow).addEventListener('click', e => { e.stopPropagation(); loadPaving(); });
  swRow.addEventListener('click', loadPaving);

  async function refreshNoise(){
    const a = S.prime.material, b = S.prime.noisy;
    const prev = $('#noisePrev'), note = $('#noiseNote');
    S.noiseArt = null; note.style.color = '';
    if(!a || !b){ prev.innerHTML = '<div class="ph">Material with noise minus material alone</div>';
      note.textContent = 'Needs both material images.'; return; }
    try{
      const res = await api('/api/noise', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ clean:a.id, noisy:b.id }) });
      S.noiseArt = res;
      const img = await loadImage(API + res.url);
      drawInto(prev, img); prev.dataset.url = API + res.url;
      note.textContent = `Average difference ${res.mean}, covering ${(res.covered_fraction*100).toFixed(1)}% of the surface. ${res.note}.`;
      log(`Isolated noise: average difference ${res.mean}.`);
    }catch(e){
      prev.innerHTML = '<div class="ph">Could not isolate noise</div>';
      note.textContent = e.message; note.style.color = 'var(--red)';
      const sub = $('#primeList [data-prime="noisy"] .imgrow-s');
      if(/sizes differ/i.test(e.message)){ sub.textContent = 'Size must match material alone'; sub.className = 'imgrow-s bad'; }
      log('Noise isolation failed: ' + e.message, 'bad');
    }
  }
  $('#noisePrev').addEventListener('click', async () => {
    if(!S.noiseArt) return;
    const img = await loadImage(API + S.noiseArt.url);
    selectRow(null); trainVp.show(img, `Isolated noise, ${img.width} × ${img.height}`);
  });

  function updatePrime(){
    const have = Object.values(S.prime).filter(Boolean).length;
    const ready = have === 3 && !!S.noiseArt;
    $('#primeCount').textContent = `${have} of 3`;
    $('#btnPrime').disabled = !ready;
    $('#primeNote').textContent = have < 3 ? 'Load all three images.'
      : !S.noiseArt ? 'Fix the problem with the two material images first.' : 'Ready to prime.';
    updateTrain(); renderInfo();
  }

  $('#btnPrime').addEventListener('click', async () => {
    $('#btnPrime').disabled = true; status('Priming…');
    log('Running the three methods on the three reference images…');
    try{
      const res = await api('/api/prime', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Priming',
        body: JSON.stringify({ material:S.prime.material.id, line:S.prime.line.id, noisy:S.prime.noisy.id }) });
      S.primeResults = res;
      const f = res.fingerprints, m = f.material.neighbours, d = f.line.dash, n = f.noise;
      log(`Material: ${m.reading}, tone ${m.mean}, local contrast ${m.local_contrast}. ${f.material.periodicity.reading}.`, 'ok');
      log(`Line: ${d.found ? d.reading : 'no dash pattern found'}, width ${d.width_px} px, paint tone ${f.line.paint.mean}.`, 'ok');
      log(`Noise: ${n.reading}, average mark thickness ${n.thickness_px} px. ${n.periodicity.reading}.`, 'ok');
      log(`Patch libraries stored: material ${f.material.patches.count}, line ${f.line.patches.count}, noise ${f.noise.patches.count} patches.`);
      log('Three trees created and saved to memory.', 'ok');
      renderPrimeResults();
      $('#primeNote').textContent = 'Priming finished. The three trees now exist in memory.';
    }catch(e){
      log(e.message, 'bad');
      if(e.body && e.body.detail) log(e.body.detail);
      $('#primeNote').textContent = e.message;
    }
    status('Ready.'); $('#btnPrime').disabled = false; renderMemory();
  });

  function renderPrimeResults(){
    const res = S.primeResults;
    if(!res) return;
    const f = res.fingerprints, m = f.material.neighbours, d = f.line.dash, n = f.noise;
    $('#primeState').textContent = 'measured';
    const sheet = (name) => res.library_previews[name]
      ? `<img src="${API + res.library_previews[name]}" alt="${name} patches" style="width:100%;border:1px solid var(--line);border-radius:4px;display:block">`
      : '<div class="ph">no patches</div>';
    $('#primeResults').innerHTML = `
      <dl class="kv">
        <dt>Material</dt><dd>${m.reading}, tone ${m.mean}</dd>
        <dt></dt><dd class="mono" style="color:var(--muted)">${f.material.periodicity.reading}</dd>
        <dt>Dash</dt><dd>${d.found ? `${d.dash_px} px dash, ${d.gap_px} px gap` : 'not found'}</dd>
        <dt>Cycle</dt><dd class="mono">${d.cycle_px || '—'} px, width ${d.width_px || '—'} px</dd>
        <dt>Wear</dt><dd>${(n.coverage*100).toFixed(1)}% of the surface, ${n.features} marks</dd>
        <dt></dt><dd class="mono" style="color:var(--muted)">${n.periodicity.reading}</dd>
      </dl>
      <div style="margin-top:10px;font-size:12px;color:var(--muted)">Patch libraries</div>
      <div class="layers" style="margin-top:6px">
        <div>${sheet('material')}<span style="font-size:11.5px;color:var(--muted)">material</span></div>
        <div>${sheet('line')}<span style="font-size:11.5px;color:var(--muted)">line</span></div>
        <div>${sheet('noise')}<span style="font-size:11.5px;color:var(--muted)">wear</span></div>
      </div>`;
  }

  /* ------------------------------------------------ pairs */
  $('#btnAddPair').addEventListener('click', () => {
    const p = { uid:++S.pairUid, mask:null, photo:null };
    S.pairs.push(p); S.selPair = p; renderPairs(); renderAlign();
    log(`Added pair ${pairNum(p)}. Load its mask and street photo.`);
  });
  // numbered by position, so deleting pair 2 makes the next pair 2 again
  const pairNum = p => S.pairs.indexOf(p) + 1;
  const pairState = p => !p.mask || !p.photo ? 'incomplete'
    : (p.mask.width === p.photo.width && p.mask.height === p.photo.height) ? 'aligned' : 'mismatch';

  function renderPairs(){
    const list = $('#pairList'); list.innerHTML = '';
    if(!S.pairs.length) list.innerHTML = '<p class="empty" style="margin:0 0 4px;">No pairs yet. Each pair is a street mask plus the matching street photo, at the same size.</p>';
    S.pairs.forEach(p => {
      const st = pairState(p);
      const el = document.createElement('div'); el.className = 'pair';
      el.innerHTML = `
        <div class="pair-h"><span class="dot ${st==='aligned'?'ok':st==='mismatch'?'bad':''}"></span>
          <strong>Pair ${pairNum(p)}</strong><span style="color:var(--muted)">${st==='aligned'?`aligned, ${fmt(p.mask.road_px)} road px`:st==='mismatch'?'sizes differ':'incomplete'}</span>
          <button class="x" title="Remove pair" aria-label="Remove pair ${pairNum(p)}">×</button></div>
        <div class="pair-slots">
          <button class="pslot" data-part="mask"><div class="thumb">+</div><span>Mask</span></button>
          <button class="pslot" data-part="photo"><div class="thumb">+</div><span>Photo</span></button>
          <button class="pslot" data-part="overlay" ${st==='aligned'?'':'disabled'}><div class="thumb">⊕</div><span>Overlay</span></button>
        </div>`;
      if(p.mask) drawInto($('[data-part=mask] .thumb', el), p.mask.img);
      if(p.photo) drawInto($('[data-part=photo] .thumb', el), p.photo.img);
      if(st === 'aligned') drawInto($('[data-part=overlay] .thumb', el), overlay(p, .45));
      $('.x', el).addEventListener('click', () => {
        S.pairs = S.pairs.filter(x => x !== p);
        if(S.selPair === p){ S.selPair = null; S.junctions = null; }
        log(`Removed pair ${pairNum(p)}.`); renderPairs(); renderAlign(); renderJunctions();
      });
      $$('.pslot', el).forEach(btn => btn.addEventListener('click', () => {
        const part = btn.dataset.part;
        S.selPair = p; renderAlign(); renderJunctions();
        if(part === 'overlay'){ selectRow(btn); trainVp.show(overlay(p, $('#overlayOpacity').value/100), `Pair ${pairNum(p)} overlay`); return; }
        const rec = p[part];
        if(!rec) return pick(f => setPairPart(p, part, f));
        selectRow(btn); trainVp.show(rec.img, `Pair ${pairNum(p)} ${part}, ${rec.width} × ${rec.height}`);
      }));
      list.appendChild(el);
    });
    $('#pairCount').textContent = `${S.pairs.filter(p => pairState(p) === 'aligned').length} aligned`;
    updateTrain(); renderInfo();
  }

  async function setPairPart(p, part, file){
    status('Uploading…');
    try{
      const rec = await upload(file, part === 'mask' ? 'pair_mask' : 'pair_photo', String(p.uid));
      p[part] = rec; S.selPair = p; S.junctions = null;
      log(`Pair ${pairNum(p)}: loaded ${part} ${rec.name} (${rec.width} × ${rec.height})`);
      const st = pairState(p);
      if(st === 'mismatch') log(`Pair ${pairNum(p)} is not aligned: mask ${p.mask.width} × ${p.mask.height}, photo ${p.photo.width} × ${p.photo.height}.`, 'bad');
      if(st === 'aligned') log(`Pair ${pairNum(p)} aligned. ${fmt(p.mask.road_px)} road pixels to learn from.`, 'ok');
      renderPairs(); renderAlign(); renderJunctions(); renderMemory();
      trainVp.show(rec.img, `Pair ${pairNum(p)} ${part}, ${rec.width} × ${rec.height}`);
      status('Ready.');
    }catch(e){ log('Upload failed: ' + e.message, 'bad'); status('Upload failed.'); }
  }

  function overlay(p, a){
    const c = document.createElement('canvas');
    c.width = p.photo.width; c.height = p.photo.height;
    const ctx = c.getContext('2d', { willReadFrequently:true });
    ctx.drawImage(p.photo.img, 0, 0);
    const mc = document.createElement('canvas');
    mc.width = p.mask.width; mc.height = p.mask.height;
    mc.getContext('2d', { willReadFrequently:true }).drawImage(p.mask.img, 0, 0);
    const md = mc.getContext('2d').getImageData(0,0,mc.width,mc.height).data;
    const out = ctx.getImageData(0,0,c.width,c.height);
    for(let i=0;i<md.length;i+=4){
      if(md[i] > 128){
        out.data[i] = out.data[i]*(1-a) + 61*a;
        out.data[i+1] = out.data[i+1]*(1-a) + 123*a;
        out.data[i+2] = out.data[i+2]*(1-a) + 224*a;
      }
    }
    ctx.putImageData(out,0,0);
    return c;
  }

  function renderAlign(){
    const p = S.selPair, prev = $('#alignPrev'), note = $('#alignNote');
    note.style.color = '';
    $('#alignWhich').textContent = p ? `pair ${pairNum(p)}` : '';
    if(!p){ prev.innerHTML = '<div class="ph">Mask drawn over the photo</div>'; note.textContent = 'Select a pair to check it.'; return; }
    const st = pairState(p);
    if(st === 'incomplete'){ prev.innerHTML = '<div class="ph">Load both the mask and the photo</div>'; note.textContent = 'Waiting for both images.'; return; }
    if(st === 'mismatch'){
      prev.innerHTML = '<div class="ph">Sizes differ</div>';
      note.textContent = `Mask is ${p.mask.width} × ${p.mask.height}, photo is ${p.photo.width} × ${p.photo.height}. Export both from the same source at the same size.`;
      note.style.color = 'var(--red)'; return;
    }
    drawInto(prev, overlay(p, $('#overlayOpacity').value/100));
    note.textContent = 'Same size. Check by eye that the blue shape sits exactly on the road.';
  }
  $('#overlayOpacity').addEventListener('input', () => {
    renderAlign();
    const lbl = $('#trainVp .vp-label').textContent;
    if(S.selPair && pairState(S.selPair) === 'aligned' && /overlay$/.test(lbl))
      trainVp.show(overlay(S.selPair, $('#overlayOpacity').value/100), lbl, true);
  });
  $('#alignPrev').addEventListener('click', () => {
    const p = S.selPair;
    if(p && pairState(p) === 'aligned') trainVp.show(overlay(p, $('#overlayOpacity').value/100), `Pair ${pairNum(p)} overlay`);
  });

  function updateTrain(){
    const aligned = S.pairs.filter(p => pairState(p) === 'aligned');
    const bad = S.pairs.filter(p => pairState(p) === 'mismatch').length;
    $('#btnTrain').disabled = !aligned.length;
    $('#trainNote').textContent = !aligned.length ? 'Add at least one aligned pair.'
      : bad ? `Ready. ${bad} misaligned pair${bad>1?'s':''} will be skipped.`
      : `Ready to train on ${aligned.length} pair${aligned.length>1?'s':''}.`;
    $('#btnJunctions').disabled = !(S.selPair && S.selPair.mask);
  }

  $('#btnTrain').addEventListener('click', async () => {
    const aligned = S.pairs.filter(p => pairState(p) === 'aligned');
    $('#btnTrain').disabled = true; status('Training…');
    log(`Training on ${aligned.length} pair${aligned.length>1?'s':''}. Junction detection runs first, so large images take a while.`);
    try{
      const res = await api('/api/train', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Training',
        body: JSON.stringify({ metres_per_pixel: +$('#trainScale').value || 0.25,
          pairs: aligned.map(p => ({ mask:p.mask.id, photo:p.photo.id })) }) });
      S.trainResults = res;
      res.pairs.forEach(r => {
        if(r.resolution && !r.resolution.ok) log(`${r.pair}: ${r.resolution.reading}`, 'bad');
        else log(`${r.pair}: ${r.resolution ? r.resolution.reading : ''}`);
        if(r.kerb_band && r.kerb_band.capped) log(`  ${r.kerb_band.reading}`);
        Object.entries(r.groups).forEach(([k,g]) => {
          const s = g.stats;
          log(`  ${k}: ${(g.share_of_road*100).toFixed(1)}% of the road, tone ${s.mean}, grain ${s.grain}, ${g.patches} patches`);
        });
      });
      if(res.already_trained) log(`  ${res.already_trained} pair${res.already_trained===1?' was':'s were'} already trained on and ${res.already_trained===1?'was':'were'} not measured again.`);
      S.rebuilt = res.rebuilt;
      (res.rebuilt.changes || []).forEach(c => log('  ' + c, /none usable/.test(c) ? 'bad' : 'ok'));
      log(`Libraries rebuilt from ${res.rebuilt.pairs} pair${res.rebuilt.pairs===1?'':'s'} in total. Earlier tree versions stay in memory.`, 'ok');
      renderTrainResults();
      $('#trainNote').textContent = 'Training finished.';
    }catch(e){ log(e.message, 'bad'); $('#trainNote').textContent = e.message; }
    status('Ready.'); $('#btnTrain').disabled = false; renderMemory();
  });

  function renderTrainResults(){
    const res = S.trainResults;
    if(!res) return;
    // every pair may already have been trained on, in which case only the
    // rebuild is news
    const r = res.pairs.length ? res.pairs[res.pairs.length - 1] : null;
    const idx = (res.rebuilt && res.rebuilt.index) || {};
    const c = {};
    Object.entries(idx).forEach(([k, rows]) => {
      const top = rows[0];
      c[k] = top ? { patches: top.patches, pixels: rows.reduce((n,r) => n + r.pixels, 0),
                     tone: '—', grain: '—', label: top.label } : {};
    });
    const total = res.rebuilt ? res.rebuilt.pairs : res.pairs.length;
    $('#trainState').textContent = `${total} pair${total === 1 ? '' : 's'}`;
    const rows = ['junction','edge','open'].map(k => {
      const g = c[k] || {};
      const ok = (g.patches || 0) > 0;
      return `<tr><td>${k}</td><td class="mono">${ok ? fmt(g.pixels) : '—'}</td>
              <td class="mono">${g.tone ?? '—'}</td><td class="mono">${g.grain ?? '—'}</td>
              <td class="mono" style="color:${ok?'var(--green)':'var(--red)'}">${g.patches ?? 0}</td></tr>`;
    }).join('');
    const sheets = Object.entries(c).filter(([k,g]) => g.preview).map(([k,g]) => g.patches
      ? `<div><img src="${API + g.preview}" alt="${k} patches" style="width:100%;border:1px solid var(--line);border-radius:4px;display:block">
         <span style="font-size:11.5px;color:var(--muted)">${k}</span></div>`
      : `<div><div class="preview"><div class="ph">none</div></div><span style="font-size:11.5px;color:var(--muted)">${k}</span></div>`).join('');
    $('#trainResults').innerHTML = `
      <div style="font-size:12.5px;color:var(--muted);margin-bottom:6px">${
        r ? `Newest: ${r.pair}, road ${r.road_width_px ?? '—'} px (${r.road_width_m ?? '—'} m), kerb band ${
             r.kerb_band ? r.kerb_band.metres + ' m' + (r.kerb_band.capped ? ', reduced to fit' : '') : '—'}${
             r.resolution && !r.resolution.ok ? '. Too coarse to learn surface detail.' : ''}`
          : 'No new pairs measured; the libraries were rebuilt from the pairs already stored.'}</div>
      <table class="dt" style="width:100%;border-collapse:collapse;font-size:12.5px">
        <tr><th style="text-align:left">part</th><th>pixels</th><th>tone</th><th>grain</th><th>patches</th></tr>
        ${rows}
      </table>
      <div style="margin-top:10px;font-size:12px;color:var(--muted)">First in each order: ${
        Object.entries(c).map(([k,g]) => `${k} ${g.label ? g.label.split(' + ')[1] : 'none'}`).join(', ')}. See the Memory tab for the full order.</div>
      <div class="layers" style="margin-top:6px">${sheets}</div>`;
  }

  /* ------------------------------------------------ junctions */
  async function detectJunctions(imageId, label, vp, prevEl){
    status('Finding junctions…');
    try{
      const res = await api('/api/junctions', { method:'POST', headers:{'Content-Type':'application/json'}, job:`${label}: finding junctions`,
        body: JSON.stringify({ image: imageId }) });
      const img = await loadImage(API + res.overlay_url);
      vp.show(img, `${label}: junctions`);
      if(prevEl){ drawInto(prevEl, img); prevEl.dataset.url = API + res.overlay_url; }
      const s = res.summary;
      log(`${label}: ${s.junctions} junctions (${Object.entries(s.by_type).map(([k,v]) => v+' '+k).join(', ')}).`, 'ok');
      log(`Cleaning: ${s.specks_filled} specks filled, ${s.dividers_merged} dividers merged at the ${s.divider_limit_px} px limit.`);
      status('Ready.');
      return res;
    }catch(e){ log('Junction detection failed: ' + e.message, 'bad'); status('Ready.'); return null; }
  }

  $('#btnJunctions').addEventListener('click', async () => {
    const p = S.selPair;
    if(!p || !p.mask) return;
    $('#btnJunctions').disabled = true;
    S.junctions = await detectJunctions(p.mask.id, `Pair ${pairNum(p)}`, trainVp, $('#juncPrev'));
    renderJunctions(); renderInfo(); renderMemory();
    $('#btnJunctions').disabled = false;
  });
  $('#juncPrev').addEventListener('click', async () => {
    const url = $('#juncPrev').dataset.url;
    if(url) trainVp.show(await loadImage(url), 'Junctions');
  });

  function renderJunctions(){
    const el = $('#juncList'), st = $('#juncState');
    if(!S.junctions){ el.innerHTML = ''; st.textContent = ''; return; }
    const s = S.junctions.summary;
    st.textContent = `${s.junctions} found`;
    const rows = Object.entries(s.by_type).map(([k,v]) =>
      `<li><span>${k}</span><span class="mono">${v}</span></li>`).join('');
    el.innerHTML = `<ul class="routes" style="margin-top:10px">${rows}
      <li><span>road width, median</span><span class="mono">${s.road_width_px_median} px</span></li>
      <li><span>street segments</span><span class="mono">${s.street_segments}</span></li></ul>`;
  }

  /* ------------------------------------------------ generate */
  const genRow = $('#genMaskRow');
  const loadGen = () => pick(async f => {
    status('Uploading…');
    try{
      await useGenMask(await upload(f, 'gen_mask'));
    }catch(e){ log('Upload failed: ' + e.message, 'bad'); status('Upload failed.'); }
  });
  // a street mask to generate from: a loaded alpha map, or an area's alpha from its coordinates
  async function useGenMask(rec){
      const otherMask = !S.gen.mask || S.gen.mask.id !== rec.id;
      S.gen.mask = rec; S.gen.junctions = null;
      $('#satTexBox').hidden = !(rec.coords && rec.coords.key);
      // another mask: the speed-up files of other masks' textures go (app/caches.py)
      if(otherMask) api('/api/caches/release', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ mask: rec.id }) })
        .then(r => { if(r && r.freed > 0) log(`Freed ${r.freed_text}: the speed-up files of earlier textures (Update 2D texture makes them again if it needs them).`); })
        .catch(() => {});
      drawInto($('.thumb', genRow), rec.img);
      $('.imgrow-s', genRow).textContent = `${rec.name}, ${rec.width} × ${rec.height}`;
      drawInto($('#genMaskPrev'), rec.img);
      genVp.show(rec.img, `Mask, ${rec.width} × ${rec.height}`);
      log(`Loaded street mask ${rec.name} (${rec.width} × ${rec.height}, ${fmt(rec.road_px)} road pixels).`);
      $('#btnAddBridge').disabled = false;
      try{
        const saved = await api('/api/bridges?mask=' + rec.id);
        S.gen.bridges = saved.bridges || []; S.gen.bridgePreview = [];
        if(S.gen.bridges.length){ log(`${S.gen.bridges.length} saved bridge(s) restored for this mask.`); previewBridges(); }
        const sc = await api('/api/scatter?mask=' + rec.id);
        S.gen.placements = sc.placements || []; S.gen.plSel = -1; maskPixels = null;
        await loadSlots(rec.id);
        await loadLanes(rec.id);
        await loadMarea(rec.id);
        await loadParea(rec.id);
        loadLayout();
        if(S.gen.placements.length) log(`${S.gen.placements.length} saved object placement(s) restored for this mask.`);
        renderObjList();
        drawBridges();
      }catch(_){}
      status('Ready.'); updateGen(); renderInfo(); renderMemory();
  }
  $('.load', genRow).addEventListener('click', e => { e.stopPropagation(); loadGen(); });

  /* ---- Coordinates: an area by its longitudes and latitudes, downloaded through Universal Maps
     Downloader as a satellite map and a street map; the street map's roads are the alpha
     (app/coords.py). The satellite is for the analysis of the next steps. */
  const CO = { running: false, job: null, area: null };
  function setInputMode(m){
    $$('#inputMode button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.mode === m)));
    $('#coordsBox').hidden = m !== 'coords';
    try{ localStorage.setItem('rta.inputMode', m); }catch(_){}
    if(m === 'coords') coEstimate();
  }
  $$('#inputMode button').forEach(b => b.addEventListener('click', () => setInputMode(b.dataset.mode)));
  const CO_FIELDS = ['coLeft', 'coRight', 'coTop', 'coBottom'];
  const coBounds = () => ({ left: +$('#coLeft').value, right: +$('#coRight').value, top: +$('#coTop').value, bottom: +$('#coBottom').value });
  const coFilled = () => CO_FIELDS.every(k => $('#' + k).value.trim() !== '');
  const fmtLen = m => m >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${Math.round(m)} m`;
  let coEstTimer = null;
  function coEstimate(){
    clearTimeout(coEstTimer);
    coEstTimer = setTimeout(async () => {
      try{ localStorage.setItem('rta.coords', JSON.stringify(Object.fromEntries(CO_FIELDS.map(k => [k, $('#' + k).value])))); }catch(_){}
      const el = $('#coEstimate');
      if(!coFilled()){ el.className = 'note'; el.textContent = 'Decimal degrees, as in the OSM tool. Zoom 20: 0.136 m a pixel in Riyadh.'; $('#btnCoords').disabled = true; return; }
      try{
        const e = await api('/api/coords/estimate', { method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ bounds: coBounds(), zoom: +$('#coZoom').value }) });
        el.className = e.ok ? 'note' : 'note bad';
        el.textContent = `${fmtLen(e.width_m)} × ${fmtLen(e.height_m)} · ${fmt(e.width_px)} × ${fmt(e.height_px)} px · ${e.mpp.toFixed(3)} m a pixel · ${fmt(e.tiles)} tiles a map`
          + (e.ok ? '' : `. Too big: ${e.problems.join('; ')}.`);
        $('#btnCoords').disabled = !e.ok || CO.running;
      }catch(err){ el.className = 'note bad'; el.textContent = err.message; $('#btnCoords').disabled = true; }
    }, 250);
  }
  CO_FIELDS.concat(['coZoom']).forEach(k => $('#' + k).addEventListener('input', coEstimate));
  async function coSettings(save){
    try{
      const s = save ? await api('/api/coords/settings', { method:'POST', headers:{'Content-Type':'application/json'},
                         body: JSON.stringify({ downloader: $('#coDownloader').value, pbf: $('#coPbf').value, zoom: +$('#coZoom').value,
                                                buildings_map: Math.max(0, Math.round(+$('#coBldMap').value || 0)) }) })
                     : await api('/api/coords/settings');
      $('#coDownloader').value = s.downloader || ''; $('#coPbf').value = s.pbf || ''; $('#coBldMap').value = s.buildings_map || 0;
      if(!save) $('#coZoom').value = s.zoom || 20;
      const from = [s.downloader_from && `downloader from ${s.downloader_from}`, s.pbf_from && `PBF from ${s.pbf_from}`].filter(Boolean);
      $('#coFilesState').textContent = s.downloader ? (s.pbf ? '· set' : '· no PBF') : '· not set';
      $('#coFilesNote').textContent = (from.length ? `Found: ${from.join(', ')}. ` : '')
        + 'Paste each file\'s full path (in Explorer: Shift + right-click the file, Copy as path). Without the PBF the alpha is made without the OSM check.';
      if(!s.downloader) $('#coFiles').open = true;
      return s;
    }catch(e){ $('#coFilesNote').textContent = e.message; return null; }
  }
  ['coDownloader', 'coPbf', 'coBldMap'].forEach(k => $('#' + k).addEventListener('change', () => coSettings(true)));
  function coBar(frac, text){
    const b = $('#coBar'); b.hidden = false;
    $('i', b).style.width = `${Math.round(Math.max(0, Math.min(1, frac || 0)) * 100)}%`;
    $('span', b).textContent = text;
  }
  async function coShowArea(area){
    CO.area = area;
    const v = Date.now().toString(36);
    $$('#coThumbs figure').forEach(f => {
      const img = new Image(); img.src = `${API}/api/coords/area/${area.key}/${f.dataset.k}?v=${v}`;
      img.alt = f.dataset.k; $('.thumb', f).innerHTML = ''; $('.thumb', f).appendChild(img);
    });
    $('#coThumbs').hidden = false;
    const a = area.alpha || {};
    const bits = [`${fmt(area.width)} × ${fmt(area.height)} px at ${area.mpp.toFixed(3)} m a pixel`];
    if(area.osm_roads != null) bits.push(`${fmt(area.osm_roads)} OSM road ways, ${Math.round((area.osm_on_road || 0) * 100)}% of them on the alpha's roads`);
    if(a.far_from_osm) bits.push(`${a.far_from_osm} white area(s) with no OSM road near left out (${fmt(Math.round(a.far_from_osm_m2))} m²)`);
    if(a.thin_lines) bits.push(`${a.thin_lines} thin line(s) left out: footpaths, parking lanes, service passages under 2 m wide (${fmt(Math.round(a.thin_m2))} m²)`);
    if(area.missing_tiles) bits.push(`${area.missing_tiles} satellite tile(s) missing`);
    $('#coNote').className = area.missing_tiles ? 'note bad' : 'note';
    $('#coNote').textContent = bits.join('; ') + '. Check: the alpha\'s edge (amber) and the OSM roads (cyan) over the satellite.';
  }
  $$('#coThumbs figure').forEach(f => f.addEventListener('click', async () => {
    if(!CO.area) return;
    const img = await loadImage(`${API}/api/coords/area/${CO.area.key}/${f.dataset.k}?v=${Date.now().toString(36)}`);
    genVp.show(img, { satellite: 'Satellite', street: 'Street map', alpha: 'Alpha', check: 'Check: the alpha\'s edge and the OSM roads over the satellite' }[f.dataset.k]);
  }));
  async function coStart(){
    if(CO.running) return;
    const s = await coSettings(true);
    if(!s || !s.downloader){ log('Coordinates: set the path of Universal Maps Downloader\'s downloader.exe first (Downloader and PBF).', 'bad'); $('#coFiles').open = true; return; }
    const job = trackJob('Coordinates: downloading the maps and making the alpha');
    CO.running = true; $('#btnCoords').disabled = true; $('#btnCoordsStop').hidden = false;
    coBar(0, 'starting…');
    const end = (ok, extra) => { job.done(ok, extra); CO.running = false; CO.job = null; $('#btnCoordsStop').hidden = true; coEstimate(); };
    try{
      const r = await api('/api/coords/start', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ bounds: coBounds(), zoom: +$('#coZoom').value, progress: job.id }) });
      CO.job = r.job;
      for(;;){
        await new Promise(res => setTimeout(res, 1000));
        const st = await api('/api/coords/job/' + r.job);
        if(st.state === 'queued' || st.state === 'running'){
          coBar(st.done, `${st.stage || 'waiting for the downloader'}${st.done != null ? ` · ${Math.floor(st.done * 100)}%` : ''}${st.left != null ? ` · about ${fmtTime(st.left)} left` : ''}`);
          continue;
        }
        if(st.state !== 'done'){ coBar(0, st.error || st.state); log('Coordinates: ' + (st.error || st.state), 'bad'); end(false, st.error || st.state); return; }
        const { area, mask } = st.result;
        coBar(1, 'done');
        mask.img = await loadImage(API + mask.url);
        await coShowArea(area);
        $('#scale').value = +area.mpp.toFixed(4);
        $('#scale').dispatchEvent(new Event('change', { bubbles: true }));
        await useGenMask(mask);
        S.gen.coords = area;
        log(`Coordinates: the alpha is the street mask (${fmt(area.width)} × ${fmt(area.height)} px, scale ${area.mpp.toFixed(4)} m a pixel): press Generate texture.`, 'ok');
        end(true, `${fmt(area.width)} × ${fmt(area.height)} px`);
        if(st.result.analysis) satFollow(area.key);
        return;
      }
    }catch(e){ coBar(0, e.message); log('Coordinates: ' + e.message, 'bad'); end(false, e.message); }
  }
  /* ---- the satellite analysis (app/sat_analysis.py): in the background once the alpha is made; its
     card on the right of the viewport, a square a tile (100 until it is looked at, then its confidence) */
  const SAT = { key: null, timer: null, data: null, job: null, cell: null };
  const SAT_NAMES = { zebra: ['zebra crossing', 'zebra crossings'], stop_line: ['stop line', 'stop lines'],
    cycle_crossing: ['cyclist crossing', 'cyclist crossings'], giveway: ['give way line', 'give way lines'],
    arrow: ['arrow', 'arrows'], parking_bay: ['row of parking bays', 'rows of parking bays'],
    lane_dash: ['lane dash', 'lane dashes'], lane_solid: ['solid lane line', 'solid lane lines'],
    car: ['car', 'cars'], unknown: ['unknown mark', 'unknown marks'] };
  const satRamp = v => { v = Math.max(0, Math.min(1, v)); const a = [220,50,50], b = [240,170,30], c = [60,200,90];
    const [p, q, t] = v < 0.6 ? [a, b, v / 0.6] : [b, c, (v - 0.6) / 0.4];
    return `rgb(${p.map((x, i) => Math.round(x + (q[i] - x) * t)).join(',')})`; };
  function satDraw(){
    const d = SAT.data, cv = $('#satTiles');
    if(!d || !d.grid || !d.grid.length){ cv.height = 1; SAT.cell = null; return; }
    const rows = Math.max(...d.grid.map(t => t.row)) + 1, cols = Math.max(...d.grid.map(t => t.col)) + 1;
    const cs = Math.max(6, Math.floor(Math.min(228 / cols, 150 / rows)));
    cv.width = cols * cs; cv.height = rows * cs; cv.style.width = `${cols * cs}px`;
    const g = cv.getContext('2d');
    g.font = `${Math.max(8, Math.floor(cs * 0.38))}px sans-serif`; g.textAlign = 'center'; g.textBaseline = 'middle';
    SAT.cell = { cs, cols, rows };
    for(const t of d.grid){
      const r = d.tiles[t.id], x = t.col * cs, y = t.row * cs;
      let fill = '#2c3138', label = '100', ink = '#7c8592';                 // not looked at yet: 100 for now
      if(r && r.state === 'no_road'){ fill = '#1d2126'; label = ''; }
      else if(r && r.state === 'failed'){ fill = '#5a2a2a'; label = '!'; ink = '#fff'; }
      else if(r && r.confidence != null){ fill = satRamp(r.confidence / 100); label = Math.round(r.confidence); ink = '#fff'; }
      g.fillStyle = fill; g.fillRect(x + 1, y + 1, cs - 2, cs - 2);
      if(cs >= 16 && label !== ''){ g.fillStyle = ink; g.fillText(String(label), x + cs / 2, y + cs / 2 + 0.5); }
      if(TEX.live && texHere()){
        // Generate texture at work: the tiles it has done framed, the others dimmed
        if(TEX.live.done.has(t.id)){ g.strokeStyle = '#fff'; g.lineWidth = 2; g.strokeRect(x + 2, y + 2, cs - 4, cs - 4); }
        else { g.fillStyle = 'rgba(10,12,14,.55)'; g.fillRect(x + 1, y + 1, cs - 2, cs - 2); }
      }
    }
  }
  /* ---- the texture made in the same tiles (app/tiled.py): while Generate texture works, the tiles done
     framed white and the others dimmed; afterwards each tile tells where its asphalt's tone came from */
  const TEX = { live: null, tiles: null, gid: null, info: null };
  const texHere = () => !!(SAT.key && satAreaKey() === SAT.key);
  window.onJobDetail = d => {
    if(!d || d.kind !== 'texture_tiles') return;
    TEX.live = { phase: d.phase, done: new Set(d.done || []), total: d.total };
    texCardState(); satDraw();
  };
  function texTilesLog(t){
    const b = t.by_tone || {};
    log(`  Made in ${t.count} tiles, the satellite analysis's ${t.tile_px} px squares (${t.with_road} with road, `
      + `${t.workers === 'here' ? 'one after another' : `${t.workers} at a time`}), one set of materials for all, so they join without a seam.`);
    if(t.tone) log(`  Asphalt tone from the satellite: ${b.satellite || 0} tile(s) seen well enough follow it, `
      + `${b.rules || 0} keep the area's usual tone (rated under ${$('#satThr').value}%).`);
    else if(+$('#satTone').value > 0) log('  Asphalt tone: one tone everywhere (the satellite analysis is not finished, or saw too little asphalt): Generate texture again once its card says done.');
  }
  async function texTilesLoad(res){
    TEX.live = null; TEX.tiles = null; TEX.gid = null; TEX.info = null;
    const t = res && res.summary && res.summary.tiles;
    if(t && t.url){
      try{
        const r = await api(t.url);
        TEX.tiles = Object.fromEntries((r.reports || []).map(x => [x.id, x]));
        TEX.gid = res.id; TEX.info = t;
      }catch(_){}
    }
    texCardState(); satDraw();
  }
  // after Update 2D texture (a new tone, say): the tiles' report as it is now
  async function texTilesRefresh(gid){
    if(!TEX.info || TEX.gid !== gid) return;
    try{
      const r = await api(TEX.info.url + '?v=' + Date.now().toString(36));
      const reps = r.reports || [], by = {};
      reps.forEach(x => { const k = x.tone || (x.road_px ? 'area' : 'no road'); by[k] = (by[k] || 0) + 1; });
      TEX.tiles = Object.fromEntries(reps.map(x => [x.id, x]));
      TEX.info = { ...TEX.info, tone: r.tone || null, by_tone: by };
    }catch(_){}
    texCardState(); satDraw();
  }
  function texCardState(){
    const box = $('#satTex'), here = texHere();
    if(TEX.live && here){
      box.hidden = false; $('#btnSatTexPic').hidden = true;
      $('#satTexNote').textContent = `Generate texture: ${TEX.live.phase === 'material' ? 'material and wear' : 'the pictures'}, `
        + `tile ${TEX.live.done.size} of ${TEX.live.total}. Framed: done.`;
      return;
    }
    const on = !!(TEX.info && here && S.gen.result && S.gen.result.id === TEX.gid);
    box.hidden = !on; $('#btnSatTexPic').hidden = !on;
    if(!on) return;
    const t = TEX.info, b = t.by_tone || {};
    $('#satTexNote').textContent = `The texture was made in these tiles: ${t.with_road} of ${t.count} with road, one set of materials. `
      + (t.tone ? `Asphalt tone: ${b.satellite || 0} from the satellite, ${b.rules || 0} the area's. Point at a tile.` : 'One asphalt tone everywhere.');
  }
  $('#btnSatTexPic').addEventListener('click', async () => {
    if(!TEX.gid) return;
    try{
      const img = await loadImage(`${API}/api/texture/tiles/${TEX.gid}?v=${Date.now().toString(36)}`);
      genVp.show(img, TEX.info && TEX.info.tone ? 'Texture tiles: green, tone from the satellite · amber, the area\'s tone (rules) · grey, no road'
                                                : 'Texture tiles: white, one tone (asphalt tone from the satellite at 0) · grey, no road');
    }catch(e){ log('Texture tiles: ' + e.message, 'bad'); }
  });
  function texTip(id){
    if(!TEX.tiles || !texHere() || !S.gen.result || S.gen.result.id !== TEX.gid) return '';
    const r = TEX.tiles[id];
    if(!r) return '';
    if(!r.road_px) return ' · texture: no road (background only)';
    if(r.tone === 'satellite') return ` · texture: asphalt tone from the satellite (×${(r.gain ?? 1).toFixed(2)})`;
    if(r.tone === 'rules') return ' · texture: the area\'s usual tone (rated under the threshold)';
    return '';
  }
  function satTip(){ $('#satTip').textContent = 'Each square a tile: 100 until it is looked at, then how well its road is seen. Point at one for why.'; }
  $('#satTiles').addEventListener('mousemove', e => {
    const d = SAT.data, c = SAT.cell;
    if(!d || !c) return;
    const rc = e.target.getBoundingClientRect();
    const col = Math.floor((e.clientX - rc.left) / c.cs), row = Math.floor((e.clientY - rc.top) / c.cs);
    const t = d.grid.find(t => t.row === row && t.col === col);
    const r = t && d.tiles[t.id];
    $('#satTip').textContent = !t ? '' : (!r ? `Tile ${t.id}: not looked at yet (100 for now).`
      : r.state === 'no_road' ? `Tile ${t.id}: no road.`
      : r.state === 'failed' ? `Tile ${t.id}: could not be analysed.`
      : `Tile ${t.id}: road ${Math.round(r.confidence)}%, markings ${Math.round(r.confidence_marks ?? r.confidence)}%`
        + ((r.reasons || []).length ? ` · ${r.reasons.map(x => `${x.why} −${x.minus}`).join('; ')}` : ' · nothing hides the road'))
      + texTip(t.id);
  });
  $('#satTiles').addEventListener('mouseleave', satTip);
  function satRender(){
    const d = SAT.data; if(!d) return;
    $('#satCard').hidden = false;
    const bar = $('#satBar');
    const frac = d.state === 'done' ? 1 : (d.total ? d.done / d.total : 0);
    $('i', bar).style.width = `${Math.round(frac * 100)}%`;
    $('span', bar).textContent = d.state === 'done' ? 'done'
      : d.state === 'failed' ? `stopped: ${d.error}`
      : d.state === 'preparing' ? 'preparing the satellite and the alpha…'
      : `${d.done} of ${d.total} tiles · ${Math.floor(frac * 100)}%` + (d.left != null ? ` · about ${fmtTime(d.left)} left` : '');
    $('#satState').textContent = d.state === 'done' ? '· done' : d.state === 'failed' ? '· stopped' : '· working';
    satDraw();
    const s = d.summary;
    if(d.state === 'done' && s){
      const order = ['zebra', 'stop_line', 'cycle_crossing', 'giveway', 'arrow', 'parking_bay', 'lane_dash', 'lane_solid'];
      const found = order.filter(k => s.counts[k]).map(k => `${fmt(s.counts[k])} ${SAT_NAMES[k][s.counts[k] === 1 ? 0 : 1]}`);
      $('#satSum').innerHTML = `Found: ${found.length ? found.join(', ') : 'no markings'}`
        + (s.counts.car ? ` (and ${fmt(s.counts.car)} cars set aside)` : '')
        + `.<br>Confidence: road <b>${s.confidence ?? '–'}%</b>, markings <b>${s.confidence_marks ?? '–'}%</b>`
        + (s.low_tiles ? `; ${s.low_tiles} tile(s) under 60%` : '') + '.';
      $('#satBtns').hidden = false;
    } else { $('#satSum').textContent = ''; $('#satBtns').hidden = true; }
    satPlaceState();
    dsCardState();
    texCardState();
  }
  async function satPoll(){
    if(!SAT.key) return;
    try{ SAT.data = await api('/api/coords/analysis/' + SAT.key); }catch(_){ SAT.data = null; }
    if(SAT.data && SAT.data.state !== 'none') satRender();
    const live = SAT.data && (SAT.data.state === 'preparing' || SAT.data.state === 'running');
    if(live){
      if(SAT.job) SAT.job.set(SAT.data.state === 'preparing' ? 'preparing' : `${SAT.data.done} of ${SAT.data.total} tiles`,
                              SAT.data.total ? SAT.data.done / SAT.data.total : null, SAT.data.left);
      SAT.timer = setTimeout(satPoll, 1000);
    } else if(SAT.data && SAT.data.state === 'done' && SAT.job){
      const s = SAT.data.summary;
      SAT.job.done(true, `confidence ${s.confidence ?? '–'}%`); SAT.job = null;
      log(`Satellite analysis: done; road confidence ${s.confidence ?? '–'}%, markings ${s.confidence_marks ?? '–'}%. Its card is on the right of the viewport.`, 'ok');
    } else if(SAT.data && SAT.data.state === 'failed' && SAT.job){ SAT.job.done(false, SAT.data.error); SAT.job = null; }
  }
  function satFollow(key){
    clearTimeout(SAT.timer);
    SAT.key = key; SAT.data = null; satTip();
    if(SAT.job) SAT.job.done(false, 'replaced');
    SAT.job = trackJob('Satellite analysis (in the background)');
    satPoll();
  }
  $('#satHead').addEventListener('click', () => $('#satCard').classList.toggle('folded'));
  $('#satTone').addEventListener('input', () => { $('#satToneVal').textContent = $('#satTone').value; });
  $$('#satBtns [data-pic]').forEach(b => b.addEventListener('click', async () => {
    if(!SAT.key) return;
    const k = b.dataset.pic;
    const img = await loadImage(`${API}/api/coords/area/${SAT.key}/${k}?v=${Date.now().toString(36)}`);
    genVp.show(img, { findings: 'Findings: every mark found, in its colour', confidence: 'Confidence of every tile',
                      satellite: 'Satellite' }[k]);
  }));
  /* ---- Save to the dataset (app/dataset.py): what the analysis found, kept as training examples, only the kinds
     you tick. Saved in the background; its progress at the bottom right of the page: one bar, and when you click it
     a bar for each kind of each area being saved */
  const DS = { jobs: [], open: false, timer: null, ended: new Set(), first: true };
  function dsCardState(){
    const d = SAT.data, on = !!(d && d.state !== 'none');
    $('#satSave').hidden = !on;
    if(!on) return;
    const done = d.state === 'done';
    const busy = DS.jobs.some(j => j.key === SAT.key && (j.state === 'running' || j.state === 'waiting'));
    $('#btnSatSave').disabled = !done || busy;
    $('#satSaveNote').textContent = !done ? 'When the analysis is done: keep what it found as training examples.'
      : busy ? 'Saving this area to the dataset: its progress is at the bottom right.'
      : 'Keep what it found as training examples; you tick which kinds.';
  }
  async function dsOpen(){
    if(!SAT.key) return;
    let r;
    try{ r = await api('/api/dataset/categories/' + SAT.key); }catch(e){ log('Dataset: ' + e.message, 'bad'); return; }
    DS.key = SAT.key;
    const skip = new Set(['car', 'unknown']);             // cars and unnamed paint: only when asked
    $('#dsList').innerHTML = r.categories.map(c => `<label class="ds-row"><input type="checkbox" data-k="${c.id}"${skip.has(c.id) ? '' : ' checked'}>`
      + `<span>${c.label}</span><small>${fmt(c.count)} here${c.in_dataset ? ` · ${fmt(c.in_dataset)} in the dataset` : ''}</small></label>`).join('');
    const kept = r.categories.reduce((a, c) => a + (c.from_area || 0), 0);
    $('#dsNote').textContent = 'What the satellite analysis of this area found. Tick what to keep as training examples: each mark as '
      + 'the satellite round it with its label (kind, size, angle, which way, how sure); road tiles as the satellite with its alpha.'
      + (kept ? ` ${fmt(kept)} from this area are kept already: a kind saved again replaces them (none twice).` : '');
    $('#dsWhere').textContent = `Kept in ${r.folder}, a folder for each kind. Saving goes on in the background.`;
    $$('#dsList input').forEach(b => b.addEventListener('change', dsAllState));
    dsAllState();
    $('#dsPop').hidden = false;
  }
  function dsAllState(){
    const bs = $$('#dsList input'), n = bs.filter(b => b.checked).length;
    $('#dsAll').checked = n === bs.length && n > 0;
    $('#dsAll').indeterminate = n > 0 && n < bs.length;
    $('#dsGo').disabled = !n;
    $('#dsGo').textContent = !n ? 'Save' : `Save ${n === bs.length ? 'all' : n} ${n === 1 ? 'kind' : 'kinds'}`;
  }
  const dsClose = () => { $('#dsPop').hidden = true; };
  $('#btnSatSave').addEventListener('click', dsOpen);
  $('#dsAll').addEventListener('change', () => { $$('#dsList input').forEach(b => { b.checked = $('#dsAll').checked; }); dsAllState(); });
  $('#dsClose').addEventListener('click', dsClose);
  $('#dsCancel').addEventListener('click', dsClose);
  addEventListener('keydown', e => { if(e.key === 'Escape' && !$('#dsPop').hidden) dsClose(); });
  $('#dsGo').addEventListener('click', async () => {
    const kinds = $$('#dsList input').filter(b => b.checked).map(b => b.dataset.k);
    if(!kinds.length || !DS.key) return;
    $('#dsGo').disabled = true;
    try{
      const j = await api('/api/dataset/save', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ key: DS.key, kinds }) });
      dsClose();
      log(`Dataset: saving ${j.total.toLocaleString()} example(s) of ${kinds.length} kind(s) in the background; its progress is at the bottom right (click it for each kind).`);
      DS.jobs = [j, ...DS.jobs.filter(q => q.id !== j.id)];
      bgRender(); dsCardState(); dsPoll();
    }catch(e){ log('Dataset not saved: ' + e.message, 'bad'); $('#dsGo').disabled = false; }
  });
  const dsLive = j => j.state === 'running' || j.state === 'waiting';
  async function dsPoll(){
    clearTimeout(DS.timer);
    try{ DS.jobs = (await api('/api/dataset/jobs')).jobs || []; }catch(_){ DS.timer = setTimeout(dsPoll, 3000); return; }
    for(const j of DS.jobs){
      if(dsLive(j) || DS.ended.has(j.id)) continue;
      DS.ended.add(j.id);
      if(DS.first) continue;                                   // finished before the page opened
      const what = j.cats.filter(c => c.kept).map(c => `${c.label.toLowerCase()} ${fmt(c.kept)}`).join(', ');
      if(j.state === 'done') log(`Dataset: ${fmt(j.kept)} example(s) of ${j.title} saved in ${fmtTime(j.elapsed)} (${what}).`, 'ok');
      else if(j.state === 'stopped') log(`Dataset: saving ${j.title} stopped; ${fmt(j.kept)} example(s) kept (${what || 'none'}).`);
      else log(`Dataset: saving ${j.title} failed: ${j.error}`, 'bad');
    }
    DS.first = false;
    bgRender(); dsCardState();
    if(DS.jobs.some(dsLive)) DS.timer = setTimeout(dsPoll, 700);
  }
  // the bottom right: one bar for every task; open, each task and each of its kinds
  function bgRender(){
    const jobs = DS.jobs;
    $('#bgTasks').hidden = !jobs.length;
    if(!jobs.length){ DS.open = false; $('#bgPanel').hidden = true; return; }
    const live = jobs.filter(dsLive), run = live.find(j => j.state === 'running');
    let title, sub, frac, cls = '';
    if(live.length){
      const total = live.reduce((a, j) => a + j.total, 0), done = live.reduce((a, j) => a + j.done, 0);
      const rate = run && run.done && run.elapsed ? run.done / run.elapsed : null;   // examples a second
      const left = rate ? (total - done) / rate : null;
      title = `Saving to the dataset${live.length > 1 ? ` · ${live.length} areas` : ''}`;
      frac = total ? done / total : 0;
      sub = `${Math.floor(frac * 100)}%` + (left != null ? ` · about ${fmtTime(Math.max(1, left))} left` : '');
    }else{
      const last = jobs[0], kept = jobs.reduce((a, j) => a + j.kept, 0);
      title = last.state === 'done' ? 'Saved to the dataset ✓' : last.state === 'stopped' ? 'Saving to the dataset stopped' : 'Saving to the dataset failed';
      sub = `${fmt(kept)} example${kept === 1 ? '' : 's'}`;
      frac = 1; cls = last.state === 'failed' ? 'bad' : 'done';
    }
    $('#bgTitle').textContent = title; $('#bgSub').textContent = sub;
    $('#bgFill').style.width = `${Math.round(frac * 100)}%`;
    $('#bgTrack').className = 'bg-track' + (cls ? ' ' + cls : '');
    $('#bgTasks').classList.toggle('open', DS.open);
    $('#bgPanel').hidden = !DS.open;
    if(!DS.open) return;
    const bar = (f, c) => `<span class="bg-track${c ? ' ' + c : ''}"><i style="width:${Math.round(f * 100)}%"></i></span>`;
    $('#bgPanel').innerHTML = jobs.map(j => {
      const f = j.total ? j.done / j.total : (dsLive(j) ? 0 : 1), state = { running: 'saving', waiting: 'waiting its turn',
        done: 'saved', stopped: 'stopped', failed: 'failed' }[j.state];
      const timeTxt = j.state === 'running' ? ` · ${fmtTime(j.elapsed)} so far` + (j.left != null ? ` · about ${fmtTime(Math.max(1, j.left))} left` : '')
        : j.state === 'waiting' ? '' : ` in ${fmtTime(j.elapsed)}`;
      return `<div class="bg-job"><div class="bg-jh"><b>${j.title}</b>`
        + (dsLive(j) ? `<button type="button" data-stop="${j.id}" title="Stop saving this area (what is saved stays)">Stop</button>`
                     : `<button type="button" data-dismiss="${j.id}" title="Take it off this list">×</button>`) + '</div>'
        + `<div class="bg-js">${state} · ${fmt(j.done)} of ${fmt(j.total)}${timeTxt}${j.error ? ' · ' + j.error : ''}</div>`
        + j.cats.map(c => `<div class="bg-cat"><b>${c.label}</b><span>${fmt(c.done)} / ${fmt(c.total)}`
            + (c.state === 'running' && c.left != null ? ` · ${fmtTime(Math.max(1, c.left))}` : c.state === 'done' ? ' ✓' : c.state === 'stopped' ? ' · stopped' : '')
            + `</span>${bar(c.total ? c.done / c.total : (c.state === 'done' ? 1 : 0), c.state === 'done' ? 'done' : c.state === 'stopped' ? 'bad' : '')}</div>`).join('')
        + '</div>';
    }).join('');
    $$('#bgPanel [data-stop]').forEach(b => b.addEventListener('click', async e => {
      e.stopPropagation();
      try{ await api('/api/dataset/cancel/' + b.dataset.stop, { method:'POST' }); }catch(_){}
      dsPoll();
    }));
    $$('#bgPanel [data-dismiss]').forEach(b => b.addEventListener('click', async e => {
      e.stopPropagation();
      try{ await api('/api/dataset/dismiss/' + b.dataset.dismiss, { method:'POST' }); }catch(_){}
      DS.jobs = DS.jobs.filter(j => j.id !== b.dataset.dismiss); bgRender();
    }));
  }
  $('#bgBar').addEventListener('click', () => { DS.open = !DS.open; bgRender(); });
  dsPoll();                                                    // what was being saved when the page opened

  /* ---- Place decals from what the satellite saw (app/observed.py): in the tiles rated at or above the
     threshold the markings it found, where it found them; under it the rules, that tile only */
  const SAT_PRESET = { zebra: ['zebra crossing', 'zebra crossings'], stop: ['stop line', 'stop lines'],
    giveway: ['give way line', 'give way lines'], cycle: ['cyclist crossing', 'cyclist crossings'], arrows: ['arrow', 'arrows'],
    stopword: ['STOP word', 'STOP words'] };
  const satCount = o => Object.entries(o || {}).filter(([, n]) => n).map(([k, n]) => `${fmt(n)} ${(SAT_PRESET[k] || SAT_NAMES[k] || [k, k])[n === 1 ? 0 : 1]}`).join(', ');
  function satAreaKey(){ return (S.gen.mask && S.gen.mask.coords && S.gen.mask.coords.key) || (S.gen.coords && S.gen.coords.key) || null; }
  async function satPlaceState(){
    const done = SAT.data && SAT.data.state === 'done';
    $('#satPlace').hidden = !done;
    if(!done) return;
    const gid = S.gen.result && S.gen.result.id;
    const ok = !!gid && satAreaKey() === SAT.key && !SAT.placing;
    $('#btnSatPlace').disabled = $('#btnSatRules').disabled = !ok;
    if(gid && SAT.settingsFor !== gid){
      // what this texture was last laid with: the satellite (and at what threshold), or the rules alone
      SAT.settingsFor = gid;
      try{ SAT.placed = (await api('/api/coords/place-decals/' + gid)).settings; }catch(_){ SAT.placed = null; }
      if(SAT.placed){ $('#satThr').value = SAT.placed.threshold; $('#satFill').checked = !!SAT.placed.fill; }
    }
    const on = ok && SAT.placed && SAT.placed.on;
    $('#btnSatPlaced').hidden = !(ok && S.gen.decals);
    $('#btnSatPlace').textContent = on ? 'Place again' : 'Place decals';
    $('#satPlaceNote').textContent = !gid ? 'Generate texture first: then Place decals lays the markings the satellite saw on it.'
      : satAreaKey() !== SAT.key ? 'The texture is not made from this area.'
      : on ? `Laid from the satellite; the rules in tiles under ${SAT.placed.threshold}%${SAT.placed.fill ? ', and where it saw none' : ''}. Update decals keeps to it.`
      : 'The decals are laid by the rules now. Place decals lays what the satellite saw where it saw it.';
  }
  async function satPlace(on){
    if(!S.gen.result || SAT.placing) return;
    const gid = S.gen.result.id, mode = $('#decalMode').value;
    const thr = Math.max(0, Math.min(100, Number($('#satThr').value) || 0));
    $('#satThr').value = thr;
    SAT.placing = true; satPlaceState();
    status(on ? 'Placing the decals the satellite saw…' : 'Placing the decals by the rules…');
    try{
      const res = await api('/api/coords/place-decals', { method:'POST', headers:{'Content-Type':'application/json'},
        job: on ? 'Placing decals from the satellite' : 'Placing decals by the rules',
        body: JSON.stringify({ generation: gid, layers: DEC.layers, mode, seed: +$('#seed').value, mark_area: MAREA.area,
                               marks: marksPayload(), threshold: thr, fill: $('#satFill').checked, on }) });
      if(!S.gen.result || S.gen.result.id !== gid) return;
      if(res.added && res.added.length){
        DEC.layers = res.decal_layers;
        log(`Decals: added the ${res.added.map(k => (SAT_PRESET[k] || [k, k])[1]).join(', ')} layer${res.added.length === 1 ? '' : 's'} for what the satellite saw (laid from it alone: the rules lay none of ${res.added.length === 1 ? 'it' : 'them'}).`);
      }
      S.gen.decals = res; S.gen.decalVersion = Date.now();
      S.gen.result.urls.result = res.urls.result;
      if(S.gen.islandTex && S.gen.islandTex.gid === gid) S.gen.islandTex.urls.full = res.urls.full;
      SAT.placed = res.settings; SAT.settingsFor = gid;
      $('#btnPlaceDecals').classList.remove('attention');
      if(res.lane_picks && S.gen.mask){
        // the satellite's lane choices are among the Lanes tool's now (yours win): already laid on the texture
        LANES.picks = res.lane_picks; S.gen.lanesPainted = JSON.stringify(LANES.picks);
        renderLanes(); drawBridges();
      }
      const o = res.observed;
      if(o){
        log(`Decals from the satellite: ${satCount(o.observed) || 'none'} laid where it saw them, in ${o.tiles.seen} tile(s) at or over ${o.threshold}%`
          + (Object.keys(o.rules).length ? `; the rules laid ${satCount(o.rules)}${o.tiles.rules ? ` (${o.tiles.rules} tile(s) under ${o.threshold}% or without road)` : ''}` : '')
          + ` (${res.seconds.toFixed(1)} s).`, 'ok');
        if(o.ends.moved || o.ends.dropped) log(`  Lane lines: they stop at ${o.ends.moved} junction edge(s) where the satellite saw the crossing or line`
          + (o.ends.dropped ? `; at ${o.ends.dropped} it saw none, so they run on to the junction` : '') + '.');
        const sk = o.skipped;
        if(sk.faint || sk.overlap) log(`  Not laid: ${[sk.faint ? `${sk.faint} too faint to be sure` : '', sk.overlap ? `${sk.overlap} lying on another found mark` : ''].filter(Boolean).join(', ')}.`);
        const ln = res.lanes;
        if(ln) log(`  Lane lines from the satellite: ${ln.read} street(s) read`
          + (ln.read ? ` (${ln.same} as the rules have them, ${ln.changed} set from what is painted${ln.none ? `, ${ln.none} of them with no lines` : ''})` : '')
          + (ln.not_seen ? `; ${ln.not_seen} not seen well enough: their lanes by the rules` : '')
          + (ln.traced ? `; on ${ln.traced} the lines are laid where they are painted (traced: their bends, drift, starts and stops)` : '')
          + '. They are in the Lanes tool as "from the satellite": set one by hand and yours wins.');
        (ln && ln.streets || []).filter(r => r.sides.join() !== r.rules.join()).slice(0, 12).forEach(r =>
          log(`    Street ${r.id}: ${r.sides[0] + r.sides[1] ? r.sides.join(' + ') + ' lanes' : 'no lines'} (the rules: ${r.rules.join(' + ')}).`));
        if(res.lanes_read_moved && res.lanes_read_moved.streets) log(`  Lane lines laid again on ${res.lanes_read_moved.streets} street(s) for their lanes (${res.lanes_read_moved.seconds.toFixed(1)} s).`);
      } else log(`Decals by the rules alone again: ${res.count.toLocaleString()} laid (${res.seconds.toFixed(1)} s)`
        + (res.lanes_read_moved && res.lanes_read_moved.streets ? `; the lanes by the rules again on ${res.lanes_read_moved.streets} street(s)` : '') + '.', 'ok');
      if(res.lanes_moved) log(`  Lane lines laid again on ${res.lanes_moved.streets} street(s) so they end at the markings (${res.lanes_moved.seconds.toFixed(1)} s).`);
      $('#decalNote').textContent = `${res.count.toLocaleString()} decals laid.`;
      try{ drawInto($('#lp-decals'), await loadImage(API + res.urls.decals)); }catch(_){}
      await showLayer(['result', 'decals', 'full'].includes(S.gen.layer) ? S.gen.layer : 'result', true);
      window.dispatchEvent(new Event('materials'));            // the 3D tab's scene is out of date
      if((S.gen.top && S.gen.top.gid === gid) || S.gen.topBusy) refreshTopView(gid, false);
    }catch(e){ log('Decals not placed: ' + e.message, 'bad'); }
    finally{ SAT.placing = false; status('Ready.'); renderDecals(); satPlaceState(); }
  }
  $('#btnSatPlace').addEventListener('click', () => satPlace(true));
  $('#btnSatRules').addEventListener('click', () => satPlace(false));
  $('#btnSatPlaced').addEventListener('click', async () => {
    if(!S.gen.result) return;
    try{ genVp.show(await loadImage(`${API}/api/coords/placed/${S.gen.result.id}?v=${Date.now().toString(36)}`),
                    'Decals: green as the satellite saw them, amber by the rules (red tiles: under the threshold)'); }
    catch(e){ log('Placed decals: ' + e.message, 'bad'); }
  });
  $('#btnCoords').addEventListener('click', coStart);
  $('#btnCoordsStop').addEventListener('click', () => { if(CO.job) api('/api/coords/cancel/' + CO.job, { method:'POST' }).catch(() => {}); });
  try{
    const saved = JSON.parse(localStorage.getItem('rta.coords') || '{}');
    CO_FIELDS.forEach(k => { if(saved[k] != null) $('#' + k).value = saved[k]; });
  }catch(_){}
  coSettings(false).then(() => { let m = 'alpha'; try{ m = localStorage.getItem('rta.inputMode') || 'alpha'; }catch(_){} setInputMode(m); });
  genRow.addEventListener('click', () => S.gen.mask ? genVp.show(S.gen.mask.img, `Mask, ${S.gen.mask.width} × ${S.gen.mask.height}`) : loadGen());
  $('#genMaskPrev').addEventListener('click', () => { if(S.gen.mask) genVp.show(S.gen.mask.img, 'Mask'); });

  function updateGen(){
    const ok = !!S.gen.mask;
    $('#btnGenerate').disabled = !ok;
    $('#btnGenJunctions').disabled = !ok;
    $('#genNote').textContent = ok ? 'Ready.' : 'Load a street mask.';
    renderParea();
  }
  $('#btnGenJunctions').addEventListener('click', async () => {
    if(!S.gen.mask) return;
    $('#btnGenJunctions').disabled = true;
    S.gen.junctions = await detectJunctions(S.gen.mask.id, 'Mask', genVp, null);
    renderInfo(); renderMemory();
    $('#btnGenJunctions').disabled = false;
  });
  function renderTextureLogs(){
    const gid=S.gen.result && S.gen.result.id;
    $('#textureLogs').hidden=!gid;
    if(gid){$('#textureSettingsLog').href='/api/texture/log/'+gid+'/settings';$('#textureUpdatesLog').href='/api/texture/log/'+gid+'/updates';}
  }
  function textureSnapshot(){
    const model = {...exportSettings()};
    delete model.decal_version;
    return JSON.parse(JSON.stringify({texture: { mask: S.gen.mask.id, scale: +$('#scale').value,
          wear: +$('#noiseAmt').value, seed: +$('#seed').value,
          marks: marksPayload(), cycle_m: MARKS.values['lane.dash_m'] + MARKS.values['lane.gap_m'],
          marking_width_m: MARKS.values['lane.width_m'],
          quality: +$('#quality').value, align_lines: S.gen.alignLines || false,
          output_scale: +$('#outScale').value, soft_edges: $('#softEdges').checked,
          grain: +$('#grainAmt').value, match_material: $('#matchMat').checked,
          line_width_mode: MARKS.opts.sizes === 'learned' ? 'learned' : 'fixed',
          materials: { street: $('#matStreet').value }, match_tone: $('#matTone').checked, lane_picks: LANES.picks,
          mark_area: MAREA.area,
          // made from coordinates (in the satellite's tiles): the asphalt's tone from the satellite, and
          // the confidence a tile needs for it (the card's "Rules in tiles under")
          ...(S.gen.mask.coords && S.gen.mask.coords.key
            ? { sat_tone: +$('#satTone').value, sat_threshold: Math.max(0, Math.min(100, Number($('#satThr').value) || 0)) } : {}) },
        model, layers: DEC.layers}));
  }
  async function updateTexture2D(quiet=false){
    if(!S.gen.result) return false;
    if(S.gen.updating) return false;
    S.gen.updating=true; $('#btnUpdate2d').disabled=true;
    const gid=S.gen.result.id;
    try{
      const res=await api('/api/texture/update',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({generation:gid,snapshot:textureSnapshot()}),job:'Updating the 2D texture'});
      if(S.gen.result.id!==gid) return false;
      if(res.generation_result){
        S.gen.result=res.generation_result; await loadIslands(S.gen.result);await loadStreets(S.gen.result);
      }
      Object.assign(S.gen.result.urls,res.urls || {});
      if(res.decals){S.gen.decals=res.decals;S.gen.decalVersion=Date.now();}
      if(res.islands) S.gen.islandTex={...res.islands,gid:S.gen.result.id};
      else if(S.gen.islandTex && res.urls.full) S.gen.islandTex.urls.full=res.urls.full;
      S.gen.updateRevision=res.revision;renderTextureLogs();
      if((res.parts || []).includes('surfaces')) texTilesRefresh(gid);
      S.gen.marksPainted=marksKey();S.gen.lanesPainted=JSON.stringify(LANES.picks);MAREA.painted=JSON.stringify(MAREA.area);
      if(res.changed.length){
        log(`2D texture updated: ${res.changed.map(c=>c.field).join(', ')}. Saved update ${res.revision}.`,'ok');
        if(S.gen.layer==='top'){S.gen.topStale=S.gen.result.id;refreshTopView(S.gen.result.id,true);}
        else await showLayer(S.gen.layer,true);
        await renderLayerThumbs();await loadStreets(S.gen.result);
        window.dispatchEvent(new Event('materials'));
      } else if(!quiet) log('2D texture: no values changed.','ok');
      renderMarks();renderDecals();renderSlots();drawBridges();return true;
    }catch(e){log('2D texture not updated: '+e.message,'bad');return false;}
    finally{S.gen.updating=false;$('#btnUpdate2d').disabled=!S.gen.result;status('Ready.');}
  }
  $('#btnUpdate2d').addEventListener('click',()=>updateTexture2D());
  window.updateTexture2D=updateTexture2D;

  $('#btnGenerate').addEventListener('click', async () => {
    if(!S.gen.mask) return;
    $('#btnGenerate').disabled = true; status('Generating…');
    log('Generating: junctions first, then material, wear and markings.');
    try{
      const res = await api('/api/generate', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Generating the texture',
        body: JSON.stringify({ ...textureSnapshot().texture, snapshot: textureSnapshot() }) });
      S.gen.result = res;
      S.gen.updateRevision=0;$('#btnUpdate2d').disabled=false;renderTextureLogs();
      S.gen.marksPainted = marksKey();                                     // the lines' settings the texture has
      S.gen.lanesPainted = JSON.stringify(res.lane_picks || []);         // the lane choices the texture has
      MAREA.painted = JSON.stringify(res.mark_area || null);               // and its markings area
      PAREA.built = pareaKey();                                             // and its project area
      renderParea();
      if(res.project_areas) log(`  Project Area: ${res.project_areas} island(s) built in as street: flat, the street material, no sidewalks or kerbs.`);
      renderMarea();
      S.gen.top = null; S.gen.decals = null;
      renderMarks();
      await loadIslands(res);
      await loadStreets(res);
      $('#lp-decals').innerHTML = '<div class="ph">—</div>';
      if(DEC.layers.length) await applyDecals();
      renderDecals();
      const s = res.summary;
      log(`Output ${s.size[0]} × ${s.size[1]} (${s.output_scale}× the mask${s.soft_edges ? ', smooth edges' : ''}), grain ${Math.round(s.grain*50)}${s.matched_tone != null ? `, colour matched to tone ${s.matched_tone}` : ''}.`);
      log(`  Road ${s.road_width_px} px wide (${s.road_width_m} m), ${s.junctions} junctions.`);
      if(s.street_material) log(`  Streets: ${s.street_material}, laid at its real size; your material's lighter and darker areas kept on it.`);
      if(s.tiles) texTilesLog(s.tiles);
      if(res.caches_freed > 0) log(`  Freed ${res.caches_freed >= 1e9 ? (res.caches_freed / 1e9).toFixed(1) + ' GB' : Math.round(res.caches_freed / 1e6) + ' MB'}: the speed-up files of earlier textures.`);
      Object.entries(res.decisions || {}).forEach(([k,d]) => log(`  ${k}: ${d.label || 'none'}${d.rank ? ` (position ${d.rank})` : ''}${d.exhausted ? ', all rejected' : ''}`));
      log(`  Lines ${s.markings.rules === 'guide' ? 'by the road marking guide' : 'all dashed alike'}${s.markings.yellow ? ', centre lines yellow' : ''}: `
        + `${s.markings.dashes} dashes placed, lines ${s.markings.width_px} px wide${
        s.markings.width_mode === 'learned' ? ` (${(s.markings.width_ratio*100).toFixed(1)}% of each street's width, learned)`
          : ' (fixed width)'}${s.markings.faint ? ': narrower than a pixel, so drawn faint, as a photo from this height shows them' : ''}.`);
      if(s.markings.lanes && Object.keys(s.markings.lanes).length)
        log('  Lanes (side 1 + side 2): ' + Object.entries(s.markings.lanes).sort()
          .map(([k, n]) => `${n} street${n === 1 ? '' : 's'} ${k === 'none' ? 'too narrow for lines' : k.endsWith(' highway')
            ? `${k.replace(' highway', '')} with an island` : k}`)
          .join(', ') + (s.markings.picked ? `; ${s.markings.picked} set in the Lanes tool` : '')
          + (s.markings.cut ? `, ${s.markings.cut} cut back to its limit` : '') + '.');
      if(s.markings.warning) log(`  Markings: ${s.markings.warning}.`, 'bad');
      if(res.junction_marks) log('  ' + junctionMarksText(res.junction_marks) + ' The lane lines stop at their markings.');
      if(MARKS.opts.sizes === 'learned' && s.markings.width_mode !== 'learned')
        log('  No line width has been learned yet: train on pairs whose photos show painted lines. Using the fixed width.', 'bad');
      (res.inner_streets || []).forEach(r => {
        if(r.problem){ log(`  Inner streets of placement ${r.placement} (${r.name}): ${r.problem}.`, 'bad'); return; }
        log(`  Inner streets of placement ${r.placement} (${r.name}): ${r.cells} space(s), ${r.road_m2} m² of road`
          + (r.connected ? `, ${r.connected} joined to a street` : '') + '.');
        const far = (r.dead_ends || 0) - (r.blocked || 0);
        if(far > 0) log(r.reach_m === 0
          ? `    ${far} street end(s) at the edge of the placement end there: Join streets up to is 0 m.`
          : `    ${far} street end(s) at the edge of the placement have no street straight ahead within ${r.reach_m ?? 200} m (Join streets up to): they end there.`, 'bad');
        if(r.blocked) log(`    ${r.blocked} street end(s) at the edge of the placement would run through objects on the way to the street: they end there.`, 'bad');
        if(r.narrowed) log(`    ${r.narrowed} space(s) had less room than a ${r.road_m} m road and two ${r.sidewalk_m} m sidewalks: the road came first, the sidewalks were narrowed${r.sidewalk_min > 0 ? ` (down to ${r.sidewalk_min} m)` : ' or left out'}.`);
        if(r.thin) log(`    ${r.thin} street(s) are narrower than ${r.thin_m} m (3 pixels at this scale, the narrowest ${r.road_min} m): they show only faintly in the texture. The 3D model has them as drawn. Wider gaps give wider streets.`, 'bad');
        if(!r.road_m2) log('    No road came out of these spaces: check that they are drawn between objects and not under them.', 'bad');
      });
      log('Done. Use the layer buttons to see each part on its own.', 'ok');
      S.review = null;
      S.defaultTarget = res.decisions && res.decisions.open
        ? { part:'open', key:res.decisions.open.key, features:{ part:'open', junction:false },
            route:{ id:res.decisions.open.route, label:res.decisions.open.label } }
        : null;
      setReviewButtons(true);
      $('#inspector').innerHTML = '<p class="empty">Click a spot to target a junction or kerb. Without a click, corrections apply to open road.</p>';
      $('#reviewNote').textContent = 'Choose what is wrong. Click a spot first if it is only a junction or kerb.';
      $('#btn3d').disabled = false;
      await showLayer('result');
      renderLayerThumbs();
      renderInfo();
      refreshTopView(res.id);
      await texTilesLoad(res);
    }catch(e){ log(e.message, 'bad'); $('#genNote').textContent = e.message; TEX.live = null; texCardState(); satDraw(); }
    status('Ready.'); $('#btnGenerate').disabled = false; renderMemory();
  });

  // the Top view: the 3D model of this texture (with the 3D model settings and the
  // materials), drawn from straight above, so the streets, sidewalks and kerbs show
  // with their materials. Shown once ready if the result is still on screen.
  // one at a time: asked again while one is being built (new island materials, say),
  // it is built once more when that one is done
  // stay: an update of the same texture (its lanes laid again): the layer on screen stays
  // A big map's 3D model takes long to build, and all of its memory: measured on a dense 12 x 9 km
  // city, about 45 s and 0.35 GB for every km² of road (23 min, 10.6 GB). Past BIG_MODEL_S its Top
  // view is built only when asked for (its layer button), never by itself after Generate texture or
  // a change, so the whole city is not being built while you work in 2D; Generate 3D scene asks first.
  // The time grows with the junctions (each one's crossings, corners, kerbs and sidewalks) more than
  // with the road's area: measured on seven maps (4 cores, sidewalks, blocks and junction markings),
  // 5 s + 0.376 s a junction + 20.3 s a km² of road, each within 6% (a 2.2 x 1.7 km map of narrow
  // streets, 1,079 junctions on 0.82 km² of road: 433 s, where the road's area alone said 37 s).
  // The larger of the two is taken. Past BIG_MODEL_S the 3D tab and the Top view build it in 1 km
  // patches (the part in view first); Download 3D model still gives one file while the road's area
  // alone says under BIG_MODEL_S (oneFile), as before.
  const BIG_MODEL_S = 60;
  const junctionCount = () => {
    const r = S.gen.result && S.gen.result.summary, j = S.gen.junctions && S.gen.junctions.summary;
    return r && r.junctions != null ? r.junctions : j && j.junctions != null ? j.junctions : null;
  };
  function modelCost(){
    const m = S.gen.mask;
    if(!m) return null;
    const mpp = mppMask(), roadPx = m.road_px || m.width * m.height * 0.28;
    const roadKm2 = roadPx * mpp * mpp / 1e6, byArea = 45 * roadKm2, nj = junctionCount();
    const s = nj == null ? byArea : Math.max(byArea, 5 + 0.376 * nj + 20.3 * roadKm2);
    return { s, big: s > BIG_MODEL_S, oneFile: byArea <= BIG_MODEL_S, junctions: nj,
             km: [m.width * mpp / 1000, m.height * mpp / 1000], roadKm2,
             gb: 0.5 + 0.35 * roadKm2,
             text: s < 90 ? `about ${Math.max(5, Math.round(s / 5) * 5)} s` : `about ${Math.round(s / 60)} min` };
  }
  window.modelCost = modelCost;
  const kmText = c => `${c.km[0].toFixed(1)} × ${c.km[1].toFixed(1)} km`;
  // after a new texture or a change: built again (small maps), else only marked to be built when asked
  function refreshTopView(gid, stay = false){
    const c = modelCost();
    if(!c || !c.big){ buildTopView(gid, stay); return; }
    const had = S.gen.top && S.gen.top.gid === gid;
    if(S.gen.topStale === gid) return;                                   // said so already
    S.gen.topStale = gid;
    $('#lp-top').innerHTML = `<div class="ph">${had ? 'out of date:<br>click to build again' : 'click to build'}<br>(${c.text})</div>`;
    log(had ? `Top view out of date: click its layer to build it again (${c.text}).`
      : `Top view not built by itself: this map is ${kmText(c)}${c.junctions != null ? ` with ${c.junctions.toLocaleString()} junctions` : ''}, `
        + `its whole 3D model would take ${c.text}; click the Top view layer `
        + 'to draw it from the model\'s 1 km patches as they are built (the 3D tab\'s).');
  }
  async function buildTopView(gid, stay = false){
    if(S.gen.topBusy){ S.gen.topAgain = gid; S.gen.topStay = stay && S.gen.topStay !== false; return; }
    S.gen.topBusy = true;
    try{ await buildTopViewNow(gid, stay); }
    finally{
      S.gen.topBusy = false;
      const again = S.gen.topAgain, st = !!S.gen.topStay; S.gen.topAgain = null; S.gen.topStay = null;
      if(again && S.gen.result && S.gen.result.id === again) buildTopView(again, st);
    }
  }
  async function buildTopViewNow(gid, stay){
    const big = modelCost();
    if(big && big.big) return buildTopViewPatches(gid, stay, big);
    $('#lp-top').innerHTML = '<div class="ph">building…</div>';
    log('Top view: building the 3D model to show the streets, sidewalks and kerbs with their materials…');
    const built = JSON.stringify(exportSettings());
    try{
      const res = await api('/api/export3d', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Building the top view',
        body: JSON.stringify({ ...exportSettings(), generation: gid, mesh: 'tiled' }) });
      if(!window.renderTopView) throw new Error('the top view drawing is not loaded');
      const canvas = await window.renderTopView(API + res.url, res.size_m);
      if(!S.gen.result || S.gen.result.id !== gid) return;               // a newer texture since
      S.gen.top = { canvas, gid, key: built };
      drawInto($('#lp-top'), canvas);
      const m = res.materials_used || {};
      const isl = res.blocks && res.blocks.island_slots ? Object.values(res.blocks.island_slots).map(v => v.name) : [];
      log(`Top view ready (${canvas.width} × ${canvas.height}, ${(res.size_m[0] / canvas.width * 100).toFixed(0)} cm per pixel): `
        + `streets ${m.street || 'your tiles'}, sidewalks ${m.sidewalk || 'your tiles'}, kerbs ${m.kerb || 'your tiles'}`
        + (isl.length ? `, islands ${isl.join(', ')}` : '') + '. Zoom in to see the materials.', 'ok');
      if(res.islands_warning) log('  Islands: ' + res.islands_warning + '.', 'bad');
      if(S.gen.layer === 'top' || (S.gen.layer === 'result' && !stay)) showLayer('top', S.gen.layer === 'top');
    }catch(e){
      $('#lp-top').innerHTML = '<div class="ph">—</div>';
      log('Top view not available: ' + e.message, 'bad');
    }
  }

  // A big map's top view, from its 3D model's 1 km patches (the 3D tab's, app/patches.py):
  // the whole map flat first, then each patch drawn into its square as it is built
  async function buildTopViewPatches(gid, stay, c){
    $('#lp-top').innerHTML = '<div class="ph">building…</div>';
    log(`Top view: this map is ${kmText(c)} (its whole 3D model would take ${c.text}), so it is drawn from its 3D model's patches of 1 km as they are built `
      + '(the same patches as the 3D tab\'s, the middle first).');
    const built = JSON.stringify(exportSettings());
    const job = trackJob('Top view from the 3D patches');
    let painter = null;
    try{
      if(!window.topViewPainter) throw new Error('the top view drawing is not loaded');
      let st = await api('/api/patches/start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...exportSettings(), generation: gid, mesh: 'tiled' }) });
      painter = window.topViewPainter(st.size_m);
      const canvas = painter.canvas, ctx = canvas.getContext('2d');
      const ground = await loadImage(API + st.ground).catch(() => null);
      if(ground) ctx.drawImage(ground, 0, 0, canvas.width, canvas.height);
      const drawn = new Set();
      for(;;){
        if(!S.gen.result || S.gen.result.id !== gid){ job.done(false, 'a newer texture'); return; }
        const todo = st.patches.filter(p => p.state === 'ready' && p.url && !drawn.has(p.id));
        for(const p of todo){ await painter.draw(API + p.url, p.m); drawn.add(p.id); }
        if(todo.length || !S.gen.top || S.gen.top.gid !== gid){
          S.gen.top = { canvas, gid, key: built };
          drawInto($('#lp-top'), canvas);
          if(S.gen.layer === 'top') showLayer('top', true);
        }
        const want = st.patches.filter(p => !['failed', 'dropped'].includes(p.state)).length;
        job.set(`${drawn.size} of ${st.total} patches drawn${st.ready < st.total ? `, ${st.ready} built` : ''}`,
          drawn.size / Math.max(1, st.total), st.left);
        if(st.finished && drawn.size >= want) break;
        if(st.error) throw new Error(st.error);
        await new Promise(r => setTimeout(r, 2000));
        st = await api('/api/patches/state/' + st.plan);
      }
      job.done(true, `${drawn.size} patches`);
      log(`Top view ready (${painter.canvas.width} × ${painter.canvas.height}, `
        + `${(st.patches.length ? c.km[0] * 1000 / painter.canvas.width * 100 : 0).toFixed(0)} cm per pixel), from ${drawn.size} patches.`, 'ok');
      if(S.gen.layer === 'top' || (S.gen.layer === 'result' && !stay)) showLayer('top', S.gen.layer === 'top');
    }catch(e){
      job.done(false, e.message);
      if(!(S.gen.top && S.gen.top.gid === gid)) $('#lp-top').innerHTML = '<div class="ph">—</div>';
      log('Top view not available: ' + e.message, 'bad');
    }finally{ if(painter) painter.dispose(); }
  }

  // the 3D model settings, shared by the export and the 3D tab's scene (ui/view3d.js)
  const exportSettings = () => ({ mesh: $('#meshMode').value,
          straightness: +$('#straight').value, spacing_m: +$('#rowSpacing').value,
          variation: +$('#variation').value, seed: +$('#seed').value,
          sidewalks: $('#swOn').checked, sw_height_cm: +$('#swHeight').value, sw_share: +$('#swShare').value,
          sw_min_m: +$('#swMin').value, sw_max_m: +$('#swMax').value,
          sw_skip_interchanges: $('#swSkip').checked,
          markings: $('#markMode').value,
          mesh_detail: $('#meshDetail').value,
          blocks: $('#blocksOn').checked,
          surface_detail: $('#surfDetail').value,
          materials: { street: $('#matStreet').value, sidewalk: $('#matSidewalk').value, kerb: $('#matKerb').value,
                       square: $('#matSquare').value },
          match_tone: $('#matTone').checked,
          island_slots: ISLS.slots,
          lane_picks: LANES.picks, mark_area: MAREA.area,
          decals: $('#decalMode').value, decal_version: S.gen.decalVersion || 0,
          scatter: S.gen.placements,
          bridges: S.gen.bridges, bridge_height_m: +$('#brHeight').value,
          bridge_ramp_m: +$('#brRamp').value, bridge_deck_m: +$('#brDeck').value });
  window.exportSettings = exportSettings;
  window.textureRevision = () => S.gen.updateRevision || 0;
  window.lastGeneration = () => S.gen.result ? S.gen.result.id : null;

  // Download 3D model: a window in the middle of the screen for the format (GLB, FBX,
  // OBJ) and what is combined; the choices are kept in this browser
  const EXPORT = (() => { try{ return { format: 'glb', ground: false, objects: false, front: '+y', patches: false, ...JSON.parse(localStorage.getItem('rta.export') || '{}') }; }
                          catch(_){ return { format: 'glb', ground: false, objects: false, front: '+y', patches: false }; } })();
  const FRONT_NAME = { '+y': '+Y', '+x': '+X', '-y': '-Y', '-x': '-X' };
  // the objects' front in the file: their own axes turned, every copy where it stood
  const exportFrontNote = () => { $('#exFrontNote').textContent = (EXPORT.front === '+y'
      ? 'Each object\'s own front is +Y, as it is here (Blender\'s green arrow). '
      : `Each object's own axes are turned so its front is ${FRONT_NAME[EXPORT.front]}; every object still stands where it is, facing its street. `)
    + (EXPORT.format === 'obj' ? 'OBJ keeps no object axes (every copy is written in place), so this changes nothing there.'
      : EXPORT.front === '+x' ? 'In Unreal, check once which way an imported object faces: its importer can turn files too (Force Front XAxis, Convert Scene).' : ''); };
  const exportNote = () => { $('#exNote').textContent = EXPORT.format === 'glb'
    ? 'GLB: one file with its textures inside. Opens in Blender, Unreal, Unity and three.js.'
    : `${EXPORT.format.toUpperCase()}: a zip with the model and a textures folder it refers to; unzip it before opening. Metres, Y up.`; };
  function openExport(){
    if(!S.gen.result) return;
    $('#exFormat').value = EXPORT.format; $('#exGround').checked = EXPORT.ground; $('#exObjects').checked = EXPORT.objects;
    $('#exFront').value = EXPORT.front; $('#exPatches').checked = !!EXPORT.patches;
    exportNote(); exportFrontNote(); $('#exportPop').hidden = false; $('#exFormat').focus();
  }
  const closeExport = () => { $('#exportPop').hidden = true; };
  ['#exFormat', '#exGround', '#exObjects', '#exFront', '#exPatches'].forEach(id => $(id).addEventListener('change', () => {
    EXPORT.format = $('#exFormat').value; EXPORT.ground = $('#exGround').checked; EXPORT.objects = $('#exObjects').checked;
    EXPORT.front = $('#exFront').value; EXPORT.patches = $('#exPatches').checked;
    try{ localStorage.setItem('rta.export', JSON.stringify(EXPORT)); }catch(_){}
    exportNote(); exportFrontNote();
  }));
  $('#exClose').addEventListener('click', closeExport);
  $('#exCancel').addEventListener('click', closeExport);
  $('#exportPop').addEventListener('pointerdown', e => { if(e.target === $('#exportPop')) closeExport(); });
  addEventListener('keydown', e => { if(e.key === 'Escape' && !$('#exportPop').hidden) closeExport(); });
  $('#btn3d').addEventListener('click', openExport);
  $('#exGo').addEventListener('click', () => { closeExport(); download3d({ ...EXPORT }); });

  async function download3d(choice){
    if(!S.gen.result) return;
    // one file as before, unless the road's area alone says the map is too big for one model
    const big = modelCost();
    if(big && !big.oneFile) return downloadPatches(big, choice);
    $('#btn3d').disabled = true; status('Building 3D model…');
    if(big && big.big) log(`Building the whole map as one 3D model to download: ${big.text}.`);
    log('Building the 3D model: tracing the road outline, triangulating, applying the texture…');
    try{
      const res = await api('/api/export3d', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Building the 3D model',
        body: JSON.stringify({ generation: S.gen.result.id, ...exportSettings() }) });
      if(res.mesh === 'tiled'){
        log(`Road model ready: ${res.quads.toLocaleString()} quads, ${res.materials} materials, ${res.dashes} dashes as their own strips, `
          + `${res.size_m[0]} × ${res.size_m[1]} m, ${(res.file_bytes/1048576).toFixed(1)} MB.`, 'ok');
        log(`  Tiles cover ${res.tile_m} m at ${res.mm_per_px} mm per pixel, laid along each street.`);
        if(res.mesh_detail === 'optimised' && res.rows_full)
          log(`  Optimised mesh: ${res.rows_kept.toLocaleString()} rows along streets instead of ${res.rows_full.toLocaleString()} (${Math.round(100 - 100*res.rows_kept/res.rows_full)}% fewer), 3 quads across instead of 4.`, 'ok');
        if(res.groups && res.groups.length){
          const tight = res.groups.filter(g => g.junctions > 1).length;
          log(`  ${tight} tight group(s) of junctions and ${res.groups.length - tight} other spot(s) covered by the traced outline (${res.fill_triangles} triangles), so no road is left bare.`);
        }
        (res.bridges || []).forEach(b => b.ok
          ? log(`  Bridge ${b.index + 1}: raised ${b.height} m over ${Math.round(b.s_out - b.s_in)} m, ramps ${Math.round(b.r_before)} m and ${Math.round(b.r_after)} m, steepest ${b.steepest_pct}%.`, 'ok')
          : log(`  Bridge ${b.index + 1}: ${(b.warnings || []).join('; ')}`, 'bad'));
        if(res.deck && res.deck.faces) log(`  Deck edges: ${res.deck.faces} faces, ${res.deck.thickness_m} m thick.`);
        if(res.scatter) log(`  Objects: ${res.scatter.copies} copies placed as instances`
          + (res.scatter.skipped_on_road ? `, ${res.scatter.skipped_on_road} left out because they were on the road` : '') + '.', 'ok');
        (res.scatter && res.scatter.problems || []).forEach(p => log('  Object left out: ' + p, 'bad'));
        if(res.streets_warning) log('  Note: ' + res.streets_warning + '.', 'bad');
        if(res.blocks && res.blocks.blocks)
          log(`  Blocks and islands: ${res.blocks.blocks} filled at ${Math.round(res.blocks.height_m*1000)/10} cm, a grid of ${res.blocks.cell_m} m quads (${(res.blocks.quads||0).toLocaleString()} quads)`
            + (res.blocks.kerb_faces ? `, with their own kerb faces (${res.blocks.kerb_faces.toLocaleString()})` : ', edges under the sidewalks') + '.');
        if(res.markings === 'painted')
          log(`  Markings painted into the road: ${res.painted_dashes} dashes on ${res.marked_quads.toLocaleString()} quads, `
            + `${res.marked_textures.length} marked texture(s) (${res.marked_textures.map(m => `${m.lines} line${m.lines === 1 ? '' : 's'} of ${m.width_cm} cm`).join(', ')}).`
            + (res.strip_dashes ? ` ${res.strip_dashes} stay as strips (highways' solid lines, bridge crossings).` : ''));
        else log(`  Markings: ${res.dashes} dashes and lines as separate strips.`);
        if(res.lane_islands)
          log(`  Highways: ${res.lane_islands.count} raised island(s), ${res.lane_islands.length_m} m in all (Median islands).`);
        if(res.lanes_cut)
          log(`  Lanes: ${res.lanes_cut} street(s) measured a little narrower in the 3D model than in the texture, so their chosen lanes were cut back to its limit there.`, 'bad');
        if(res.materials_used && Object.keys(res.materials_used).length)
          log('  Scanned materials: ' + Object.entries(res.materials_used).map(([p, n]) => `${p} ${n}`).join(', ') + '.', 'ok');
        if(res.surface && res.surface.length)
          log(`  Surface detail (bump and roughness): ${res.surface.map(s => s === 'grain' ? 'from the tiles\' grain' : s.replace(/^scan: /, 'scanned ')).join(', ')}.`);
        if(res.sidewalk) log(`  Sidewalks: ${res.sidewalk.top_quads.toLocaleString()} quads on top, raised ${Math.round(res.sidewalk.height_m*100)} cm, with ${res.sidewalk.kerb_faces.toLocaleString()} kerb faces.`);
        else if($('#swOn').checked) log('  No sidewalks: build the tiles first (Memory tab), so there is a sidewalk material.', 'bad');
        const rc = res.repeat_check;
        if(rc) log(`  Repetition check on a ${rc.street_m} m street: repeat peak ${rc.tiles_only} with tiles alone, `
          + `${rc.with_variation} with the variation layer (near 0 means not visible).`, rc.with_variation < 0.1 ? 'ok' : 'bad');
        log('  The variation layer is stored as vertex colours (glTF COLOR_0), which the glTF standard multiplies into the base colour. '
          + 'If the road looks evenly toned in Blender or Unreal, the importer did not connect them: multiply the Color Attribute (Blender) or Vertex Color (Unreal) into Base Color.');
      } else if(res.mesh === 'quads'){
        log(`Road model ready: ${res.quads.toLocaleString()} quads (${res.streets} streets, ${res.junctions} junctions), `
          + `${res.size_m[0]} × ${res.size_m[1]} m, ${(res.file_bytes/1048576).toFixed(1)} MB.`, 'ok');
        if(res.interchange_triangles) log(`  Interchanges stay as ${res.interchange_triangles} traced triangles for now: they are multi-level.`);
        log('  In Blender, select the mesh, Tab, then Face > Triangles to Quads (Alt+J) to see the quads.');
      } else {
        log(`Road model ready: ${res.triangles.toLocaleString()} triangles, ${res.size_m[0]} × ${res.size_m[1]} m, `
          + `${(res.file_bytes/1048576).toFixed(1)} MB.`, 'ok');
      }
      let url = res.url;
      const front = choice.front || '+y';
      if(choice.format !== 'glb' || choice.ground || choice.objects || front !== '+y'){
        status(`Writing ${choice.format.toUpperCase()}…`);
        const cv = await api('/api/export3d/convert', { method:'POST', headers:{'Content-Type':'application/json'},
          body: JSON.stringify({ generation: S.gen.result.id, format: choice.format, combine_ground: choice.ground, combine_objects: choice.objects, front }) });
        url = cv.url;
        log(`  ${choice.format.toUpperCase()}: ${cv.objects} object(s)`
          + (choice.ground ? ', the streets, sidewalks and kerbs as one ("Streets")' : '')
          + (choice.objects ? ', every placed object as one ("Objects")' : cv.copies ? `, ${cv.copies.toLocaleString()} object copies each on their own` : '')
          + (cv.file_bytes ? `, ${(cv.file_bytes / 1048576).toFixed(1)} MB` : '')
          + (choice.format === 'glb' ? '.' : ': a zip with the model and its textures folder; unzip it, then import the model.'), 'ok');
        if(front !== '+y' && choice.format !== 'obj') log(`  Objects' front: ${FRONT_NAME[front]}. Each object's own axes are turned so its front is ${FRONT_NAME[front]}; every object stands where it was, facing its street.`, 'ok');
      }
      log(choice.format === 'fbx' ? '  Units are metres, Y up. Blender: File > Import > FBX. Unreal: drag the .fbx into the Content Browser (its textures come along).'
        : choice.format === 'obj' ? '  Units are metres, Y up. Blender: File > Import > Wavefront (.obj). Unreal: drag the .obj into the Content Browser.'
        : '  Units are metres, Y up. Opens directly in Blender (File > Import > glTF) and Unreal (drag into the Content Browser).');
      const link = document.createElement('a');
      link.href = API + url; link.download = '';
      document.body.appendChild(link); link.click(); link.remove();
    }catch(e){ log('3D export failed: ' + e.message, 'bad'); }
    status('Ready.'); $('#btn3d').disabled = false; renderMemory();
  }

  // a big map's 3D model: its 1 km patches (the 3D tab's), each a GLB in the whole map's
  // coordinates, in one zip; those not built yet are built now (watch them in the 3D tab)
  async function downloadPatches(c, choice){
    choice = choice || { ...EXPORT };
    $('#btn3d').disabled = true;
    try{
      const st = await api('/api/patches/start', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ generation: S.gen.result.id, ...exportSettings() }) });
      if(choice.patches && (choice.format !== 'glb' || choice.ground || choice.objects))
        log('As patch files: the GLBs as they are built (the format and combine choices are for the one model).');
      if(st.ready < st.total){
        log(`This map is ${kmText(c)}: its 3D model comes in ${st.total} patches of 1 km, too big as one model. `
          + `${st.ready} of them are built; the rest are being built now`
          + (st.left ? ` (about ${fmtTime(Math.round(st.left))} left)` : '') + '. Press Download 3D model again when they are done '
          + '(the 3D tab shows them as they come).');
        return;
      }
      let href;
      if(choice.patches){
        log(`3D model in ${st.total} patches of 1 km: each a GLB in the whole map's coordinates (metres, the map's middle at 0), `
          + 'so imported together they fit as they are.', 'ok');
        href = `/api/patches/zip/${st.plan}`;
      } else {
        // one model by kind: the patches joined, each part one object
        status('Joining the patches into one model…');
        const m = await api(`/api/patches/model/${st.plan}`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
          job: 'Joining the patches into one 3D model',
          body: JSON.stringify({ format: choice.format, combine_ground: choice.ground, combine_objects: choice.objects, front: choice.front || '+y' }) });
        log(`3D model ready as one ${choice.format.toUpperCase()}: its ${m.patches} patches of 1 km joined, by kind`
          + (m.parts ? ` (${m.parts.join(', ')}${m.copies ? `, and ${m.copies.toLocaleString()} object copies` : ''})` : '')
          + (choice.ground ? '; the streets, sidewalks and kerbs as one ("Streets")' : '')
          + `, ${(m.file_bytes / 1048576).toFixed(0)} MB.`, 'ok');
        if(choice.format !== 'glb') log('  A zip with the model and its textures folder: unzip it, then import the model.');
        href = m.url;
      }
      const link = document.createElement('a');
      link.href = API + href; link.download = '';
      document.body.appendChild(link); link.click(); link.remove();
    }catch(e){ log('3D export failed: ' + e.message, 'bad'); }
    finally{ $('#btn3d').disabled = false; }
  }

  // keep: the same layer drawn again (its lanes laid again, say): the zoom and place stay
  async function showLayer(name, keep){
    const res = S.gen.result;
    if(!res) return;
    if(name === 'top' && (!(S.gen.top && S.gen.top.gid === res.id) || S.gen.topStale === res.id)){
      // not built (a big map) or out of date: built now that it is asked for
      if(S.gen.topStale === res.id && !S.gen.topBusy){
        const c = modelCost();
        S.gen.topStale = null;
        if(!c.big) log(`Top view: building the 3D model of the whole map (${kmText(c)}): ${c.text}.`);
        buildTopView(res.id);
      }
      if(!(S.gen.top && S.gen.top.gid === res.id)) return;
    }
    const it = S.gen.islandTex;
    if((name === 'islands' || name === 'full') && !it) return;
    if(name === 'decals' && !S.gen.decals) return;
    S.gen.layer = name;
    const img = name === 'top' ? S.gen.top.canvas
      : await loadImage(API + (name === 'decals' ? S.gen.decals.urls.decals : it && (name === 'islands' || name === 'full') ? it.urls[name] : res.urls[name]));
    genVp.show(img, name === 'top' ? `Top view of the 3D model, ${img.width} × ${img.height}`
      : name === 'islands' ? `Islands texture, ${img.width} × ${img.height} (the roads transparent)`
      : name === 'full' ? `Roads + islands, ${img.width} × ${img.height}`
      : name === 'decals' ? `Decals, ${img.width} × ${img.height}` : `${name}, ${img.width} × ${img.height}`,
      keep || ['islands', 'full', 'decals'].includes(name));
    drawBridges();
    $$('.layer').forEach(b => b.setAttribute('aria-pressed', b.dataset.layer === name));
  }

  async function renderLayerThumbs(){
    const res = S.gen.result;
    if(!res) return;
    for(const [key, el] of [['material','#lp-material'], ['wear','#lp-noise'], ['markings','#lp-markings']]){
      try{ drawInto($(el), await loadImage(API + res.urls[key])); }catch(e){}
    }
  }

  $$('.layer').forEach(b => b.addEventListener('click', () => {
    const map = { material:'material', noise:'wear', markings:'markings', top:'top', islands:'islands', full:'full', decals:'decals' };
    if(!S.gen.result) return;
    const name = map[b.dataset.layer];
    showLayer(S.gen.layer === name ? 'result' : name);
  }));

  /* ------------------------------------------------ review */
  genVp.canvas.addEventListener('click', async e => {
    if(GROUPS.picking || !S.gen.result || genVp.wasDrag && genVp.wasDrag()) return;
    const c = e.currentTarget, r = c.getBoundingClientRect();
    // in the texture's own pixels (the top view is drawn finer than the texture)
    const size = S.gen.result.summary && S.gen.result.summary.size;
    const tw = S.gen.layer === 'top' && size ? size[0] : c.width, th = S.gen.layer === 'top' && size ? size[1] : c.height;
    const x = Math.floor((e.clientX - r.left) / r.width * tw);
    const y = Math.floor((e.clientY - r.top) / r.height * th);
    try{
      const res = await api('/api/inspect', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ generation: S.gen.result.id, x, y }) });
      S.review = res.on_road ? res : null;
      renderInspector(res, x, y);
      setReviewButtons(!!res.on_road);
      $('#reviewNote').style.color = '';
      $('#reviewNote').textContent = res.on_road
        ? 'If this spot looks wrong, choose the problem and try the next route.'
        : res.reading;
    }catch(err){ log(err.message, 'bad'); }
  });

  function setReviewButtons(on){
    // the options unlock as soon as there is a result to judge
    const ready = !!S.gen.result;
    $('#problem').disabled = !ready;
    $('#btnAccept').disabled = !ready;
    $('#btnFlag').disabled = !ready;
    if(!ready){ $('#problem').value = ''; $('#correctionNote').textContent = ''; }
    // and the action stays out of reach until a problem is chosen
    $('#btnNext').disabled = !ready || !$('#problem').value;
  }

  $('#problem').addEventListener('change', () => {
    const c = (S.corrections || []).find(x => x.id === $('#problem').value);
    $('#correctionNote').textContent = c ? `Will try ${c.explain}.` : '';
    $('#btnNext').disabled = !$('#problem').value || !S.gen.result;
  });

  function renderInspector(res, x, y){
    if(!res.on_road){
      $('#inspector').innerHTML = `<p class="empty">${res.reading}</p>`;
      return;
    }
    const m = res.layers.material;
    const rt = m.route || {};
    const pct = Math.round((rt.confidence || 0) * 100);
    const rows = m.candidates.map(c => `<li class="${c.rejected ? 'tried' : (c.chosen ? 'cur' : '')}">
        <span>${c.rank}. ${c.label}</span>
        <span class="mono">${c.sitting_out ? 'sitting out' : (c.chosen ? 'in use' : Math.round((c.confidence||0)*100) + '%')}${
          c.rejections ? `, rejected ${c.rejections}×` : ''}</span></li>`).join('');
    $('#inspector').innerHTML = `
      <dl class="kv">
        <dt>Spot</dt><dd class="mono">${x}, ${y}</dd>
        <dt>Part</dt><dd>${res.part}${res.junction ? `, junction ${res.junction.id} (${res.junction.type})` : ''}</dd>
        <dt>To kerb</dt><dd class="mono">${res.edge_m} m</dd>
        <dt>Library</dt><dd>${rt.label || '—'}</dd>
        <dt>Position</dt><dd class="mono">${rt.rank ?? '—'} of ${m.candidates.length}</dd>
        <dt>Confidence</dt><dd class="mono">${pct}%<div class="bar"><i style="width:${pct}%;background:${
          pct>=70?'var(--green)':pct>=50?'var(--blue-ink)':'var(--red)'}"></i></div></dd>
        <dt>Patches</dt><dd class="mono">${rt.patches ?? '—'} of ${rt.size_px ?? '—'} px</dd>
      </dl>
      <div style="margin-top:10px;font-weight:600;font-size:12.5px">Read in this order for ${res.part}</div>
      <ul class="routes">${rows}</ul>
      ${m.exhausted ? '<p class="empty" style="color:var(--red);margin-top:8px">Every library has been rejected here. It is waiting for your decision.</p>' : ''}
      <div class="mono" style="margin-top:8px;font-size:11px;color:var(--faint)">${m.key}</div>`;
  }

  async function sendReview(outcome){
    const m = (S.review && S.review.layers.material) || S.defaultTarget;
    if(!m || !m.route || !m.route.id){ log('Nothing to review yet. Generate a result first.', 'bad'); return; }
    const correction = $('#problem').value;
    if(outcome === 'rejected' && !correction){
      $('#reviewNote').textContent = 'Choose what is wrong first.';
      $('#reviewNote').style.color = 'var(--red)';
      return;
    }
    $('#reviewNote').style.color = '';
    try{
      const res = await api('/api/review', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ part:m.part, key:m.key, features:m.features,
          route:m.route.id, outcome, correction: correction || null }) });
      const c = res.correction;
      const what = c && !c.rejects_library ? 'corrected' : outcome;
      log(`${c ? c.label : m.route.label + ' ' + outcome}${c ? ` (${m.part})` : ` for ${m.part}`}.`,
          outcome === 'accepted' ? 'ok' : (what === 'corrected' ? '' : 'bad'));
      if(res.moved){
        const mv = res.moved;
        log(`  Confidence now ${Math.round(mv.confidence*100)}%, position ${mv.rank_before} -> ${mv.rank_after}${
          outcome === 'rejected' ? `, rejected ${mv.rejections} time${mv.rejections===1?'':'s'}` : ''}.`);
        if(outcome === 'rejected' && mv.rank_after === mv.rank_before)
          log('  It sits out the next generation, then returns unless more rejections push it down.');
      }
      if(c && c.rejects_library) log(res.exhausted ? '  Every library has been rejected for this part.' : `  Next for ${m.part}: ${res.next_label}.`, res.exhausted ? 'bad' : '');
      // clear the choice so the same correction is not applied twice by accident
      $('#problem').value = ''; $('#correctionNote').textContent = ''; $('#btnNext').disabled = true;
      renderMemory();

      if(c){
        applyAdjustment(c);
        log(`  Applying: ${c.explain}. Generating again.`);
        $('#reviewNote').textContent = 'Correction applied. Generating again.';
        await $('#btnGenerate').click();
      } else if(outcome === 'accepted'){
        $('#reviewNote').textContent = 'Recorded. Click another spot to keep reviewing.';
      } else {
        $('#reviewNote').textContent = 'Flagged for your decision.';
      }
    }catch(e){ log(e.message, 'bad'); }
  }

  $('#grainAmt').addEventListener('input', () => { $('#grainVal').textContent = $('#grainAmt').value; });
  $('#straight').addEventListener('input', () => { $('#straightVal').textContent = $('#straight').value; });
  $('#meshMode').addEventListener('change', () => {
    const mode = $('#meshMode').value;
    $('#straight').disabled = mode === 'traced'; $('#rowSpacing').disabled = mode === 'traced';
    $('#variation').disabled = mode !== 'tiled';
  });
  $('#variation').addEventListener('input', () => { $('#variationVal').textContent = $('#variation').value; });

  function applyAdjustment(c){
    const adj = c.adjust || {};
    if(adj.grain){
      const el = $('#grainAmt');
      const before = +el.value;
      el.value = Math.max(0, Math.min(100, before + adj.grain));
      $('#grainVal').textContent = el.value;
      log(`  Surface grain ${before} -> ${el.value}.`);
    }
    if(adj.wear){
      const el = $('#noiseAmt');
      const before = +el.value;
      el.value = Math.max(0, Math.min(100, before + adj.wear));
      log(`  Wear ${before} -> ${el.value}.`);
    }
    if(adj.quality){
      const el = $('#quality');
      const before = +el.value;
      el.value = Math.max(1, Math.min(3, before + adj.quality));
      if(+el.value === before) log('  Quality is already at its highest.', 'bad');
      else log(`  Quality ${before} -> ${el.value}. This takes longer to generate.`);
    }
    if(adj.align_lines){
      if(S.gen.alignLines) log('  Lines are already being aligned.', 'bad');
      S.gen.alignLines = true;
    }
  }

  $('#btnNext').addEventListener('click', () => sendReview('rejected'));
  $('#btnAccept').addEventListener('click', () => sendReview('accepted'));
  $('#btnFlag').addEventListener('click', () => sendReview('flagged'));


  /* ------------------------------------------------ bridges */
  // Rectangles are kept in mask pixels, so they stay put whatever size the
  // viewer is showing (the mask, or a result generated at 2x or 4x).
  S.gen.bridges = []; S.gen.bridgePreview = []; S.gen.bridgeSel = -1;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const layer = $('#bridgeLayer');
  const shown = () => S.gen.mask ? genVp.canvas.width / S.gen.mask.width : 1;   // display pixels per mask pixel
  const mppMask = () => (+$('#scale').value || 1.1);

  function el(name, attrs, parent){
    const e = document.createElementNS(SVGNS, name);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    if(parent) parent.appendChild(e);
    return e;
  }
  function corners(b){
    const a = b.angle * Math.PI / 180, ux = Math.cos(a), uy = Math.sin(a), vx = -uy, vy = ux;
    const L = b.length / 2, W = b.width / 2;
    return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([i, j]) => [b.cx + ux*L*i + vx*W*j, b.cy + uy*L*i + vy*W*j]);
  }

  // The island outlines and numbers, the lanes' lines and labels and the objects of automatic
  // placements: thousands of shapes on a city
  // (1,500 islands, 1,000 streets), drawn on a canvas the size of the view, between the texture and
  // the handles: only what is in view, again after each move or zoom (a frame or two), crisp at any
  // zoom. As SVG the browser drew every one of them again at each zoom step: 2 s a step on a city.
  const ovl = document.createElement('canvas');
  ovl.className = 'vp-overlay';
  $('#genVp .stage').insertBefore(ovl, layer);
  let ovlFrame = 0;
  const scheduleOverlay = () => { if(!ovlFrame) ovlFrame = requestAnimationFrame(() => { ovlFrame = 0; drawOverlay(); }); };
  $('#genVp').addEventListener('vpview', scheduleOverlay);
  new ResizeObserver(scheduleOverlay).observe($('#genVp'));             // resized, or shown again (its tab)
  function drawOverlay(){
    const want = !!S.gen.mask && ((ISLS.data && (ISLS.show || ISLS.pick >= 0 || AUTO.pick >= 0)) || (LANES.on && LANES.streets) || hasAuto()
                                  || (PAREA.data && (PAREA.on || PAREA.sel.size > 0)) || layShown());
    ovl.hidden = !want;
    if(!want) return;
    const root = $('#genVp'), r = root.getBoundingClientRect(), { s, tx, ty } = genVp.view();
    const dpr = window.devicePixelRatio || 1, W = Math.max(1, Math.round(r.width * dpr)), H = Math.max(1, Math.round(r.height * dpr));
    if(ovl.width !== W || ovl.height !== H){ ovl.width = W; ovl.height = H; }
    // the stage is moved and scaled (tx, ty, s): the overlay undoes it, so it lies on the view
    ovl.style.width = r.width + 'px'; ovl.style.height = r.height + 'px';
    ovl.style.transform = `translate(${-tx / s}px, ${-ty / s}px) scale(${1 / s})`;
    const ctx = ovl.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, W, H);
    // mask pixels to the overlay's: x * k + ox; what of the mask is in view; the handle size the
    // SVG drawing uses (a share of the texture's width) as it shows on screen
    const f = shown(), k = dpr * s * f, ox = dpr * tx, oy = dpr * ty;
    const v = { k, ox, oy, x0: -ox / k, y0: -oy / k, x1: (W - ox) / k, y1: (H - oy) / k, dpr,
                hr: Math.max(4, genVp.canvas.width / 260) * dpr * s, minFont: 7 * dpr };
    drawIslandsOn(ctx, v);
    drawPareaOn(ctx, v);
    drawBuildingsOn(ctx, v);
    drawAutoOn(ctx, v);
    drawLanesOn(ctx, v);
    drawHighwayGroups(ctx, v);
  }
  // a line (mask pixels) as a path on the overlay, its points closer than half a pixel there left out
  function tracePath(ctx, v, pts, close){
    let lx = Infinity, ly = Infinity;
    pts.forEach((q, i) => {
      const x = q[0] * v.k + v.ox, y = q[1] * v.k + v.oy;
      if(i === 0) ctx.moveTo(x, y);
      else if(Math.abs(x - lx) + Math.abs(y - ly) >= 0.5 || i === pts.length - 1) ctx.lineTo(x, y);
      else return;
      lx = x; ly = y;
    });
    if(close) ctx.closePath();
  }
  const inView = (v, b, pad = 0) => !(b[2] + pad < v.x0 || b[0] - pad > v.x1 || b[3] + pad < v.y0 || b[1] - pad > v.y1);
  function overlayLabels(ctx, labels, outline){
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
    labels.forEach(([x, y, size, text, fill]) => {
      ctx.font = `700 ${size}px system-ui, sans-serif`;
      ctx.lineWidth = size * outline; ctx.strokeStyle = 'rgba(0,0,0,.8)'; ctx.strokeText(text, x, y);
      ctx.fillStyle = fill || '#fff'; ctx.fillText(text, x, y);
    });
  }

  function drawBridges(){
    const f = shown();
    layer.setAttribute('width', genVp.canvas.width); layer.setAttribute('height', genVp.canvas.height);
    layer.innerHTML = '';
    drawOverlay();
    if(!S.gen.mask) return;
    drawLanes(f);
    drawMarea(f, Math.max(4, genVp.canvas.width / 260));
    const road = Math.max(3, 9 / mppMask() * f);                  // roughly a road's width, for the lines
    // ramps and deck along the road, from the last preview
    S.gen.bridgePreview.forEach(p => {
      if(!p.ok) return;
      p.ramps.forEach(r => r.length > 1 && el('polyline', { points: r.map(q => `${q[0]*f},${q[1]*f}`).join(' '),
        fill:'none', stroke:'rgba(175,178,182,.6)', 'stroke-width': road, 'stroke-linecap':'round', 'stroke-linejoin':'round' }, layer));
      p.deck.length > 1 && el('polyline', { points: p.deck.map(q => `${q[0]*f},${q[1]*f}`).join(' '),
        fill:'none', stroke:'rgba(216,96,76,.65)', 'stroke-width': road, 'stroke-linecap':'butt' }, layer);
    });
    const hr = Math.max(4, genVp.canvas.width / 260);
    S.gen.bridges.forEach((b, i) => {
      const pts = corners(b).map(q => [q[0]*f, q[1]*f]);
      const sel = i === S.gen.bridgeSel;
      const g = el('g', {}, layer);
      const poly = el('polygon', { points: pts.map(q => q.join(',')).join(' '), class:'grab',
        fill:'rgba(216,96,76,.12)', stroke:'#D8604C', 'stroke-width': sel ? hr*0.7 : hr*0.45 }, g);
      poly.addEventListener('pointerdown', e => startDrag(e, i, 'move'));
      pts.forEach((q, k) => {
        const c = el('circle', { cx:q[0], cy:q[1], r:hr, class:'corner', fill:'#fff', stroke:'#D8604C', 'stroke-width': hr*0.35 }, g);
        c.addEventListener('pointerdown', e => startDrag(e, i, 'size'));
      });
      const a = b.angle * Math.PI / 180;
      const rx = (b.cx + Math.cos(a) * (b.length/2)) * f + Math.cos(a) * hr * 5;
      const ry = (b.cy + Math.sin(a) * (b.length/2)) * f + Math.sin(a) * hr * 5;
      el('line', { x1:(b.cx + Math.cos(a)*b.length/2)*f, y1:(b.cy + Math.sin(a)*b.length/2)*f, x2:rx, y2:ry,
        stroke:'#D8604C', 'stroke-width': hr*0.35 }, g);
      const spin = el('circle', { cx:rx, cy:ry, r:hr*1.15, class:'spin', fill:'#D8604C' }, g);
      spin.addEventListener('pointerdown', e => startDrag(e, i, 'rotate'));
    });
    renderBridgeList();
    drawScatter(f, hr);
  }

  function toMask(e){
    const r = genVp.canvas.getBoundingClientRect(), f = shown();
    return [(e.clientX - r.left) / r.width * genVp.canvas.width / f,
            (e.clientY - r.top) / r.height * genVp.canvas.height / f];
  }

  let drag = null;
  function startDrag(e, i, mode){
    e.stopPropagation(); e.preventDefault();
    S.gen.bridgeSel = i;
    const b = S.gen.bridges[i];
    drag = { i, mode, start: toMask(e), orig: { ...b } };
  }
  addEventListener('pointermove', e => {
    if(!drag) return;
    const b = S.gen.bridges[drag.i], o = drag.orig, p = toMask(e);
    if(drag.mode === 'move'){
      b.cx = o.cx + p[0] - drag.start[0]; b.cy = o.cy + p[1] - drag.start[1];
    } else if(drag.mode === 'size'){
      const a = o.angle * Math.PI / 180, dx = p[0] - o.cx, dy = p[1] - o.cy;
      b.length = Math.max(4, 2 * Math.abs(dx*Math.cos(a) + dy*Math.sin(a)));
      b.width = Math.max(4, 2 * Math.abs(-dx*Math.sin(a) + dy*Math.cos(a)));
    } else {
      b.angle = Math.atan2(p[1] - o.cy, p[0] - o.cx) * 180 / Math.PI;
    }
    drawBridges();
  });
  addEventListener('pointerup', () => {
    if(!drag) return;
    drag = null;
    bridgesChanged();
  });

  let previewTimer = null;
  function bridgesChanged(){
    drawBridges();
    if(!S.gen.mask) return;
    api('/api/bridges', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mask: S.gen.mask.id, bridges: S.gen.bridges }) }).catch(() => {});
    clearTimeout(previewTimer);
    previewTimer = setTimeout(previewBridges, 250);
  }

  async function previewBridges(){
    if(!S.gen.mask || !S.gen.bridges.length){ S.gen.bridgePreview = []; drawBridges(); return; }
    status('Finding the bridge roads…');
    try{
      const res = await api('/api/bridges/preview', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ mask: S.gen.mask.id, scale: mppMask(), bridges: S.gen.bridges,
          straightness: +$('#straight').value, spacing_m: +$('#rowSpacing').value,
          height_m: +$('#brHeight').value, ramp_m: +$('#brRamp').value }) });
      S.gen.bridgePreview = res.bridges;
      res.bridges.forEach(p => {
        if(!p.ok) log(`Bridge ${p.index + 1}: ${p.warnings.join('; ')}`, 'bad');
        else {
          log(`Bridge ${p.index + 1}: deck ${p.deck_m} m, ramps ${p.ramp_m.join(' m and ')} m, steepest ${p.steepest_pct}%.`, 'ok');
          p.warnings.forEach(w => log('  ' + w, 'bad'));
        }
      });
    }catch(e){ log('Bridge preview failed: ' + e.message, 'bad'); }
    drawBridges(); status('Ready.');
  }

  function renderBridgeList(){
    const list = $('#bridgeList');
    $('#bridgeCount').textContent = S.gen.bridges.length ? `${S.gen.bridges.length}` : '';
    list.innerHTML = '';
    S.gen.bridges.forEach((b, i) => {
      const p = S.gen.bridgePreview.find(q => q.index === i);
      const row = document.createElement('div');
      row.className = 'bridge-item' + (i === S.gen.bridgeSel ? ' sel' : '');
      row.innerHTML = `<span>Bridge ${i + 1}<span style="color:var(--muted)"> · ${
        p ? (p.ok ? `${p.deck_m} m deck` : 'no road found') : `${Math.round(b.length * mppMask())} m`}</span></span>
        <button class="x" title="Remove bridge" aria-label="Remove bridge ${i + 1}">×</button>`;
      row.addEventListener('click', () => { S.gen.bridgeSel = i; drawBridges(); });
      row.querySelector('.x').addEventListener('click', e => {
        e.stopPropagation();
        S.gen.bridges.splice(i, 1); S.gen.bridgeSel = -1;
        log(`Removed bridge ${i + 1}.`); bridgesChanged();
      });
      list.appendChild(row);
    });
  }

  $('#btnAddBridge').addEventListener('click', () => {
    if(!S.gen.mask) return;
    const m = mppMask();
    // start in the middle of what is on screen
    const r = $('#genVp').getBoundingClientRect(), c = genVp.canvas.getBoundingClientRect(), f = shown();
    const cx = ((r.left + r.width/2) - c.left) / c.width * genVp.canvas.width / f;
    const cy = ((r.top + r.height/2) - c.top) / c.height * genVp.canvas.height / f;
    S.gen.bridges.push({ cx: Math.min(Math.max(cx, 0), S.gen.mask.width), cy: Math.min(Math.max(cy, 0), S.gen.mask.height),
                         length: 60 / m, width: 25 / m, angle: 0 });
    S.gen.bridgeSel = S.gen.bridges.length - 1;
    log(`Bridge ${S.gen.bridges.length} added. Place it along the road that should go over.`);
    bridgesChanged();
  });
  ['#brHeight', '#brRamp'].forEach(id => $(id).addEventListener('change', () => bridgesChanged()));

  // redraw whenever the viewer shows a different image or size
  const _show = genVp.show;
  genVp.show = (...args) => { _show(...args); drawBridges(); };


  /* ------------------------------------------------ islands */
  // The areas between the roads, numbered by Generate texture (app/islands.py), drawn on
  // the map with their numbers. Material slots under Blocks and islands: each a material
  // (a ball, as the other parts) and the islands picked for it; Pick, then click islands.
  const ISLAND_COLOURS = ['230,159,0', '86,180,233', '0,158,115', '240,228,66', '204,121,167', '213,94,0', '0,114,178', '170,170,170'];
  const slotColour = i => ISLAND_COLOURS[i % ISLAND_COLOURS.length];
  let slotsTimer = null, slotsMask = null;
  // the island a point (mask pixels) lies in: its outline, even-odd over its rings
  function islandAt(x, y){ return islandIn(ISLS.data, x, y); }
  function islandIn(d, x, y){
    if(!d) return 0;
    for(const isl of d.islands){
      const b = isl.bbox;
      if(x < b[0] || y < b[1] || x > b[2] || y > b[3]) continue;
      let inside = false;
      for(const r of isl.rings) for(let i = 0, j = r.length - 1; i < r.length; j = i++){
        const [xi, yi] = r[i], [xj, yj] = r[j];
        if((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
      }
      if(inside) return isl.id;
    }
    return 0;
  }
  // each slot's islands (numbers), and each island's slot (a later slot wins, as the server)
  function slotIslands(){
    const of = new Map(), lists = ISLS.slots.map(() => []);
    if(!ISLS.data) return { of, lists };
    ISLS.slots.forEach((s, i) => (s.picks || []).forEach(p => { const k = islandAt(p[0], p[1]); if(k) of.set(k, i); }));
    of.forEach((i, k) => lists[i].push(k));
    lists.forEach(l => l.sort((a, b) => a - b));
    return { of, lists };
  }
  // "1, 4, 7-9": compact, and back
  function idText(ids){
    const out = [];
    for(let i = 0; i < ids.length; i++){
      let j = i; while(j + 1 < ids.length && ids[j + 1] === ids[j] + 1) j++;
      out.push(j > i + 1 ? `${ids[i]}-${ids[j]}` : j === i + 1 ? `${ids[i]}, ${ids[j]}` : `${ids[i]}`); i = j;
    }
    return out.join(', ');
  }
  function parseIds(text){
    const n = ISLS.data ? ISLS.data.count : 0, out = new Set();
    for(const part of String(text).split(/[,;\s]+/)){
      const m = part.match(/^(\d+)(?:-(\d+))?$/); if(!m) continue;
      const a = +m[1], b = m[2] ? +m[2] : a;
      for(let k = Math.min(a, b); k <= Math.max(a, b) && k <= n; k++) if(k >= 1) out.add(k);
    }
    return [...out];
  }
  const islandById = id => ISLS.data && ISLS.data.islands[id - 1] && ISLS.data.islands[id - 1].id === id
    ? ISLS.data.islands[id - 1] : (ISLS.data ? ISLS.data.islands.find(i => i.id === id) : null);
  // give islands to a slot (or to none, slot -1): out of every other slot first
  function assign(ids, slot){
    const drop = new Set(ids);
    ISLS.slots.forEach(s => { s.picks = (s.picks || []).filter(p => !drop.has(islandAt(p[0], p[1]))); });
    if(slot >= 0 && ISLS.slots[slot]) ids.forEach(k => { const isl = islandById(k); if(isl) ISLS.slots[slot].picks.push(isl.pt.slice()); });
  }
  function renderSlots(){
    const box = $('#islandSlots'); box.innerHTML = '';
    const { lists } = slotIslands();
    ISLS.slots.forEach((s, i) => {
      const row = document.createElement('div'); row.className = 'isl-slot';
      const cur = slotMaterial(i), n = lists[i].length;
      row.innerHTML = `<span class="isl-sw" style="background:rgb(${slotColour(i)})"></span>
        <button type="button" class="matChip" data-part="island" data-slot="${i}"><img alt=""><span class="mc-t"><small></small><b></b></span></button>
        <button type="button" class="isl-pick" aria-pressed="${ISLS.pick === i}">Pick</button>
        <button type="button" class="isl-x" title="Remove this slot (its islands go back to the Squares material)">×</button>
        <input class="isl-ids" spellcheck="false" placeholder="${ISLS.data ? 'no islands: Pick, or type numbers' : 'Generate texture to number the islands'}">
        <div class="isl-mix" hidden>
          <button type="button" class="matChip isl-mixchip" data-part="island" data-slot="${i}" data-mix="1"><img alt=""><span class="mc-t"><small>In patches through it</small><b></b></span></button>
          <label title="How much of the ground the second material covers, in patches">Amount <output></output><input type="range" class="isl-amt" min="0" max="80" step="5"></label>
          <label title="How big its patches are, about">Patches <select class="isl-size"><option value="4">4 m</option><option value="8">8 m</option><option value="10">10 m</option><option value="16">16 m</option><option value="25">25 m</option><option value="40">40 m</option></select></label>
        </div>`;
      const chip = $('.matChip', row);
      $('small', chip).textContent = `Slot ${i + 1} · ${n} island${n === 1 ? '' : 's'}`;
      $('b', chip).textContent = cur.name;
      if(cur.ball) $('img', chip).src = cur.ball;
      chip.title = cur.m ? `${cur.m.title} by ${(cur.m.authors || []).join(', ')}, Poly Haven, ${cur.m.licence}; ${cur.m.size_m[0]} m across. Click to change`
                         : 'Laid as the islands in no slot. Click to choose a material, such as sand';
      chip.addEventListener('click', e => { e.stopPropagation(); window.openMatPop(chip); });
      const pickBtn = $('.isl-pick', row);
      pickBtn.disabled = !ISLS.data;
      pickBtn.title = ISLS.data ? 'Click islands on the map to give them this material; click one again to take it out' : 'Generate texture first: it numbers the islands';
      pickBtn.addEventListener('click', () => setPick(ISLS.pick === i ? -1 : i));
      $('.isl-x', row).addEventListener('click', () => {
        ISLS.slots.splice(i, 1);
        if(ISLS.pick === i) ISLS.pick = -1; else if(ISLS.pick > i) ISLS.pick--;
        log(`Island material slot ${i + 1} removed.`); ISLS.changed();
      });
      // ground: a second material in patches (gravel through sand), the colour and shine drifting
      const mp = mixPartner(i), mixBox = $('.isl-mix', row);
      if(mp){
        mixBox.hidden = false;
        const mchip = $('.isl-mixchip', row);
        $('b', mchip).textContent = mp.m ? (mp.auto ? `${mp.m.name} (suggested)` : mp.m.name) : 'None';
        if(mp.m) $('img', mchip).src = API + `/api/materials/${mp.m.id}/ball`; else $('img', mchip).style.visibility = 'hidden';
        mchip.title = 'A second ground material in patches through these islands, the higher stones of either showing at the edges (3D tab and Render photo). Click to change';
        mchip.addEventListener('click', e => { e.stopPropagation(); window.openMatPop(mchip); });
        const amt = $('.isl-amt', row), out = $('output', amt.parentNode), size = $('.isl-size', row);
        amt.value = Math.round(mp.mix.amount * 100); out.textContent = amt.value + '%';
        amt.disabled = size.disabled = !mp.m;
        size.value = String(mp.mix.size_m);
        if(size.value !== String(mp.mix.size_m)) size.add(new Option(`${mp.mix.size_m} m`, String(mp.mix.size_m), true, true));
        amt.addEventListener('input', () => { out.textContent = amt.value + '%'; });
        amt.addEventListener('change', () => { ISLS.slots[i].mix = Object.assign(slotMix(i), { amount: +amt.value / 100 }); ISLS.changed(); });
        size.addEventListener('change', () => { ISLS.slots[i].mix = Object.assign(slotMix(i), { size_m: +size.value }); ISLS.changed(); });
      }
      const ids = $('.isl-ids', row);
      ids.value = idText(lists[i]); ids.disabled = !ISLS.data;
      ids.title = 'The islands in this slot, by number: type them (such as 3, 7, 12-15) and press Enter';
      ids.addEventListener('change', () => {
        const want = parseIds(ids.value), gone = lists[i].filter(k => !want.includes(k));
        assign(gone, -1); assign(want, i);
        log(`Slot ${i + 1}: ${want.length} island(s).`); ISLS.changed();
      });
      box.appendChild(row);
    });
    const d = ISLS.data;
    $('#islandCount').textContent = d ? `${d.count} island${d.count === 1 ? '' : 's'}` : 'no islands yet';
    $('#btnShowIslands').disabled = !d;
    $('#btnShowIslands').setAttribute('aria-pressed', ISLS.show);
    $('#btnIslandsTex').disabled = !d || !S.gen.result;
    // made once: the button updates it, laying again only the islands whose material changed
    const itx = S.gen.islandTex && S.gen.result && S.gen.islandTex.gid === S.gen.result.id;
    $('#btnIslandsTex').textContent = itx ? 'Update islands texture' : 'Generate islands texture';
    $('#btnIslandsTex').classList.toggle('attention', !!(itx && S.gen.islandTex.slots !== JSON.stringify(ISLS.slots)));
    if(d){
      const free = d.count - [...slotIslands().of.keys()].length;
      $('#islandNote').textContent = ISLS.pick >= 0
        ? `Picking for slot ${ISLS.pick + 1}: click islands on the map to add them, click again to take them out. Pick or Esc to finish.`
        : `${free} island(s) in no slot: they use the Squares, blocks and islands material (${materialOf('square').name}). `
          + 'Generate islands texture makes a texture of only the islands; the 3D model and Top view use the slots too.';
    }
  }
  function setPick(i){
    if(i >= 0 && LANES.on) lanesMode(false);
    if(i >= 0){ AUTO.pick = -1; pareaStop(); }
    ISLS.pick = i;
    if(i >= 0) log(`Pick islands for slot ${i + 1} (${slotMaterial(i).name}): click an island to add it, click it again to take it out.`);
    renderSlots(); drawBridges();
  }
  ISLS.changed = quiet => {
    renderSlots(); drawBridges();
    S.gen.placements.forEach(p => {
      if(p.auto && autoOf(p).islands === 'slot' && autoLaidFor.has(p) && autoLaidFor.get(p) !== JSON.stringify(autoIslands(p))) relayAutoSoon(p);
    });
    if(quiet === true) return;
    window.dispatchEvent(new Event('materials'));
    if(!S.gen.mask) return;
    clearTimeout(slotsTimer);
    const mask = S.gen.mask.id, slots = JSON.parse(JSON.stringify(ISLS.slots));
    slotsTimer = setTimeout(() => api('/api/islands', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mask, slots }) }).catch(e => log('Island slots not saved: ' + e.message, 'bad')), 300);
  };
  window.addEventListener('materials', () => renderSlots());
  // the slots of a mask, kept on the server with its picture
  async function loadSlots(maskId){
    slotsMask = maskId; ISLS.slots = []; ISLS.data = null; ISLS.pick = -1;
    try{
      const r = await api('/api/islands?mask=' + maskId);
      if(slotsMask !== maskId) return;
      ISLS.slots = (r.slots || []).map(s => ({ material: s.material, picks: s.picks || [] }));
      if(ISLS.slots.length) log(`${ISLS.slots.length} island material slot(s) restored for this mask. Generate texture to number its islands.`);
    }catch(_){}
    renderSlots();
  }
  // the numbered islands of a texture just generated
  async function loadIslands(res){
    ISLS.data = null; ISLS.pick = -1; S.gen.islandTex = null;
    ['#lp-islands', '#lp-full'].forEach(id => { $(id).innerHTML = '<div class="ph">—</div>'; });
    $('#islandSaves').hidden = true;
    if(res.islands && res.islands.url){
      try{
        ISLS.data = await api(res.islands.url);
        const { lists } = slotIslands();
        log(`Islands: ${ISLS.data.count} found between the roads, numbered 1 to ${ISLS.data.count} in reading order`
          + (ISLS.slots.length ? `; slots: ${lists.map((l, i) => `${i + 1}: ${l.length}`).join(', ')}` : '')
          + '. Show islands to see them, or add a slot under Blocks and islands and Pick.', 'ok');
      }catch(e){ log('Islands not loaded: ' + e.message, 'bad'); }
    }
    renderSlots(); drawBridges();
    // automatic placements: laid again on these islands
    if(ISLS.data) S.gen.placements.forEach(p => { if(p.auto && p.auto.gen !== res.id) relayAuto(p); });
  }
  // the islands' outlines (each in its slot's colour) and numbers, on the overlay (drawOverlay)
  function drawIslandsOn(ctx, v){
    const d = ISLS.data;
    if(!d || !(ISLS.show || ISLS.pick >= 0 || AUTO.pick >= 0)) return;
    const { of } = slotIslands(), labels = [], mpp = mppMask();
    // picking islands for an automatic placement: its islands blue, the rest plain
    const au = AUTO.pick >= 0 && S.gen.placements[AUTO.pick] ? new Set(autoIslands(S.gen.placements[AUTO.pick]) || []) : null;
    for(const isl of d.islands){
      if(!inView(v, isl.bbox)) continue;
      const s = au ? -1 : of.has(isl.id) ? of.get(isl.id) : -1, mine = au ? au.has(isl.id) : ISLS.pick >= 0 && s === ISLS.pick;
      ctx.beginPath();
      isl.rings.forEach(r => tracePath(ctx, v, r, true));
      ctx.fillStyle = au ? (mine ? 'rgba(70,130,220,.5)' : 'rgba(255,255,255,.07)')
        : s >= 0 ? `rgba(${slotColour(s)},${mine ? .62 : .42})` : (ISLS.pick >= 0 ? 'rgba(255,255,255,.07)' : 'rgba(255,255,255,.03)');
      ctx.fill('evenodd');
      ctx.strokeStyle = au ? (mine ? '#4682DC' : 'rgba(255,190,60,.9)') : s >= 0 ? `rgb(${slotColour(s)})` : 'rgba(255,190,60,.9)';
      ctx.lineWidth = Math.max(v.dpr, v.hr * (mine ? 0.45 : 0.25));
      ctx.stroke();
      // its number where it is widest, sized to fit; none too small to read
      const r = isl.r_m / mpp * v.k, fs = Math.max(v.hr * 1.1, Math.min(v.hr * 3.2, r * 0.9));
      if(r > v.hr * 0.5 && fs >= v.minFont) labels.push([isl.pt[0] * v.k + v.ox, isl.pt[1] * v.k + v.oy, fs, String(isl.id)]);
    }
    overlayLabels(ctx, labels, 0.18);
  }
  // Pick: a click on an island puts it in the slot being picked (or takes it out)
  function pickIsland(id){
    const isl = ISLS.data.islands.find(q => q.id === id);
    if(!isl) return;
    const { of } = slotIslands(), s = of.has(id) ? of.get(id) : -1, slot = ISLS.pick, had = s === slot;
    assign([id], had ? -1 : slot);
    log(had ? `Island ${id} taken out of slot ${slot + 1}.` : `Island ${id} (${Math.round(isl.area_m2).toLocaleString()} m²) into slot ${slot + 1}: ${slotMaterial(slot).name}.`);
    ISLS.changed();
  }
  // ------------------------------------------------ lanes (app/lanes.py: the same equation)
  // Every street's lanes are split evenly between its two sides, as many as its width
  // holds: limit = floor((width - island) / 3 m), the island 1 m from 22 m. A highway
  // (Road markings, Highways: from 18 m) has 3 lanes each way however wide it is, widened
  // to fill it. The Lanes tool gives a street its own: click it, set side 1 and side 2,
  // never more than its limit together. Kept with the mask, by a point on each street,
  // as the island slots.
  const laneHighway = W => W >= LN.HIGHWAY - LN.TOL;
  const laneLimit = W => Math.max(0, Math.floor((W + LN.TOL - (laneHighway(W) ? LN.ISLAND : 0)) / LN.LANE));
  const isHighway = W => MARKS.opts.highways === 'on' && MARKS.values['highway.min_m'] != null
    && W >= MARKS.values['highway.min_m'] - LN.TOL;
  function laneSides(W, want){
    const lim = laneLimit(W);
    if(!want && isHighway(W)){
      const n = Math.min(Math.max(1, Math.trunc(MARKS.values['highway.lanes'] || 3)), Math.floor(lim / 2));
      return [n, n, false];
    }
    if(!want) return [Math.floor(lim / 2), Math.floor(lim / 2), false];
    let a = Math.max(0, Math.trunc(want[0])), b = Math.max(0, Math.trunc(want[1])), cut = false;
    while(a + b > lim){ if(a >= b) a--; else b--; cut = true; }
    return [a, b, cut];
  }
  // a street's name in the log: a highway's whole centreline (app/highways.py) is numbered from 1,000,000
  const streetName = s => s.id >= 1000000 ? `Highway ${s.id - 1000000 + 1}` : `Street ${s.id}`;
  // with the road marking guide's rules (Road markings, Lines): the line between the two
  // directions is the centre line, broken with one lane each way, solid with two, double from three
  const laneRules = () => MARKS.opts.rules === 'guide' ? 'guide' : 'simple';
  const centreKind = (a, b) => Math.max(a, b) <= 1 ? 'cdash' : Math.max(a, b) === 2 ? 'csolid' : 'cdouble';
  function laneLayout(W, want){
    const [a, b, cut] = laneSides(W, want), lanes = a + b, half = W / 2;
    const out = { limit: laneLimit(W), sides: [a, b], cut, lanes, lines: [], island: null, kind: 'none', lane_m: W,
      hw: isHighway(W) };
    if(lanes < 2) return out;
    const guide = laneRules() === 'guide';
    const ei = guide && MARKS.values['edge.inset_m'] != null ? MARKS.values['edge.inset_m'] + MARKS.values['edge.width_m'] / 2 : LN.EDGE;
    if(laneHighway(W) && a >= 1 && b >= 1){
      const s = (W - LN.ISLAND) / lanes, i0 = a * s;
      for(let k = 1; k < a; k++) out.lines.push([s * k - half, 'dash']);
      for(let k = 1; k < b; k++) out.lines.push([i0 + LN.ISLAND + s * k - half, 'dash']);
      [ei, i0 - ei, i0 + LN.ISLAND + ei, W - ei].forEach(x => out.lines.push([x - half, 'solid']));
      out.lines.sort((p, q) => p[0] - q[0]);
      return Object.assign(out, { kind: 'highway', lane_m: s, island: [i0 - half, i0 + LN.ISLAND - half] });
    }
    const s = W / lanes;
    for(let k = 1; k < lanes; k++) out.lines.push([s * k - half, guide && a >= 1 && b >= 1 && k === a ? centreKind(a, b) : 'dash']);
    return Object.assign(out, { kind: 'road', lane_m: s });
  }
  // a street's line moved d mask pixels to its side (-ty, tx), as the texture and the model move it
  function offsetLine(line, d){
    const n = line.length;
    return line.map((p, i) => {
      const a = line[Math.max(0, i - 1)], b = line[Math.min(n - 1, i + 1)];
      let tx = b[0] - a[0], ty = b[1] - a[1]; const L = Math.hypot(tx, ty) || 1; tx /= L; ty /= L;
      return [p[0] - ty * d, p[1] + tx * d];
    });
  }
  function distToLine(p, line){
    let best = Infinity;
    for(let i = 0; i + 1 < line.length; i++){
      const a = line[i], b = line[i + 1], dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy;
      const t = L2 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2)) : 0;
      best = Math.min(best, Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy));
    }
    return line.length === 1 ? Math.hypot(p[0] - line[0][0], p[1] - line[0][1]) : best;
  }
  function midOf(line){
    let total = 0; for(let i = 0; i + 1 < line.length; i++) total += Math.hypot(line[i+1][0] - line[i][0], line[i+1][1] - line[i][1]);
    let left = total / 2;
    for(let i = 0; i + 1 < line.length; i++){
      const L = Math.hypot(line[i+1][0] - line[i][0], line[i+1][1] - line[i][1]);
      if(left <= L){ const t = L ? left / L : 0; return [line[i][0] + (line[i+1][0] - line[i][0]) * t, line[i][1] + (line[i+1][1] - line[i][1]) * t]; }
      left -= L;
    }
    return line[0].slice();
  }
  // which street each choice is for (the nearest, if the point lies on it), as the server matches them
  function laneMatch(){
    const of = new Map(), st = LANES.streets || [];
    LANES.picks.forEach((p, k) => {
      let best = -1, bd = Infinity;
      st.forEach((s, i) => { const d = distToLine(p.pt, s.line); if(d < bd){ bd = d; best = i; } });
      if(best >= 0 && bd <= st[best].width_m / 2 / LANES.mpp + 1 / LANES.mpp) of.set(best, k);
    });
    return of;
  }
  function streetLayout(i, of){
    const s = LANES.streets[i], k = (of || laneMatch()).get(i);
    return { s, k, L: laneLayout(s.width_m, k != null ? LANES.picks[k].sides : null) };
  }
  let lanesTimer = null;
  function lanesChanged(){
    renderLanes(); drawBridges();
    window.dispatchEvent(new Event('materials'));                    // the 3D tab's scene is out of date
    if(!S.gen.mask) return;
    clearTimeout(lanesTimer);
    const mask = S.gen.mask.id, picks = JSON.parse(JSON.stringify(LANES.picks));
    lanesTimer = setTimeout(() => api('/api/lanes', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mask, picks }) }).catch(e => log('Lane choices not saved: ' + e.message, 'bad')), 300);
  }
  // Done: only the lines of the streets whose lanes changed are laid again, on the texture as
  // it is (the server keeps what that needs from Generate texture): a second or so, not a new
  // texture. One at a time; asked again meanwhile, it runs once more when that one is done
  async function paintLanes(){
    const res = S.gen.result;
    if(!res || !LANES.streets) return;
    const want = JSON.stringify(LANES.picks);
    if(want === S.gen.lanesPainted) return;
    if(LANES.painting){ LANES.paintAgain = true; return; }
    LANES.painting = true;
    status('Laying the changed lanes…');
    try{
      const r = await api('/api/lanes/repaint', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ generation: res.id, lane_picks: JSON.parse(want), mark_area: MAREA.area }) });
      if(S.gen.result === res){
        S.gen.lanesPainted = want;
        if(r.changed.length){
          res.urls.result = r.urls.result; res.urls.markings = r.urls.markings;
          if(r.urls.full && S.gen.islandTex) S.gen.islandTex.urls.full = r.urls.full;
          log(`Lanes laid again on ${r.changed.map(c => `street ${c.id} (${c.sides.join(' + ')}${c.even ? ', even' : ''})`).join(', ')}`
            + ` in ${r.seconds.toFixed(1)} s: only their lines, the rest of the texture as it was.`, 'ok');
          if(['result', 'markings', 'full'].includes(S.gen.layer)) await showLayer(S.gen.layer, true);
          renderLayerThumbs();
          if((S.gen.top && S.gen.top.gid === res.id) || S.gen.topBusy) refreshTopView(res.id, true);
        }
      }
    }catch(e){
      log('Lanes not laid on the texture: ' + e.message, 'bad');
      if(e.status === 409 && S.gen.result === res) S.gen.lanesPainted = want;      // said once
    }
    status('Ready.');
    LANES.painting = false;
    if(LANES.paintAgain){ LANES.paintAgain = false; paintLanes(); }
  }
  function lanesMode(on){
    LANES.on = on && !!LANES.streets;
    if(LANES.on && MAREA.on){ MAREA.on = false; renderMarea(); }
    if(LANES.on && ISLS.pick >= 0) setPick(-1);
    if(LANES.on){ AUTO.pick = -1; pareaStop(); }
    if(!LANES.on){ LANES.sel = -1; paintLanes(); }
    else log('Set lanes: click a street to choose how many lanes each side has (the two sides show in blue and orange). Esc or Set lanes to finish.');
    renderLanes(); drawBridges();
  }
  function setSides(i, a, b){
    const of = laneMatch(), s = LANES.streets[i];
    const [ca, cb] = laneSides(s.width_m, [a, b]);
    let k = of.get(i);
    if(k == null){ LANES.picks.push({ pt: midOf(s.line).map(v => Math.round(v * 10) / 10), sides: [ca, cb] }); }
    else { LANES.picks[k].sides = [ca, cb]; delete LANES.picks[k].src; delete LANES.picks[k].trace; }   // set by hand: yours, laid by the rules
    lanesChanged();
  }
  function setEven(i){
    const k = laneMatch().get(i);
    if(k != null){ LANES.picks.splice(k, 1); log(`${streetName(LANES.streets[i])}: back to the even split.`); lanesChanged(); }
  }
  function renderLanes(){
    const has = !!LANES.streets;
    $('#btnLanes').disabled = !has;
    $('#btnLanes').setAttribute('aria-pressed', LANES.on);
    $('#btnLanes').title = has ? 'Click streets on the map to choose their lanes' : 'Generate texture first: it finds the streets';
    $('#btnLanesEven').disabled = !LANES.picks.length;
    const of = laneMatch();
    // the street being set
    const ed = $('#laneEdit');
    ed.hidden = !(LANES.on && LANES.sel >= 0 && has);
    if(!ed.hidden){
      const i = LANES.sel, { s, k, L } = streetLayout(i, of);
      const [a, b] = L.sides, full = a + b >= L.limit;
      const what = L.lanes < 2 ? 'one lane, no lines'
        : `${a} + ${b} lane${a + b === 1 ? '' : 's'} of ${L.lane_m.toFixed(2)} m`
          + (L.kind === 'highway' ? ', a raised island between the sides, solid lines at the kerbs and the island' : '')
          + (a === 0 || b === 0 ? ' (one-way)' : '');
      ed.innerHTML = `<div class="le-h">${streetName(s)} · ${s.width_m.toFixed(1)} m wide · holds at most ${L.limit} lane${L.limit === 1 ? '' : 's'}</div>
        <div class="le-sides">
          <span><span class="le-sw" style="background:rgb(70,140,255)"></span>Side 1</span>
          <span class="le-step"><button type="button" data-s="0" data-d="-1" ${a <= 0 ? 'disabled' : ''}>−</button><b>${a}</b><button type="button" data-s="0" data-d="1" ${full ? 'disabled' : ''}>+</button></span><span></span>
          <span><span class="le-sw" style="background:rgb(255,150,60)"></span>Side 2</span>
          <span class="le-step"><button type="button" data-s="1" data-d="-1" ${b <= 0 ? 'disabled' : ''}>−</button><b>${b}</b><button type="button" data-s="1" data-d="1" ${full ? 'disabled' : ''}>+</button></span><span></span>
        </div>
        <div class="le-out">${k == null ? 'Even split: ' : 'Your choice: '}${what}.${full && L.limit ? ' That is its limit.' : ''}</div>
        ${L.cut ? '<div class="le-cut">More than this street can hold: cut back to its limit.</div>' : ''}
        <div style="margin-top:6px;display:flex;gap:6px"><button type="button" class="le-even" ${k == null ? 'disabled' : ''}>Even</button><button type="button" class="le-done" title="Lay this street's lines on the texture: only its lines, in a second or so">Done</button></div>`;
      ed.querySelectorAll('.le-step button').forEach(btn => btn.addEventListener('click', () => {
        const sd = [a, b]; sd[+btn.dataset.s] += +btn.dataset.d;
        setSides(i, sd[0], sd[1]);
      }));
      $('.le-even', ed).addEventListener('click', () => setEven(i));
      $('.le-done', ed).addEventListener('click', () => { LANES.sel = -1; renderLanes(); drawBridges(); paintLanes(); });
    }
    // the streets given their own lanes
    const list = $('#lanesList'); list.innerHTML = '';
    const owner = new Map([...of.entries()].map(([i, k]) => [k, i]));
    LANES.picks.forEach((p, k) => {
      const row = document.createElement('div'); row.className = 'll-row';
      const i = owner.get(k);
      row.innerHTML = `<span></span><button type="button" class="ll-x" title="Back to the even split">×</button>`;
      const lanesTxt = sd => sd[0] + sd[1] ? `${sd[0]} + ${sd[1]} lanes` : 'no lines';
      $('span', row).textContent = (i != null
        ? `${streetName(LANES.streets[i])} (${LANES.streets[i].width_m.toFixed(1)} m): ${lanesTxt(laneSides(LANES.streets[i].width_m, p.sides))}`
        : `${lanesTxt(p.sides)} at (${Math.round(p.pt[0])}, ${Math.round(p.pt[1])})${LANES.streets ? ': on no street now' : ''}`)
        + (p.src === 'satellite' ? ' · from the satellite' : '');
      row.addEventListener('click', () => { if(i != null){ LANES.sel = i; if(!LANES.on) lanesMode(true); else { renderLanes(); drawBridges(); } } });
      $('.ll-x', row).addEventListener('click', e => { e.stopPropagation(); LANES.picks.splice(k, 1); LANES.sel = -1; lanesChanged(); paintLanes(); });
      list.appendChild(row);
    });
  }
  // the street being set: its two sides, lines and island, as SVG over the overlay (the other streets'
  // lines and labels are on the overlay: drawLanesOn)
  function drawLanes(f){
    const st = LANES.streets;
    if(!st || !LANES.on || LANES.sel < 0 || !st[LANES.sel]) return;
    const of = laneMatch(), g = el('g', {}, layer), top = el('g', {}, layer);
    const pts = l => l.map(q => `${(q[0] * f).toFixed(1)},${(q[1] * f).toFixed(1)}`).join(' ');
    const hr = Math.max(4, genVp.canvas.width / 260), m2px = f / LANES.mpp;   // display pixels per metre
    const s = st[LANES.sel];
    if(s.line.length < 2) return;
    const { L } = streetLayout(LANES.sel, of);
    const fs = Math.max(hr * 1.2, Math.min(hr * 8, s.width_m * m2px * 0.42));       // a label that fits the road
    // its two sides, in two colours, each as wide as its lanes (split at the island, or at the
    // line between the sides), and its lines and island as they will be laid
    const half = s.width_m / 2 / LANES.mpp, [a, b] = L.sides;
    const split = (L.island ? (L.island[0] + L.island[1]) / 2 : (a + b ? a * L.lane_m - s.width_m / 2 : 0)) / LANES.mpp;
    const bands = [[-half, split, 'rgba(70,140,255,.40)', '1'], [split, half, 'rgba(255,150,60,.40)', '2']];
    bands.forEach(([d0, d1, c, t]) => {
      if(d1 - d0 < 1e-6) return;
      el('polygon', { points: pts(offsetLine(s.line, d0).concat(offsetLine(s.line, d1).reverse())), fill: c, stroke: 'none' }, g);
      const p = midOf(offsetLine(s.line, (d0 + d1) / 2));
      const tx = el('text', { x: p[0] * f, y: p[1] * f, 'font-size': Math.min(fs * 1.4, (d1 - d0) * f * 0.8 + hr),
                              'stroke-width': fs * 0.2, class: 'lane-lbl' }, top);
      tx.textContent = t;
    });
    if(L.island) el('polygon', { points: pts(offsetLine(s.line, L.island[0] / LANES.mpp).concat(offsetLine(s.line, L.island[1] / LANES.mpp).reverse())),
                                fill: 'rgba(200,196,186,.95)', stroke: 'rgba(90,88,84,.9)', 'stroke-width': Math.max(1, 0.1 * m2px) }, g);
    // each line as it will be laid: dashed or whole, the centre line yellow with the guide's colours, double from three lanes
    const yel = MARKS.opts.colours === 'guide' && laneRules() === 'guide';
    L.lines.forEach(([x, kind]) => {
      const centre = kind[0] === 'c', dashed = kind === 'dash' || kind === 'cdash';
      const offs = kind === 'cdouble' ? [x - 0.15, x + 0.15] : [x];
      offs.forEach(o => el('polyline', { points: pts(offsetLine(s.line, o / LANES.mpp)), fill: 'none',
        stroke: centre && yel ? '#E8AC24' : '#fff', 'stroke-width': Math.max(1, 0.3 * m2px),
        'stroke-dasharray': dashed ? `${3 * m2px} ${6 * m2px}` : 'none' }, g));
    });
  }
  // every street's centre line and what it has (2|2, yellow when set in the tool), on the overlay
  function drawLanesOn(ctx, v){
    const st = LANES.streets;
    if(!st || !LANES.on) return;
    const of = laneMatch(), labels = [], m2px = v.k / LANES.mpp;          // overlay pixels per metre
    ctx.lineCap = 'butt';
    st.forEach((s, i) => {
      if(s.line.length < 2) return;
      const b = s.bbox || (s.bbox = lineBox(s.line)), pad = s.width_m / LANES.mpp;
      if(!inView(v, b, pad)) return;
      const { k, L } = streetLayout(i, of), sel = i === LANES.sel;
      ctx.beginPath(); tracePath(ctx, v, s.line, false);
      ctx.strokeStyle = k != null ? 'rgba(255,200,80,.95)' : 'rgba(255,255,255,.55)';
      ctx.lineWidth = Math.max(v.dpr, v.hr * (sel ? 0.18 : 0.3));
      ctx.setLineDash(sel ? [] : [v.hr, v.hr * 0.7]);
      ctx.stroke();
      if(sel) return;
      const fs = Math.max(v.hr * 1.2, Math.min(v.hr * 8, s.width_m * m2px * 0.42));
      if(fs < v.minFont) return;
      const p = s.mid || (s.mid = midOf(s.line));
      labels.push([p[0] * v.k + v.ox, p[1] * v.k + v.oy, fs, L.lanes < 2 ? '–' : `${L.sides[0]}|${L.sides[1]}`, k != null ? '#ffd27a' : '#fff']);
    });
    ctx.setLineDash([]);
    // the lines the satellite traced (app/observed.py): laid where they are painted
    LANES.picks.forEach(p => (p.trace || []).forEach(t => {
      if(!t.pts || t.pts.length < 2) return;
      ctx.beginPath(); tracePath(ctx, v, t.pts, false);
      ctx.strokeStyle = 'rgba(90,220,255,.9)'; ctx.lineWidth = Math.max(v.dpr, v.hr * 0.22); ctx.stroke();
    }));
    overlayLabels(ctx, labels, 0.2);
  }
  function saveHighwayGroups(){
    if(!S.gen.mask) return;
    api('/api/highway-groups',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mask:S.gen.mask.id,picks:GROUPS.picks,enabled:GROUPS.enabled})}).catch(e=>log(e.message,'bad'));
  }
  function groupOf(s){
    return GROUPS.picks.find(p => distToLine(p.pt, s.line) <= s.width_m / 2 / LANES.mpp);
  }
  function groupHighway(s){ return s.id >= 1000000 || s.width_m >= MARKS.values['highway.min_m'] - 0.3; }
  function drawHighwayGroups(ctx, v){
    if(!GROUPS.picking) return;
    for(const s of LANES.streets || []){
      if(!groupHighway(s) || s.line.length < 2) continue;
      const p = groupOf(s);
      ctx.beginPath(); tracePath(ctx,v,s.line,false);
      ctx.strokeStyle = p ? (p.group === 1 ? 'rgba(255,255,255,.7)' : 'rgba(232,172,36,.8)') : 'rgba(60,165,255,.65)';
      ctx.lineWidth = Math.max(6*v.dpr,s.width_m/LANES.mpp*v.k*.7); ctx.stroke();
      const q=midOf(s.line);overlayLabels(ctx,[[q[0]*v.k+v.ox,q[1]*v.k+v.oy,Math.max(12*v.dpr,v.hr*2),p ? 'Group '+p.group : 'Highway']],.2);
    }
  }
  function lineBox(line){
    const b = [Infinity, Infinity, -Infinity, -Infinity];
    line.forEach(q => { b[0] = Math.min(b[0], q[0]); b[1] = Math.min(b[1], q[1]); b[2] = Math.max(b[2], q[0]); b[3] = Math.max(b[3], q[1]); });
    return b;
  }
  // A click on the map: in Pick, the island under it; in the Lanes tool, the street under it (found by
  // their outlines and lines, as the overlay draws them, not by a shape each to click on). Before the
  // map's own click (the review), which then does not see it; a handle's own clicks are left alone
  const onHandle = e => e.target.closest && e.target.closest('#bridgeLayer .grab, #bridgeLayer .corner, #bridgeLayer .spin, #bridgeLayer .pt');
  $('#genVp').addEventListener('click', e => {
    if(!S.gen.mask || (genVp.wasDrag && genVp.wasDrag()) || e.target.closest('.vp-tools') || onHandle(e)) return;
    const [x, y] = toMask(e);
    if(GROUPS.picking){
      const highways=(LANES.streets || []).filter(groupHighway);
      const s=highways.sort((a,b)=>distToLine([x,y],a.line)-distToLine([x,y],b.line))[0];
      if(s && distToLine([x,y],s.line) <= s.width_m/2/LANES.mpp+2){
        const old=groupOf(s);GROUPS.picks=GROUPS.picks.filter(p=>p!==old);
        if(!old || old.group!==GROUPS.picking) GROUPS.picks.push({pt:midOf(s.line),group:GROUPS.picking});
        saveHighwayGroups();renderMarks();drawBridges();window.dispatchEvent(new Event('materials'));
      }
      e.stopImmediatePropagation(); return;
    }
    if(PAREA.on){
      e.stopPropagation();
      const id = PAREA.data ? islandIn(PAREA.data, x, y) : 0;
      if(id) pareaToggle(id, x, y);
      return;
    }
    if(AUTO.pick >= 0 && ISLS.data){
      const id = islandAt(x, y);
      if(id){ e.stopPropagation(); autoPickIsland(id); return; }
    }
    if(ISLS.pick >= 0 && ISLS.data){
      const id = islandAt(x, y);
      if(id){ e.stopPropagation(); pickIsland(id); return; }
    }
    if(LANES.on && LANES.streets){
      const i = streetAt(x, y);
      if(i >= 0){
        e.stopPropagation();
        if(LANES.sel >= 0 && LANES.sel !== i) paintLanes();                     // the street left: its lanes laid
        LANES.sel = i; renderLanes(); drawBridges();
        return;
      }
    }
    // an automatic placement's object (on the overlay): selects its placement; in its own
    // Turn single objects or Draw spaced the press has done it already
    const hit = autoCopyAt(x, y);
    if(hit){
      e.stopPropagation();
      if(S.gen.edit && S.gen.edit.i === hit[0]) return;
      if(S.gen.edit) S.gen.edit = null;
      S.gen.plSel = hit[0]; drawBridges();
    }
  }, true);
  // and a hand over what a click would pick
  $('#genVp').addEventListener('pointermove', e => {
    const stage = $('#genVp .stage');
    let hit = false;
    if(S.gen.mask && !e.buttons && !onHandle(e) && (((ISLS.pick >= 0 || AUTO.pick >= 0) && ISLS.data) || (LANES.on && LANES.streets) || hasAuto())){
      const [x, y] = toMask(e);
      hit = ((ISLS.pick >= 0 || AUTO.pick >= 0) && ISLS.data && islandAt(x, y) > 0) || (LANES.on && LANES.streets && streetAt(x, y) >= 0)
        || !!autoCopyAt(x, y);
    }
    stage.style.cursor = hit ? 'pointer' : '';
  });
  // the street a point (mask pixels) is on: the nearest whose road (or a finger's width) it lies on
  function streetAt(x, y){
    const st = LANES.streets || [], f = shown(), hr = Math.max(4, genVp.canvas.width / 260);
    let best = -1, bd = Infinity;
    st.forEach((s, i) => {
      if(s.line.length < 2) return;
      const tol = Math.max(hr * 2 / f, s.width_m / LANES.mpp) / 2, b = s.bbox || (s.bbox = lineBox(s.line));
      if(x < b[0] - tol || x > b[2] + tol || y < b[1] - tol || y > b[3] + tol) return;
      const d = distToLine([x, y], s.line);
      if(d <= tol && d < bd){ bd = d; best = i; }
    });
    return best;
  }
  async function loadLanes(maskId){
    GROUPS.picks = []; GROUPS.picking = 0;
    try{
      const groups=await api('/api/highway-groups?mask='+maskId);
      if(S.gen.mask && S.gen.mask.id===maskId){ GROUPS.picks=groups.picks || [];GROUPS.enabled=groups.enabled!==false; }
    }catch(_){}
    renderMarks();
    LANES.mask = maskId; LANES.picks = []; LANES.streets = null; LANES.sel = -1; LANES.on = false;
    try{
      const r = await api('/api/lanes?mask=' + maskId);
      if(LANES.mask !== maskId) return;
      LANES.picks = r.picks || [];
      if(LANES.picks.length) log(`${LANES.picks.length} street(s) with their own lanes restored for this mask.`);
    }catch(_){}
    renderLanes();
  }
  async function loadStreets(res){
    LANES.streets = null; LANES.sel = -1;
    if(res.streets && res.streets.url){
      try{
        const d = await api(res.streets.url);
        LANES.streets = d.streets || []; LANES.mpp = +d.mpp || mppMask();
      }catch(e){ log('Streets for the Lanes tool not loaded: ' + e.message, 'bad'); }
    }
    if(!LANES.streets) LANES.on = false;
    renderLanes(); drawBridges();
  }
  $('#btnLanes').addEventListener('click', () => lanesMode(!LANES.on));
  $('#btnLanesEven').addEventListener('click', () => {
    if(!LANES.picks.length) return;
    log(`All ${LANES.picks.length} street choice(s) cleared: every street splits its lanes evenly.`);
    LANES.picks = []; LANES.sel = -1; lanesChanged(); paintLanes();
  });
  addEventListener('keydown', e => { if(e.key === 'Escape' && LANES.on && !MAT.open) lanesMode(false); });

  // ------------------------------------------------ markings area (app/lanes.py)
  // One rectangle on the map: lane lines and decals only inside it (a dash whole or not
  // at all, by its middle; a solid line up to the edge; a decal by its middle). Kept with
  // the mask. Set before Generate texture, the texture is made with it; set or changed
  // afterwards, what now lies outside is taken away and what comes inside is laid, the
  // rest of the texture as it is. { cx, cy, w, h (mask pixels), angle (degrees) }.
  function mareaCorners(a){
    const r = a.angle * Math.PI / 180, ux = Math.cos(r), uy = Math.sin(r), vx = -uy, vy = ux;
    return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([i, j]) => [a.cx + ux*a.w/2*i + vx*a.h/2*j, a.cy + uy*a.w/2*i + vy*a.h/2*j]);
  }
  function renderMarea(){
    const has = !!S.gen.mask, a = MAREA.area;
    $('#btnMarea').disabled = !has;
    $('#btnMarea').setAttribute('aria-pressed', MAREA.on);
    $('#btnMarea').textContent = MAREA.on ? 'Done' : a ? 'Edit markings area' : 'Markings area';
    $('#btnMareaAll').disabled = !a;
    const m = mppMask();
    $('#mareaNote').textContent = a
      ? `Lane lines and decals only inside the area: ${Math.round(a.w * m)} × ${Math.round(a.h * m)} m`
        + `${Math.round(a.angle) % 360 ? `, turned ${Math.round(((a.angle % 360) + 360) % 360)}°` : ''}.`
        + (MAREA.on ? ' Drag it to move it, a corner to size it, the round handle to turn it; Done or Esc to finish.' : '')
        + (S.gen.result ? ' Changes are laid on the texture as you make them: markings outside are taken away, nothing else is generated again.'
                        : ' Generate texture lays the markings inside it only.')
      : 'Markings everywhere. Markings area puts a rectangle on the map: lane lines and decals only inside it (move it, size it from its corners, turn it from its handle).';
  }
  function drawMarea(f, hr){
    const a = MAREA.area;
    if(!a) return;
    // handles about 7 screen pixels whatever the zoom (the layer's pixels are the texture's)
    const shownW = genVp.canvas.getBoundingClientRect().width;
    if(shownW > 0) hr = Math.max(hr, 7 * genVp.canvas.width / shownW);
    const pts = mareaCorners(a).map(q => [q[0] * f, q[1] * f]);
    const g = el('g', {}, layer), poly = pts.map(q => q.join(',')).join(' ');
    if(MAREA.on){
      // outside it darkened: no markings there
      const W = genVp.canvas.width, H = genVp.canvas.height;
      el('path', { d: `M0,0H${W}V${H}H0Z M${pts.map(q => q.join(',')).join('L')}Z`, 'fill-rule': 'evenodd',
                   fill: 'rgba(0,0,0,.38)', 'pointer-events': 'none' }, g);
    }
    const body = el('polygon', { points: poly, fill: MAREA.on ? 'rgba(80,200,170,.10)' : 'none', stroke: '#50C8AA',
      'stroke-width': hr * (MAREA.on ? 0.55 : 0.35), 'stroke-dasharray': MAREA.on ? 'none' : `${hr * 1.6} ${hr}`,
      class: MAREA.on ? 'grab' : '', 'pointer-events': MAREA.on ? 'all' : 'none' }, g);
    const tl = pts[0];
    const lab = el('text', { x: tl[0] + hr, y: tl[1] - hr * 0.8, 'font-size': hr * 1.8, fill: '#50C8AA', class: 'lane-lbl',
                             'stroke-width': hr * 0.35, 'pointer-events': 'none' }, g);
    lab.textContent = 'Markings area';
    if(!MAREA.on) return;
    body.addEventListener('pointerdown', e => mareaStart(e, 'move'));
    pts.forEach(q => {
      const c = el('circle', { cx: q[0], cy: q[1], r: hr, class: 'corner', fill: '#fff', stroke: '#50C8AA', 'stroke-width': hr * 0.35 }, g);
      c.addEventListener('pointerdown', e => mareaStart(e, 'size'));
    });
    const r = a.angle * Math.PI / 180, ex = (a.cx + Math.cos(r) * a.w / 2) * f, ey = (a.cy + Math.sin(r) * a.w / 2) * f;
    const rx = ex + Math.cos(r) * hr * 5, ry = ey + Math.sin(r) * hr * 5;
    el('line', { x1: ex, y1: ey, x2: rx, y2: ry, stroke: '#50C8AA', 'stroke-width': hr * 0.35 }, g);
    const spin = el('circle', { cx: rx, cy: ry, r: hr * 1.15, class: 'spin', fill: '#50C8AA' }, g);
    spin.addEventListener('pointerdown', e => mareaStart(e, 'rotate'));
  }
  function mareaStart(e, mode){
    e.stopPropagation(); e.preventDefault();
    MAREA.drag = { mode, start: toMask(e), orig: { ...MAREA.area } };
  }
  addEventListener('pointermove', e => {
    const d = MAREA.drag;
    if(!d) return;
    const a = MAREA.area, o = d.orig, p = toMask(e);
    if(d.mode === 'move'){ a.cx = o.cx + p[0] - d.start[0]; a.cy = o.cy + p[1] - d.start[1]; }
    else if(d.mode === 'size'){
      const r = o.angle * Math.PI / 180, dx = p[0] - o.cx, dy = p[1] - o.cy;
      a.w = Math.max(8, 2 * Math.abs(dx * Math.cos(r) + dy * Math.sin(r)));
      a.h = Math.max(8, 2 * Math.abs(-dx * Math.sin(r) + dy * Math.cos(r)));
    }else a.angle = Math.atan2(p[1] - o.cy, p[0] - o.cx) * 180 / Math.PI;
    renderMarea(); drawBridges();
  });
  addEventListener('pointerup', () => {
    if(!MAREA.drag) return;
    MAREA.drag = null;
    mareaChanged();
  });
  function mareaMode(on){
    MAREA.on = on && !!S.gen.mask;
    if(MAREA.on){
      if(LANES.on) lanesMode(false);
      if(ISLS.pick >= 0) setPick(-1);
      AUTO.pick = -1;
      pareaStop();
      if(!MAREA.area){
        // a first rectangle: the middle half of the mask
        const W = S.gen.mask.width, H = S.gen.mask.height;
        MAREA.area = { cx: W / 2, cy: H / 2, w: W / 2, h: H / 2, angle: 0 };
        log('Markings area: lane lines and decals only inside the rectangle. Move it, size it from its corners, turn it from its round handle.');
        mareaChanged();
      }
    }
    renderMarea(); drawBridges();
  }
  // kept with the mask, the 3D tab told, and laid on the texture as it is
  function mareaChanged(){
    renderMarea(); drawBridges();
    window.dispatchEvent(new Event('materials'));
    if(!S.gen.mask) return;
    clearTimeout(MAREA.timer);
    const mask = S.gen.mask.id, area = MAREA.area ? { ...MAREA.area } : null;
    MAREA.timer = setTimeout(() => api('/api/markarea', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mask, area }) }).catch(e => log('Markings area not saved: ' + e.message, 'bad')), 300);
    paintMarea();
  }
  // only what the change reaches: lines of the streets its edges cross or that come in or go
  // out, decals that come in or go out. One at a time; asked again meanwhile, once more after
  async function paintMarea(){
    const res = S.gen.result;
    if(!res) return;
    const want = JSON.stringify(MAREA.area);
    if(want === MAREA.painted) return;
    if(MAREA.busy){ MAREA.again = true; return; }
    MAREA.busy = true;
    status('Laying the markings area on the texture…');
    try{
      const r = await api('/api/markarea/apply', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ generation: res.id, area: JSON.parse(want) }) });
      if(S.gen.result === res){
        MAREA.painted = want;
        res.urls.result = r.urls.result; res.urls.markings = r.urls.markings;
        if(r.lanes) S.gen.lanesPainted = JSON.stringify(LANES.picks);
        if(r.decals && S.gen.decals){ S.gen.decals.urls = r.decals.urls; S.gen.decals.count = r.decals.count; }
        if(S.gen.islandTex && S.gen.islandTex.gid === res.id) S.gen.islandTex.urls.full = r.urls.full;
        const n = r.lanes ? r.lanes.changed.length : 0, dp = r.decals && r.decals.partial;
        const dTxt = r.decals ? (dp ? (dp.changed ? `, ${dp.gone} decal(s) taken away and ${dp.changed - dp.gone} laid` : ', decals as they were')
                                    : `, ${r.decals.count} decal(s) laid`) : '';
        log(`Markings area ${JSON.parse(want) ? 'laid' : 'taken away: markings everywhere again'} in ${r.seconds.toFixed(1)} s: `
          + `lines of ${n} street(s) laid again${dTxt}; the rest of the texture as it was, nothing generated again.`, 'ok');
        if(['result', 'markings', 'full', 'decals'].includes(S.gen.layer)) await showLayer(S.gen.layer, true);
        renderLayerThumbs();
        if(S.gen.decals) try{ drawInto($('#lp-decals'), await loadImage(API + r.urls.decals)); }catch(_){}
        if((S.gen.top && S.gen.top.gid === res.id) || S.gen.topBusy) refreshTopView(res.id, true);
      }
    }catch(e){
      log('Markings area not laid on the texture: ' + e.message, 'bad');
      if(e.status === 409 && S.gen.result === res) MAREA.painted = want;
    }
    status('Ready.');
    MAREA.busy = false;
    if(MAREA.again){ MAREA.again = false; paintMarea(); }
  }
  async function loadMarea(maskId){
    MAREA.mask = maskId; MAREA.area = null; MAREA.on = false; MAREA.painted = 'null';
    try{
      const r = await api('/api/markarea?mask=' + maskId);
      if(MAREA.mask !== maskId) return;
      MAREA.area = r.area || null;
      if(MAREA.area) log('Markings area restored for this mask: lane lines and decals only inside it.');
    }catch(_){}
    renderMarea();
  }
  $('#btnMarea').addEventListener('click', () => mareaMode(!MAREA.on));
  // its handles stay the same size on screen as you zoom
  let mareaFrame = 0;
  $('#genVp').addEventListener('vpzoom', () => {
    if(!MAREA.on || !MAREA.area || mareaFrame) return;
    mareaFrame = requestAnimationFrame(() => { mareaFrame = 0; drawBridges(); });
  });
  $('#btnMareaAll').addEventListener('click', () => {
    if(!MAREA.area) return;
    MAREA.area = null; MAREA.on = false;
    log('Markings area taken away: lane lines and decals everywhere.');
    mareaChanged();
  });
  addEventListener('keydown', e => { if(e.key === 'Escape' && MAREA.on && !MAT.open) mareaMode(false); });

  /* ---- Project Area: the islands (blocks, squares, traffic islands) picked for the project become street:
     flat, the street material, no sidewalks or kerbs, and none added round them (server.py _inner_streets).
     Select islands, click them, Done: the texture is made again with them at once. Kept with the mask by a
     point in each island, as the island slots */
  const pareaKey = () => JSON.stringify([...PAREA.picks].map(p => p.map(v => Math.round(v * 10) / 10)).sort());
  const pareaScale = () => +$('#scale').value || 1.1;
  function renderParea(){
    const has = !!S.gen.mask, n = PAREA.sel.size;
    $('#btnParea').disabled = !has || PAREA.busy;
    $('#btnParea').setAttribute('aria-pressed', PAREA.on);
    $('#btnParea').textContent = PAREA.on ? 'Done' : 'Select islands';
    $('#btnPareaClear').disabled = !has || !n || PAREA.busy;
    $('#pareaCount').textContent = n ? `${n} island${n === 1 ? '' : 's'}` : '';
    const isl = PAREA.data ? PAREA.data.islands.filter(q => PAREA.sel.has(q.id)) : [];
    const m2 = isl.reduce((a, q) => a + q.area_m2, 0);
    const changed = S.gen.result && pareaKey() !== PAREA.built;
    $('#pareaNote').textContent = !has ? 'Load a street mask, then pick the islands (blocks, squares, traffic islands) your project takes.'
      : PAREA.on ? `Click islands to add them to the Project Area, click again to take them out (${n} picked`
          + (n ? `, ${Math.round(m2).toLocaleString()} m²` : '') + '). Done (or Esc) builds them in at once.'
      : !n ? 'None. Select islands, click the blocks or islands your project takes, then Done: they become street (flat, the street material, no sidewalks or kerbs) and the texture is made again with them.'
      : `${n} island${n === 1 ? '' : 's'} (${isl.map(q => q.id).join(', ')}; ${Math.round(m2).toLocaleString()} m²) made street: flat, the street material, no sidewalks or kerbs.`
        + (changed ? ' Changed since the texture: Select islands, then Done, builds them in (or Generate texture).'
                   : !S.gen.result ? ' Generate texture builds them in.' : '');
  }
  // every island to pick from (the mask without a project area, numbered as its texture would be), and the picks
  async function loadPareaData(){
    if(!S.gen.mask) return;
    const mask = S.gen.mask.id, scale = pareaScale();
    if(PAREA.data && PAREA.mask === mask && PAREA.scale === scale) return;
    const r = await api(`/api/projectareas?mask=${mask}&scale=${scale}`);
    if(!S.gen.mask || S.gen.mask.id !== mask) return;
    PAREA.data = { islands: r.islands || [], count: r.count }; PAREA.scale = scale; PAREA.mask = mask;
    PAREA.picks = r.picks || []; PAREA.sel = new Set(r.selected || []);
  }
  async function loadParea(maskId){
    PAREA.mask = maskId; PAREA.picks = []; PAREA.sel = new Set(); PAREA.data = null; PAREA.on = false; PAREA.scale = null;
    try{
      await loadPareaData();
      if(PAREA.sel.size) log(`Project Area: ${PAREA.sel.size} island(s) restored for this mask.`);
    }catch(_){}
    PAREA.built = S.gen.result ? pareaKey() : '[]';
    renderParea(); drawBridges();
  }
  // the island under a click, in or out; kept on the server at once (as a point well inside it)
  async function pareaToggle(id, x, y){
    const d = PAREA.data, isl = d.islands.find(q => q.id === id);
    const had = PAREA.sel.has(id);
    const picks = had ? PAREA.picks.filter(p => islandIn(d, p[0], p[1]) !== id) : [...PAREA.picks, [x, y]];
    if(had) PAREA.sel.delete(id); else PAREA.sel.add(id);
    renderParea(); drawBridges();
    try{
      const r = await api('/api/projectareas', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ mask: S.gen.mask.id, picks, scale: pareaScale() }) });
      PAREA.picks = r.picks || []; PAREA.sel = new Set(r.selected || []);
      log(had ? `Island ${id} taken out of the Project Area.`
              : `Island ${id} (${Math.round(isl.area_m2).toLocaleString()} m²) into the Project Area: it becomes street.`);
    }catch(e){ log('Project Area not saved: ' + e.message, 'bad'); }
    renderParea(); drawBridges();
  }
  // Done: what changed is built in at once (the texture made again with the same settings, then everything
  // that follows a Generate texture: decals, islands, the Top view)
  function pareaBuild(){
    if(!S.gen.result || pareaKey() === PAREA.built) return;
    log(`Project Area changed: making the texture again with ${PAREA.sel.size} island(s) as street…`);
    $('#btnGenerate').click();
  }
  // another tool taken up: the selecting ends, nothing built (Select islands, then Done, builds it in)
  function pareaStop(){
    if(!PAREA.on) return;
    PAREA.on = false; renderParea(); drawBridges();
  }
  async function pareaMode(on){
    if(on && !S.gen.mask) return;
    if(on){
      if(MAREA.on) mareaMode(false);
      if(LANES.on) lanesMode(false);
      if(ISLS.pick >= 0) setPick(-1);
      AUTO.pick = -1;
      PAREA.busy = true; renderParea();
      try{ await loadPareaData(); }catch(e){ log('Islands not found: ' + e.message, 'bad'); PAREA.busy = false; renderParea(); return; }
      PAREA.busy = false;
      PAREA.on = true;
      log(`Project Area: ${PAREA.data.count} island(s) on the map. Click the ones your project takes, click again to take one out; Done when finished.`);
    }else{
      PAREA.on = false;
      pareaBuild();
    }
    renderParea(); drawBridges();
  }
  $('#btnParea').addEventListener('click', () => pareaMode(!PAREA.on));
  $('#btnPareaClear').addEventListener('click', async () => {
    if(!S.gen.mask || !PAREA.sel.size) return;
    try{
      const r = await api('/api/projectareas', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ mask: S.gen.mask.id, picks: [], scale: pareaScale() }) });
      PAREA.picks = r.picks || []; PAREA.sel = new Set(r.selected || []);
      log('Project Area cleared: its islands are islands again.');
    }catch(e){ log('Project Area not saved: ' + e.message, 'bad'); }
    renderParea(); drawBridges();
    if(!PAREA.on) pareaBuild();
  });
  addEventListener('keydown', e => { if(e.key === 'Escape' && PAREA.on && !MAT.open) pareaMode(false); });

  /* ---- Layout (app/footprints.py, app/layout.py): the buildings of an area made from coordinates, read
     from its street map, and an object on each: Create layout from dataset opens a window of the objects
     (imported, and Dynamic creation's buildings); each building takes the ticked one whose shape fits it
     best, its width and depth stretched to the building's, its height its own, its front to the street.
     The layout is a placement like an automatic one (its objects on the map, Turn single objects, Draw
     spaced), laid again by pressing Create layout from dataset again */
  const LAY = { data: null, show: false, mask: null, cream: true };
  const layCoords = () => !!(S.gen.mask && S.gen.mask.coords && S.gen.mask.coords.key);
  function layShown(){ return !!(LAY.show && LAY.data && LAY.data.coords && LAY.data.count && LAY.mask === (S.gen.mask || {}).id); }
  async function loadLayout(){
    LAY.data = null; LAY.mask = S.gen.mask ? S.gen.mask.id : null;
    renderLayout();
    if(!S.gen.mask) return;
    const mask = S.gen.mask.id;
    try{
      const d = await api('/api/layout/buildings?mask=' + mask);
      if(!S.gen.mask || S.gen.mask.id !== mask) return;
      LAY.data = d;
    }catch(e){ LAY.data = { error: e.message }; }
    renderLayout(); drawBridges();
  }
  function renderLayout(){
    const d = LAY.data, on = !!(d && d.coords);
    $('#btnLayout').disabled = !on || !d.count;
    $('#layShow').disabled = !on || !d.count;
    $('#layCount').textContent = on ? `${fmt(d.count)}` : '';
    const th = $('#layThumb');
    th.hidden = !on || !d.count;
    if(on && d.count){
      const src = `${API}${d.preview}?v=${d.key}${d.count}`;
      if($('img', th).dataset.src !== src){ $('img', th).src = src; $('img', th).dataset.src = src; }
    }
    $('#layNote').textContent = !S.gen.mask ? 'Load an area with Coordinates: its buildings are read from its street map.'
      : d && d.error ? `Layout: ${d.error}`
      : !d ? (layCoords() ? 'Reading the buildings…' : 'This mask is not from coordinates: Layout reads the buildings of an area loaded with Coordinates (from its street map).')
      : !on ? 'This mask is not from coordinates: Layout reads the buildings of an area loaded with Coordinates (from its street map).'
      : !d.count ? 'No buildings on this area\'s map (its colours may differ from the ones expected: workspace/coords/palette.json, "buildings").'
      : `${fmt(d.count)} buildings read from the ${d.source === 'buildings_map.png' ? 'buildings map' : 'street map'}`
        + (d.cream ? ` (${fmt(d.cream)} cream: shops, halls, stands)` : '')
        + (d.edge ? `, ${d.edge} cut by the map's edge` : '') + '. Create layout from dataset puts an object on each.';
  }
  $('#layShow').addEventListener('change', () => { LAY.show = $('#layShow').checked; drawBridges(); });
  $('#layThumb').addEventListener('click', async () => {
    if(!LAY.data || !LAY.data.preview) return;
    const img = await loadImage(`${API}${LAY.data.preview}?v=${Date.now().toString(36)}`);
    genVp.show(img, 'Buildings read from the map: each one\'s rectangle (grey: cut by the map\'s edge), a tick on its front');
  });
  // on the overlay: each building's rectangle (teal: cream on the map), a tick on its front (grey, dashed: cut by the map's edge)
  function drawBuildingsOn(ctx, v){
    if(!layShown()) return;
    const X = q => q[0] * v.k + v.ox, Y = q => q[1] * v.k + v.oy, lw = Math.max(v.dpr, v.hr * 0.14);
    for(const b of LAY.data.buildings){
      const r = Math.max(b.w_m, b.d_m) / mppMask();
      if(b.cx + r < v.x0 || b.cx - r > v.x1 || b.cy + r < v.y0 || b.cy - r > v.y1) continue;
      ctx.beginPath(); b.rect.forEach((q, k) => k ? ctx.lineTo(X(q), Y(q)) : ctx.moveTo(X(q), Y(q))); ctx.closePath();
      const col = b.edge ? '170,170,170' : b.fill === 'cream' ? '40,190,200' : '255,170,60';      // teal: a cream building
      ctx.fillStyle = `rgba(${col},${b.edge ? .15 : .16})`; ctx.fill();
      if(b.edge) ctx.setLineDash([v.hr * 0.5, v.hr * 0.4]);
      ctx.strokeStyle = `rgba(${col},.95)`; ctx.lineWidth = lw; ctx.stroke(); ctx.setLineDash([]);
      const a = b.a * DEG, f = [Math.sin(a), -Math.cos(a)], reach = b.d_m / 2 / mppMask();
      ctx.beginPath(); ctx.moveTo(X([b.cx, b.cy]), Y([b.cx, b.cy]));
      ctx.lineTo(X([b.cx + f[0] * reach, 0]), Y([0, b.cy + f[1] * reach]));
      ctx.strokeStyle = 'rgba(220,60,60,.95)'; ctx.lineWidth = lw * 1.4; ctx.stroke();
    }
  }
  // the window: the objects, and Dynamic creation's buildings (made objects as they are laid)
  async function layOpen(){
    if(!LAY.data || !LAY.data.count) return;
    let blds = [];
    try{ blds = (await api('/api/buildings')).buildings || []; }catch(_){}
    const objs = S.gen.objects.filter(o => !o.foliage && o.format !== 'building');
    const row = (kind, id, name, size, on) => `<label class="ds-row"><input type="checkbox" data-kind="${kind}" data-id="${id}"${on ? ' checked' : ''}>`
      + `<span>${esc(name)}</span><small>${size}</small></label>`;
    const sizeOf = o => { const k = o.scale || 1; return `${(o.width_m * k).toFixed(1)} × ${(o.depth_m * k).toFixed(1)} m, ${(o.height_m * k).toFixed(1)} m high`; };
    const had = new Set(((S.gen.placements.find(p => p.layout) || {}).layout || {}).objects || []);
    let html = '';
    if(objs.length) html += '<div class="ds-group">Imported objects</div>' + objs.map(o => row('object', o.id, o.name, sizeOf(o), had.has(o.id))).join('');
    if(blds.length) html += '<div class="ds-group">Dynamic creation</div>' + blds.map(b => {
      const o = b.object ? objById(b.object) : null;
      return row('building', b.id, b.name, o ? sizeOf(o) : `${b.levels} level${b.levels === 1 ? '' : 's'}, made an object when laid`, b.object && had.has(b.object));
    }).join('');
    $('#layList').innerHTML = html || '<div class="note">No objects yet: import one (Objects), or make a building in Dynamic creation.</div>';
    const d = LAY.data;
    $('#layWhere').textContent = `${fmt(d.count)} buildings on this area's map${d.cream ? `, ${fmt(d.cream)} of them cream` : ''}`
      + `${d.edge ? `, ${d.edge} cut by the map's edge` : ''}.`;
    $('#layEdge').checked = true;
    $('#layCreamRow').hidden = !d.cream;
    $('#layCream').checked = LAY.cream;
    $$('#layList input').forEach(b => b.addEventListener('change', layAllState));
    layAllState();
    $('#layPop').hidden = false;
  }
  function layAllState(){
    const bs = $$('#layList input'), n = bs.filter(b => b.checked).length;
    $('#layAll').checked = n === bs.length && n > 0;
    $('#layAll').indeterminate = n > 0 && n < bs.length;
    $('#layGo').disabled = !n;
    $('#layGo').textContent = !n ? 'OK' : `OK: lay ${n} ${n === 1 ? 'object' : 'objects'}`;
  }
  const layClose = () => { $('#layPop').hidden = true; };
  $('#btnLayout').addEventListener('click', layOpen);
  $('#layAll').addEventListener('change', () => { $$('#layList input').forEach(b => { b.checked = $('#layAll').checked; }); layAllState(); });
  $('#layClose').addEventListener('click', layClose);
  $('#layCancel').addEventListener('click', layClose);
  addEventListener('keydown', e => { if(e.key === 'Escape' && !$('#layPop').hidden) layClose(); });
  $('#layGo').addEventListener('click', async () => {
    const picked = $$('#layList input').filter(b => b.checked);
    if(!picked.length || !S.gen.mask) return;
    $('#layGo').disabled = true;
    try{
      const r = await api('/api/layout/create', { method:'POST', headers:{'Content-Type':'application/json'}, job: 'Laying objects on the buildings',
        body: JSON.stringify({ mask: S.gen.mask.id, skip_edge: $('#layEdge').checked, cream: LAY.cream = $('#layCream').checked,
          objects: picked.filter(b => b.dataset.kind === 'object').map(b => b.dataset.id),
          buildings: picked.filter(b => b.dataset.kind === 'building').map(b => b.dataset.id) }) });
      layClose();
      if(r.made.length) await loadObjects();                    // Dynamic creation's buildings, now objects
      const used = [...new Set(r.items.map(it => it.object))];
      const p = { object: used[0] || picked[0].dataset.id, cx: S.gen.mask.width / 2, cy: S.gen.mask.height / 2, angle: 0, nx: 1, ny: 1,
                  gap_x: 0, gap_y: 0, seed: 1, layout: { key: r.key, objects: Object.keys(r.note.names) }, items: r.items };
      // one layout an area: laid again, it takes the place of the one before
      const at = S.gen.placements.findIndex(q => q.layout);
      if(at >= 0) S.gen.placements[at] = p; else S.gen.placements.push(p);
      S.gen.plSel = at >= 0 ? at : S.gen.placements.length - 1; S.gen.edit = null;
      saveScatter(); renderObjList();
      const n = r.note;
      log(`Layout: ${fmt(n.laid)} of ${fmt(n.buildings)} buildings got an object: `
        + Object.entries(n.by_object).sort((a, b) => b[1] - a[1]).map(([id, k]) => `${k} × ${n.names[id] || id}`).join(', ')
        + (n.edge ? `; ${n.edge} cut by the map's edge left out` : '') + (n.fill ? `; ${fmt(n.fill)} cream left out` : '')
        + (n.small ? `; ${n.small} too small (under 2 m a side)` : '') + '.', 'ok');
      log('  Each object\'s width and depth are its building\'s, its height its own, its front to the building\'s street. Generate 3D scene or Download 3D model to see them.');
      if(r.made.length) log(`  Made objects of ${r.made.map(m => m.name).join(', ')} (Dynamic creation), as Use as object does.`);
    }catch(e){ log('Layout: ' + e.message, 'bad'); }
    finally{ $('#layGo').disabled = false; }
  });
  // the islands' sizes (and which small ones count) go by the scale: found again for a new one
  $('#scale').addEventListener('change', async () => {
    if(!S.gen.mask || !PAREA.data || PAREA.scale === pareaScale()) return;
    try{ await loadPareaData(); }catch(_){}
    renderParea(); drawBridges();
  });
  // on the overlay: while selecting, every island (picked ones filled, with their numbers); else the picked ones' outlines
  function drawPareaOn(ctx, v){
    const d = PAREA.data;
    if(!d || !(PAREA.on || PAREA.sel.size)) return;
    const labels = [], mpp = mppMask();
    for(const isl of d.islands){
      const mine = PAREA.sel.has(isl.id);
      if((!PAREA.on && !mine) || !inView(v, isl.bbox)) continue;
      ctx.beginPath();
      isl.rings.forEach(r => tracePath(ctx, v, r, true));
      if(PAREA.on){
        ctx.fillStyle = mine ? 'rgba(232,162,58,.55)' : 'rgba(255,255,255,.06)';
        ctx.fill('evenodd');
      }
      ctx.setLineDash(PAREA.on ? [] : [6 * v.dpr, 5 * v.dpr]);
      ctx.strokeStyle = mine ? '#E8A23A' : 'rgba(255,255,255,.55)';
      ctx.lineWidth = Math.max(v.dpr, v.hr * (mine ? 0.45 : 0.22));
      ctx.stroke();
      ctx.setLineDash([]);
      const r = isl.r_m / mpp * v.k, fs = Math.max(v.hr * 1.1, Math.min(v.hr * 3.2, r * 0.9));
      if(PAREA.on && r > v.hr * 0.5 && fs >= v.minFont) labels.push([isl.pt[0] * v.k + v.ox, isl.pt[1] * v.k + v.oy, fs, String(isl.id)]);
    }
    overlayLabels(ctx, labels, 0.18);
  }
  $('#btnAddSlot').addEventListener('click', () => {
    const first = MAT.lib && MAT.lib.materials.find(m => m.kind === 'ground');
    ISLS.slots.push({ material: first ? first.id : 'same', picks: [] });
    const i = ISLS.slots.length - 1;
    log(`Island material slot ${i + 1} added (${slotMaterial(i).name}): click its ball to choose the material, then Pick its islands.`);
    if(ISLS.data) ISLS.pick = i;
    ISLS.changed();
  });
  $('#btnShowIslands').addEventListener('click', () => { ISLS.show = !ISLS.show; renderSlots(); drawBridges(); });
  addEventListener('keydown', e => { if(e.key === 'Escape' && ISLS.pick >= 0 && !MAT.open) setPick(-1); });
  $('#btnIslandsTex').addEventListener('click', async () => {
    if(!S.gen.result || !ISLS.data) return;
    const btn = $('#btnIslandsTex'); btn.disabled = true; status('Generating the islands texture…');
    const gid = S.gen.result.id, slots = JSON.stringify(ISLS.slots);
    const again = S.gen.islandTex && S.gen.islandTex.gid === gid;
    log(again ? 'Updating the islands texture: only the islands whose material changed are laid again…'
              : 'Generating the islands texture: only the islands, each with its slot\'s material, at real size…');
    try{
      const res = await api('/api/islands/texture', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Generating the islands texture',
        body: JSON.stringify({ generation: gid, island_slots: ISLS.slots, materials: { square: $('#matSquare').value },
                               match_tone: $('#matTone').checked, seed: +$('#seed').value }) });
      if(!S.gen.result || S.gen.result.id !== gid) return;
      S.gen.islandTex = Object.assign(res, { gid, slots });
      renderSlots();
      const pt = res.partial;
      if(pt) log(pt.islands.length
        ? `Islands texture updated in ${res.seconds.toFixed(1)} s: ${pt.islands.length} island(s) laid again with their new material`
          + ` (${pt.islands.slice(0, 12).join(', ')}${pt.islands.length > 12 ? '…' : ''}); the other ${res.islands - pt.islands.length} kept as they were.`
        : 'Islands texture: no island changed its material; the texture is as it was.', 'ok');
      log(`Islands texture ${pt ? 'now' : 'ready'} (${res.size[0]} × ${res.size[1]}, the road texture's size): `
        + res.by_slot.map(b => `${b.islands} island(s) ${b.name}${b.slot >= 0 ? ` (slot ${b.slot + 1})` : ''}`).join(', ') + '.', pt ? '' : 'ok');
      if(!pt) log('  The roads are transparent in it, so it lies under the road texture with no gap; Roads + islands shows the two together.');
      $('#saveIslands').href = API + res.downloads.islands; $('#saveFull').href = API + res.downloads.full;
      $('#islandSaves').hidden = false;
      try{ drawInto($('#lp-islands'), await loadImage(API + res.urls.islands)); drawInto($('#lp-full'), await loadImage(API + res.urls.full)); }catch(_){}
      await showLayer(pt && ['islands', 'full'].includes(S.gen.layer) ? S.gen.layer : 'full', !!pt && ['islands', 'full'].includes(S.gen.layer));
      // the top view shows the 3D model's islands too: brought up to date
      if(!S.gen.top || S.gen.top.gid !== gid || S.gen.top.key !== JSON.stringify(exportSettings())) refreshTopView(gid);
    }catch(e){ log('Islands texture failed: ' + e.message, 'bad'); }
    status('Ready.'); btn.disabled = false; renderSlots();
  });
  renderSlots();

  /* ------------------------------------------------ road markings (app/roadmarks.py) */
  // The road marking guide's values (UTTIPEC, Delhi 2009, after IRC 35): the lines'
  // sizes and colours here (Generate texture lays them), the junction markings as
  // presets of the Decals panel. Every value shows the guide's beside it; with the
  // chain on, changing one sets the others of its group to the guide's values times
  // the same factor (app/roadmarks.py's chain, these few lines again), so the design
  // keeps the guide's proportions; off, each changes on its own.
  const MARKS = { spec: null, opts: {}, values: {}, chain: true };
  const LINK_SVG = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">'
    + '<path d="M6.3 9.7l3.4-3.4"/><path d="M7.2 4.6l1.1-1.1a2.7 2.7 0 0 1 3.8 3.8L11 8.4"/><path d="M8.8 11.4l-1.1 1.1a2.7 2.7 0 0 1-3.8-3.8L5 7.6"/></svg>';
  const UNLINK_SVG = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true">'
    + '<path d="M7.2 4.6l1.1-1.1a2.7 2.7 0 0 1 3.8 3.8L11 8.4"/><path d="M8.8 11.4l-1.1 1.1a2.7 2.7 0 0 1-3.8-3.8L5 7.6"/><path d="M2.5 2.5l2 2M13.5 13.5l-2-2"/></svg>';
  function guideOf(f, opts){
    let v = f.guide;
    Object.entries(f.guide_if || {}).forEach(([o, alt]) => { if((opts || {})[o]) v = alt; });
    return v;
  }
  function rangeOf(f, opts){
    let r = f.range || null;
    Object.entries(f.range_if || {}).forEach(([o, alt]) => { if((opts || {})[o]) r = alt; });
    return r;
  }
  const clampF = (f, v) => { v = Math.min(f.max, Math.max(f.min, +v)); return f.int ? Math.round(v) : Math.round(v * 10000) / 10000; };
  const num = v => String(Math.round(v * 1000) / 1000);
  // a value set, and with the chain on its group following it: [values, the keys that followed]
  function chainSet(fields, values, key, v, opts, linked){
    const out = { ...values }, f = fields.find(x => x.key === key), moved = [];
    out[key] = clampF(f, v);
    const base = guideOf(f, opts);
    if(linked && f.link && base){
      const k = out[key] / base;
      fields.forEach(h => {
        if(h.link === f.link && h.key !== key){ out[h.key] = clampF(h, guideOf(h, opts) * k); moved.push(h.key); }
      });
    }
    return [out, moved];
  }
  const offGuide = (fields, values, opts) => fields.filter(f => Math.abs(+values[f.key] - guideOf(f, opts)) > 1e-6);
  const outOfRange = (fields, values, opts) => fields.filter(f => {
    const r = rangeOf(f, opts); return r && (+values[f.key] < r[0] - 1e-6 || +values[f.key] > r[1] + 1e-6);
  });
  function keepMarks(){
    try{ localStorage.setItem('rta.marks', JSON.stringify({ opts: MARKS.opts, values: MARKS.values, chain: MARKS.chain })); }catch(_){}
  }
  const GROUPS = { picks: [], picking: 0, enabled: true };
  const marksPayload = () => ({ ...MARKS.opts, values: { ...MARKS.values }, centre_groups: GROUPS.enabled ? GROUPS.picks : [], group_mpp: mppMask() });
  // what decides the lines in the texture (the drive side only moves the junction markings)
  addEventListener('keydown', e => { if(e.key === 'Escape' && GROUPS.picking){ GROUPS.picking=0;renderMarks();drawBridges(); } });
  const marksKey = () => JSON.stringify([MARKS.opts.standard, MARKS.opts.speed, MARKS.opts.rules, MARKS.opts.colours, MARKS.opts.sizes, MARKS.opts.highways,
    MARKS.opts.posted_speed, MARKS.values, GROUPS.enabled, GROUPS.picks]);
  async function loadMarks(){
    try{ MARKS.spec = await api('/api/marks/spec'); }
    catch(e){ $('#rmBox').innerHTML = '<div class="note">The road marking guide could not be loaded: ' + esc(e.message) + '</div>'; return; }
    let kept = null;
    try{ kept = JSON.parse(localStorage.getItem('rta.marks') || 'null'); }catch(_){}
    const L = MARKS.spec.lines;
    L.opts.forEach(o => { MARKS.opts[o.key] = kept && kept.opts && o.choices.some(c => c[0] === kept.opts[o.key]) ? kept.opts[o.key] : o.default; });
    L.fields.forEach(f => { MARKS.values[f.key] = kept && kept.values && isFinite(kept.values[f.key]) ? clampF(f, kept.values[f.key]) : f.guide; });
    MARKS.chain = kept ? kept.chain !== false : true;
    renderMarks(); renderPresetPick(); renderDecals();
  }
  const fieldOf = key => MARKS.spec.lines.fields.find(f => f.key === key);
  // the lines' table: each kind of line, its dash, gap and width
  const LINE_ROWS = [
    ['Centre line', 'between the two directions', ['centre.dash_m', 'centre.gap_m', 'centre.width_m'], 'guide'],
    ['Double centre line', 'three lanes or more each way', [null, 'double.gap_m', 'double.width_m'], 'guide'],
    ['Lane lines', 'between lanes going the same way', ['lane.dash_m', 'lane.gap_m', 'lane.width_m'], 'all'],
    ['Edge lines', 'beside the kerbs of wide roads', ['edge.inset_m', null, 'edge.width_m'], 'all'],
  ];
  const COLS = ['dash_m', 'gap_m', 'width_m'];
  function chainButton(on, title){
    return `<button type="button" class="rm-chain" aria-pressed="${on}" title="${esc(title)}">${on ? LINK_SVG + 'Linked' : UNLINK_SVG + 'Unlinked'}</button>`;
  }
  function warnBox(fields, values, opts, label, resetId){
    const off = offGuide(fields, values, opts), bad = outOfRange(fields, values, opts);
    const ours = fields.every(f => f.src === 'ours');
    if(!off.length) return ours
      ? '<div class="rm-ok"><b>✓</b> Our values: this marking is not in the guide, so these are common practice.</div>'
      : '<div class="rm-ok"><b>✓</b> The guide\'s values: the right ones for a good design. Change them only if you need to.</div>';
    const items = off.map(f => {
      const r = rangeOf(f, opts), out = bad.includes(f), unit = f.int ? '' : f.unit ? ' ' + f.unit : ' m';
      return `<li>${esc(label(f))}: ${num(values[f.key])}${unit} (${f.src === 'ours' ? 'ours' : 'the guide'}: ${num(guideOf(f, opts))}${unit}`
        + `${r ? `, ${num(r[0])} to ${num(r[1])}` : ''})${out ? ' <b>outside the guide</b>' : ''}</li>`;
    }).join('');
    return `<div class="rm-warn"><b>⚠ Not the guide's values.</b> The guide's are the right ones; changed values can give a bad design:`
      + `<ul>${items}</ul><button type="button" class="btn secondary" data-reset="${resetId}">Back to the guide's values</button></div>`;
  }
  function renderMarks(){
    const box = $('#rmBox');
    if(!MARKS.spec || !box) return;
    const o = MARKS.opts, L = MARKS.spec.lines, guide = o.rules === 'guide', learned = o.sizes === 'learned';
    const speed = +o.speed || 60, postedSpeed = +o.posted_speed || 60, dash = speed >= 100 ? 6 : speed >= 80 ? 3 : 2;
    const autoValues = o.standard === 'saudi' && o.sizes === 'saudi' ? {
      'centre.dash_m':dash, 'centre.gap_m':dash*2, 'centre.width_m':.15,
      'lane.dash_m':dash, 'lane.gap_m':dash*2, 'lane.width_m':speed>=100?.2:.15,
      'double.width_m':speed>=80?.15:.1, 'edge.width_m':postedSpeed>100?.2:.15
    } : {};
    const sel = (key) => {
      const d = L.opts.find(x => x.key === key);
      return `<select data-opt="${key}">${d.choices.map(([v, t]) => `<option value="${v}"${o[key] === v ? ' selected' : ''}>${esc(t)}</option>`).join('')}</select>`;
    };
    const COLNAME = { dash_m: 'Dash', gap_m: 'Gap', width_m: 'Width' };
    const cell = (key, col) => {
      if(!key) return '<div></div>';
      const f = fieldOf(key), v = autoValues[key] ?? MARKS.values[key], g = guideOf(f), isOff = Math.abs(v - g) > 1e-6;
      const name = key === 'setback_m' ? 'Distance' : key === 'edge.inset_m' ? 'From kerb'
        : key.endsWith('.' + col) && key !== 'double.gap_m' ? COLNAME[col] : f.label;
      const dis = ((learned || o.standard === 'saudi' && o.sizes === 'saudi') && ['centre','lane','double','edge'].includes(f.group) && key !== 'double.gap_m' && key !== 'edge.inset_m') || (f.group === 'highway' && o.highways !== 'on') ? ' disabled' : '';
      return `<label class="rm-in${isOff ? ' off' : ''}" data-k="${key}" title="${esc(f.note || '')} (${esc(MARKS.spec.sources[f.src])})">`
        + `<span>${esc(name)}${f.int ? '' : ' (m)'}</span>`
        + `<input type="number" step="${f.int ? 1 : f.min < 0.1 ? 0.01 : 0.1}" min="${f.min}" max="${f.max}" value="${num(v)}"${dis}>`
        + `<small>${key in autoValues ? 'Saudi '+num(v) : (f.src === 'ours' ? 'ours' : 'custom guide')+' '+num(g)}</small></label>`;
    };
    const rows = LINE_ROWS.filter(r => guide || r[3] === 'all').map(([name, sub, keys]) =>
      `<div class="rm-kind"><div class="rm-kn">${esc(name)} <small>${esc(sub)}</small></div>`
      + `<div class="rm-3">${keys.map((k, c) => cell(k, COLS[c])).join('')}</div></div>`).join('');
    const sb = fieldOf('setback_m');
    const fields = L.fields.filter(f => guide || !(f.group === 'centre' || f.group === 'double'));
    const stale = S.gen.result && S.gen.marksPainted && S.gen.marksPainted !== marksKey();
    box.innerHTML = `<div class="rm-h"><div>Road markings<span class="rm-src">${o.standard === 'saudi' ? 'Saudi SHC 602, Chapter 4' : 'UTTIPEC guide, Delhi 2009'}</span></div>
        ${chainButton(MARKS.chain, MARKS.chain ? 'Linked: changing a length changes the other lengths, and a width the other widths, '
          + 'in the guide\'s proportions. Press to change each on its own' : 'Unlinked: each value changes on its own. Press to keep the guide\'s proportions')}</div>
      <label class="rm-l">Standard${sel('standard')}</label>
      <label class="rm-l">Design speed (km/h)${sel('speed')}</label>
      <label class="rm-l">Posted / approach speed (km/h)${sel('posted_speed')}</label>
      <div class="note">Saudi mode uses white centre lines and yellow edge lines. Speed-based sizes follow SHC 602 Tables 4-1, 4-2, 4-3 and 4-6. Custom dimensions remain available. Traffic priority, signs and pedestrian demand cannot be read from an alpha map.</div>
      <label class="rm-l">Lines${sel('rules')}</label>
      <div class="rm-l">Traffic drives on the<div class="seg" data-opt-seg="drive">
        <button type="button" class="seg-b" data-v="right" aria-pressed="${o.drive === 'right'}">Right</button>
        <button type="button" class="seg-b" data-v="left" aria-pressed="${o.drive === 'left'}">Left</button></div></div>
      <label class="rm-l">Colours${sel('colours')}</label>
      <label class="rm-l">Dash and line sizes${sel('sizes')}</label>
      ${rows}
      <div class="rm-kind"><div class="rm-kn">Minimum road width for markings</div><div class="rm-3">${cell('marking.min_m', 'x')}<div></div><div></div></div>
        <div class="note">The same cutoff applies to lane markings and procedural crossings in 2D and 3D. Highway width below controls classification, not this cutoff.</div></div>
      <div class="rm-kind"><div class="rm-kn">Highway centre-line groups</div>
        <label class="dc-chk"><input id="groupsOn" type="checkbox"${GROUPS.enabled ? ' checked' : ''}> Use selected highway groups</label>
        ${[1,2].map(g => `<div class="rm-row">Group ${g}: ${g === 1 ? 'double white line' : o.standard === 'saudi' ? 'single white centre line' : 'single yellow centre line'}
          <button type="button" class="btn secondary" data-group="${g}"${!S.gen.result || !GROUPS.enabled ? ' disabled' : ''}>${GROUPS.picking === g ? 'Done' : 'Pick'}</button>
          <small>${GROUPS.picks.filter(p => p.group === g).length} highways</small></div>`).join('')}
        <div class="note">Saudi colours follow the selected standard; choose Custom / previous guide to retain the yellow centre-line style. Group assignments are manual, not inferred no-passing restrictions.</div>
        <div class="note">Pick highlights all detected highways. Click to assign; click the same group again to remove. A highway belongs to one group.</div>
      </div>
      <div class="rm-kind"><div class="rm-kn">${esc(sb.label)}</div><div class="rm-3">${cell('setback_m', 'x')}<div></div><div></div></div></div>
      <label class="rm-l">Highways${sel('highways')}</label>
      <div class="rm-kind"><div class="rm-kn">Highways <small>streets this wide or wider: their lines run on past side
        streets and T-junctions, with no crossing or stop line across them there</small></div>
        <div class="rm-3">${cell('highway.min_m', 'x')}${cell('highway.lanes', 'x')}<div></div></div></div>
      ${!guide ? '<div class="note">Every line between two lanes is dashed alike, with the lane lines\' sizes, white (as before).</div>' : ''}
      ${learned ? '<div class="note">Widths and the dash\'s share of dash + gap come from your training images; the dash and gap together stay as set.</div>' : ''}
      ${warnBox(fields, MARKS.values, null, f => `${{ centre: 'Centre line', double: 'Double centre line', lane: 'Lane lines', edge: 'Edge lines', ends: 'Lines end', highway: 'Highways', eligibility: 'Marking eligibility' }[f.group]} ${f.label.toLowerCase()}`, 'lines')}
      ${stale ? '<div class="rm-stale">These differ from the texture\'s lines: press Update 2D texture to lay them.</div>' : ''}`;
    $('#groupsOn', box).addEventListener('change', e => {
      GROUPS.enabled = e.target.checked; GROUPS.picking = 0; saveHighwayGroups(); keepMarks(); renderMarks(); drawBridges();
    });
    $$('[data-group]', box).forEach(b => b.addEventListener('click', () => {
      GROUPS.picking = GROUPS.picking === +b.dataset.group ? 0 : +b.dataset.group;
      if(GROUPS.picking){ LANES.on = false; ISLS.pick = -1; AUTO.pick = -1; MAREA.on = false; }
      renderMarks(); drawBridges();
    }));
    // events
    $('.rm-chain', box).addEventListener('click', () => { MARKS.chain = !MARKS.chain; keepMarks(); renderMarks(); });
    $$('select[data-opt]', box).forEach(s => s.addEventListener('change', () => {
      MARKS.opts[s.dataset.opt] = s.value; keepMarks(); renderMarks(); marksMoved(s.dataset.opt);
      if(['standard','speed','posted_speed'].includes(s.dataset.opt)) renderDecals();
    }));
    $$('[data-opt-seg] button', box).forEach(b => b.addEventListener('click', () => {
      MARKS.opts.drive = b.dataset.v; keepMarks(); renderMarks(); marksMoved('drive');
    }));
    $$('.rm-in input', box).forEach(inp => inp.addEventListener('change', () => {
      const key = inp.closest('.rm-in').dataset.k, v = parseFloat(inp.value);
      if(!isFinite(v)){ renderMarks(); return; }
      const [vals, moved] = chainSet(L.fields, MARKS.values, key, v, null, MARKS.chain);
      MARKS.values = vals; keepMarks(); renderMarks();
      moved.forEach(k => { const c = $(`.rm-in[data-k="${k}"]`, box); if(c) c.classList.add('flash'); });
      if(moved.length) log(`Road markings: ${fieldLabel(key)} ${num(vals[key])} m; linked, so ${moved.map(k => `${fieldLabel(k)} ${num(vals[k])} m`).join(', ')} (the guide's proportions).`);
      if(LANES.on) drawBridges();
    }));
    const rb = $('[data-reset="lines"]', box);
    if(rb) rb.addEventListener('click', () => {
      L.fields.forEach(f => { MARKS.values[f.key] = f.guide; });
      keepMarks(); renderMarks(); log('Road markings: back to the guide\'s values.', 'ok');
    });
  }
  const fieldLabel = key => {
    const f = fieldOf(key);
    return `${{ centre: 'centre line', double: 'double centre line', lane: 'lane lines', edge: 'edge lines', ends: 'lines end', highway: 'highways', eligibility: 'marking eligibility' }[f.group]} ${f.label.toLowerCase()}`;
  };
  // the drive side and the colours move or change the junction markings: lay them again
  function marksMoved(what){
    if(['standard', 'speed', 'posted_speed', 'drive', 'colours'].includes(what) && DEC.layers.some(L => L.preset) && S.gen.decals){
      $('#btnPlaceDecals').classList.add('attention');
      $('#decalNote').textContent = 'The road markings changed: press Update decals to lay the junction markings again.';
    }
    if(['rules', 'highways'].includes(what) && LANES.on) drawBridges();
  }
  // "54 junction approaches (...), 1 roundabout, 1 big junction"
  function junctionMarksText(j){
    const a = j.approaches || {}, n = Object.entries(a).reduce((s, [k, v]) => s + (k === 'ring' ? 0 : v), 0);
    const parts = [];
    if(a.major) parts.push(`${a.major} on main roads`);
    if(a.minor) parts.push(`${a.minor} on side streets`);
    if(a.through) parts.push(`${a.through} on highways running on through (nothing laid across them)`);
    if(a.equal) parts.push(`${a.equal} where no road is the main one`);
    if(a.entry) parts.push(`${a.entry} roundabout entries`);
    return `Junction markings: ${n} street ends at junctions (${parts.join(', ') || 'none'}), ${j.roundabouts} roundabout${j.roundabouts === 1 ? '' : 's'}`
      + ` (its ring left clear), ${j.big_junctions} big open junction${j.big_junctions === 1 ? '' : 's'}`
      + (j.highway_through ? `; highways' lines run on through ${j.highway_through} junction${j.highway_through === 1 ? '' : 's'}.` : '.');
  }

  // ---- the presets in the Decals panel
  const PRESET_THUMB = new Map();                  // a layer as set -> its picture's URL
  function renderPresetPick(){
    const s = $('#rmPreset');
    if(!MARKS.spec || !s) return;
    const P = MARKS.spec.presets;
    s.innerHTML = `<option value="set">The junction set: ${MARKS.spec.junction_set.map(k => P[k].short.toLowerCase()).join(', ')}</option>`
      + `<optgroup label="One marking">${MARKS.spec.order.map(k => `<option value="${k}">${esc(P[k].label)}</option>`).join('')}</optgroup>`;
  }
  function newPreset(k){
    const P = MARKS.spec.presets[k], values = {}, opts = {};
    P.opts.forEach(o => { opts[o.key] = o.default; });
    P.fields.forEach(f => { values[f.key] = guideOf(f, opts); });
    return { preset: k, decal: 'preset:' + k, values, opts, chain: true, off: false };
  }
  $('#btnAddPreset').addEventListener('click', () => {
    if(!MARKS.spec) return;
    const v = $('#rmPreset').value, keys = v === 'set' ? MARKS.spec.junction_set : [v];
    keys.forEach(k => DEC.layers.push(newPreset(k)));
    log(`Road markings added: ${keys.map(k => MARKS.spec.presets[k].label).join(', ')}, at the guide's measurements. `
      + (S.gen.result ? 'Press Update decals to lay them.' : 'Generate texture lays them.'), 'ok');
    decalsChanged();
  });
  function presetWhere(L){
    const o = L.opts, v = presetDisplayValues(L), both = (a, b) => a && b ? 'every junction and roundabout entry' : a ? 'every junction' : b ? 'roundabout entries' : 'nowhere';
    switch(L.preset){
      case 'zebra': return `${num(v.length_m)} m wide, ${num(v.stripe_m)} m stripes; ${both(o.junctions, o.roundabouts)}`;
      case 'stop': return `${o.double ? 'double' : 'single'}, ${num(v.dist_m)} m behind the crossing; ${o.minor_only ? 'side streets' : o.signals ? 'every street into a junction' : 'every street into a junction (at T-junctions the side street)'}${o.signals ? ', traffic lights' : ''}`;
      case 'giveway': return `${[o.roundabouts && 'roundabout entries', o.minor && 'side streets'].filter(Boolean).join(' and ') || 'nowhere'}`;
      case 'arrows': return `${v.count} in each lane coming in, every ${num(v.every_m)} m, ${num(v.length_m)} m long; ${both(o.junctions, o.roundabouts)}`;
      case 'stopword': return `in each lane, ${num(v.dist_m)} m before the stop line, letters ${num(v.height_m)} m`;
      case 'cycle': return `${num(v.spacing_m)} m track beside the crossing; ${both(o.junctions, o.roundabouts)}`;
      case 'box': return o.all ? 'inside every junction' : `inside big open junctions (over ${num(v.big_m2)} m² and ${num(v.open_x)}× a plain crossing)`;
    }
    return '';
  }
  async function presetThumb(L, img){
    const preview = { ...L, values: presetDisplayValues(L) };
    const key = JSON.stringify([L.preset, preview.values, L.opts, MARKS.opts.colours]);
    if(PRESET_THUMB.has(key)){ img.src = PRESET_THUMB.get(key); return; }
    try{
      const r = await fetch(API + '/api/marks/preview', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ layer: preview, colours: MARKS.opts.colours }) });
      if(!r.ok) return;
      const url = URL.createObjectURL(await r.blob());
      PRESET_THUMB.set(key, url); img.src = url;
    }catch(_){}
  }
  function presetDisplayValues(L){
    const v = { ...L.values };
    if(MARKS.opts.standard !== 'saudi') return v;
    if(L.preset === 'zebra') Object.assign(v, {stripe_m:.5, gap_m:.5, length_m:Math.max(2,v.length_m), rb_back_m:Math.max(6,v.rb_back_m)});
    if(L.preset === 'stop') Object.assign(v, {width_m:.3, dist_m:1});
    if(L.preset === 'giveway') Object.assign(v, {seg_m:.5, gap_m:.2});
    if(L.preset === 'stopword' && +(MARKS.opts.posted_speed || MARKS.opts.speed) > 50) v.height_m = Math.max(2.5,v.height_m);
    return v;
  }
  function presetCard(L, i, box){
    if(!MARKS.spec) return;
    const P = MARKS.spec.presets[L.preset];
    if(!P) return;
    const card = document.createElement('div'); card.className = 'dc-layer rm-card';
    const fields = P.fields;
    // a layer kept from an earlier version: the values and options added since at their defaults
    L.opts = L.opts || {}; L.values = L.values || {};
    P.opts.forEach(o => { if(!(o.key in L.opts)) L.opts[o.key] = o.default; });
    fields.forEach(f => { if(!isFinite(L.values[f.key])) L.values[f.key] = guideOf(f, L.opts); });
    const displayed = presetDisplayValues(L);
    const prescribed = MARKS.opts.standard === 'saudi' ? {
      zebra:['stripe_m','gap_m'], stop:['width_m','dist_m'], giveway:['seg_m','gap_m']
    }[L.preset] || [] : [];
    card.innerHTML = `<div class="dc-head"><img class="dc-thumb" alt=""><div><b></b><small></small></div>
        <button type="button" class="dc-x" title="Remove this road marking layer">×</button></div>
      <div class="rm-row"><select data-switch title="Which road marking this layer is">${MARKS.spec.order.map(k =>
        `<option value="${k}"${k === L.preset ? ' selected' : ''}>${esc(MARKS.spec.presets[k].label)}</option>`).join('')}</select></div>
      <div class="rm-row">${chainButton(L.chain !== false, L.chain !== false ? 'Linked: values that belong together keep the guide\'s proportions. Press to change each on its own'
          : 'Unlinked: each value changes on its own. Press to keep the guide\'s proportions')}
        <label class="rm-on" title="Lay this layer, or keep it without laying it"><input type="checkbox"${L.off ? '' : ' checked'}> Laid on the map</label></div>
      <div class="rm-fields">${fields.map(f => {
        const g = guideOf(f, L.opts), v = displayed[f.key], isOff = Math.abs(v - g) > 1e-6, r = rangeOf(f, L.opts);
        return `<label class="rm-in${isOff ? ' off' : ''}" data-k="${f.key}" title="${esc(f.note || '')} (${esc(MARKS.spec.sources[f.src])})">`
          + `<span>${esc(f.label)}${f.int || f.unit ? '' : ' (m)'}</span>`
          + `<input type="number" step="${f.int ? 1 : f.min < 0.1 ? 0.05 : 0.5}" min="${f.min}" max="${f.max}" value="${num(v)}"${prescribed.includes(f.key) ? ' disabled' : ''}>`
          + `<small>${f.src === 'ours' ? 'ours' : 'guide'} ${num(g)}${r && (r[0] !== r[1]) ? ` (${num(r[0])}–${num(r[1])})` : ''}</small></label>`;
      }).join('')}</div>
      ${P.opts.map(o => o.choices
        ? `<div class="rm-row"><select data-o="${o.key}" title="${esc(o.label)}">${o.choices.map(([v, t]) => `<option value="${v}"${L.opts[o.key] === v ? ' selected' : ''}>${esc(t)}</option>`).join('')}</select></div>`
        : `<label class="dc-chk"><input type="checkbox" data-o="${o.key}"${L.opts[o.key] ? ' checked' : ''}> ${esc(o.label)}</label>`).join('')}
      ${warnBox(fields, L.values, L.opts, f => f.label, 'p' + i)}
      <div class="rm-note">${MARKS.opts.standard === 'saudi' ? esc(MARKS.spec.saudi_references[L.preset] || 'Custom geometry; no verified Saudi dimensions for this preset.') + '. Saudi mode overrides prescribed dimensions; relative fit may adapt the shown base sizes. Placement assumes the enabled crossing / traffic-control options are intended.' : esc(P.note)}</div>`;
    $('b', card).textContent = P.label;
    $('small', card).textContent = L.off ? 'off: kept, not laid' : presetWhere(L);
    presetThumb(L, $('.dc-thumb', card));
    $('.dc-x', card).addEventListener('click', () => {
      DEC.layers.splice(i, 1); log(`Road marking layer ${P.label} removed.`); decalsChanged();
    });
    $('.rm-on input', card).addEventListener('change', e => { L.off = !e.target.checked; decalsChanged(); });
    $('[data-switch]', card).addEventListener('change', e => {
      const n = newPreset(e.target.value); n.off = L.off; DEC.layers[i] = n;
      log(`Road marking layer is now ${MARKS.spec.presets[n.preset].label}, at the guide's measurements.`); decalsChanged();
    });
    $('.rm-chain', card).addEventListener('click', () => { L.chain = L.chain === false; decalsChanged(); });
    $$('.rm-in input', card).forEach(inp => inp.addEventListener('change', () => {
      const key = inp.closest('.rm-in').dataset.k, v = parseFloat(inp.value);
      if(!isFinite(v)){ renderDecals(); return; }
      const [vals, moved] = chainSet(fields, L.values, key, v, L.opts, L.chain !== false);
      L.values = vals;
      if(moved.length) log(`${P.label}: ${fields.find(f => f.key === key).label.toLowerCase()} ${num(vals[key])}; linked, so `
        + moved.map(k => `${fields.find(f => f.key === k).label.toLowerCase()} ${num(vals[k])}`).join(', ') + ' (the guide\'s proportions).');
      decalsChanged();
      moved.forEach(k => { const c = $$(`.rm-card .rm-in[data-k="${k}"]`)[0]; if(c) c.classList.add('flash'); });
    }));
    $$('[data-o]', card).forEach(c => c.addEventListener('change', () => {
      const k = c.dataset.o, before = { ...L.opts };
      L.opts[k] = c.type === 'checkbox' ? c.checked : c.value;
      // a value whose guide depends on this choice (the stop line with traffic lights: 1 m) follows it, if it was the guide's
      fields.forEach(f => {
        if(f.guide_if && Math.abs(L.values[f.key] - guideOf(f, before)) < 1e-6) L.values[f.key] = guideOf(f, L.opts);
      });
      decalsChanged();
    }));
    const rb = $(`[data-reset="p${i}"]`, card);
    if(rb) rb.addEventListener('click', () => {
      fields.forEach(f => { L.values[f.key] = guideOf(f, L.opts); });
      log(`${P.label}: back to the guide's values.`, 'ok'); decalsChanged();
    });
    box.appendChild(card);
  }

  /* ------------------------------------------------ decals */
  // Pictures laid on the streets (app/decals.py). Each imported decal is a layer with
  // where it goes: randomly along the streets, or at the junctions' edges (a share
  // of them, all of them, or two opposite ones per junction). Kept on the server for
  // every map; Place decals (and Generate texture) lays them on the generated texture
  const DEC = { decals: [], layers: [], timer: null };
  function fitAllDecals(enabled){
    DEC.layers.forEach(L => {
      if(L.preset){ L.opts = L.opts || {}; L.opts.relative_fit = enabled; }
      else L.relative_fit = enabled;
    });
    decalsChanged();
    log(enabled ? 'Relative scale and road rotation enabled for all decal layers. Press Update 2D texture.' : 'Relative fit disabled for all decal layers. Press Update 2D texture.');
  }
  $('#btnFitAllDecals').addEventListener('click', () => fitAllDecals(true));
  $('#btnFixedAllDecals').addEventListener('click', () => fitAllDecals(false));
  const decalById = id => DEC.decals.find(d => d.id === id);
  async function loadDecals(){
    try{ const r = await api('/api/decals'); DEC.decals = r.decals || []; DEC.layers = r.layers || []; }catch(_){}
    renderDecals();
  }
  function decalsChanged(){
    clearTimeout(DEC.timer);
    const layers = JSON.parse(JSON.stringify(DEC.layers));
    DEC.timer = setTimeout(() => api('/api/decals/layers', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ layers }) }).catch(e => log('Decal layers not saved: ' + e.message, 'bad')), 300);
    renderDecals();
    $('#decalNote').textContent = !S.gen.result ? 'Generate texture lays them.'
      : S.gen.decals ? 'Changed: press Update decals. Only the decals that change are painted again; the rest of the texture stays.'
      : 'Changed: press Place decals to lay them.';
    if(S.gen.decals) $('#btnPlaceDecals').classList.add('attention');
  }
  // a layer's places, in words (the card's second line)
  function decalWhere(L){
    const w = [];
    if(L.inter) w.push(L.all_edges ? 'every junction edge' : L.even_edges ? 'two opposite edges per junction' : `${L.chance}% of junction edges`);
    if(L.streets) w.push(`streets (weight ${L.streets_w})`);
    if(L.jn) w.push(`junctions (weight ${L.jn_w})`);
    if(L.edge) w.push(`${L.edge_side} kerb at junctions (weight ${L.edge_w})`);
    return w.length ? w.join(', ') : 'off';
  }
  const DEC_JUNC = ['junctions', 'all_edges', 'even_edges'];
  function renderDecals(){
    const box = $('#decalList'); box.innerHTML = '';
    DEC.layers.forEach((L, i) => {
      if(L.preset){ presetCard(L, i, box); return; }
      const d = decalById(L.decal); if(!d) return;
      const card = document.createElement('div'); card.className = 'dc-layer';
      card.innerHTML = `<div class="dc-head"><img class="dc-thumb" alt=""><div><b></b><small></small></div>
          <button type="button" class="dc-x" title="Remove this decal layer">×</button></div>
        <label class="dc-chk" title="Width is the artwork size on a 6 m reference road. Scale uniformly for each road and align to its local direction; overrides random turning while on."><input type="checkbox" data-k="relative_fit"> Fit scale and rotation to each road</label>
        <label class="dc-chk" title="Across the street, right where it meets a junction, its top towards the junction (a crossing, a stop line)"><input type="checkbox" data-k="inter"> Place intersections</label>
        <div class="dc-sub" data-for="inter">
          <label class="dc-chk" title="At a random share of the junctions' edges"><input type="checkbox" data-k="junctions"> Place at junctions</label>
          <label class="dc-chk" title="At every edge of every junction"><input type="checkbox" data-k="all_edges"> At all edges</label>
          <label class="dc-chk" title="At two opposite edges of every junction: the two streets most in line"><input type="checkbox" data-k="even_edges"> Even edges</label>
          <div class="dc-nums">
            <label data-only="chance" title="The share of junction edges that get one">Share of edges (%)<input type="number" step="5" min="0" max="100" data-n="chance"></label>
            <label title="0: right at the junction's edge. More: further back along the street">Back from the edge (m)<input type="number" step="0.5" min="0" data-n="back_m"></label>
          </div>
          <label class="dc-chk" title="Scaled to the road's width at that edge (a crossing from kerb to kerb)"><input type="checkbox" data-k="fit"> Fit the road's width</label>
        </div>
        <label class="dc-chk" title="Anywhere on the road, streets and junctions alike, at random"><input type="checkbox" data-k="streets"> Place on streets</label>
        <div class="dc-sub" data-for="streets">
          <label class="dc-w" title="More weight, more decals: 50 is about one per 400 m² of road, 100 twice as many">Weight<input type="range" min="0" max="300" step="5" data-w="streets_w"><output></output></label>
        </div>
        <label class="dc-chk" title="Inside the junctions only: T, Y, crossroads and any other junction"><input type="checkbox" data-k="jn"> Place on junctions</label>
        <div class="dc-sub" data-for="jn">
          <label class="dc-w" title="More weight, more decals: 100 is one per junction on average, 50 one in two junctions, 200 two each">Weight<input type="range" min="0" max="300" step="5" data-w="jn_w"><output></output></label>
        </div>
        <label class="dc-chk" title="Next to the junctions, against the right or the left kerb of each street, its top towards the junction (an arrow before a junction)"><input type="checkbox" data-k="edge"> Place junction edge</label>
        <div class="dc-sub" data-for="edge">
          <div class="dc-side" title="Which kerb, as seen driving along the street towards the junction">
            <label class="dc-chk"><input type="radio" value="right"> Right</label>
            <label class="dc-chk"><input type="radio" value="left"> Left</label>
          </div>
          <label class="dc-w" title="More weight, more decals: 100 is one at every street end, 50 at half of them, 200 two one behind the other">Weight<input type="range" min="0" max="300" step="5" data-w="edge_w"><output></output></label>
        </div>
        <div class="dc-nums">
          <label data-only="width" title="When relative fitting is on, this is the width on a 6 m reference road; the picture keeps its proportions">Width / reference width (m)<input type="number" step="0.1" min="0.1" data-n="width_m"></label>
        </div>
        <label class="dc-chk" data-only="spin" title="On streets and junctions: turned any way (a manhole cover, a stain); off, it lies along the street, facing either way"><input type="checkbox" data-k="spin"> Turn randomly</label>`;
      $('.dc-thumb', card).src = API + d.url;
      $('b', card).textContent = d.name;
      const fitted = L.inter && L.fit && !(L.streets || L.jn || L.edge);
      const relative = !!L.relative_fit;
      const len = L.width_m * d.height / d.width;
      $('small', card).textContent = (relative ? 'relative to each road (6 m reference)' : fitted ? 'road width × in proportion' : `${L.width_m} m × ${len.toFixed(1)} m`) + ', ' + decalWhere(L);
      $$('[data-k]', card).forEach(c => {
        const k = c.dataset.k; c.checked = !!L[k];
        c.addEventListener('change', () => {
          L[k] = c.checked;
          // the three intersection choices: one at a time, and one of them always on
          if(DEC_JUNC.includes(k)){
            if(c.checked) DEC_JUNC.forEach(o => { if(o !== k) L[o] = false; });
            else if(!DEC_JUNC.some(o => L[o])) L[k] = true;
          }
          if(k === 'inter' && c.checked && !DEC_JUNC.some(o => L[o])) L.all_edges = true;
          decalsChanged();
        });
      });
      // right or left kerb: one of the two
      const side = 'dcside' + i;
      $$('.dc-side input', card).forEach(r => {
        r.name = side; r.checked = (L.edge_side || 'right') === r.value;
        r.addEventListener('change', () => { if(r.checked){ L.edge_side = r.value; decalsChanged(); } });
      });
      $$('[data-n]', card).forEach(n => {
        n.value = L[n.dataset.n];
        n.addEventListener('change', () => { const v = parseFloat(n.value); if(isFinite(v)) L[n.dataset.n] = v; decalsChanged(); });
      });
      $$('[data-w]', card).forEach(n => {
        const out = n.nextElementSibling;
        n.value = L[n.dataset.w] ?? 50; out.textContent = n.value;
        n.addEventListener('input', () => { out.textContent = n.value; });
        n.addEventListener('change', () => { L[n.dataset.w] = +n.value; decalsChanged(); });
      });
      $$('[data-for]', card).forEach(e => { e.hidden = !L[e.dataset.for]; });
      $$('[data-only]', card).forEach(e => {
        const o = e.dataset.only;
        e.hidden = o === 'chance' ? !L.junctions : o === 'spin' ? L.relative_fit || !(L.streets || L.jn) : o === 'width' ? fitted : false;
      });
      $('.dc-x', card).addEventListener('click', async () => {
        DEC.layers.splice(i, 1);
        log(`Decal layer ${d.name} removed.`);
        if(!DEC.layers.some(x => x.decal === d.id)){
          try{ await api('/api/decals/' + d.id, { method:'DELETE' }); DEC.decals = DEC.decals.filter(x => x.id !== d.id); }catch(_){}
        }
        decalsChanged();
      });
      box.appendChild(card);
    });
    $('#decalCount').textContent = DEC.layers.length ? String(DEC.layers.length) : '';
    satPlaceState();
    $('#btnPlaceDecals').disabled = !S.gen.result || !(DEC.layers.length || S.gen.decals);
    // laid once: the button updates them, painting again only where decals change
    $('#btnPlaceDecals').textContent = S.gen.decals ? 'Update decals' : 'Place decals';
    $('#btnPlaceDecals').title = S.gen.decals
      ? 'Lay the decals again where they changed: only the decals added, moved or removed are painted again, the rest of the texture stays'
      : 'Lay the decals on the generated texture (Generate texture lays them too)';
  }
  $('#btnImportDecal').addEventListener('click', () => { $('#decalFile').value = ''; $('#decalFile').click(); });
  $('#decalFile').addEventListener('change', async () => {
    const f = $('#decalFile').files[0]; if(!f) return;
    const fd = new FormData(); fd.append('file', f);
    try{
      const d = await api('/api/decals/import', { method:'POST', body: fd });
      DEC.decals.push(d);
      const wide = d.width >= 1.5 * d.height;
      // a wide picture (a crossing): across every junction edge, kerb to kerb; else here and there on the streets
      DEC.layers.push({ decal: d.id, width_m: wide ? 3 : 1, spin: false,
                        inter: wide, junctions: false, all_edges: true, even_edges: false, chance: 50, back_m: 0, fit: wide,
                        streets: !wide, streets_w: 50, jn: false, jn_w: 50, edge: false, edge_side: 'right', edge_w: 100 });
      log(`Decal ${d.name} imported (${d.width} × ${d.height} px${d.see_through ? '' : ', no transparent background: it will show as a rectangle'}): `
        + 'set where it goes, then Place decals.', d.see_through ? 'ok' : 'bad');
      decalsChanged();
    }catch(e){ log('Decal not imported: ' + e.message, 'bad'); }
  });
  // "12 at intersections, 30 on streets": a layer's decals by where they lie
  const DEC_AT = { intersection: 'at intersections', edge: 'at junction edges', junction: 'in junctions', street: 'on streets' };
  const decalCounted = l => {
    const parts = Object.keys(DEC_AT).filter(k => l[k]);
    return parts.length > 1 ? ' (' + parts.map(k => `${l[k]} ${DEC_AT[k]}`).join(', ') + ')' : parts.length ? ' ' + DEC_AT[parts[0]] : '';
  };
  // lay the decals on the generated texture: their places, the Decals layer, and the
  // road texture with them painted in or not (Decals, under 3D model)
  async function applyDecals(){
    if(!S.gen.result || !(DEC.layers.length || S.gen.decals)) return;
    const gid = S.gen.result.id, mode = $('#decalMode').value;
    status('Laying the decals…');
    try{
      const res = await api('/api/decals/apply', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ generation: gid, layers: DEC.layers, mode, seed: +$('#seed').value, mark_area: MAREA.area,
                               marks: marksPayload() }) });
      if(!S.gen.result || S.gen.result.id !== gid) return;
      S.gen.decals = res; S.gen.decalVersion = Date.now();
      S.gen.result.urls.result = res.urls.result;
      if(S.gen.islandTex && S.gen.islandTex.gid === gid) S.gen.islandTex.urls.full = res.urls.full;
      $('#btnPlaceDecals').classList.remove('attention');
      const pt = res.partial;
      if(pt && !pt.changed){
        log(`Decals: nothing changed (${res.count.toLocaleString()} laid as before); the texture is as it was.`, 'ok');
        $('#decalNote').textContent = `${res.count.toLocaleString()} decals laid.`;
        status('Ready.'); renderDecals(); return;
      }
      if(res.lanes_moved) log(`Lane lines laid again on ${res.lanes_moved.streets} street(s) so they end at the junction markings `
        + `(${res.lanes_moved.seconds.toFixed(1)} s).`, 'ok');
      S.gen.decalDrive = MARKS.opts.drive;
      if(pt) log(`Decals updated in ${res.seconds.toFixed(1)} s: ${pt.changed - pt.gone} laid where they changed, ${pt.gone} taken away; `
        + `only their places were painted again (${(pt.share * 100).toFixed(1)}% of the texture), the rest stays as it was.`, 'ok');
      log(`Decals: ${res.count.toLocaleString()} laid, ${mode === 'painted' ? 'painted into the road texture and part of the road in the 3D model'
                                                            : 'on a layer of their own and an object of their own in the 3D model'}`
        + (res.layers.length ? ': ' + res.layers.map(l => `${l.count} ${l.name}` + decalCounted(l)).join(', ') : '') + '.', pt ? '' : 'ok');
      $('#decalNote').textContent = `${res.count.toLocaleString()} decals laid.`;
      try{ drawInto($('#lp-decals'), await loadImage(API + res.urls.decals)); }catch(_){}
      if(['result', 'decals', 'full'].includes(S.gen.layer)) await showLayer(S.gen.layer, !!pt);
      window.dispatchEvent(new Event('materials'));            // the 3D tab's scene is out of date
      if((S.gen.top && S.gen.top.gid === gid) || S.gen.topBusy) refreshTopView(gid, !!pt);
    }catch(e){ log('Decals not laid: ' + e.message, 'bad'); }
    status('Ready.'); renderDecals();
  }
  $('#btnPlaceDecals').addEventListener('click', () => updateTexture2D());
  $('#decalMode').addEventListener('change', () => { if(S.gen.decals) updateTexture2D(); });

  /* ------------------------------------------------ objects: layers and placements */
  // An object is a layer. A placement is a rectangle of copies of one object:
  // nx copies along the object's width (X), ny along its depth (Y), with a gap
  // between neighbours. Rectangles are kept in mask pixels, sizes in metres.
  S.gen.objects = []; S.gen.objSel = null; S.gen.folSel = null; S.gen.placements = []; S.gen.plSel = -1;
  // Objects and foliage work alike: layers (one object or plant each), packages
  // (several mixed) and placements, foliage ones marked foliage. Each has its own
  // panel; the placement panel moves to the panel of the selected placement
  const KIND = {
    objects: { foliage: false, list: '#objList', count: '#objCount', add: '#btnAddObj', pkgList: '#pkgList', pkgCount: '#pkgCount',
               sel: 'objSel', host: '#objPlHost', dup: '#btnDupPl', flip: '#btnMirrorPl', word: 'object', words: 'objects' },
    foliage: { foliage: true, list: '#folList', count: '#folCount', add: '#btnAddFol', pkgList: '#folPkgList', pkgCount: '#folPkgCount',
               sel: 'folSel', host: '#folPlHost', dup: '#btnDupFol', flip: '#btnMirrorFol', word: 'plant', words: 'plants' },
  };
  const kindOf = x => x && x.foliage ? KIND.foliage : KIND.objects;
  let maskPixels = null;

  function objById(id){ return S.gen.objects.find(o => o.id === id); }
  function objSize(o){
    const k = o.scale || 1, w = o.width_m * k, d = o.depth_m * k;
    return (o.turn || 0) % 2 ? [d, w, o.height_m * k] : [w, d, o.height_m * k];   // a quarter turn swaps them
  }
  const esc = t => String(t).replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));

  // Packages: several objects mixed in one placement. Each slot is an imported
  // object with a weight; each slot has its own colour on the map.
  S.gen.packages = [];
  const pkgById = id => S.gen.packages.find(k => k.id === id);
  const SLOT_COLOURS = ['216,96,76', '76,163,107', '201,162,39', '142,107,201', '63,167,201', '201,107,160', '122,143,61', '181,112,58'];
  function pkgSlots(pk){
    // the slots whose object is there, with footprints in metres (w along a row, d across)
    return (pk ? pk.slots : []).map((sl, k) => {
      const o = objById(sl.object); if(!o) return null;
      const [w, d] = objSize(o);
      return { o, w, d, weight: sl.weight, turn: o.turn || 0, colour: SLOT_COLOURS[k % SLOT_COLOURS.length] };
    }).filter(Boolean);
  }

  // A curved placement: copies along a smooth line through its points. These
  // are the same steps as app/curves.py (keep the two alike), so the map shows
  // exactly what the 3D model gets. Between points the line is a centripetal
  // Catmull-Rom curve: it passes through every point and never loops.
  const CURVE_STEPS = 24;                                   // samples per span between two points
  function curveSamples(P){
    if(P.length < 2) return P.map(q => q.slice());
    const n = P.length;
    const ext = [[2*P[0][0] - P[1][0], 2*P[0][1] - P[1][1]], ...P,
                 [2*P[n-1][0] - P[n-2][0], 2*P[n-1][1] - P[n-2][1]]];
    const mix = (a, wa, b, wb, den) => [(wa*a[0] + wb*b[0]) / den, (wa*a[1] + wb*b[1]) / den];
    const out = [];
    for(let i = 1; i < ext.length - 2; i++){
      const p0 = ext[i-1], p1 = ext[i], p2 = ext[i+1], p3 = ext[i+2];
      const t0 = 0;
      const t1 = t0 + Math.max(Math.sqrt(Math.hypot(p1[0]-p0[0], p1[1]-p0[1])), 1e-6);
      const t2 = t1 + Math.max(Math.sqrt(Math.hypot(p2[0]-p1[0], p2[1]-p1[1])), 1e-6);
      const t3 = t2 + Math.max(Math.sqrt(Math.hypot(p3[0]-p2[0], p3[1]-p2[1])), 1e-6);
      for(let k = 0; k < CURVE_STEPS; k++){
        const t = t1 + (t2 - t1) * k / CURVE_STEPS;
        const a1 = mix(p0, t1 - t, p1, t - t0, t1 - t0);
        const a2 = mix(p1, t2 - t, p2, t - t1, t2 - t1);
        const a3 = mix(p2, t3 - t, p3, t - t2, t3 - t2);
        const b1 = mix(a1, t2 - t, a2, t - t0, t2 - t0);
        const b2 = mix(a2, t3 - t, a3, t - t1, t3 - t1);
        out.push(mix(b1, t2 - t, b2, t - t1, t2 - t1));
      }
    }
    out.push(P[n-1].slice());
    return out;
  }
  // the line through the points: its samples, their spacing, distance along it, length
  function lineOf(P){
    const S = curveSamples(P), seg = [], s = [0];
    for(let k = 0; k + 1 < S.length; k++){ seg.push(Math.hypot(S[k+1][0]-S[k][0], S[k+1][1]-S[k][1])); s.push(s[k] + seg[k]); }
    return { S, seg, s, L: s[s.length - 1] };
  }
  // the point at distance `at` along the line, and the line's direction there
  function lineAt(ln, at){
    const { S, seg, s, L } = ln;
    if(!seg.length || L <= 0) return [S.length ? S[0].slice() : [0, 0], 0];
    let k = 0;
    while(k + 1 < s.length && s[k+1] <= at) k++;
    k = Math.min(Math.max(k, 0), seg.length - 1);
    const t = seg[k] > 0 ? (at - s[k]) / seg[k] : 0;
    const pos = [S[k][0] + (S[k+1][0] - S[k][0])*t, S[k][1] + (S[k+1][1] - S[k][1])*t];
    let j = k;                                              // the direction of the nearest span with length
    while(j < seg.length - 1 && seg[j] <= 1e-12) j++;
    while(j > 0 && seg[j] <= 1e-12) j--;
    return [pos, Math.atan2(S[j+1][1] - S[j][1], S[j+1][0] - S[j][0])];
  }
  // Several curve lines (each with as many points): the first shapes the front
  // row, the last the back row, the others spread evenly between; a row between
  // two lines follows a blend of the two, point by point (5 rows on 2 lines: the
  // second row is 3/4 of the first line and 1/4 of the second). Each row's line
  // and its points (null with one line); rows on the same line share it
  function rowLines(P, ny){
    if(!Array.isArray(P[0][0]) || P.length === 1){
      const ln = lineOf(Array.isArray(P[0][0]) ? P[0] : P);
      return { lines: Array(ny).fill(ln), pts: null };
    }
    const m = P.length, lines = [], pts = [], made = new Map();
    for(let r = 0; r < ny; r++){
      const x = ny > 1 ? r * (m - 1) / (ny - 1) : 0;      // where the row is among the lines
      const a = Math.min(Math.floor(x), m - 2), f = x - a, key = `${a}:${f}`;
      if(!made.has(key)){
        const q = f === 0 ? P[a].map(z => z.slice()) : P[a].map((z, j) => [(1 - f)*z[0] + f*P[a+1][j][0], (1 - f)*z[1] + f*P[a+1][j][1]]);
        made.set(key, [q, lineOf(q)]);
      }
      pts.push(made.get(key)[0]); lines.push(made.get(key)[1]);
    }
    return { lines, pts };
  }
  // a distance along line la to the same place along line lb: lines with as many
  // points have as many samples, and the same sample on each is the same place
  function carry(la, lb, at){
    if(la === lb || !la.seg.length) return at;
    let k = 0;
    while(k + 1 < la.s.length && la.s[k+1] <= at) k++;
    k = Math.min(Math.max(k, 0), la.seg.length - 1);
    const t = la.seg[k] > 0 ? (at - la.s[k]) / la.seg[k] : 0;
    return lb.s[k] + t * lb.seg[k];
  }
  // a footprint's size along its row and across it once turned by deg degrees:
  // a turned object takes its whole turned outline, so the spaces around it stay
  // as set (neighbours and the rows behind move instead)
  function turned(w, d, deg){
    if(!deg) return [w, d];
    const t = deg * Math.PI / 180, c = Math.abs(Math.cos(t)), s = Math.abs(Math.sin(t));
    return [w * c + d * s, w * s + d * c];
  }
  const turnAt = (turns, r, i) => +((turns || {})[`${r + 1}-${i + 1}`] || 0);
  // Objects alignment, as in app/curves.py: align = { rows, last, columns, exclude }.
  // rows: every second row turned round; last: the last row turned round (back
  // again if rows had turned it); columns: the objects at either end of a row
  // face out of that end, in every row (first column 270 degrees clockwise from
  // row 1's direction, last 90; swapped when the line is flipped or the rectangle
  // mirrored); exclude: the first and last rows keep their ends as their row.
  // A row turned round lines its fronts up on its back edge, the side it faces
  const aligned = al => !!al && !!(al.rows || al.last || al.columns);
  const turnedRound = (al, r, ny) => !!(al.rows && r % 2) !== !!(al.last && r === ny - 1);
  const endsOut = (al, r, ny) => !!al.columns && !(al.exclude && (r === 0 || r === ny - 1));
  function autoTurn(al, r, i, n, ny, flip){
    // n: the objects in the row, or null while it is being filled (not the last yet)
    if(!al) return 0;
    if(endsOut(al, r, ny) && (n === null || n > 1)){
      if(i === 0) return flip ? 90 : 270;
      if(n !== null && i === n - 1) return flip ? 270 : 90;
    }
    return turnedRound(al, r, ny) ? 180 : 0;
  }
  const DEG = Math.PI / 180;
  // Random transform (foliage), as in app/curves.py: jitter = { seed, scale:[min,max],
  // rotate:[min,max] (degrees, clockwise), offset:[min,max] (metres), overlap }. Each
  // copy, by its row and column, has its own random numbers: its scale, turn, and
  // offset (a distance in a random direction). With overlap off each copy takes the
  // room of its scaled and turned outline with its offset on every side
  function jitterOf(jitter, r, i){
    if(!jitter) return [1, 0, 0, 0];
    const seed = rowSeed(((jitter.seed ?? 1) >>> 0 ^ 0x3C6EF372) >>> 0, r);
    const rnd = rng((seed ^ Math.imul(i + 1, 0x85EBCA6B)) >>> 0);
    const [s0, s1] = (jitter.scale || [1, 1]).map(x => Math.max(0.01, +x)).sort((a, b) => a - b);
    const [a0, a1] = (jitter.rotate || [0, 0]).map(x => +x).sort((a, b) => a - b);
    const [o0, o1] = (jitter.offset || [0, 0]).map(x => Math.max(0, +x)).sort((a, b) => a - b);
    return [s0 + rnd() * (s1 - s0), a0 + rnd() * (a1 - a0), o0 + rnd() * (o1 - o0), rnd() * 2 * Math.PI];
  }
  const roomy = jitter => !!jitter && jitter.overlap === false;
  function sizeOf(w, d, deg, jitter, r, i){
    // a copy's room along its row and across it (see app/curves.py _size)
    if(!roomy(jitter)) return turned(w, d, deg);
    const [s, rot, off] = jitterOf(jitter, r, i), [ww, dd] = turned(w * s, d * s, deg + rot);
    return [ww + 2 * off, dd + 2 * off];
  }
  // copies fill the line at the gap, centred on it; each turns with the curve:
  // its X along the line, its front (+Y) to the left of the line's direction,
  // or the right when flipped. With random spaces or turned copies each row has
  // its own gaps and sizes. Returns [x, y, angle] per copy (angle: its X, without
  // its own turn) and the copies in each row.
  function curveCopies(P, w, d, gx, gy, ny, flip, spaces, layout, turns, align, jitter){
    const { lines, pts } = rowLines(P, ny), spots = [];
    const hasTurns = turns && Object.keys(turns).length;
    if(!spaces && !hasTurns && !aligned(align) && !roomy(jitter)){
      const { fronts, Ly } = rowFronts(d, ny, rowGaps(spaces, ny - 1, gy));
      const lay = newLayout(layout, P, flip, fronts, Array(ny).fill(d), pts);
      const step = Math.max(w + gx, 1e-9), counts = [];
      for(let r = 0; r < ny; r++){
        const ln = lines[r], L = ln.L;
        const n = Math.max(1, Math.floor((L + gx) / step + 1e-6));
        const first = (L - (n*w + (n - 1)*gx)) / 2 + w / 2;
        counts.push(n);
        for(let i = 0; i < n; i++){
          lay.rows[r].items.push([first + i*step - w/2, first + i*step + w/2]);
          let [pos, ang] = lineAt(ln, Math.min(Math.max(first + i*step, 0), L));
          if(flip) ang += Math.PI;
          const v = [-Math.sin(ang), Math.cos(ang)], off = fronts[r] + d/2;   // across the line, towards the copy's back
          spots.push([pos[0] + v[0]*off, pos[1] + v[1]*off, ang]);
          lay.ids.push(`${r + 1}-${i + 1}`);
        }
      }
      return { spots, counts, n: Math.max(...counts), L: lines[0].L, Ly, samples: lines[0].S, lines, layout: lay };
    }
    // each row first: its copies' (turned) sizes, as many as fit with the gaps
    const rows = [];
    for(let r = 0; r < ny; r++){
      const L = lines[r].L;
      const tt = (i, n = null) => autoTurn(align, r, i, n, ny, flip) + turnAt(turns, r, i);
      const gap = spacer(spaces, r, gx), sizes = [sizeOf(w, d, tt(0), jitter, r, 0)], gaps = [0];
      let used = Math.max(sizes[0][0], 1e-6);
      while(gaps.length < 5000){
        const nxt = sizeOf(w, d, tt(gaps.length), jitter, r, gaps.length), g = gap();
        if(used + g + Math.max(nxt[0], 1e-6) > L + 1e-9) break;   // the row is full: always at least one
        gaps.push(g); sizes.push(nxt); used += g + Math.max(nxt[0], 1e-6);
      }
      if(align && endsOut(align, r, ny)){
        // the last one faces out of its end too: turned, and as many as still fit
        for(;;){
          const n = sizes.length;
          sizes[n - 1] = sizeOf(w, d, tt(n - 1, n), jitter, r, n - 1);
          used = Math.max(sizes[0][0], 1e-6);
          for(let i = 1; i < n; i++) used += gaps[i] + Math.max(sizes[i][0], 1e-6);
          if(n === 1 || used <= L + 1e-9) break;
          gaps.pop(); sizes.pop();
        }
      }
      rows.push([gaps, sizes, used]);
    }
    // then the rows, each as deep as its deepest (turned) copy
    const depths = rows.map(([, sz]) => Math.max(d, ...sz.map(z => z[1])));
    const { fronts, Ly } = rowFronts(depths, ny, rowGaps(spaces, ny - 1, gy), d);
    const lay = newLayout(layout, P, flip, fronts, depths, pts), counts = [];
    rows.forEach(([gaps, sizes, used], r) => {
      const ln = lines[r], L = ln.L, n = gaps.length, back = !!align && turnedRound(align, r, ny);
      counts.push(n);
      let at = (L - used) / 2;
      gaps.forEach((g, i) => {
        const [ww, dd] = sizes[i];
        at += g;
        let [pos, ang] = lineAt(ln, Math.min(Math.max(at + ww/2, 0), L));
        lay.rows[r].items.push([at, at + Math.max(ww, 1e-6)]);
        at += Math.max(ww, 1e-6);
        if(flip) ang += Math.PI;
        // fronts in line on the row's front edge, or its back edge when turned round
        const v = [-Math.sin(ang), Math.cos(ang)], off = back ? fronts[r] + depths[r] - dd/2 : fronts[r] + dd/2;
        spots.push([pos[0] + v[0]*off, pos[1] + v[1]*off, ang + autoTurn(align, r, i, n, ny, flip) * DEG]);
        lay.ids.push(`${r + 1}-${i + 1}`);
      });
    });
    return { spots, counts, n: Math.max(...counts), L: lines[0].L, Ly, samples: lines[0].S, lines, layout: lay };
  }
  // a rectangle of copies: nx along its width in each of ny rows, centred on c,
  // rows along angle a. With random spaces or turned copies each row has its own
  // gaps, sizes and length, and each row is as deep as its deepest copy.
  // Mirrored, the rows are laid out from the other end, the copies facing the same way
  function gridCopies(c, a, w, d, nx, ny, gx, gy, spaces, layout, turns, mirror, align, jitter){
    const u = [Math.cos(a), Math.sin(a)], v = [-u[1], u[0]], du = mirror ? [-u[0], -u[1]] : u;
    const rows = [];
    for(let r = 0; r < ny; r++){
      const gap = spacer(spaces, r, gx), gaps = [0];
      for(let i = 1; i < nx; i++) gaps.push(gap());
      const sizes = gaps.map((_, i) => sizeOf(w, d, autoTurn(align, r, i, nx, ny, !!mirror) + turnAt(turns, r, i), jitter, r, i));
      let Lr = 0;
      gaps.forEach((g, i) => { Lr += g; Lr += sizes[i][0]; });
      rows.push([gaps, sizes, Lr]);
    }
    const depths = rows.map(([, sz]) => Math.max(d, ...sz.map(z => z[1])));
    const { fronts, Ly } = rowFronts(depths, ny, rowGaps(spaces, ny - 1, gy), d);
    const spots = [], lengths = [], starts = [];
    rows.forEach(([gaps, sizes, Lr], r) => {
      lengths.push(Lr);
      let at = -Lr / 2;
      const row = [], back = !!align && turnedRound(align, r, ny);
      gaps.forEach((g, i) => {
        const [ww, dd] = sizes[i];
        at += g;
        const o1 = at + ww/2, o2 = back ? fronts[r] + depths[r] - dd/2 : fronts[r] + dd/2;
        spots.push([c[0] + du[0]*o1 + v[0]*o2, c[1] + du[1]*o1 + v[1]*o2, a + autoTurn(align, r, i, nx, ny, !!mirror) * DEG]);
        row.push([at, at + ww]);
        at += ww;
      });
      starts.push(row);
    });
    // the rectangle's middle line, as long as its longest row
    const half = Math.max(...lengths) / 2;
    const lay = newLayout(layout, [[c[0] - du[0]*half, c[1] - du[1]*half], [c[0] + du[0]*half, c[1] + du[1]*half]], !!mirror, fronts, depths);
    for(let r = 0; r < ny; r++){
      lay.rows[r].items = starts[r].map(([x0, x1]) => [x0 + half, x1 + half]);
      for(let i = 0; i < nx; i++) lay.ids.push(`${r + 1}-${i + 1}`);
    }
    return { spots, lengths, Ly, layout: lay };
  }

  // A package mixes several objects in one placement: each row is filled with
  // random picks (weighted, never the same object twice in a row), each taking
  // its own width with the same gap between every pair, the run centred; fronts
  // in line on the row's front edge. Same steps and the same random numbers as
  // app/curves.py (mulberry32, bit for bit), so the map shows the exported mix.
  function rng(seed){
    let a = seed | 0;
    return () => {
      a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, a | 1);
      t = t + Math.imul(t ^ t >>> 7, t | 61) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  const rowSeed = (seed, row) => (seed ^ Math.imul(row + 1, 0x9E3779B9)) >>> 0;   // each row its own stream
  function pickSlot(rnd, weights, prev){
    let cand = weights.map((w, k) => w > 0 ? k : -1).filter(k => k >= 0);
    if(cand.length > 1 && cand.includes(prev)) cand = cand.filter(k => k !== prev);
    let total = 0;
    for(const k of cand) total += weights[k];
    const r = rnd() * total;
    let acc = 0;
    for(const k of cand){ acc += weights[k]; if(r < acc) return k; }
    return cand[cand.length - 1];
  }
  // Random spaces: each gap along a row a random distance between the X min and
  // max, each gap between two rows one between the Y min and max (metres). Same
  // steps and random numbers as app/curves.py. spaces = { on, x:[min,max], y:[min,max], seed };
  // only used when on. Their own seed, apart from a package's mix.
  const spacesOf = p => (p.spaces && p.spaces.on) ? p.spaces : null;
  function spacer(spaces, row, even){
    if(!spaces) return () => even;
    const [lo, hi] = spaces.x.map(x => Math.max(0, +x)).sort((a, b) => a - b);
    const rnd = rng(rowSeed(((spaces.seed >>> 0) ^ 0x5BD1E995) >>> 0, row));
    return () => lo + rnd() * (hi - lo);
  }
  function rowGaps(spaces, n, even){
    if(!spaces) return Array(Math.max(n, 0)).fill(even);
    const [lo, hi] = spaces.y.map(x => Math.max(0, +x)).sort((a, b) => a - b);
    const rnd = rng(((spaces.seed >>> 0) ^ 0xA5A5A5A5) >>> 0), out = [];
    for(let i = 0; i < n; i++) out.push(lo + rnd() * (hi - lo));
    return out;
  }
  // with base (the rows' depth before any copy was turned) the rows keep their
  // front: a row made deeper by a turned copy pushes only the rows behind it back
  function rowFronts(depth, ny, gaps, base){
    const depths = Array.isArray(depth) ? depth : Array(ny).fill(depth);   // one for every row, or one per row
    let Ly = 0;
    for(const dd of depths) Ly += dd;
    for(const g of gaps) Ly += g;
    let Ly0 = Ly;
    if(base !== undefined && base !== null){
      Ly0 = 0;
      for(let r = 0; r < ny; r++) Ly0 += base;
      for(const g of gaps) Ly0 += g;
    }
    const fronts = [];
    let f = -Ly0 / 2;
    for(let r = 0; r < ny; r++){ if(r) f += depths[r - 1] + gaps[r - 1]; fronts.push(f); }
    return { fronts, Ly };
  }
  // slots: [w, d, weight] per object. Returns [x, y, angle, slot] per copy
  function packageCopies(P, slots, gx, gy, ny, flip, seed, spaces, layout, turns, align, jitter){
    const { lines, pts } = rowLines(P, ny);
    let weights = slots.map(q => Math.max(0, +q[2] || 0));
    if(!weights.some(w => w > 0)) weights = slots.map(() => 1);
    const D = Math.max(...slots.map(q => q[1]));          // rows are as deep as the deepest object
    const rows = [];
    for(let r = 0; r < ny; r++){
      const L = lines[r].L;
      const rnd = rng(rowSeed(seed, r)), gap = spacer(spaces, r, gx), picks = [], gaps = [], sizes = [];
      const tt = (i, n = null) => autoTurn(align, r, i, n, ny, flip) + turnAt(turns, r, i);
      let used = 0, prev = null;
      while(picks.length < 5000){
        const k = pickSlot(rnd, weights, prev);
        const [ww, dd] = sizeOf(slots[k][0], slots[k][1], tt(picks.length), jitter, r, picks.length);
        const g = picks.length ? gap() : 0;
        const need = Math.max(ww, 1e-6) + g;
        if(picks.length && used + need > L + 1e-9) break;      // the row is full: always at least one
        picks.push(k); gaps.push(g); sizes.push([ww, dd]); used += need; prev = k;
      }
      if(align && endsOut(align, r, ny)){
        // the last one faces out of its end too: turned, and as many as still fit
        for(;;){
          const n = picks.length, k = picks[n - 1];
          sizes[n - 1] = sizeOf(slots[k][0], slots[k][1], tt(n - 1, n), jitter, r, n - 1);
          used = 0;
          for(let i = 0; i < n; i++) used += Math.max(sizes[i][0], 1e-6) + gaps[i];
          if(n === 1 || used <= L + 1e-9) break;
          picks.pop(); gaps.pop(); sizes.pop();
        }
      }
      rows.push([picks, gaps, sizes, used]);
    }
    // each row as deep as the package's deepest object, or a deeper turned one in it
    const depths = rows.map(([, , sz]) => Math.max(D, ...sz.map(z => z[1])));
    const { fronts, Ly } = rowFronts(depths, ny, rowGaps(spaces, ny - 1, gy), D);
    const lay = newLayout(layout, P, flip, fronts, depths, pts);
    const spots = [], counts = [];
    rows.forEach(([picks, gaps, sizes, used], r) => {
      const ln = lines[r], L = ln.L, n = picks.length, back = !!align && turnedRound(align, r, ny);
      counts.push(n);
      let at = (L - used) / 2;
      picks.forEach((k, i) => {
        const [ww, dd] = sizes[i];
        at += gaps[i];
        let [pos, ang] = lineAt(ln, Math.min(Math.max(at + ww/2, 0), L));
        lay.rows[r].items.push([at, at + Math.max(ww, 1e-6)]);
        lay.ids.push(`${r + 1}-${i + 1}`);
        at += Math.max(ww, 1e-6);
        if(flip) ang += Math.PI;
        // fronts in line on the row's front edge, or its back edge when turned round
        const v = [-Math.sin(ang), Math.cos(ang)], off = back ? fronts[r] + depths[r] - dd/2 : fronts[r] + dd/2;
        spots.push([pos[0] + v[0]*off, pos[1] + v[1]*off, ang + autoTurn(align, r, i, n, ny, flip) * DEG, k]);
      });
    });
    return { spots, counts, L: lines[0].L, D, Ly, samples: lines[0].S, lines, layout: lay };
  }

  // Layout and cells, as in app/curves.py: every placement is rows of objects
  // along a line; objects have ids "row-column" from the top left. The spaces
  // split into cells for inner streets: x between neighbours in a row, y between
  // two rows beside the objects, j junctions where an x space meets a y space.
  // With several curve lines each row has its own line (its points), and its
  // distances are along it; a cell's are along its row's (y and j: the front one's)
  function newLayout(layout, points, flip, fronts, depths, rowPoints){
    const lay = layout || {}, first = Array.isArray(points[0][0]) ? points[0] : points;
    Object.assign(lay, { line: first.map(q => [+q[0], +q[1]]), flip: !!flip,
                         rows: fronts.map((f, r) => ({ front: f, depth: depths[r], items: [] })), ids: [] });
    if(rowPoints) lay.rows.forEach((row, r) => { row.line = rowPoints[r].map(q => [+q[0], +q[1]]); });
    return lay;
  }
  // row r's line: its own with several curve lines, else the layout's (cache: reused)
  function rowLine(lay, r, cache = new Map()){
    const key = lay.rows[r].line ? r : -1;
    if(!cache.has(key)) cache.set(key, lineOf(key >= 0 ? lay.rows[r].line : lay.line));
    return cache.get(key);
  }
  function cellsOf(lay, minM = 0.05, cache = new Map()){
    const rows = lay.rows, out = [];
    rows.forEach((row, r) => {
      const it = row.items;
      for(let i = 0; i + 1 < it.length; i++)
        if(it[i+1][0] - it[i][1] > minM)
          out.push({ id: `x${r + 1}-${i + 1}`, kind: 'x', row: r, s: [it[i][1], it[i+1][0]], o: [row.front, row.front + row.depth] });
    });
    for(let r = 0; r + 1 < rows.length; r++){
      const a = rows[r], b = rows[r + 1], o0 = a.front + a.depth, o1 = b.front;
      if(o1 - o0 <= minM || !a.items.length || !b.items.length) continue;
      let bi = b.items;
      if(a.line || b.line){
        // the row behind on a line of its own: its objects where they are along the front one's
        const la = rowLine(lay, r, cache), lb = rowLine(lay, r + 1, cache);
        bi = bi.map(([s0, s1]) => [carry(lb, la, s0), carry(lb, la, s1)]);
      }
      const lo = Math.min(a.items[0][0], bi[0][0]), hi = Math.max(a.items[a.items.length - 1][1], bi[bi.length - 1][1]);
      const gaps = [];
      for(const it of [a.items, bi])
        for(let i = 0; i + 1 < it.length; i++) if(it[i+1][0] - it[i][1] > minM) gaps.push([it[i][1], it[i+1][0]]);
      gaps.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
      const merged = [];
      for(const g of gaps){
        if(merged.length && g[0] <= merged[merged.length - 1][1]) merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], g[1]);
        else merged.push(g.slice());
      }
      let cur = lo, ky = 0, kj = 0;
      for(let [g0, g1] of merged){
        g0 = Math.max(g0, lo); g1 = Math.min(g1, hi);
        if(g0 - cur > minM){ ky++; out.push({ id: `y${r + 1}-${ky}`, kind: 'y', row: r, s: [cur, g0], o: [o0, o1] }); }
        kj++; out.push({ id: `j${r + 1}-${kj}`, kind: 'j', row: r, s: [g0, g1], o: [o0, o1] });
        cur = g1;
      }
      if(hi - cur > minM){ ky++; out.push({ id: `y${r + 1}-${ky}`, kind: 'y', row: r, s: [cur, hi], o: [o0, o1] }); }
    }
    return out;
  }
  // with far, another row's line, the far edge runs along that one, at the same places
  function stripPolygon(lay, s0, s1, o0, o1, step = 1.0, ln = null, far = null){
    ln = ln || lineOf(lay.line);
    const n = Math.max(1, Math.ceil((s1 - s0) / step - 1e-9)), sign = lay.flip ? -1 : 1, near = [], back = [];
    for(let k = 0; k <= n; k++){
      const at = Math.min(Math.max(s0 + (s1 - s0) * k / n, 0), ln.L);
      let [pos, ang] = lineAt(ln, at), v = [-Math.sin(ang) * sign, Math.cos(ang) * sign];
      near.push([pos[0] + v[0]*o0, pos[1] + v[1]*o0]);
      if(far && far !== ln){
        [pos, ang] = lineAt(far, Math.min(Math.max(carry(ln, far, at), 0), far.L));
        v = [-Math.sin(ang) * sign, Math.cos(ang) * sign];
      }
      back.push([pos[0] + v[0]*o1, pos[1] + v[1]*o1]);
    }
    return near.concat(back.reverse());
  }
  // a cell's area: along its row's line, a y or j cell reaching across to the next row's
  function cellPolygon(lay, c, step, cache = new Map()){
    const r = c.row || 0, ln = rowLine(lay, r, cache);
    const far = c.kind !== 'x' && r + 1 < lay.rows.length ? rowLine(lay, r + 1, cache) : ln;
    return stripPolygon(lay, c.s[0], c.s[1], c.o[0], c.o[1], step, ln, far);
  }

  function roadAt(x, y){
    // the mask itself: white is road. Read once per mask
    if(!S.gen.mask || !S.gen.mask.img) return false;
    if(!maskPixels || maskPixels.src !== S.gen.mask.id){
      const c = document.createElement('canvas'); c.width = S.gen.mask.width; c.height = S.gen.mask.height;
      const cx = c.getContext('2d'); cx.drawImage(S.gen.mask.img, 0, 0);
      maskPixels = { src: S.gen.mask.id, w: c.width, h: c.height, d: cx.getImageData(0, 0, c.width, c.height).data };
    }
    const xi = Math.round(x), yi = Math.round(y);
    if(xi < 0 || yi < 0 || xi >= maskPixels.w || yi >= maskPixels.h) return false;
    return maskPixels.d[(yi * maskPixels.w + xi) * 4] > 127;
  }

  const isCurve = p => Array.isArray(p.path) && p.path.length >= 2;
  // a curve's lines in mask pixels: the first (path), then any more curve lines,
  // each with as many points
  const curveLinesOf = p => [p.path].concat((p.lines || []).filter(q => q.length === p.path.length));
  function setCurveLines(p, all){ p.path = all[0]; if(all.length > 1) p.lines = all.slice(1); else delete p.lines; }
  // in metres for the maths: the one line, or all of them
  function curvePoints(p, m){
    const all = curveLinesOf(p).map(q => q.map(z => [z[0]*m, z[1]*m]));
    return all.length > 1 ? all : all[0];
  }

  function placementGeom(p){
    const m = mppMask();
    // one copy: its footprint in mask pixels (w, d in metres), turned so its X points along a
    const copyAt = (cx, cy, a, w, d, slot, id) => {
      if(p.jitter){
        // a plant's random offset across its row's frame, its random turn and scale
        const [r, i] = id.split('-').map(x => +x - 1), [s, rot, off, th] = jitterOf(p.jitter, r, i);
        cx += off * (Math.cos(th) * Math.cos(a) - Math.sin(th) * Math.sin(a)) / m;
        cy += off * (Math.cos(th) * Math.sin(a) + Math.sin(th) * Math.cos(a)) / m;
        a += rot * DEG; w *= s; d *= s;
      }
      a += turnOf(p, id);                                   // a single object turned on its own
      const u = [Math.cos(a), Math.sin(a)], v = [-u[1], u[0]], hw = w / m / 2, hd = d / m / 2;
      const pts = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([sx, sy]) => [cx + u[0]*sx*hw + v[0]*sy*hd, cy + u[1]*sx*hw + v[1]*sy*hd]);
      const onRoad = pts.concat([[cx, cy]]).some(q => roadAt(q[0], q[1]));
      return { cx, cy, a, u, v, pts, onRoad, w: w / m, d: d / m, slot: slot || null, id, empty: isEmpty(p, id) };
    };
    // the band the rows cover beside a curve, from o0 to o1 across it (towards the
    // copies' backs): blue between copies is the gap. With line2 (the back row's
    // own line) the far side runs along that one
    const along = (line, k) => {
      const a = line[Math.max(k - 1, 0)], b = line[Math.min(k + 1, line.length - 1)], len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      return [(b[0] - a[0]) / len, (b[1] - a[1]) / len];
    };
    const band = (line, o0, o1, flip, line2) => {
      const near = [], far = [], sg = flip ? -1 : 1;
      line.forEach((q, k) => {
        const t = along(line, k);
        near.push([q[0] - t[1]*sg*o0, q[1] + t[0]*sg*o0]);
      });
      (line2 || line).forEach((q, k) => {
        const t = along(line2 || line, k);
        far.push([q[0] - t[1]*sg*o1, q[1] + t[0]*sg*o1]);
      });
      return near.concat(far.reverse());
    };
    // a curve's outline, from the front row's front to the back row's back, each
    // on its row's line, and its curve lines as drawn: one line is the middle
    // line; with more, each sits on the row it shapes (between two rows when it
    // falls between), its points where they bend that row
    const curveShape = r => {
      const lay = r.layout, rows = lay.rows, ny = rows.length, sg = p.flip ? -1 : 1;
      const px = S => S.map(q => [q[0] / m, q[1] / m]);
      const line = px(r.samples), last = rows[ny - 1], own = r.lines[ny - 1] !== r.lines[0];
      const outer = own ? band(line, rows[0].front / m, (last.front + last.depth) / m, !!p.flip, px(r.lines[ny - 1].S))
                        : band(line, front0(lay), front0(lay) + r.Ly / m, !!p.flip);
      const all = curveLinesOf(p), k = all.length, cen = i => rows[i].front + rows[i].depth / 2;
      const centre = x => { const a = Math.min(Math.floor(x), ny - 1), b = Math.min(a + 1, ny - 1), f = x - a; return (1 - f)*cen(a) + f*cen(b); };
      const curveLines = all.map((pts, li) => {
        const off = k > 1 ? centre(ny > 1 ? li * (ny - 1) / (k - 1) : 0) / m : 0;
        const S = px(curveSamples(pts.map(q => [q[0]*m, q[1]*m]))), T = S.map((_, j) => along(S, j));
        const shown = off ? S.map((q, j) => [q[0] - T[j][1]*sg*off, q[1] + T[j][0]*sg*off]) : S;
        const at = j => Math.min(j * CURVE_STEPS, S.length - 1);
        return { samples: shown, off, handles: off ? pts.map((_, j) => shown[at(j)]) : pts.map(q => q.slice()),
                 tangents: pts.map((_, j) => T[at(j)]) };
      });
      return { line, outer, curveLines, Ly: r.Ly / m, lengths_m: r.lines.map(l => l.L) };
    };
    // a rectangle's outline: Lx long, across from o0 to o1 (centred when not given)
    const box = (a, Lx, Ly, o0 = -Ly/2) => {
      const u = [Math.cos(a), Math.sin(a)], v = [-u[1], u[0]];
      return { u, v, Lx, Ly, outer: [[-1, o0], [1, o0], [1, o0 + Ly], [-1, o0 + Ly]].map(([sx, o]) =>
        [p.cx + u[0]*sx*Lx/2 + v[0]*o, p.cy + u[1]*sx*Lx/2 + v[1]*o]) };
    };
    const front0 = lay => lay.rows[0].front / m;            // the first row's front, in mask pixels
    if(p.layout){
      // a layout (app/layout.py): an object on each building, its front to the building's street, its
      // own width and depth stretched to the building's ("w", "d" metres), as the 3D export takes them
      // (app/placements.py); each object its own colour. A quarter-turned object's frame is turned back
      // by its turn, so its front still faces the street: its footprint along that frame swaps
      const ids = [...new Set((p.items || []).map(it => it.object))], slots = {}, copies = [];
      ids.forEach((id, k) => { const o = objById(id); if(o) slots[id] = { o, turn: o.turn || 0, colour: SLOT_COLOURS[k % SLOT_COLOURS.length] }; });
      (p.items || []).forEach(it => {
        const sl = slots[it.object]; if(!sl) return;
        const odd = sl.turn % 2 === 1;
        const c = copyAt(it.x, it.y, (it.a - 90 * (sl.turn % 4)) * DEG, odd ? it.d : it.w, odd ? it.w : it.d, sl, it.id);
        c.onRoad = false;                     // on its building, by the map
        copies.push(c);
      });
      return { auto: true, layout: true, slots: Object.values(slots), copies };
    }
    if(p.auto){
      // an automatic placement: the objects it laid on the islands, as the 3D export takes them
      // (app/placements.py). "a" is where the front (+Y) faces: a quarter-turned object's frame
      // is turned back by its turn, so its front still faces the street
      const slots = p.package ? pkgSlots(pkgById(p.package)) : null, copies = [];
      (p.items || []).forEach(it => {
        const o = objById(it.object); if(!o) return;
        const [w, d] = objSize(o), slot = slots ? slots.find(sl => sl.o.id === it.object) || null : null;
        const c = copyAt(it.x, it.y, (it.a - 90 * ((o.turn || 0) % 4)) * DEG, w, d, slot, it.id);
        c.onRoad = false;                     // on its island by the rule (a mask pixel is coarser on a city)
        copies.push(c);
      });
      return { auto: true, pkg: !!p.package, slots, copies };
    }
    if(p.package){
      // a package: its mix along a line, straight through the rectangle or curved,
      // worked out in metres exactly as the 3D export does, then back to mask pixels
      const slots = pkgSlots(pkgById(p.package)); if(!slots.length) return null;
      const curve = isCurve(p), a = p.angle * Math.PI / 180, Lx = p.length || 0;
      // a rectangle is a straight line; mirrored, it is laid out from the other end, facing the same way
      const straight = [[p.cx - Math.cos(a)*Lx/2, p.cy - Math.sin(a)*Lx/2], [p.cx + Math.cos(a)*Lx/2, p.cy + Math.sin(a)*Lx/2]].map(q => [q[0]*m, q[1]*m]);
      const P = curve ? curvePoints(p, m) : p.mirror ? straight.reverse() : straight;
      const r = packageCopies(P, slots.map(s => [s.w, s.d, s.weight]),
                              p.gap_x, p.gap_y, p.ny, curve ? !!p.flip : !!p.mirror, p.seed >>> 0, spacesOf(p), null, p.turns, p.align, p.jitter);
      const copies = r.spots.map(([x, y, ang, k], n) => copyAt(x / m, y / m, ang, slots[k].w, slots[k].d, slots[k], r.layout.ids[n]));
      const Ly = r.Ly / m;
      const base = { pkg: true, slots, copies, counts: r.counts, length_m: r.L, layout: r.layout };
      if(curve) return { ...base, curve: true, ...curveShape(r) };
      return { ...base, ...box(a, Lx, Ly, front0(r.layout)) };
    }
    const o = objById(p.object); if(!o) return null;
    const [w, d] = objSize(o);
    if(isCurve(p)){
      // worked out in metres, as the 3D export does, then back to mask pixels
      const r = curveCopies(curvePoints(p, m), w, d, p.gap_x, p.gap_y, p.ny, !!p.flip, spacesOf(p), null, p.turns, p.align, p.jitter);
      return { curve: true, copies: r.spots.map(([x, y, a], n) => copyAt(x / m, y / m, a, w, d, null, r.layout.ids[n])),
               counts: r.counts, along: r.n, length_m: r.L, w: w/m, d: d/m, layout: r.layout, ...curveShape(r) };
    }
    // a rectangle of copies, in metres as the 3D export does; with random spaces
    // each row has its own length, and the rectangle is as long as the longest
    const a = p.angle * Math.PI / 180;
    const r = gridCopies([p.cx*m, p.cy*m], a, w, d, p.nx, p.ny, p.gap_x, p.gap_y, spacesOf(p), null, p.turns, !!p.mirror, p.align, p.jitter);
    const copies = r.spots.map(([x, y, ang], n) => copyAt(x / m, y / m, ang, w, d, null, r.layout.ids[n]));
    return { ...box(a, Math.max(...r.lengths) / m, r.Ly / m, front0(r.layout)), w: w/m, d: d/m, copies, layout: r.layout };
  }
  const turnOf = (p, id) => ((p.turns || {})[id] || 0) * Math.PI / 180;
  // Draw spaced: single objects made empty spots, by id; the rest of the grid stays put
  const isEmpty = (p, id) => !!(p.empty && p.empty.includes(id));
  function setEmpty(p, id, on){
    const s = new Set(p.empty || []);
    if(on) s.add(id); else s.delete(id);
    if(s.size) p.empty = [...s].sort(); else delete p.empty;
  }

  // the cells of a placement's spaces, as polygons in mask pixels
  function spaceCells(p, g0){
    if(!g0 || !g0.layout) return [];
    const m = mppMask(), cache = new Map();                 // the rows' lines
    return cellsOf(g0.layout, 0.05, cache).map(c => ({ ...c, poly: cellPolygon(g0.layout, c, 1.0, cache).map(q => [q[0] / m, q[1] / m]) }));
  }

  // editing a curve: points are kept in mask pixels, like the rectangles. A
  // point added or removed goes on (or off) every curve line, at the same place
  function curveCentre(p){
    // cx, cy stay meaningful (the middle of the points) for anything that reads them
    const all = curveLinesOf(p).flat();
    p.cx = all.reduce((s, q) => s + q[0], 0) / all.length;
    p.cy = all.reduce((s, q) => s + q[1], 0) / all.length;
  }
  function curveLine(pts){
    // a line's samples in mask pixels: CURVE_STEPS per span between two points
    const m = mppMask();
    return curveSamples(pts.map(q => [q[0]*m, q[1]*m])).map(q => [q[0]/m, q[1]/m]);
  }
  // the point at sample k + t (0 <= t <= 1) of a line's samples
  const sampleAt = (S, k, t) => [S[k][0] + (S[k+1][0] - S[k][0])*t, S[k][1] + (S[k+1][1] - S[k][1])*t];
  function insertPoint(p, span, where){
    // where(line's samples, its span's first sample) gives the new point on each line
    setCurveLines(p, curveLinesOf(p).map(pts => {
      const out = pts.map(q => q.slice());
      out.splice(span + 1, 0, where(curveLine(pts), span * CURVE_STEPS));
      return out;
    }));
  }
  function addCurvePointNear(p, q, g0){
    // the nearest spot on a line as drawn goes in between the two points of its
    // span, on every line at the same place, so the lines keep their shape until
    // the new point is dragged
    let best = null;
    (g0.curveLines || []).forEach((cl, li) => {
      const S = cl.samples;
      for(let k = 0; k + 1 < S.length; k++){
        const a = S[k], b = S[k+1], dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx*dx + dy*dy;
        const t = l2 > 0 ? Math.min(Math.max(((q[0] - a[0])*dx + (q[1] - a[1])*dy) / l2, 0), 1) : 0;
        const dd = Math.hypot(a[0] + dx*t - q[0], a[1] + dy*t - q[1]);
        if(!best || dd < best.dd) best = { dd, li, k, t };
      }
    });
    if(!best) return false;
    const span = Math.min(Math.floor(best.k / CURVE_STEPS), p.path.length - 2);
    const pts = curveLinesOf(p)[best.li], c = sampleAt(curveLine(pts), best.k, best.t);
    const near = i => Math.hypot(pts[i][0] - c[0], pts[i][1] - c[1]) < 1.5;
    if(near(span) || near(span + 1)) return false;                       // already a point there
    insertPoint(p, span, (S, k0) => sampleAt(S, best.k, best.t));
    return true;
  }
  function addCurvePointMiddle(p){
    // in the middle of the longest stretch between two points (over all the lines)
    const lines = curveLinesOf(p).map(curveLine);
    const spanLen = (S, i) => { let len = 0; for(let k = i*CURVE_STEPS; k < (i + 1)*CURVE_STEPS; k++) len += Math.hypot(S[k+1][0] - S[k][0], S[k+1][1] - S[k][1]); return len; };
    let bestSpan = 0, bestLen = -1;
    for(let i = 0; i + 1 < p.path.length; i++){
      const len = lines.reduce((sum, S) => sum + spanLen(S, i), 0);
      if(len > bestLen){ bestLen = len; bestSpan = i; }
    }
    insertPoint(p, bestSpan, (S, k0) => {
      // half way along this line's own stretch
      const half = spanLen(S, bestSpan) / 2;
      let run = 0;
      for(let k = k0; k < k0 + CURVE_STEPS; k++){
        const l = Math.hypot(S[k+1][0] - S[k][0], S[k+1][1] - S[k][1]);
        if(run + l >= half) return sampleAt(S, k, l > 0 ? (half - run) / l : 0);
        run += l;
      }
      return S[k0 + CURVE_STEPS].slice();
    });
  }
  function removeCurvePoint(p, j){
    setCurveLines(p, curveLinesOf(p).map(pts => pts.filter((_, i) => i !== j)));
  }
  // the line t of the way through the curve lines (0 the first, 1 the last), point by point
  function blendLines(all, t){
    const m = all.length;
    if(m === 1) return all[0].map(z => z.slice());
    const x = t * (m - 1), a = Math.min(Math.floor(x), m - 2), f = x - a;
    return all[a].map((z, j) => [(1 - f)*z[0] + f*all[a+1][j][0], (1 - f)*z[1] + f*all[a+1][j][1]]);
  }
  function spreadLines(p, k){
    // k curve lines, spread over the rows like the ones there: the first and last stay
    const all = curveLinesOf(p);
    setCurveLines(p, k === 1 ? [blendLines(all, 0.5)] : Array.from({ length: k }, (_, i) => blendLines(all, i / (k - 1))));
  }

  function drawScatter(f, hr){
    S.gen.placements.forEach((p, i) => {
      if(laid(p)) return;                                    // on the overlay (drawAutoOn)
      const g0 = placementGeom(p); if(!g0) return;
      const sel = i === S.gen.plSel;
      // an edit mode for this placement: 'streets' (draw inner streets) or 'objects' (turn single objects)
      const ed = S.gen.edit && S.gen.edit.i === i ? S.gen.edit.kind : null;
      const g = el('g', {}, layer);
      const P = q => `${q[0]*f},${q[1]*f}`;
      // the whole area in blue: what shows between the copies is the gap
      const area = el('polygon', { points: g0.outer.map(P).join(' '), class: ed ? '' : 'grab',
        fill:'rgba(70,130,220,.35)', stroke:'#4682DC', 'stroke-width': sel ? hr*0.6 : hr*0.35 }, g);
      if(!ed) area.addEventListener('pointerdown', e => startScatterDrag(e, i, 'move'));
      if(ed === 'streets') drawCells(g, p, g0, f, hr, P);
      g0.copies.forEach(c => {
        // a package's copies in their slot's colour, so the mix shows
        const col = c.slot ? c.slot.colour : p.foliage ? '76,163,107' : '216,96,76';
        if(c.empty || ed === 'spaces'){
          // an empty spot (Draw spaced): its outline dashed; in that mode every copy is clickable
          const poly = el('polygon', { points: c.pts.map(P).join(' '),
            fill: c.empty ? 'rgba(255,255,255,.06)' : `rgba(${col},.6)`, stroke: c.empty ? 'rgba(255,255,255,.85)' : `rgb(${col})`,
            'stroke-width': hr * 0.25, 'stroke-dasharray': c.empty ? `${hr*0.8},${hr*0.6}` : 'none',
            'pointer-events': ed === 'spaces' ? 'all' : 'none', class: ed === 'spaces' ? 'pt' : '' }, g);
          if(ed === 'spaces'){
            poly.addEventListener('pointerdown', e => spaceAt(e, i, c.id, !c.empty, true));
            poly.addEventListener('pointerenter', e => { if(S.gen.edit.paint != null && (e.buttons & 1)) spaceAt(e, i, c.id, S.gen.edit.paint, false); });
          }
          return;
        }
        const picked = ed === 'objects' && S.gen.edit.sel.has(c.id);
        const poly = el('polygon', { points: c.pts.map(P).join(' '), fill: c.onRoad ? 'rgba(120,120,120,.55)' : `rgba(${col},.6)`,
          stroke: picked ? '#FFD23F' : c.onRoad ? '#999' : `rgb(${col})`, 'stroke-width': picked ? hr*0.7 : hr*0.25,
          'pointer-events': ed === 'objects' ? 'all' : 'none', class: ed === 'objects' ? 'pt' : '' }, g);
        if(ed === 'objects') poly.addEventListener('pointerdown', e => pickObject(e, c.id));
        if(c.onRoad){
          el('line', { x1:c.pts[0][0]*f, y1:c.pts[0][1]*f, x2:c.pts[2][0]*f, y2:c.pts[2][1]*f, stroke:'#ddd', 'stroke-width': hr*0.25, 'pointer-events':'none' }, g);
          el('line', { x1:c.pts[1][0]*f, y1:c.pts[1][1]*f, x2:c.pts[3][0]*f, y2:c.pts[3][1]*f, stroke:'#ddd', 'stroke-width': hr*0.25, 'pointer-events':'none' }, g);
        }
      });
      const ob = p.package ? null : objById(p.object), turn = (ob && ob.turn) || 0;
      if(g0.curve){
        if(!g0.pkg) p.nx = g0.along;                         // copies along the line, for the panel and the file
        // the curve lines, dashed: one is the middle line; more sit on the rows they shape
        g0.curveLines.forEach(cl => el('polyline', { points: cl.samples.map(P).join(' '), fill:'none', stroke:'#fff', 'stroke-width': hr*0.3,
          'stroke-dasharray': `${hr*1.2},${hr*0.8}`, 'pointer-events':'none' }, g));
      }
      if((g0.curve || g0.pkg) && ed !== 'objects' && ed !== 'spaces' && !p.foliage){
        // a small arrow on each copy's front: Blender's +Y, turned with the curve and the object
        // (not on plants, whose random turns would make them point every way)
        g0.copies.forEach(c => {
          if(c.empty) return;
          const th = c.a + (c.slot ? c.slot.turn : turn) * Math.PI / 2, fd = [Math.sin(th), -Math.cos(th)], sd = [Math.cos(th), Math.sin(th)];
          const reach = Math.abs(fd[0]*c.u[0] + fd[1]*c.u[1]) * c.w/2 + Math.abs(fd[0]*c.v[0] + fd[1]*c.v[1]) * c.d/2;
          const fx = c.cx + fd[0]*reach, fy = c.cy + fd[1]*reach, s = Math.min(hr*1.4/f, Math.min(c.w, c.d) / 3);
          el('polygon', { points: [[fx + fd[0]*s*1.4, fy + fd[1]*s*1.4], [fx - sd[0]*s, fy - sd[1]*s], [fx + sd[0]*s, fy + sd[1]*s]].map(P).join(' '),
            fill:'#fff', 'pointer-events':'none' }, g);
        });
      }
      if(ed === 'objects'){
        // each object's own axis: a green arrow from its centre to its front, its
        // +Y (Blender's green arrow), turned with it; and its id, row-column from
        // the top left, at 30% of the object's smaller side, towards its back
        g0.copies.forEach(c => {
          if(c.empty) return;
          const th = c.a + (c.slot ? c.slot.turn : turn) * Math.PI / 2, fd = [Math.sin(th), -Math.cos(th)], sd = [Math.cos(th), Math.sin(th)];
          const reach = Math.abs(fd[0]*c.u[0] + fd[1]*c.u[1]) * c.w/2 + Math.abs(fd[0]*c.v[0] + fd[1]*c.v[1]) * c.d/2;
          const side = Math.min(c.w, c.d), head = Math.min(reach * 0.35, side * 0.22);
          const tip = [c.cx + fd[0]*reach*0.92, c.cy + fd[1]*reach*0.92], base = [tip[0] - fd[0]*head, tip[1] - fd[1]*head];
          el('line', { x1: c.cx*f, y1: c.cy*f, x2: base[0]*f, y2: base[1]*f, stroke:'#3DBE5C', 'stroke-width': Math.max(1.5, side*f*0.05),
            'stroke-linecap':'round', 'pointer-events':'none' }, g);
          el('polygon', { points: [tip, [base[0] - sd[0]*head*0.6, base[1] - sd[1]*head*0.6], [base[0] + sd[0]*head*0.6, base[1] + sd[1]*head*0.6]].map(P).join(' '),
            fill:'#3DBE5C', 'pointer-events':'none' }, g);
          el('circle', { cx: c.cx*f, cy: c.cy*f, r: Math.max(1.5, side*f*0.04), fill:'#3DBE5C', 'pointer-events':'none' }, g);
          const back = Math.min(reach * 0.5, side * 0.3);
          const t = el('text', { x: (c.cx - fd[0]*back)*f, y: (c.cy - fd[1]*back)*f, 'text-anchor':'middle', 'dominant-baseline':'central',
            'font-size': side * f * 0.3, fill:'#fff', stroke:'rgba(0,0,0,.6)', 'stroke-width': side * f * 0.03,
            'paint-order':'stroke', 'pointer-events':'none' }, g);
          t.textContent = c.id;
        });
      }
      if(ed) return;                                         // no handles while editing
      if(g0.curve){
        // the points: drag to bend, double-click one in the middle to remove it (from every line)
        g0.curveLines.forEach((cl, li) => cl.handles.forEach((q, k) => {
          const end = k === 0 || k === cl.handles.length - 1;
          const c = el('circle', { cx:q[0]*f, cy:q[1]*f, r: end ? hr : hr*0.85, class:'pt',
            fill: end ? '#fff' : '#4682DC', stroke: end ? '#4682DC' : '#fff', 'stroke-width': hr*0.35 }, g);
          c.dataset.pt = `${i}/${li}-${k}`;                // placement / line-point
          c.addEventListener('pointerdown', e => startScatterDrag(e, i, 'point', [li, k], cl.tangents[k]));
        }));
        return;
      }
      if(!g0.pkg){
        // the front: Blender's +Y, the top edge at rotation 0, turned with the object
        const th = (p.angle + 90 * turn) * Math.PI / 180;
        const fd = [Math.sin(th), -Math.cos(th)], sd = [Math.cos(th), Math.sin(th)];
        const reach = Math.abs(fd[0]*g0.u[0] + fd[1]*g0.u[1]) * g0.Lx/2 + Math.abs(fd[0]*g0.v[0] + fd[1]*g0.v[1]) * g0.Ly/2;
        const fx = p.cx + fd[0]*reach, fy = p.cy + fd[1]*reach;
        const tip = [fx + fd[0]*hr*2.2/f, fy + fd[1]*hr*2.2/f];
        const l = [fx - sd[0]*hr*1.2/f, fy - sd[1]*hr*1.2/f], r = [fx + sd[0]*hr*1.2/f, fy + sd[1]*hr*1.2/f];
        el('polygon', { points: [tip, l, r].map(P).join(' '), fill:'#fff', 'pointer-events':'none' }, g);
      }
      g0.outer.forEach(q => {
        const c = el('circle', { cx:q[0]*f, cy:q[1]*f, r:hr*0.9, class:'corner', fill:'#fff', stroke:'#4682DC', 'stroke-width': hr*0.3 }, g);
        c.addEventListener('pointerdown', e => startScatterDrag(e, i, 'size'));
      });
      const rx = (p.cx + g0.u[0]*g0.Lx/2)*f + g0.u[0]*hr*4, ry = (p.cy + g0.u[1]*g0.Lx/2)*f + g0.u[1]*hr*4;
      el('line', { x1:(p.cx + g0.u[0]*g0.Lx/2)*f, y1:(p.cy + g0.u[1]*g0.Lx/2)*f, x2:rx, y2:ry, stroke:'#4682DC', 'stroke-width': hr*0.3 }, g);
      const spin = el('circle', { cx:rx, cy:ry, r:hr, class:'spin', fill:'#4682DC' }, g);
      spin.addEventListener('pointerdown', e => startScatterDrag(e, i, 'rotate'));
    });
    renderPlacementBox();
  }

  // Inner streets: the cells of the spaces, grey, or green when drawn as a street.
  // Press on a cell and drag across others: they all take the opposite of the first
  function drawCells(g, p, g0, f, hr, P){
    const on = new Set((p.streets && p.streets.cells) || []);
    spaceCells(p, g0).forEach(c => {
      const poly = el('polygon', { points: c.poly.map(P).join(' '), class:'pt',
        fill: on.has(c.id) ? 'rgba(60,190,90,.85)' : 'rgba(150,150,150,.6)',
        stroke: on.has(c.id) ? '#2E9B4F' : '#777', 'stroke-width': hr*0.2 }, g);
      poly.dataset.cell = c.id;
      const paint = () => {
        const st = p.streets, has = st.cells.includes(c.id);
        if(S.gen.edit.paint && has !== S.gen.edit.paint.to){
          st.cells = S.gen.edit.paint.to ? st.cells.concat([c.id]) : st.cells.filter(x => x !== c.id);
          poly.setAttribute('fill', S.gen.edit.paint.to ? 'rgba(60,190,90,.85)' : 'rgba(150,150,150,.6)');
          poly.setAttribute('stroke', S.gen.edit.paint.to ? '#2E9B4F' : '#777');
        }
      };
      poly.addEventListener('pointerdown', e => {
        e.stopPropagation(); e.preventDefault();
        p.streets = streetsOf(p);
        S.gen.edit.paint = { to: !p.streets.cells.includes(c.id) };
        paint();
      });
      poly.addEventListener('pointerenter', paint);
    });
  }
  const streetsOf = p => ({ cells: [], sidewalk_m: 2, road_m: 6, corner_m: 4, reach_m: 200, markings: true, ...(p.streets || {}) });
  // a space's width across the street it makes, and the sidewalk it gets: the road
  // comes first, up to its road width, then the sidewalks (as app/streets.py)
  const spaceWidth = c => { const along = c.s[1] - c.s[0], across = c.o[1] - c.o[0]; return c.kind === 'x' ? along : c.kind === 'y' ? across : Math.min(along, across); };
  function fittedSidewalk(sw, width, road){
    width = Math.max(width, 0);
    return Math.min(sw, (width - Math.min(road, width)) / 2);
  }
  addEventListener('pointerup', () => {
    if(S.gen.edit && S.gen.edit.kind === 'streets' && S.gen.edit.paint){ S.gen.edit.paint = null; saveScatter(); }
  });
  // Single objects: click to pick one, shift-click to add or remove more
  // Draw spaced: a press empties an object (or brings an empty spot's object back);
  // dragging on does the same to every copy passed over; saved when the button is let go
  function spaceAt(e, i, id, on, first){
    e.stopPropagation(); if(first) e.preventDefault();
    const p = S.gen.placements[i]; if(!p) return;
    if(first){ S.gen.edit.paint = on; S.gen.edit.changed = 0; }
    if(isEmpty(p, id) === on) return;
    setEmpty(p, id, on); S.gen.edit.changed++;
    drawBridges();
  }
  addEventListener('pointerup', () => {
    const ed = S.gen.edit;
    if(!ed || ed.kind !== 'spaces' || ed.paint == null) return;
    const p = S.gen.placements[ed.i], n = ed.changed, on = ed.paint;
    ed.paint = null;
    if(!n || !p) return;
    log(`${n} object${n > 1 ? 's' : ''} ${on ? (n > 1 ? 'made empty spots' : 'made an empty spot') : 'brought back'}; ${(p.empty || []).length} empty spot(s) in this placement.`);
    saveScatter(); renderPlacementBox();
  });

  function pickObject(e, id){
    e.stopPropagation(); e.preventDefault();
    // an angle still being typed belongs to the objects picked so far: apply it first
    if(document.activeElement === $('#objTurn')) $('#objTurn').blur();
    const sel = S.gen.edit.sel;
    if(e.shiftKey){ if(sel.has(id)) sel.delete(id); else sel.add(id); }
    else { sel.clear(); sel.add(id); }
    drawBridges();
  }

  let sdrag = null;
  // double-clicks are told apart here: the map is redrawn on every press, so
  // the browser's own dblclick never reaches the same element twice
  let lastPress = null;
  function isDoublePress(i, what, q){
    const now = performance.now(), prev = lastPress;
    lastPress = { t: now, i, what, q };
    const dbl = prev && now - prev.t < 400 && prev.i === i && prev.what === what
      && Math.hypot(q[0] - prev.q[0], q[1] - prev.q[1]) * shown() < 8;
    if(dbl) lastPress = null;
    return dbl;
  }
  function startScatterDrag(e, i, mode, k, tangent){
    // k: for a curve's point, [its line, its index]; tangent: its line's direction there
    e.stopPropagation(); e.preventDefault();
    if(S.gen.edit && S.gen.edit.i !== i) S.gen.edit = null;
    S.gen.plSel = i;
    const p = S.gen.placements[i], q = toMask(e);
    if(isCurve(p) && isDoublePress(i, mode + (k ?? ''), q)){
      if(mode === 'point' && k[1] > 0 && k[1] < p.path.length - 1){
        removeCurvePoint(p, k[1]); curveCentre(p); log('Point removed.'); saveScatter(); return;
      }
      const g0 = mode === 'move' ? placementGeom(p) : null;
      if(g0 && g0.curve && addCurvePointNear(p, q, g0)){
        curveCentre(p); log('Point added: drag it to bend the line.'); saveScatter(); return;
      }
    }
    sdrag = { i, mode, k, tangent, start: q, orig: { ...p, lines: isCurve(p) ? curveLinesOf(p).map(l => l.map(r => r.slice())) : undefined } };
    drawBridges();
  }
  addEventListener('pointermove', e => {
    if(!sdrag) return;
    const p = S.gen.placements[sdrag.i], o = sdrag.orig, q = toMask(e);
    if(sdrag.mode === 'move' && o.lines){
      const dx = q[0] - sdrag.start[0], dy = q[1] - sdrag.start[1];
      setCurveLines(p, o.lines.map(l => l.map(r => [r[0] + dx, r[1] + dy]))); curveCentre(p);
    } else if(sdrag.mode === 'point'){
      // along its line the point moves on its own; across it, the points at the
      // same place on the other curve lines move with it
      const [li, j] = sdrag.k, t = sdrag.tangent || [1, 0];
      const dx = q[0] - sdrag.start[0], dy = q[1] - sdrag.start[1], a = dx*t[0] + dy*t[1];
      const across = [dx - a*t[0], dy - a*t[1]];
      setCurveLines(p, o.lines.map((l, n) => l.map((r, k) => k !== j ? r.slice()
        : n === li ? [r[0] + dx, r[1] + dy] : [r[0] + across[0], r[1] + across[1]])));
      curveCentre(p);
    } else if(sdrag.mode === 'move'){
      p.cx = o.cx + q[0] - sdrag.start[0]; p.cy = o.cy + q[1] - sdrag.start[1];
    } else if(sdrag.mode === 'size'){
      const m = mppMask(), a = o.angle * Math.PI / 180, dx = q[0] - o.cx, dy = q[1] - o.cy;
      const wantX = 2 * Math.abs(dx*Math.cos(a) + dy*Math.sin(a)) * m, wantY = 2 * Math.abs(-dx*Math.sin(a) + dy*Math.cos(a)) * m;
      // with random spaces, copies and rows are counted with the middle of their range
      const sp = spacesOf(p), mid = k => (+sp[k][0] + +sp[k][1]) / 2;
      const gx = sp ? mid('x') : p.gap_x, gy = sp ? mid('y') : p.gap_y;
      if(p.package){
        // a package fills the length it is given; rows are as deep as its deepest object.
        // It never gets shorter than its widest object
        const slots = pkgSlots(pkgById(p.package)); if(!slots.length) return;
        p.length = Math.max(wantX, ...slots.map(s => s.w)) / m;
        p.ny = Math.max(1, Math.round((wantY + gy) / (Math.max(...slots.map(s => s.d)) + gy)));
      } else {
        // stretch adds whole copies; the rectangle never goes below one object
        const ob = objById(p.object); if(!ob) return;
        const [w, d] = objSize(ob);
        p.nx = Math.max(1, Math.round((wantX + gx) / (w + gx)));
        p.ny = Math.max(1, Math.round((wantY + gy) / (d + gy)));
      }
    } else {
      p.angle = Math.atan2(q[1] - o.cy, q[0] - o.cx) * 180 / Math.PI;
    }
    drawBridges();
  });
  addEventListener('pointerup', () => { if(sdrag){ sdrag = null; saveScatter(); } });

  function saveScatter(){
    drawBridges();
    if(!S.gen.mask) return;
    api('/api/scatter', { method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ mask: S.gen.mask.id, placements: S.gen.placements }) }).catch(() => {});
  }

  function renderPlacementBox(){
    const box = $('#plBox');
    const p = S.gen.placements[S.gen.plSel];
    // duplicate and flip act on the selected placement, from its own panel
    Object.values(KIND).forEach(K => { $(K.dup).disabled = $(K.flip).disabled = !p || kindOf(p) !== K || laid(p); });
    if(AUTO.pick >= 0 && AUTO.pick !== S.gen.plSel) AUTO.pick = -1;      // picking belongs to the selected placement
    if(!p){ box.hidden = true; S.gen.edit = null; return; }
    box.hidden = false;
    // an automatic placement: its own settings, none of a rectangle's
    box.classList.toggle('auto', laid(p));
    $('#autoBox').hidden = !p.auto;
    $('#objEditNote').textContent = p.layout ? 'Click an object to pick it, Shift-click to pick more. Ids are b and the number of its building.'
      : p.auto ? 'Click an object to pick it, Shift-click to pick more. Ids are island-number: the island, then its objects in order round it.'
      : 'Click an object to pick it, Shift-click to pick more. Ids are row-column, counted from the top left.';
    // in the panel of its kind; foliage has no random spaces, curve lines, inner
    // streets, single turns or alignment, and objects have no random transform
    if(box.parentElement !== $(kindOf(p).host)) $(kindOf(p).host).appendChild(box);
    $$('[data-only]', box).forEach(e => { e.hidden = e.dataset.only !== (p.foliage ? 'foliage' : 'objects'); });
    if(p.foliage) S.gen.edit = null;
    // the edit modes belong to the selected placement
    if(S.gen.edit && S.gen.edit.i !== S.gen.plSel) S.gen.edit = null;
    const ed = S.gen.edit ? S.gen.edit.kind : null;
    $('#btnDrawStreets').textContent = ed === 'streets' ? 'Done drawing streets' : 'Draw inner streets';
    $('#streetBox').hidden = ed !== 'streets';
    $('#btnEditObjs').textContent = ed === 'objects' ? 'Done turning objects' : 'Turn single objects';
    $('#objEditBox').hidden = ed !== 'objects';
    $('#btnSpaceObjs').textContent = ed === 'spaces' ? 'Done drawing spaces' : 'Draw spaced';
    $('#btnSpaceObjs').classList.toggle('attention', ed === 'spaces');
    $('#spaceBox').hidden = ed !== 'spaces';
    const nEmpty = (p.empty || []).length;
    $('#btnSpaceReset').disabled = !nEmpty;
    $('#spaceInfo').textContent = nEmpty ? `${nEmpty} empty spot${nEmpty > 1 ? 's' : ''}: ${p.empty.slice(0, 12).join(', ')}${nEmpty > 12 ? '…' : ''}.` : '';
    const st = streetsOf(p);
    if(document.activeElement !== $('#stSidewalk')) $('#stSidewalk').value = st.sidewalk_m;
    if(document.activeElement !== $('#stRoad')) $('#stRoad').value = st.road_m;
    if(document.activeElement !== $('#stCorner')) $('#stCorner').value = st.corner_m;
    if(document.activeElement !== $('#stReach')) $('#stReach').value = st.reach_m;
    $('#stMarkings').checked = !!st.markings;
    // a plant's random transform: the ranges, overlap, and what they do
    if(p.foliage){
      const jt = jitterSet(p);
      [['#jScMin', jt.scale[0]], ['#jScMax', jt.scale[1]], ['#jRotMin', jt.rotate[0]], ['#jRotMax', jt.rotate[1]],
       ['#jOffMin', jt.offset[0]], ['#jOffMax', jt.offset[1]]].forEach(([id, v]) => { if(document.activeElement !== $(id)) $(id).value = v; });
      $('#jOverlap').checked = jt.overlap !== false;
      const rg = (r, u) => `${Math.min(...r)}${u}–${Math.max(...r)}${u}`, parts = [];
      if(jt.scale[0] !== 1 || jt.scale[1] !== 1) parts.push(`scaled ${rg(jt.scale, '×')}`);
      if(+jt.rotate[0] || +jt.rotate[1]) parts.push(`turned ${rg(jt.rotate, '°')}`);
      if(+jt.offset[0] || +jt.offset[1]) parts.push(`moved ${rg(jt.offset, ' m')} from its spot`);
      $('#jInfo').textContent = parts.length ? `Each plant ${parts.join(', ')}; ${jt.overlap !== false ? 'big ones may overlap' : 'none overlap: each takes the room it needs'}.`
        : 'No random transform: set a min and a max above.';
    }
    // objects alignment: three switches and the checkbox, and which way each row faces
    const al = alignOf(p);
    [['#btnAlignRows', 'rows'], ['#btnAlignLast', 'last'], ['#btnAlignCols', 'columns']].forEach(([id, key]) =>
      $(id).setAttribute('aria-pressed', al[key] ? 'true' : 'false'));
    $('#alignExclude').checked = al.exclude; $('#alignExclude').disabled = !al.columns;
    const between = p.ny > 3 ? `rows 2 to ${p.ny - 1}` : p.ny === 3 ? 'row 2' : 'no row';
    $('#alignInfo').textContent = !aligned(al) ? ''
      : `Rows face ${Array.from({ length: p.ny }, (_, r) => turnedRound(al, r, p.ny) ? '↓' : '↑').join(' ')} (↑ as placed)`
        + (al.columns ? `; the ends face out in ${al.exclude ? between : 'every row'}.` : '.');
    const turned = Object.keys(p.turns || {}).length, picked = ed === 'objects' ? [...S.gen.edit.sel] : [];
    if(picked.length && document.activeElement !== $('#objTurn')) $('#objTurn').value = (p.turns || {})[picked[0]] || 0;
    $('#objEditInfo').textContent = (picked.length ? `Picked: ${picked.join(', ')}. ` : (ed === 'objects' ? 'Nothing picked yet. ' : ''))
      + (turned ? `${turned} object${turned > 1 ? 's' : ''} turned on their own.` : '');
    const o = p.package ? null : objById(p.object), pk = p.package ? pkgById(p.package) : null;
    const g0 = laid(p) ? autoGeom(p) : placementGeom(p);
    // the drawn streets, and the road the narrowest of them gets
    let stNote = '';
    if(st.cells.length && g0 && g0.layout){
      const drawn = cellsOf(g0.layout).filter(c => st.cells.includes(c.id) && c.kind !== 'j');
      if(drawn.length){
        const w = Math.min(...drawn.map(spaceWidth)), f = fittedSidewalk(st.sidewalk_m, w, st.road_m), m = mppMask();
        stNote = ` The narrowest is ${w.toFixed(1)} m: ${f > 0 ? `a ${(w - 2*f).toFixed(1)} m road between ${f.toFixed(f < st.sidewalk_m ? 2 : 1)} m sidewalks`
          : `road from side to side, no room for sidewalks`}.`
          + (w - 2*f < 3 * m ? ` Roads under ${(3 * m).toFixed(1)} m (3 pixels at this scale) show only faintly in the texture; the 3D model has them as drawn.` : '');
      }
    }
    $('#stInfo').textContent = st.cells.length
      ? `${st.cells.length} space${st.cells.length > 1 ? 's' : ''} drawn as streets, ${st.markings ? 'with' : 'without'} markings.${stNote} Press Generate to build them.` : '';
    const onRoad = g0 ? g0.copies.filter(c => c.onRoad && !c.empty).length : 0;
    const curve = isCurve(p);
    $('#plTitle').textContent = `${p.foliage ? 'Foliage placement' : 'Placement'} ${S.gen.plSel + 1}: `
      + (p.package ? (pk ? `package ${pk.name}` : 'missing package') : (o ? o.name : 'missing object')) + (curve ? ', curved' : '');
    $('#btnShuffle').hidden = !p.package;
    if(p.layout){ renderLayoutBox(p, g0); return; }
    if(p.auto){ renderAutoBox(p, g0, o, pk, onRoad); return; }
    $('#gapXLabel').textContent = curve ? 'Gap along the line (m)' : 'Gap along X (m)';
    $('#gapYLabel').textContent = curve ? 'Gap between rows (m)' : 'Gap along Y (m)';
    $('#plCurveBox').hidden = !curve; $('#btnCurve').hidden = curve;
    if(document.activeElement !== $('#gapX')) $('#gapX').value = p.gap_x;
    if(document.activeElement !== $('#gapY')) $('#gapY').value = p.gap_y;
    // random spaces: the ranges stay with the placement, even when switched off
    const sp = p.spaces || { on: false, x: [1, 4], y: [2, 6] }, on = !!(p.spaces && p.spaces.on);
    [['#spXmin', sp.x[0]], ['#spXmax', sp.x[1]], ['#spYmin', sp.y[0]], ['#spYmax', sp.y[1]]].forEach(([id, v]) => {
      if(document.activeElement !== $(id)) $(id).value = v;
    });
    $('#gapX').disabled = $('#gapY').disabled = on;
    $('#gapX').title = $('#gapY').title = on ? 'Spaces are random: set their range below, or press Even spaces' : '';
    $('#btnEvenSpaces').disabled = !on;
    const spNote = on ? `, random spaces X ${sp.x[0]}–${sp.x[1]} m, Y ${sp.y[0]}–${sp.y[1]} m` : '';
    if(document.activeElement !== $('#plRows')) $('#plRows').value = p.ny;
    // curve lines: at most one per row
    const nLines = curve ? curveLinesOf(p).length : 0;
    $('#btnAddLine').disabled = nLines >= p.ny; $('#btnRemoveLine').disabled = nLines < 2;
    $('#btnAddLine').title = nLines >= p.ny ? 'A curve line per row at most: add rows first'
      : 'Another curve line: the first shapes the front row, the last the back row, the others spread between; rows between two lines follow a blend. At most one per row';
    $('#plLinesInfo').textContent = nLines > 1 ? `${nLines} curve lines over ${p.ny} rows: the first shapes the front row, the last the back row.` : '';
    const lineNote = !curve || !g0 ? '' : nLines > 1
      ? `, lines ${Math.min(...g0.lengths_m).toFixed(1)}–${Math.max(...g0.lengths_m).toFixed(1)} m, ${p.path.length} points each`
      : `, line ${g0.length_m.toFixed(1)} m, ${p.path.length} points`;
    const nEmptied = g0 ? g0.copies.filter(c => c.empty).length : 0;
    const road = (onRoad ? `, ${onRoad} on the road (left out)` : '') + (nEmptied ? `, ${nEmptied} empty spot${nEmptied > 1 ? 's' : ''} (Draw spaced)` : '');
    if(p.package){
      // how many of each object the mix holds
      const n = {};
      (g0 ? g0.copies : []).forEach(c => { if(!c.empty) n[c.slot.o.name] = (n[c.slot.o.name] || 0) + 1; });
      $('#plInfo').textContent = !g0 ? 'This package has no objects yet: import some into its slots.'
        : `${g0.copies.length} copies in ${p.ny} row${p.ny > 1 ? 's' : ''}: ` + Object.entries(n).map(([k, v]) => `${v} ${k}`).join(', ')
          + road + (curve ? lineNote : `, area ${(g0.Lx * mppMask()).toFixed(1)} × ${(g0.Ly * mppMask()).toFixed(1)} m`) + spNote;
      return;
    }
    const even = g0 && g0.counts && g0.counts.every(n => n === g0.counts[0]);
    $('#plInfo').textContent = (curve && g0
      ? (even ? `${g0.along} along the line × ${p.ny} row${p.ny > 1 ? 's' : ''} = ${g0.copies.length} copies`
              : `${g0.copies.length} copies in ${p.ny} rows (${g0.counts.join(', ')})`)
        + road + lineNote
      : `${p.nx} × ${p.ny} = ${p.nx * p.ny} copies${road}`
        + (g0 ? `, area ${(g0.Lx * mppMask()).toFixed(1)} × ${(g0.Ly * mppMask()).toFixed(1)} m` : '')) + spNote;
  }
  // a layout's panel: how many buildings each object stands on
  function renderLayoutBox(p, g0){
    $('#plTitle').textContent = `Placement ${S.gen.plSel + 1}: layout on the buildings`;
    $('#plCurveBox').hidden = true; $('#btnCurve').hidden = true; $('#btnShuffle').hidden = true;
    const n = {}, copies = g0 ? g0.copies : [];
    copies.forEach(c => { if(!c.empty) n[c.slot.o.name] = (n[c.slot.o.name] || 0) + 1; });
    const nEmptied = copies.filter(c => c.empty).length, missing = (p.items || []).length - copies.length;
    $('#plInfo').textContent = `${(p.items || []).length} buildings: ` + (Object.entries(n).map(([k, v]) => `${v} ${k}`).join(', ') || 'no object')
      + (nEmptied ? `, ${nEmptied} left empty (Draw spaced)` : '') + (missing > 0 ? `, ${missing} whose object is gone` : '')
      + '. Each object\'s width and depth are its building\'s, its height its own; Create layout from dataset lays them again.';
  }
  // an automatic placement's panel: its settings, its islands, and what it laid
  function renderAutoBox(p, g0, o, pk, onRoad){
    const au = autoOf(p), res = S.gen.result, note = autoNote.get(p), mine = AUTO.pick === S.gen.plSel;
    $('#plTitle').textContent += ', automatic on islands';
    $('#plCurveBox').hidden = true; $('#btnCurve').hidden = true;
    [['#auWeight', au.weight], ['#auSetback', au.setback], ['#auGap', au.gap]].forEach(([id, val]) => { if(document.activeElement !== $(id)) $(id).value = val; });
    $('#auIslands').value = au.islands;
    // the material slots, by their material
    const sl = $('#auSlot'), key = ISLS.slots.map((x, i) => slotMaterial(i).name).join('|');
    if(sl.dataset.key !== key){
      sl.dataset.key = key;
      sl.innerHTML = ISLS.slots.length ? ISLS.slots.map((x, i) => `<option value="${i}">Slot ${i + 1}: ${esc(slotMaterial(i).name)}</option>`).join('')
        : '<option value="0">No slots yet</option>';
    }
    sl.value = String(Math.min(au.slot || 0, Math.max(0, ISLS.slots.length - 1))); sl.disabled = !ISLS.slots.length;
    $('#auSlotRow').hidden = au.islands !== 'slot';
    $('#btnAuPick').hidden = au.islands !== 'picked';
    $('#btnAuPick').setAttribute('aria-pressed', mine ? 'true' : 'false');
    $('#btnAuPick').textContent = mine ? 'Done picking' : 'Pick islands';
    $('#btnAuPick').disabled = !ISLS.data;
    $('#btnAuLay').disabled = !res || !ISLS.data;
    const ids = autoIslands(p) || [], k = ids.length;
    $('#autoInfo').textContent = !ISLS.data ? 'Generate texture first: it finds and numbers the islands, and the objects are laid on them then.'
      : au.islands === 'all' ? `Every island: ${ISLS.data.count}.`
      : au.islands === 'slot' ? (!ISLS.slots.length ? 'No material slots yet: add one under Blocks and islands, and Pick its islands.'
          : `Slot ${(au.slot || 0) + 1}: ` + (k ? `${k} island${k > 1 ? 's' : ''} (${idText(ids)}).` : 'no islands yet: Pick some for it under Blocks and islands.'))
      : k ? `Picked: ${idText(ids)}.` : 'No islands picked yet: press Pick islands and click them on the map.';
    // what it laid, and how
    const all = g0 ? g0.copies : [], live = all.filter(c => !c.empty), per = {};
    live.forEach(c => { const nm = c.slot ? c.slot.o.name : (o ? o.name : 'object'); per[nm] = (per[nm] || 0) + 1; });
    const nEmpty = all.length - live.length, laid = (p.items || []).length;
    const stale = laid && res && au.gen && au.gen !== res.id;
    $('#plInfo').textContent = !laid
      ? (!ISLS.data ? 'No objects yet.' : !note ? 'Laying…' : !note.islands ? 'No objects: no islands to lay them on yet.'
          : 'No objects: none of these islands has room for one, with this setback and gap.')
      : `${live.length} object${live.length === 1 ? '' : 's'}` + (p.package ? ': ' + Object.entries(per).map(([nm, n]) => `${n} ${nm}`).join(', ') : '')
        + (note ? ` on ${note.with} of ${note.islands} island${note.islands === 1 ? '' : 's'}` : '')
        + (onRoad ? `, ${onRoad} on the road (left out)` : '') + (nEmpty ? `, ${nEmpty} empty spot${nEmpty > 1 ? 's' : ''} (Draw spaced)` : '')
        + '. ' + (stale ? 'Laid on an earlier texture: press Lay again.'
          : `Each with its front to its street, ${au.setback} m from the kerb, at least ${au.gap} m apart; up to ${au.weight} per side.`);
  }
  // the two edit modes: drawing inner streets, and turning single objects
  S.gen.edit = null;
  $('#btnDrawStreets').addEventListener('click', () => {
    const i = S.gen.plSel, p = S.gen.placements[i]; if(!p) return;
    if(S.gen.edit && S.gen.edit.kind === 'streets'){
      S.gen.edit = null;
      const n = streetsOf(p).cells.length;
      log(n ? `${n} space(s) drawn as inner streets. Press Generate to build them into the streets.` : 'No inner streets drawn.');
    } else {
      p.streets = streetsOf(p); S.gen.edit = { kind: 'streets', i, paint: null };
      log('Draw inner streets: click a space (or press and drag across several) to make it a street; click again to take it back.');
    }
    drawBridges();
  });
  [['#stSidewalk', 'sidewalk_m', 0.5], ['#stRoad', 'road_m', 1], ['#stCorner', 'corner_m', 0], ['#stReach', 'reach_m', 0, 2000]].forEach(([id, key, lo, hi = Infinity]) => $(id).addEventListener('input', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    p.streets = { ...streetsOf(p), [key]: Math.min(hi, Math.max(lo, +$(id).value || 0)) };
    clearTimeout(gapTimer); gapTimer = setTimeout(saveScatter, 300);
  }));
  $('#stMarkings').addEventListener('change', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    p.streets = { ...streetsOf(p), markings: $('#stMarkings').checked }; saveScatter();
  });
  $('#btnSpaceObjs').addEventListener('click', () => {
    const i = S.gen.plSel, p = S.gen.placements[i]; if(!p) return;
    if(S.gen.edit && S.gen.edit.kind === 'spaces') S.gen.edit = null;
    else { S.gen.edit = { kind: 'spaces', i, paint: null }; log('Draw spaced: click an object to make it an empty spot, click a spot to bring its object back; press and drag across several.'); }
    drawBridges(); renderPlacementBox();
  });
  $('#btnSpaceReset').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !(p.empty || []).length) return;
    log(`${p.empty.length} empty spot(s) given their objects back.`);
    delete p.empty; saveScatter(); renderPlacementBox();
  });
  $('#btnEditObjs').addEventListener('click', () => {
    const i = S.gen.plSel, p = S.gen.placements[i]; if(!p) return;
    if(S.gen.edit && S.gen.edit.kind === 'objects') S.gen.edit = null;
    else { S.gen.edit = { kind: 'objects', i, sel: new Set() }; log('Turn single objects: click one to pick it, Shift-click to pick more.'); }
    drawBridges();
  });
  // a single object's own rotation, in degrees clockwise on the map, kept by its id
  function setTurn(p, id, deg){
    deg = ((deg % 360) + 360) % 360;
    p.turns = { ...(p.turns || {}) };
    if(deg) p.turns[id] = deg; else delete p.turns[id];
  }
  const turnPicked = f => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !S.gen.edit || S.gen.edit.kind !== 'objects' || !S.gen.edit.sel.size) return;
    S.gen.edit.sel.forEach(id => setTurn(p, id, f((p.turns || {})[id] || 0)));
    saveScatter();
  };
  $('#objTurn').addEventListener('change', () => { const v = +$('#objTurn').value || 0; turnPicked(() => v); });
  $('#btnTurnL').addEventListener('click', () => turnPicked(t => t - 90));
  $('#btnTurnR').addEventListener('click', () => turnPicked(t => t + 90));
  $('#btnTurnReset').addEventListener('click', () => turnPicked(() => 0));
  addEventListener('keydown', e => { if(e.key === 'Escape' && S.gen.edit){ S.gen.edit = null; drawBridges(); } });
  addEventListener('keydown', e => { if(e.key === 'Escape' && AUTO.pick >= 0 && !MAT.open){ AUTO.pick = -1; drawBridges(); } });
  // Foliage: random scale, rotation and offset per plant, between a min and a max
  // (app/curves.py); the random values are kept with the placement until New random set
  const jitterSet = p => ({ seed: 1, scale: [1, 1], rotate: [0, 0], offset: [0, 0], overlap: true, ...(p.jitter || {}) });
  [['#jScMin', 'scale', 0, 0.01], ['#jScMax', 'scale', 1, 0.01], ['#jRotMin', 'rotate', 0, null], ['#jRotMax', 'rotate', 1, null],
   ['#jOffMin', 'offset', 0, 0], ['#jOffMax', 'offset', 1, 0]].forEach(([id, key, k, lo]) => $(id).addEventListener('input', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.foliage) return;
    const v = parseFloat($(id).value); if(!Number.isFinite(v)) return;
    const jt = jitterSet(p);
    jt[key] = jt[key].slice(); jt[key][k] = lo === null ? v : Math.max(lo, v);
    p.jitter = jt; drawBridges(); clearTimeout(gapTimer); gapTimer = setTimeout(saveScatter, 300);
  }));
  $('#jOverlap').addEventListener('change', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.foliage) return;
    p.jitter = { ...jitterSet(p), overlap: $('#jOverlap').checked };
    log(p.jitter.overlap ? 'Overlap on: the plants keep their spots and vary on them.' : 'Overlap off: each plant takes the room of its size, turn and offset, so none overlap.');
    saveScatter();
  });
  $('#btnJitter').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.foliage) return;
    p.jitter = { ...jitterSet(p), seed: (Math.random() * 4294967296) >>> 0 }; log('New random set.'); saveScatter();
  });
  // Objects alignment: on/off switches kept with the placement (app/curves.py)
  const alignOf = p => ({ rows: false, last: false, columns: false, exclude: false, ...(p.align || {}) });
  function setAlign(p, key, on){
    const al = { ...alignOf(p), [key]: on };
    if(Object.values(al).some(Boolean)) p.align = al; else delete p.align;
    saveScatter();
  }
  const ALIGN_LOG = {
    rows: ['Alternate rows: row 1 as placed, row 2 turned round, row 3 as row 1, and so on.', 'Alternate rows off.'],
    last: ['Flip last row: the last row faces the other way.', 'Flip last row off.'],
    columns: ['Flip columns: the objects at either end of each row face out of that end (left 270°, right 90°).', 'Flip columns off.'],
  };
  [['#btnAlignRows', 'rows'], ['#btnAlignLast', 'last'], ['#btnAlignCols', 'columns']].forEach(([id, key]) => $(id).addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    const on = !alignOf(p)[key];
    log(ALIGN_LOG[key][on ? 0 : 1]); setAlign(p, key, on);
  }));
  $('#alignExclude').addEventListener('change', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    const on = $('#alignExclude').checked;
    log(on ? 'Exclude rows from columns: the first and last rows keep their ends facing as their row.' : 'The first and last rows have their ends facing out too.');
    setAlign(p, 'exclude', on);
  });
  $('#plRows').addEventListener('input', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    p.ny = Math.max(1, Math.round(+$('#plRows').value || 1));
    drawBridges(); clearTimeout(gapTimer); gapTimer = setTimeout(saveScatter, 300);
  });
  $('#plRows').addEventListener('change', () => {
    // never more curve lines than rows: fewer rows spread the lines over them again
    const p = S.gen.placements[S.gen.plSel]; if(!p || !isCurve(p)) return;
    const k = curveLinesOf(p).length;
    if(k > p.ny){ spreadLines(p, p.ny); log(`${p.ny} curve line${p.ny > 1 ? 's' : ''} now: never more than rows.`); saveScatter(); }
  });
  $('#btnAddLine').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !isCurve(p)) return;
    const k = curveLinesOf(p).length;
    if(k >= p.ny){ log('A curve line per row at most: add rows first.', 'warn'); return; }
    spreadLines(p, k + 1);
    log(k === 1 ? 'Two curve lines: the first shapes the front row, the second the back row. Drag a point along its line to move it alone, across to move the points beside it on the other line too.'
                : `${k + 1} curve lines, spread over the ${p.ny} rows.`);
    saveScatter();
  });
  $('#btnRemoveLine').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !isCurve(p)) return;
    const k = curveLinesOf(p).length; if(k < 2) return;
    spreadLines(p, k - 1); log(k === 2 ? 'One curve line again: the rows run beside it.' : `${k - 1} curve lines.`); saveScatter();
  });
  // Duplicate: the same placement with all its settings, just behind the original, selected to drag
  ['#btnDupPl', '#btnDupFol'].forEach(id => $(id).addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    const g0 = placementGeom(p), q = JSON.parse(JSON.stringify(p)), m = mppMask();
    if(g0){
      let v = g0.v;                                         // a rectangle: towards its backs
      if(g0.curve){
        // a curve: across its first line at the middle, towards the backs
        const S = g0.line, k = Math.floor(S.length / 2), a = S[Math.max(k - 1, 0)], b = S[Math.min(k + 1, S.length - 1)];
        const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1, sg = p.flip ? -1 : 1;
        v = [-(b[1] - a[1]) / len * sg, (b[0] - a[0]) / len * sg];
      }
      // moved by its whole extent that way and a gap, so the two never overlap, however it bends
      const proj = g0.outer.map(q => q[0]*v[0] + q[1]*v[1]);
      const step = Math.max(...proj) - Math.min(...proj) + Math.max(p.gap_y, 2) / m;
      const dx = v[0] * step, dy = v[1] * step;
      if(isCurve(q)){ setCurveLines(q, curveLinesOf(q).map(l => l.map(r => [r[0] + dx, r[1] + dy]))); curveCentre(q); }
      else { q.cx += dx; q.cy += dy; }
    }
    S.gen.placements.push(q); S.gen.plSel = S.gen.placements.length - 1; S.gen.edit = null;
    log(`Placement ${S.gen.plSel + 1}: a copy of placement ${S.gen.placements.indexOf(p) + 1}, just behind it. Drag it where it goes.`);
    saveScatter();
  }));
  // Flip: the mirror image along the rows. The order of the objects reverses, a
  // curve bends the other way, single turns mirror (30° becomes -30°); fronts
  // still face the same side and the models themselves are not mirrored
  ['#btnMirrorPl', '#btnMirrorFol'].forEach(id => $(id).addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    if(isCurve(p)){
      // every curve line mirrored across the line square to the first one's ends,
      // through the middle; the rows still laid out from the same end, so flipped
      const all = curveLinesOf(p), A = all[0][0], B = all[0][all[0].length - 1];
      let u = [B[0] - A[0], B[1] - A[1]];
      if(Math.hypot(...u) < 1e-6) u = [all[0][1][0] - A[0], all[0][1][1] - A[1]];
      const len = Math.hypot(...u) || 1; u = [u[0] / len, u[1] / len];
      const c = [0, 0];
      all.forEach(l => { c[0] += (l[0][0] + l[l.length - 1][0]) / 2 / all.length; c[1] += (l[0][1] + l[l.length - 1][1]) / 2 / all.length; });
      setCurveLines(p, all.map(l => l.map(q => {
        const t = (q[0] - c[0])*u[0] + (q[1] - c[1])*u[1];
        return [q[0] - 2*t*u[0], q[1] - 2*t*u[1]];
      })));
      p.flip = !p.flip; curveCentre(p);
    } else if(p.mirror) delete p.mirror;
    else p.mirror = true;
    Object.keys(p.turns || {}).forEach(id => setTurn(p, id, -p.turns[id]));
    log('Placement flipped: its mirror image, left to right along the rows.');
    saveScatter();
  }));
  $('#btnCurve').addEventListener('click', () => {
    // the rectangle's middle line, end to end: the same copies, now on a line that can bend
    const p = S.gen.placements[S.gen.plSel]; if(!p || isCurve(p)) return;
    const g0 = placementGeom(p); if(!g0) return;
    const hx = g0.u[0] * g0.Lx / 2, hy = g0.u[1] * g0.Lx / 2;
    p.path = [[p.cx - hx, p.cy - hy], [p.cx + hx, p.cy + hy]]; p.flip = false;
    if(p.mirror){ p.path.reverse(); p.flip = true; delete p.mirror; }      // laid out from the same end as before
    log('Curve: double-click the line (or press Add point) to add a point, drag points to bend it, '
      + 'double-click a point to remove it. Drag an end to make the line longer or shorter.');
    saveScatter();
  });
  // Randomize spaces: a new random set within the ranges; editing a range keeps
  // the set and stretches it to the new range; Even spaces goes back to the gaps
  const spaceRange = () => ({ x: [Math.max(0, +$('#spXmin').value || 0), Math.max(0, +$('#spXmax').value || 0)],
                              y: [Math.max(0, +$('#spYmin').value || 0), Math.max(0, +$('#spYmax').value || 0)] });
  $('#btnRandSpaces').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    p.spaces = { on: true, ...spaceRange(), seed: (Math.random() * 4294967296) >>> 0 };
    log(`Random spaces: X ${p.spaces.x[0]}–${p.spaces.x[1]} m, Y ${p.spaces.y[0]}–${p.spaces.y[1]} m. Press again for another set.`);
    saveScatter();
  });
  $('#btnEvenSpaces').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.spaces) return;
    p.spaces.on = false; log('Even spaces: the gaps above apply again.'); saveScatter();
  });
  ['#spXmin', '#spXmax', '#spYmin', '#spYmax'].forEach(id => $(id).addEventListener('input', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    p.spaces = { on: false, seed: (Math.random() * 4294967296) >>> 0, ...(p.spaces || {}), ...spaceRange() };
    drawBridges(); clearTimeout(gapTimer); gapTimer = setTimeout(saveScatter, 300);
  }));
  $('#btnShuffle').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.package) return;
    p.seed = (Math.random() * 4294967296) >>> 0; log('New mix.');
    if(p.auto) relayAuto(p); else saveScatter();
  });
  $('#btnAddPt').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !isCurve(p)) return;
    addCurvePointMiddle(p); curveCentre(p); log('Point added: drag it to bend the line.'); saveScatter();
  });
  $('#btnFlipPl').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !isCurve(p)) return;
    p.flip = !p.flip; log(`Copies now face the ${p.flip ? 'other' : 'first'} side of the line.`); saveScatter();
  });
  $('#btnStraight').addEventListener('click', () => {
    // back to a rectangle along the line from its first point to its last, facing the same way
    // and laid out from the same end; with several curve lines, along the middle of them
    const p = S.gen.placements[S.gen.plSel]; if(!p || !isCurve(p)) return;
    const mid = blendLines(curveLinesOf(p), 0.5);
    const A = mid[0], B = mid[mid.length - 1], m = mppMask(), len = Math.hypot(B[0] - A[0], B[1] - A[1]);
    p.cx = (A[0] + B[0]) / 2; p.cy = (A[1] + B[1]) / 2;
    p.angle = Math.atan2(B[1] - A[1], B[0] - A[0]) * 180 / Math.PI + (p.flip ? 180 : 0);
    if(p.package) p.length = len;                                        // a package fills the length
    else {
      const o = objById(p.object), w = o ? objSize(o)[0] : 1;
      p.nx = Math.max(1, Math.floor((len * m + p.gap_x) / Math.max(w + p.gap_x, 1e-9) + 1e-6));
    }
    if(p.flip) p.mirror = true; else delete p.mirror;
    delete p.path; delete p.flip; delete p.lines;
    log('Placement is a straight rectangle again.'); saveScatter();
  });
  ['#gapX', '#gapY'].forEach((id, k) => $(id).addEventListener('input', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p) return;
    const val = Math.max(0, +$(id).value || 0);
    if(k === 0) p.gap_x = val; else p.gap_y = val;
    drawBridges(); clearTimeout(gapTimer); gapTimer = setTimeout(saveScatter, 300);
  }));
  let gapTimer = null;
  $('#btnRemovePl').addEventListener('click', () => {
    if(S.gen.plSel < 0) return;
    if(AUTO.pick >= 0) AUTO.pick = -1;
    S.gen.placements.splice(S.gen.plSel, 1); S.gen.plSel = -1;
    log('Placement removed.'); saveScatter();
  });

  /* ------------------------------------------------ automatic placement on the islands */
  // An object, or a package's mix, laid by rule (app/autoplace.py) on the islands Generate
  // texture numbered: every island, the ones picked, or a material slot's. One in the middle
  // of each side, its front (+Y) to the street, set back from the kerb; the weight is objects
  // per side. The server lays them; the placement keeps them ("items": centre in mask pixels,
  // "a": where the front faces, clockwise from the top of the map), and lays them again when
  // its settings, its islands or the texture change. Drawn on the overlay, at screen size and
  // only what is in view: a city takes thousands of them
  const AUTO_DEFAULT = { weight: 1, islands: 'all', setback: 3, gap: 2 };
  // a placement laid copy by copy: an automatic one (on the islands) or a layout (on the buildings)
  function laid(p){ return !!(p && (p.auto || p.layout)); }
  const autoOf = p => ({ ...AUTO_DEFAULT, ...(p.auto || {}) });
  const autoAsk = new WeakMap(), autoNote = new WeakMap(), autoTimer = new WeakMap(), autoLaidFor = new WeakMap(), autoKey = new WeakMap();
  const AUTO_BIG = 200;                 // islands from which the console follows the laying (a city: a few seconds)
  // the islands it lays on (numbers), or null for every one
  function autoIslands(p){
    const au = autoOf(p);
    if(au.islands === 'all') return null;
    if(!ISLS.data) return [];
    if(au.islands === 'slot') return slotIslands().lists[au.slot || 0] || [];
    return [...new Set((au.picks || []).map(q => islandAt(q[0], q[1])).filter(Boolean))].sort((a, b) => a - b);
  }
  const autoName = p => p.package ? (pkgById(p.package) || {}).name || 'the package' : (objById(p.object) || {}).name || 'the object';
  async function relayAuto(p){
    const res = S.gen.result;
    if(!p.auto || !S.gen.placements.includes(p)) return;
    if(!res || !ISLS.data){ renderPlacementBox(); return; }        // laid once Generate texture numbers the islands
    const ids = autoIslands(p), n = (autoAsk.get(p) || 0) + 1;
    autoAsk.set(p, n);
    if(ids && !ids.length){
      // no islands picked, or none in its slot, yet: nothing to lay
      p.items = []; p.auto = { ...p.auto, gen: res.id };
      autoNote.set(p, { islands: 0, with: 0, objects: 0 }); autoLaidFor.set(p, '[]');
      saveScatter(); return;
    }
    if(!autoKey.has(p)) autoKey.set(p, Math.random().toString(36).slice(2, 12));
    const body = { generation: res.id, seed: p.seed >>> 0, auto: autoOf(p), island_ids: ids || [], key: autoKey.get(p) };
    if(p.package) body.package = p.package; else body.object = p.object;
    const many = (ids ? ids.length : ISLS.data.count) >= AUTO_BIG;
    status('Laying objects on the islands…');
    try{
      const r = await api('/api/autoplace', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body),
                                              ...(many ? { job: 'Laying objects on the islands' } : {}) });
      if(autoAsk.get(p) !== n || !S.gen.placements.includes(p)) return;                 // asked again since, or removed
      p.items = r.items; p.auto = { ...p.auto, gen: r.generation };
      autoNote.set(p, r.note); autoLaidFor.set(p, JSON.stringify(ids));
      // single turns and empty spots stay with the ids still laid
      const keep = new Set(r.items.map(it => it.id));
      if(p.turns){ Object.keys(p.turns).forEach(k => { if(!keep.has(k)) delete p.turns[k]; }); if(!Object.keys(p.turns).length) delete p.turns; }
      if(p.empty){ p.empty = p.empty.filter(k => keep.has(k)); if(!p.empty.length) delete p.empty; }
      const nt = r.note, idx = S.gen.placements.indexOf(p) + 1, none = nt.islands - nt.with;
      log(`Placement ${idx}: ${nt.objects} × ${autoName(p)} on ${nt.with} of ${nt.islands} island${nt.islands === 1 ? '' : 's'}`
        + (none ? ` (${none} too small for one)` : '') + `, ${autoOf(p).weight} per side at most, in ${r.seconds.toFixed(2)} s.`, nt.objects ? 'ok' : 'warn');
      saveScatter();
    }catch(e){
      if(autoAsk.get(p) === n) log('Automatic placement: ' + e.message, 'bad');
    }
    status('Ready.');
  }
  function relayAutoSoon(p){
    clearTimeout(autoTimer.get(p));
    autoTimer.set(p, setTimeout(() => relayAuto(p), 300));
    drawBridges();
  }
  // the automatic placements an object is in (alone or in its package), laid again when it changes
  const autoUses = (p, oid) => p.auto && (p.object === oid || (p.package && ((pkgById(p.package) || {}).slots || []).some(sl => sl.object === oid)));
  const relayAutoUsing = oid => S.gen.placements.forEach(p => { if(autoUses(p, oid)) relayAutoSoon(p); });
  const relayAutoPackage = pid => S.gen.placements.forEach(p => { if(p.auto && p.package === pid) relayAutoSoon(p); });
  function addAuto(base){
    if(!S.gen.mask) return;
    const p = { ...base, cx: S.gen.mask.width / 2, cy: S.gen.mask.height / 2, angle: 0, nx: 1, ny: 1, gap_x: 0, gap_y: 0,
                seed: (Math.random() * 4294967296) >>> 0, auto: { ...AUTO_DEFAULT }, items: [] };
    S.gen.placements.push(p); S.gen.plSel = S.gen.placements.length - 1; S.gen.edit = null;
    if(!S.gen.result || !ISLS.data) log(`${autoName(p)}: automatic placement added. Press Generate texture: it finds the islands, and the objects are laid on them then.`, 'warn');
    saveScatter(); relayAuto(p);
  }
  $('#btnAutoObj').addEventListener('click', () => { if(S.gen.objSel) addAuto({ object: S.gen.objSel }); });

  // its objects where they stand (mask pixels), worked out once for each change; the overlay
  // draws them again at every move and zoom
  const autoGeomCache = new WeakMap();
  function autoGeom(p){
    const objs = S.gen.objects.map(o => `${o.id}:${o.scale || 1}:${o.turn || 0}`).join(',');
    const sig = [mppMask(), JSON.stringify(p.turns || {}), (p.empty || []).join(','), objs,
                 p.package ? JSON.stringify((pkgById(p.package) || {}).slots || []) : '', !!maskPixels].join('|');
    const c = autoGeomCache.get(p);
    if(c && c.items === p.items && c.sig === sig) return c.g0;
    const g0 = placementGeom(p);
    autoGeomCache.set(p, { items: p.items, sig, g0 });
    return g0;
  }
  // the copy of an automatic placement under a point (mask pixels): [placement, copy], the top one
  function autoCopyAt(x, y, only){
    for(let i = S.gen.placements.length - 1; i >= 0; i--){
      const p = S.gen.placements[i];
      if((only != null && i !== only) || !laid(p) || !(p.items || []).length) continue;
      const g0 = autoGeom(p); if(!g0) continue;
      for(const c of g0.copies){
        const dx = x - c.cx, dy = y - c.cy;
        if(Math.abs(dx*c.u[0] + dy*c.u[1]) <= c.w/2 && Math.abs(dx*c.v[0] + dy*c.v[1]) <= c.d/2) return [i, c];
      }
    }
    return null;
  }
  function hasAuto(){ return S.gen.placements.some(p => laid(p) && (p.items || []).length); }
  // on the overlay (drawOverlay): each object's footprint in its colour (a package's slot
  // colour), the arrow on its front; in Turn single objects its axis and id, in Draw spaced
  // its empty spots dashed. The selected placement's outlines drawn heavier
  function drawAutoOn(ctx, v){
    S.gen.placements.forEach((p, i) => {
      if(!laid(p) || !(p.items || []).length) return;
      const g0 = autoGeom(p); if(!g0) return;
      const sel = i === S.gen.plSel, ed = S.gen.edit && S.gen.edit.i === i ? S.gen.edit.kind : null;
      const X = q => q[0] * v.k + v.ox, Y = q => q[1] * v.k + v.oy;
      const ob = p.package ? null : objById(p.object), turn0 = (ob && ob.turn) || 0;
      const lw = Math.max(v.dpr, v.hr * (sel ? 0.3 : 0.16));
      const path = pts => { ctx.beginPath(); pts.forEach((q, k) => k ? ctx.lineTo(X(q), Y(q)) : ctx.moveTo(X(q), Y(q))); ctx.closePath(); };
      for(const c of g0.copies){
        const r = (c.w + c.d) / 2;
        if(c.cx + r < v.x0 || c.cx - r > v.x1 || c.cy + r < v.y0 || c.cy - r > v.y1) continue;
        const col = c.slot ? c.slot.colour : '216,96,76', small = Math.min(c.w, c.d) * v.k;
        path(c.pts);
        if(c.empty){
          ctx.fillStyle = 'rgba(255,255,255,.06)'; ctx.fill();
          ctx.setLineDash([v.hr * 0.5, v.hr * 0.4]); ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = lw; ctx.stroke(); ctx.setLineDash([]);
          continue;
        }
        const picked = ed === 'objects' && S.gen.edit.sel.has(c.id);
        ctx.fillStyle = c.onRoad ? 'rgba(120,120,120,.55)' : `rgba(${col},.6)`; ctx.fill();
        ctx.strokeStyle = picked ? '#FFD23F' : c.onRoad ? '#999' : sel ? '#fff' : `rgb(${col})`;
        ctx.lineWidth = picked ? Math.max(2 * v.dpr, lw * 2) : lw; ctx.stroke();
        if(c.onRoad){
          ctx.beginPath(); ctx.moveTo(X(c.pts[0]), Y(c.pts[0])); ctx.lineTo(X(c.pts[2]), Y(c.pts[2]));
          ctx.moveTo(X(c.pts[1]), Y(c.pts[1])); ctx.lineTo(X(c.pts[3]), Y(c.pts[3])); ctx.strokeStyle = '#ddd'; ctx.lineWidth = lw; ctx.stroke();
        }
        if(small < 6 * v.dpr || ed === 'spaces') continue;                  // too small on screen for its arrow
        // its front, its +Y: the object's own turn on top of where the placement turned it
        const th = c.a + (c.slot ? c.slot.turn : turn0) * Math.PI / 2, fd = [Math.sin(th), -Math.cos(th)], sd = [Math.cos(th), Math.sin(th)];
        const reach = Math.abs(fd[0]*c.u[0] + fd[1]*c.u[1]) * c.w/2 + Math.abs(fd[0]*c.v[0] + fd[1]*c.v[1]) * c.d/2;
        if(ed === 'objects'){
          // its axis, a green arrow from its middle to its front, and its id (island-number) towards its back
          const side = Math.min(c.w, c.d), head = Math.min(reach * 0.35, side * 0.22);
          const tip = [c.cx + fd[0]*reach*0.92, c.cy + fd[1]*reach*0.92], base = [tip[0] - fd[0]*head, tip[1] - fd[1]*head];
          ctx.beginPath(); ctx.moveTo(X([c.cx, c.cy]), Y([c.cx, c.cy])); ctx.lineTo(X(base), Y(base));
          ctx.strokeStyle = '#3DBE5C'; ctx.lineWidth = Math.max(1.5 * v.dpr, side * v.k * 0.05); ctx.lineCap = 'round'; ctx.stroke(); ctx.lineCap = 'butt';
          path([tip, [base[0] - sd[0]*head*0.6, base[1] - sd[1]*head*0.6], [base[0] + sd[0]*head*0.6, base[1] + sd[1]*head*0.6]]);
          ctx.fillStyle = '#3DBE5C'; ctx.fill();
          const back = Math.min(reach * 0.5, side * 0.3), fs = side * v.k * 0.3;
          if(fs >= v.minFont){
            ctx.font = `${fs}px system-ui, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.lineWidth = side * v.k * 0.03; ctx.strokeStyle = 'rgba(0,0,0,.6)'; ctx.fillStyle = '#fff';
            const tx = X([c.cx - fd[0]*back, 0]), ty = Y([0, c.cy - fd[1]*back]);
            ctx.strokeText(c.id, tx, ty); ctx.fillText(c.id, tx, ty);
          }
          continue;
        }
        const fx = c.cx + fd[0]*reach, fy = c.cy + fd[1]*reach, s = Math.min(v.hr * 1.4 / v.k, Math.min(c.w, c.d) / 3);
        path([[fx + fd[0]*s*1.4, fy + fd[1]*s*1.4], [fx - sd[0]*s, fy - sd[1]*s], [fx + sd[0]*s, fy + sd[1]*s]]);
        ctx.fillStyle = '#fff'; ctx.fill();
      }
    });
  }
  // Pick islands: a click on an island lays objects on it, or takes them away
  function autoPickMode(i){
    AUTO.pick = i;
    if(i >= 0){
      if(LANES.on) lanesMode(false);
      if(ISLS.pick >= 0) setPick(-1);
      if(MAREA.on) mareaMode(false);
      log('Pick islands: click an island to lay objects on it, click it again to take them away. Pick islands or Esc to finish.');
    }
    drawBridges();
  }
  function autoPickIsland(id){
    const p = S.gen.placements[AUTO.pick], isl = islandById(id);
    if(!p || !p.auto || !isl) return;
    const was = p.auto.picks || [], picks = was.filter(q => islandAt(q[0], q[1]) !== id), had = picks.length !== was.length;
    if(!had) picks.push(isl.pt.slice());
    p.auto = { ...p.auto, picks };
    log(had ? `Island ${id} taken out.` : `Island ${id} (${Math.round(isl.area_m2).toLocaleString()} m²) picked.`);
    relayAutoSoon(p);
  }
  [['#auWeight', 'weight', 1, 50, true], ['#auSetback', 'setback', 0, 50], ['#auGap', 'gap', 0, 50]].forEach(([id, key, lo, hi, whole]) =>
    $(id).addEventListener('input', () => {
      const p = S.gen.placements[S.gen.plSel]; if(!p || !p.auto) return;
      const val = parseFloat($(id).value); if(!Number.isFinite(val)) return;
      p.auto = { ...p.auto, [key]: Math.min(hi, Math.max(lo, whole ? Math.round(val) : val)) };
      relayAutoSoon(p);
    }));
  $('#auIslands').addEventListener('change', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.auto) return;
    const mode = $('#auIslands').value;
    p.auto = { ...p.auto, islands: mode };
    if(mode === 'slot' && p.auto.slot == null) p.auto.slot = 0;
    if(mode === 'picked' && !(p.auto.picks || []).length && ISLS.data) autoPickMode(S.gen.plSel);   // nothing picked yet: pick now
    else if(mode !== 'picked' && AUTO.pick >= 0) autoPickMode(-1);
    if(mode === 'slot' && !ISLS.slots.length) log('No material slots yet: add one under Blocks and islands and Pick its islands.', 'warn');
    relayAutoSoon(p); renderPlacementBox();
  });
  $('#auSlot').addEventListener('change', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.auto) return;
    p.auto = { ...p.auto, slot: +$('#auSlot').value || 0 }; relayAutoSoon(p);
  });
  $('#btnAuPick').addEventListener('click', () => {
    if(!ISLS.data){ log('Generate texture first: it numbers the islands.', 'warn'); return; }
    autoPickMode(AUTO.pick === S.gen.plSel ? -1 : S.gen.plSel);
  });
  $('#btnAuLay').addEventListener('click', () => {
    const p = S.gen.placements[S.gen.plSel]; if(!p || !p.auto) return;
    if(!S.gen.result || !ISLS.data){ log('Generate texture first: it finds the islands.', 'warn'); return; }
    relayAuto(p);
  });
  // Turn single objects and Draw spaced on an automatic placement: its objects are on the
  // overlay, so a press is matched to the one under it here, before the map's own drag
  $('#genVp').addEventListener('pointerdown', e => {
    const ed = S.gen.edit;
    if(!S.gen.mask || !ed || (ed.kind !== 'objects' && ed.kind !== 'spaces') || e.button !== 0 || onHandle(e) || e.target.closest('.vp-tools')) return;
    const p = S.gen.placements[ed.i]; if(!p || !laid(p)) return;
    const [x, y] = toMask(e), hit = autoCopyAt(x, y, ed.i); if(!hit) return;
    if(ed.kind === 'objects') pickObject(e, hit[1].id);
    else spaceAt(e, ed.i, hit[1].id, !hit[1].empty, true);
  }, true);
  addEventListener('pointermove', e => {
    // Draw spaced: dragging on empties (or brings back) every object passed over
    const ed = S.gen.edit;
    if(!ed || ed.kind !== 'spaces' || ed.paint == null || !(e.buttons & 1)) return;
    const p = S.gen.placements[ed.i]; if(!p || !laid(p)) return;
    const [x, y] = toMask(e), hit = autoCopyAt(x, y, ed.i);
    if(hit) spaceAt(e, ed.i, hit[1].id, ed.paint, false);
  });

  function renderObjList(){
    renderLayers(KIND.objects); renderLayers(KIND.foliage); renderPkgList();
  }
  function renderLayers(K){
    // objects (or plants) in a package show in its slots
    const layers = S.gen.objects.filter(o => !o.package && !!o.foliage === K.foliage);
    $(K.count).textContent = layers.length ? `${layers.length}` : '';
    const list = $(K.list); list.innerHTML = '';
    layers.forEach(o => {
      const [w, d, hgt] = objSize(o);
      const big = Math.max(w, d, hgt) > 60;
      const row = document.createElement('div');
      row.className = 'bridge-item' + (o.id === S.gen.objSel ? ' sel' : '');
      row.style.flexWrap = 'wrap';
      row.innerHTML = `<span style="flex:1">${o.name} <span style="color:var(--muted)">· ${w.toFixed(2)} × ${d.toFixed(2)} × ${hgt.toFixed(2)} m</span>
          ${big ? '<span style="color:var(--amber)"> · very large: check the scale</span>' : ''}</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="color:var(--muted);font-size:11.5px">scale</span>
          <input type="number" value="${o.scale || 1}" step="0.01" min="0.0001" style="width:62px;background:var(--field);border:1px solid var(--line);border-radius:4px;padding:2px 4px" title="Scale">
          <button class="x turn" title="Turn the ${K.word} a quarter turn within its rectangle" aria-label="Turn ${o.name}" style="font-size:13px">↻ ${90 * ((o.turn || 0) % 4)}°</button>
          <button class="x" title="Delete ${K.word}" aria-label="Delete ${o.name}">×</button></span>`;
      row.addEventListener('click', e => {
        if(e.target.tagName === 'INPUT' || e.target.classList.contains('x')) return;
        S.gen[K.sel] = o.id; $(K.add).disabled = !S.gen.mask; renderObjList();
      });
      row.querySelector('input').addEventListener('change', async ev => {
        const sc = Math.max(0.0001, +ev.target.value || 1);
        try{
          const res = await api('/api/objects/' + o.id, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ scale: sc }) });
          Object.assign(o, res); log(`${o.name}: scale ×${sc}, now ${(o.width_m*sc).toFixed(2)} × ${(o.depth_m*sc).toFixed(2)} m.`);
          renderObjList(); drawBridges(); relayAutoUsing(o.id);
        }catch(err){ log(err.message, 'bad'); }
      });
      row.querySelector('.turn').addEventListener('click', async ev => {
        ev.stopPropagation();
        try{
          const res = await api('/api/objects/' + o.id, { method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ turn: ((o.turn || 0) + 1) % 4 }) });
          Object.assign(o, res); log(`${o.name}: turned to ${90 * o.turn}°.`);
          renderObjList(); drawBridges(); saveScatter();
        }catch(err){ log(err.message, 'bad'); }
      });
      row.querySelector('.x:not(.turn)').addEventListener('click', async ev => {
        ev.stopPropagation();
        try{
          await api('/api/objects/' + o.id, { method:'DELETE' });
          S.gen.objects = S.gen.objects.filter(x => x.id !== o.id);
          S.gen.placements = S.gen.placements.filter(p => p.object !== o.id);
          if(S.gen[K.sel] === o.id) S.gen[K.sel] = null;
          S.gen.plSel = -1; log(`Deleted ${o.name} and its placements.`);
          renderObjList(); saveScatter();
        }catch(err){ log(err.message, 'bad'); }
      });
      list.appendChild(row);
    });
    $(K.add).disabled = !(S.gen[K.sel] && S.gen.mask);
    if(!K.foliage) $('#btnAutoObj').disabled = $(K.add).disabled;
  }

  // a building made an object in the Dynamic creation tab (ui/building.js): the lists again
  window.addEventListener('objects-changed', () => loadObjects());
  async function loadObjects(){
    try{
      S.gen.objects = (await api('/api/objects')).objects;
      S.gen.packages = (await api('/api/packages')).packages;
      renderObjList(); drawBridges();
    }catch(_){}
  }
  loadObjects();

  /* ------------------------------------------------ packages */
  const jsonPost = (url, body) => api(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  function renderPkgList(){ renderPkgs(KIND.objects); renderPkgs(KIND.foliage); }
  function renderPkgs(K){
    const list = $(K.pkgList); list.innerHTML = '';
    const pkgs = S.gen.packages.filter(pk => !!pk.foliage === K.foliage);
    $(K.pkgCount).textContent = pkgs.length ? `${pkgs.length}` : '';
    pkgs.forEach(pk => {
      const slots = pkgSlots(pk);
      const box = document.createElement('div');
      box.className = 'pkg';
      box.innerHTML = `
        <div style="display:flex;gap:6px;align-items:center">
          <input class="pkg-name" value="${esc(pk.name)}" aria-label="Package name">
          <button class="x pkg-del" title="Delete this package and its objects" aria-label="Delete package ${esc(pk.name)}"
            style="background:none;border:none;color:var(--faint);cursor:pointer;font-size:15px">×</button>
        </div>
        <div class="pkg-slots"></div>
        ${slots.length ? '' : `<div class="note">No ${K.words} yet: import some into this package.</div>`}
        <div class="btn-row">
          <button class="btn secondary pkg-import">Import ${K.word}</button>
          <button class="btn secondary pkg-place" ${S.gen.mask && slots.length ? '' : 'disabled'}
            title="${S.gen.mask ? '' : 'Load a street mask first'}">Place on the map</button>
        </div>
        ${K.foliage ? '' : `<button class="btn secondary pkg-auto" ${S.gen.mask && slots.length ? '' : 'disabled'}
          title="Lay the package's mix on the islands by rule, each object picked by its weight, its front (+Y) to the street">Place automatically on islands</button>`}`;
      const holder = $('.pkg-slots', box);
      slots.forEach(sl => {
        const o = sl.o, [w, d, hgt] = objSize(o);
        const row = document.createElement('div');
        row.className = 'bridge-item slot';
        row.style.flexWrap = 'wrap'; row.style.marginTop = '6px';
        row.innerHTML = `<span style="flex:1;min-width:0"><span class="swatch" style="background:rgb(${sl.colour})"></span>${esc(o.name)}
            <span style="color:var(--muted)">· ${w.toFixed(1)} × ${d.toFixed(1)} × ${hgt.toFixed(1)} m</span></span>
          <span style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">
            <span style="color:var(--muted);font-size:11.5px">weight</span>
            <input type="number" class="wt" value="${sl.weight}" step="0.5" min="0" title="How often this object is picked: 2 is twice as often as 1, 0 never">
            <span style="color:var(--muted);font-size:11.5px">scale</span>
            <input type="number" class="sc" value="${o.scale || 1}" step="0.01" min="0.0001" title="Scale">
            <button class="x turn" title="Turn the object a quarter turn" aria-label="Turn ${esc(o.name)}" style="font-size:13px">↻ ${90 * ((o.turn || 0) % 4)}°</button>
            <button class="x del" title="Remove this object from the package" aria-label="Remove ${esc(o.name)}">×</button></span>`;
        $('.wt', row).addEventListener('change', async ev => {
          try{
            const res = await jsonPost('/api/packages/' + pk.id, { weights: { [o.id]: Math.max(0, +ev.target.value || 0) } });
            Object.assign(pk, res); drawBridges(); relayAutoPackage(pk.id);
          }catch(err){ log(err.message, 'bad'); }
        });
        $('.sc', row).addEventListener('change', async ev => {
          try{
            Object.assign(o, await jsonPost('/api/objects/' + o.id, { scale: Math.max(0.0001, +ev.target.value || 1) }));
            renderPkgList(); drawBridges(); relayAutoPackage(pk.id);
          }catch(err){ log(err.message, 'bad'); }
        });
        $('.turn', row).addEventListener('click', async () => {
          try{
            Object.assign(o, await jsonPost('/api/objects/' + o.id, { turn: ((o.turn || 0) + 1) % 4 }));
            log(`${o.name}: turned to ${90 * o.turn}°.`); renderPkgList(); drawBridges();
          }catch(err){ log(err.message, 'bad'); }
        });
        $('.del', row).addEventListener('click', async () => {
          try{
            await api('/api/objects/' + o.id, { method:'DELETE' });
            log(`Removed ${o.name} from ${pk.name}.`); await loadObjects(); relayAutoPackage(pk.id);
          }catch(err){ log(err.message, 'bad'); }
        });
        holder.appendChild(row);
      });
      $('.pkg-name', box).addEventListener('change', async ev => {
        try{ Object.assign(pk, await jsonPost('/api/packages/' + pk.id, { name: ev.target.value })); drawBridges(); }
        catch(err){ log(err.message, 'bad'); }
      });
      $('.pkg-del', box).addEventListener('click', async () => {
        try{
          await api('/api/packages/' + pk.id, { method:'DELETE' });
          S.gen.placements = S.gen.placements.filter(p => p.package !== pk.id); S.gen.plSel = -1;
          log(`Deleted package ${pk.name}, its objects and its placements.`);
          await loadObjects(); saveScatter();
        }catch(err){ log(err.message, 'bad'); }
      });
      $('.pkg-import', box).addEventListener('click', () => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.accept = '.glb,.obj,.fbx'; inp.multiple = true;
        inp.addEventListener('change', async () => {
          const files = [...inp.files]; if(!files.length) return;
          const front = await askFront(files, `into ${pk.name}`);
          if(!front){ log('Import cancelled.'); return; }
          for(const f of files){
            status('Importing object…'); log(`Importing ${f.name} into ${pk.name}…`);
            try{
              const fd = new FormData(); fd.append('file', f); fd.append('package', pk.id); fd.append('front', front);
              const r = await fetch(API + '/api/objects/import', { method:'POST', body: fd });
              const res = await r.json();
              if(!r.ok) throw new Error(res.detail || 'import failed');
              log(`Imported ${res.name} into ${pk.name}: ${res.width_m} × ${res.depth_m} × ${res.height_m} m.${frontNote(res, front)}`, 'ok');
              if(Math.max(res.width_m, res.depth_m, res.height_m) > 60)
                log('  That is very large for an object: the file\'s units were probably off. Set its scale (for example 0.01).', 'bad');
            }catch(e){ log('Import failed: ' + e.message, 'bad'); }
          }
          status('Ready.'); await loadObjects(); relayAutoPackage(pk.id);
        });
        inp.click();
      });
      if(!K.foliage) $('.pkg-auto', box).addEventListener('click', () => { if(slots.length) addAuto({ package: pk.id, length: 0 }); });
      $('.pkg-place', box).addEventListener('click', () => {
        if(!S.gen.mask || !slots.length) return;
        // in the middle of the view, long enough for about three objects, with a mix of its own
        const r = $('#genVp').getBoundingClientRect(), c = genVp.canvas.getBoundingClientRect(), f = shown();
        const cx = ((r.left + r.width/2) - c.left) / c.width * genVp.canvas.width / f;
        const cy = ((r.top + r.height/2) - c.top) / c.height * genVp.canvas.height / f;
        S.gen.placements.push({ package: pk.id, cx: Math.min(Math.max(cx, 0), S.gen.mask.width), cy: Math.min(Math.max(cy, 0), S.gen.mask.height),
          angle: 0, length: 3 * Math.max(...slots.map(s => s.w)) / mppMask(), nx: 1, ny: 1, gap_x: 0, gap_y: 0,
          seed: (Math.random() * 4294967296) >>> 0, ...(K.foliage ? newFoliage() : {}) });
        S.gen.plSel = S.gen.placements.length - 1;
        log(`Placed ${pk.name}. Drag a corner to make it longer or add rows; Shuffle the mix for another random order.`);
        saveScatter();
      });
      list.appendChild(box);
    });
  }
  [['#btnCreatePkg', KIND.objects], ['#btnCreateFolPkg', KIND.foliage]].forEach(([id, K]) => $(id).addEventListener('click', async () => {
    try{
      const res = await jsonPost('/api/packages', { foliage: K.foliage });
      S.gen.packages.push(res); log(`Created ${res.name}. Import ${K.words} into it, then place it on the map.`);
      renderPkgList();
    }catch(err){ log(err.message, 'bad'); }
  }));

  // a new foliage placement: no random transform until its ranges are set
  const newFoliage = () => ({ foliage: true, jitter: { seed: (Math.random() * 4294967296) >>> 0, scale: [1, 1], rotate: [0, 0], offset: [0, 0], overlap: true } });
  // Importing: which way the file's front faces (+Y as here, +X as Unreal's, -Y, -X), asked each
  // time, the last answer ready; the object is turned so its front is +Y like every object here
  let frontAsked = null;
  function askFront(files, what){
    $('#importTitle').textContent = `Import ${what}`;
    $('#imFiles').textContent = files.length === 1 ? files[0].name : `${files.length} files: ${files.map(f => f.name).join(', ')}`;
    let last = '+y';
    try{ last = localStorage.getItem('rta.import.front') || '+y'; }catch(_){}
    $$('input[name="imFront"]').forEach(r => { r.checked = r.value === last; });
    $('#importPop').hidden = false;
    $('#imGo').focus();
    return new Promise(res => { frontAsked = res; });
  }
  function answerFront(ok){
    if(!frontAsked) return;
    const pick = ($$('input[name="imFront"]').find(r => r.checked) || {}).value || '+y', done = frontAsked;
    frontAsked = null; $('#importPop').hidden = true;
    if(ok) try{ localStorage.setItem('rta.import.front', pick); }catch(_){}
    done(ok ? pick : null);
  }
  $('#imGo').addEventListener('click', () => answerFront(true));
  ['#imCancel', '#imClose'].forEach(id => $(id).addEventListener('click', () => answerFront(false)));
  $('#importPop').addEventListener('pointerdown', e => { if(e.target === $('#importPop')) answerFront(false); });
  addEventListener('keydown', e => { if(e.key === 'Escape' && !$('#importPop').hidden) answerFront(false); });
  const frontNote = (res, front) => front && front !== '+y' ? ` Its front was ${FRONT_NAME[front]} in the file: turned to face +Y like every object here.` : '';

  [['#btnImportObj', KIND.objects], ['#btnImportFol', KIND.foliage]].forEach(([id, K]) => $(id).addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file'; inp.accept = '.glb,.obj,.fbx';
    inp.addEventListener('change', async () => {
      const f = inp.files[0]; if(!f) return;
      const front = await askFront([f], K.word);
      if(!front){ log(`Import of ${f.name} cancelled.`); return; }
      status(`Importing ${K.word}…`); log(`Importing ${f.name}…`);
      try{
        const fd = new FormData(); fd.append('file', f); fd.append('front', front);
        if(K.foliage) fd.append('foliage', '1');
        const r = await fetch(API + '/api/objects/import', { method:'POST', body: fd });
        const res = await r.json();
        if(!r.ok) throw new Error(res.detail || 'import failed');
        S.gen.objects.push(res); S.gen[K.sel] = res.id;
        log(`Imported ${res.name}: ${res.width_m} × ${res.depth_m} × ${res.height_m} m, ${res.triangles.toLocaleString()} triangles.${frontNote(res, front)}`, 'ok');
        if(res.frame) log(`  Kept ${res.frame}: the arrow on the map points to its +Y.`);
        if(Math.max(res.width_m, res.depth_m, res.height_m) > 60)
          log('  That is very large for an object: the file\'s units were probably off. Set its scale (for example 0.01).', 'bad');
        renderObjList();
      }catch(e){ log('Import failed: ' + e.message, 'bad'); }
      status('Ready.');
    });
    inp.click();
  }));

  [['#btnAddObj', KIND.objects], ['#btnAddFol', KIND.foliage]].forEach(([id, K]) => $(id).addEventListener('click', () => {
    const oid = S.gen[K.sel];
    if(!oid || !S.gen.mask) return;
    const r = $('#genVp').getBoundingClientRect(), c = genVp.canvas.getBoundingClientRect(), f = shown();
    const cx = ((r.left + r.width/2) - c.left) / c.width * genVp.canvas.width / f;
    const cy = ((r.top + r.height/2) - c.top) / c.height * genVp.canvas.height / f;
    S.gen.placements.push({ object: oid, cx: Math.min(Math.max(cx, 0), S.gen.mask.width),
                            cy: Math.min(Math.max(cy, 0), S.gen.mask.height), angle: 0, nx: 1, ny: 1, gap_x: 0, gap_y: 0,
                            ...(K.foliage ? newFoliage() : {}) });
    S.gen.plSel = S.gen.placements.length - 1;
    log(`Placed ${objById(oid).name}. Drag a corner to add copies; set the gaps in the panel.`);
    saveScatter();
  }));

  /* ------------------------------------------------ memory panel */
  async function renderMemory(){
    if(!S.online) return;
    try{
      const m = await api('/api/memory');
      S.memory = m;
      $('#memState').textContent = `${m.steps} steps`;
      $('#memSummary').innerHTML = `<dl class="kv">
        <dt>Steps recorded</dt><dd class="mono">${fmt(m.steps)}</dd>
        <dt>Images stored</dt><dd class="mono">${fmt(m.images)}</dd>
        <dt>Situations known</dt><dd class="mono">${fmt(m.situations)}</dd>
        <dt>Routes tried</dt><dd class="mono">${fmt(m.attempts)}</dd>
        <dt>Accepted</dt><dd class="mono">${fmt(m.accepted)}</dd>
        <dt>Rejected</dt><dd class="mono">${fmt(m.rejected)}</dd>
        <dt>Tree versions</dt><dd class="mono">${fmt(m.tree_versions)}</dd></dl>
        <div style="margin-top:8px;font-size:11.5px;color:var(--muted);word-break:break-all">Workspace: <span class="mono">${m.workspace || ''}</span></div>
        ${m.fresh && !m.trained_pairs ? '<div style="margin-top:4px;font-size:11.5px;color:var(--red)">This workspace was created new at this start. If you expected your earlier training, copy your old workspace folder here or set its path in workspace.txt.</div>' : ''}`;
      renderTree(curTree);
    }catch(e){ /* server went away; the console already says so */ }
  }

  let curTree = 'material';
  async function renderTree(name){
    curTree = name;
    const el = $('#treeView');
    if(!S.memory || !S.memory.trees.includes(name)){
      el.innerHTML = `<p class="empty">No ${name} tree yet. It is created when priming runs.</p>`;
      return;
    }
    try{
      const trees = await api('/api/memory/trees');
      el.innerHTML = `<pre class="mono" style="white-space:pre-wrap;font-size:11.5px;margin:0">${
        JSON.stringify(trees[name], null, 1)}</pre>`;
    }catch(e){ el.innerHTML = '<p class="empty">Could not read the tree.</p>'; }
  }
  $$('.tree-tabs .chip').forEach(b => b.addEventListener('click', () => {
    $$('.tree-tabs .chip').forEach(x => x.setAttribute('aria-pressed', x === b));
    renderTree(b.dataset.tree);
  }));

  /* ------------------------------------------------ memory tab */
  async function renderPairs2(){
    try{
      const [{pairs}, m] = await Promise.all([api('/api/pairs'), api('/api/memory')]);
      $('#pairsState').textContent = `${pairs.length} pair${pairs.length===1?'':'s'}`;
      $('#memSummary2').innerHTML = `<dl class="kv">
        <dt>Steps recorded</dt><dd class="mono">${fmt(m.steps)}</dd>
        <dt>Trained pairs</dt><dd class="mono">${fmt(m.trained_pairs)}</dd>
        <dt>Situations known</dt><dd class="mono">${fmt(m.situations)}</dd>
        <dt>Corrections</dt><dd class="mono">${fmt(m.attempts)}</dd>
        <dt>Tree versions</dt><dd class="mono">${fmt(m.tree_versions)}</dd></dl>
        <div style="margin-top:8px;font-size:11.5px;color:var(--muted);word-break:break-all">Workspace: <span class="mono">${m.workspace || ''}</span></div>
        ${m.fresh && !m.trained_pairs ? '<div style="margin-top:4px;font-size:11.5px;color:var(--red)">This workspace was created new at this start. If you expected your earlier training, copy your old workspace folder here or set its path in workspace.txt.</div>' : ''}`;

      if(!pairs.length){ $('#pairsTable').innerHTML = '<p class="empty">No pairs yet. Train on a pair in the Train tab.</p>'; }
      else {
        const rows = pairs.map(p => {
          const g = k => p.groups[k] || {};
          const cell = k => g(k).usable ? `${fmt(g(k).pixels)} px, ${g(k).patches} patches` : '<span style="color:var(--red)">too little</span>';
          return `<tr>
            <td>${p.mask}<div style="color:var(--muted);font-size:12px">${p.photo}</div></td>
            <td class="mono">${p.scale}</td>
            <td class="mono">${fmt(p.road_px)}</td>
            <td class="mono">${p.junctions}</td>
            <td>${cell('junction')}</td><td>${cell('edge')}</td><td>${cell('open')}</td>
            <td><button class="load" data-del="${p.id}">Delete</button></td></tr>`;
        }).join('');
        $('#pairsTable').innerHTML = `<table class="dt" style="width:100%;border-collapse:collapse">
          <tr><th style="text-align:left">pair</th><th>scale</th><th>road px</th><th>junctions</th>
              <th>junction</th><th>kerb</th><th>open</th><th></th></tr>${rows}</table>
          <div class="note">Deleting a pair rebuilds the libraries from the rest. No re-analysis, so it is instant.</div>`;
        $$('#pairsTable [data-del]').forEach(b => b.addEventListener('click', () => deletePair(b.dataset.del)));
      }
      renderLibraries();
      renderTiles();
    }catch(e){ log(e.message, 'bad'); }
  }

  async function renderLibraries(){
    try{
      const [idx, lines] = await Promise.all([api('/api/index'), api('/api/lines')]);
      const total = Math.max(...Object.values(idx).map(r => r.length), 0);
      $('#libState').textContent = `${total} pair${total===1?'':'s'}`;
      $('#libList').innerHTML = Object.entries(idx).map(([part, rows]) => `
        <div style="margin-bottom:10px">
          <div style="font-size:12.5px;font-weight:600">${part}</div>
          ${rows.length ? `<ul class="routes">${rows.map(r => `<li class="${r.usable ? '' : 'tried'}">
              <span>${r.rank}. ${r.label.split(' + ')[1] || r.label}</span>
              <span class="mono">${Math.round(r.confidence*100)}%${r.rejections ? `, ✕${r.rejections}` : ''}${
                r.sitting_out ? ', out' : ''}</span></li>`).join('')}</ul>`
            : '<div style="font-size:12px;color:var(--red)">no pairs, priming is used</div>'}
        </div>`).join('') + `
        <div style="margin-bottom:10px">
          <div style="font-size:12.5px;font-weight:600">lines</div>
          <div style="font-size:12px;color:var(--muted)">${lines.width_ratio
            ? `${(lines.width_ratio*100).toFixed(1)}% of each street's width, from ${lines.pairs} pair${lines.pairs===1?'':'s'}`
            : 'not learned yet: no painted lines found in the pairs'}</div>
        </div>`;
    }catch(e){ $('#libList').innerHTML = '<p class="empty">Could not read the index.</p>'; }
  }

  async function renderTiles(){
    try{
      const t = await api('/api/tiles');
      const parts = Object.entries(t.tiles || {});
      const n = parts.reduce((k, [, v]) => k + v.length, 0);
      if(t.settings){
        $('#tPhoto').value = t.settings.photo_width_m; $('#tSize').value = t.settings.tile_m;
        $('#tPx').value = t.settings.px; $('#tVar').value = t.settings.variants;
        if(t.settings.sidewalk_width_m) $('#tSwPhoto').value = t.settings.sidewalk_width_m;
      }
      if(!n){ $('#tilesState').textContent = ''; $('#tilesView').innerHTML = '<p class="empty">Tiles are built after priming and training.</p>'; return; }
      $('#tilesState').textContent = `${n} tiles, ${t.mm_per_px} mm per pixel`;
      $('#tilesView').innerHTML = parts.map(([part, list]) => `
        <div style="margin-bottom:10px">
          <div style="font-size:12.5px;font-weight:600;margin-bottom:4px">${{edge:'kerb band', kerbstone:'kerb stone and face'}[part] || part}${
            part === 'sidewalk' && list[0] && list[0].method ? ` <span style="font-weight:400;color:var(--muted)">(${list[0].method})</span>` : ''}</div>
          <div style="display:grid;grid-template-columns:repeat(${list.length},1fr);gap:8px">
            ${list.map(v => `<div>
              <div title="Repeating 4 × 4, as it will be seen" style="aspect-ratio:1;border:1px solid var(--line);border-radius:4px;
                   background:url(${API + v.url}) 0 0 / 25% 25% repeat"></div>
              <div class="mono" style="font-size:11px;color:var(--muted);margin-top:3px">v${v.variant} · ${v.seam == null
                ? `pattern-aligned, ${v.repeats ? v.repeats.join(' × ') + ' repeats' : ''}`
                : `seam ${v.seam} · noticeable ${v.noticeability}`}</div>
            </div>`).join('')}
          </div>
        </div>`).join('');
    }catch(e){ $('#tilesView').innerHTML = '<p class="empty">Could not read the tiles.</p>'; }
  }

  $('#btnTiles').addEventListener('click', async () => {
    $('#btnTiles').disabled = true; status('Building tiles…');
    log('Building material tiles from the priming close-up…');
    try{
      const res = await api('/api/tiles', { method:'POST', headers:{'Content-Type':'application/json'}, job:'Building the tiles',
        body: JSON.stringify({ photo_width_m: +$('#tPhoto').value, tile_m: +$('#tSize').value,
          px: +$('#tPx').value, variants: +$('#tVar').value, sidewalk_width_m: +$('#tSwPhoto').value }) });
      const n = Object.values(res.tiles).reduce((k, v) => k + v.length, 0);
      log(`${n} tiles built, ${res.mm_per_px} mm per pixel on the road.`, 'ok');
      renderTiles(); renderMemory();
    }catch(e){ log(e.message, 'bad'); }
    status('Ready.'); $('#btnTiles').disabled = false;
  });

  async function deletePair(id){
    try{
      log('Deleting pair and rebuilding the libraries…');
      const res = await api('/api/pairs/' + id, { method:'DELETE' });
      S.rebuilt = res.rebuilt;
      res.rebuilt.changes.forEach(c => log('  ' + c, /none usable/.test(c) ? 'bad' : 'ok'));
      log(`Order rebuilt from ${res.rebuilt.pairs} pair${res.rebuilt.pairs===1?'':'s'}.`, 'ok');
      renderPairs2(); renderMemory();
    }catch(e){ log(e.message, 'bad'); }
  }

  $('#btnRecalc').addEventListener('click', async () => {
    $('#btnRecalc').disabled = true;
    try{
      log('Recalculating the libraries from the stored pairs…');
      const res = await api('/api/recalculate', { method:'POST' });
      S.rebuilt = res.rebuilt;
      res.rebuilt.changes.forEach(c => log('  ' + c, /none usable/.test(c) ? 'bad' : 'ok'));
      log(`Order rebuilt from ${res.rebuilt.pairs} pair${res.rebuilt.pairs===1?'':'s'}.`, 'ok');
      renderPairs2(); renderMemory();
    }catch(e){ log(e.message, 'bad'); }
    $('#btnRecalc').disabled = false;
  });

  /* ------------------------------------------------ info panel */
  const mpp = () => (+$('#scale').value || 1.1);
  function renderInfo(){
    if(S.tab === 'train'){
      $('#infoTitle').textContent = 'Training info';
      const aligned = S.pairs.filter(p => pairState(p) === 'aligned');
      const px = aligned.reduce((n,p) => n + (p.mask.road_px || 0), 0);
      const j = S.junctions && S.junctions.summary;
      $('#info').innerHTML = `<dl class="kv">
        <dt>Prime images</dt><dd>${Object.values(S.prime).filter(Boolean).length} of 3</dd>
        <dt>Noise isolated</dt><dd>${S.noiseArt ? 'yes' : 'no'}</dd>
        <dt>Pairs aligned</dt><dd>${aligned.length} of ${S.pairs.length}</dd>
        <dt>Road pixels</dt><dd>${fmt(px)}</dd>
        <dt>Junctions</dt><dd>${j ? j.junctions : 'not detected'}</dd>
        <dt>Road width</dt><dd>${j ? j.road_width_px_median + ' px' : '—'}</dd></dl>`;
    } else {
      $('#infoTitle').textContent = 'Result info';
      const m = S.gen.mask, j = S.gen.junctions && S.gen.junctions.summary;
      $('#info').innerHTML = `<dl class="kv">
        <dt>Mask</dt><dd>${m ? `${m.width} × ${m.height}` : 'none'}</dd>
        <dt>Road pixels</dt><dd>${m ? fmt(m.road_px) : '0'}</dd>
        <dt>Real size</dt><dd>${m ? `${(m.width*mpp()).toFixed(0)} × ${(m.height*mpp()).toFixed(0)} m` : '—'}</dd>
        <dt>Junctions</dt><dd>${j ? j.junctions : 'not detected'}</dd>
        <dt>Dashes</dt><dd>${S.gen.result ? S.gen.result.summary.markings.dashes : '—'}</dd>
        <dt>Result</dt><dd>${S.gen.result ? 'generated, showing ' + (S.gen.layer || 'result') : 'not generated'}</dd></dl>`;
    }
  }
  $('#scale').addEventListener('input', renderInfo);

  renderPairs(); renderInfo(); updateGen(); updatePrime(); boot();
})();
