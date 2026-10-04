const UI_VERSION = '2026.10.04-streets5';   // must match VERSION in server.py
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

  async function api(path, opts){
    const res = await fetch(API + path, opts);
    let body = null;
    try { body = await res.json(); } catch(e) { body = null; }
    if(!res.ok){
      const msg = (body && (body.error || body.detail)) || `${res.status} ${res.statusText}`;
      const err = new Error(msg); err.status = res.status; err.body = body; throw err;
    }
    return body;
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
    }catch(e){
      $('.engine').innerHTML = '<i style="background:var(--red)"></i>Server not running';
      log('No server. Start it with: python server.py', 'bad');
    }
  }

  $('#uiVersion').textContent = 'v' + UI_VERSION;

  /* ------------------------------------------------ tabs */
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    S.tab = t.dataset.tab;
    $$('.tab').forEach(x => x.setAttribute('aria-selected', x === t));
    $('#pane-train').hidden = S.tab !== 'train';
    $('#pane-generate').hidden = S.tab !== 'generate';
    $('#pane-memory').hidden = S.tab !== 'memory';
    $('#pane-view3d').hidden = S.tab !== 'view3d';
    window.dispatchEvent(new CustomEvent('view3d', { detail: S.tab === 'view3d' }));   // ui/view3d.js draws only while showing
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
    const apply = () => {
      stage.style.transform = `translate(${tx}px,${ty}px) scale(${s})`;
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
    return { canvas, show, fit, wasDrag: () => moved };
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
      const res = await api('/api/tiles', { method:'POST', headers:{'Content-Type':'application/json'}, body: '{}' });
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
      const res = await api('/api/prime', { method:'POST', headers:{'Content-Type':'application/json'},
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
      const res = await api('/api/train', { method:'POST', headers:{'Content-Type':'application/json'},
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
      const res = await api('/api/junctions', { method:'POST', headers:{'Content-Type':'application/json'},
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
      const rec = await upload(f, 'gen_mask');
      S.gen.mask = rec; S.gen.junctions = null;
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
        if(S.gen.placements.length) log(`${S.gen.placements.length} saved object placement(s) restored for this mask.`);
        renderObjList();
        drawBridges();
      }catch(_){}
      status('Ready.'); updateGen(); renderInfo(); renderMemory();
    }catch(e){ log('Upload failed: ' + e.message, 'bad'); status('Upload failed.'); }
  });
  $('.load', genRow).addEventListener('click', e => { e.stopPropagation(); loadGen(); });
  genRow.addEventListener('click', () => S.gen.mask ? genVp.show(S.gen.mask.img, `Mask, ${S.gen.mask.width} × ${S.gen.mask.height}`) : loadGen());
  $('#genMaskPrev').addEventListener('click', () => { if(S.gen.mask) genVp.show(S.gen.mask.img, 'Mask'); });

  function updateGen(){
    const ok = !!S.gen.mask;
    $('#btnGenerate').disabled = !ok;
    $('#btnGenJunctions').disabled = !ok;
    $('#genNote').textContent = ok ? 'Ready.' : 'Load a street mask.';
  }
  $('#btnGenJunctions').addEventListener('click', async () => {
    if(!S.gen.mask) return;
    $('#btnGenJunctions').disabled = true;
    S.gen.junctions = await detectJunctions(S.gen.mask.id, 'Mask', genVp, null);
    renderInfo(); renderMemory();
    $('#btnGenJunctions').disabled = false;
  });
  $('#btnGenerate').addEventListener('click', async () => {
    if(!S.gen.mask) return;
    $('#btnGenerate').disabled = true; status('Generating…');
    log('Generating: junctions first, then material, wear and markings.');
    try{
      const res = await api('/api/generate', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ mask: S.gen.mask.id, scale: +$('#scale').value,
          wear: +$('#noiseAmt').value, seed: +$('#seed').value,
          cycle_m: +$('#cycleM').value, marking_width_m: +$('#markW').value,
          quality: +$('#quality').value, align_lines: S.gen.alignLines || false,
          output_scale: +$('#outScale').value, soft_edges: $('#softEdges').checked,
          grain: +$('#grainAmt').value, match_material: $('#matchMat').checked,
          line_width_mode: $('#lineMode').value }) });
      S.gen.result = res;
      const s = res.summary;
      log(`Output ${s.size[0]} × ${s.size[1]} (${s.output_scale}× the mask${s.soft_edges ? ', smooth edges' : ''}), grain ${Math.round(s.grain*50)}${s.matched_tone != null ? `, colour matched to tone ${s.matched_tone}` : ''}.`);
      log(`  Road ${s.road_width_px} px wide (${s.road_width_m} m), ${s.junctions} junctions.`);
      Object.entries(res.decisions || {}).forEach(([k,d]) => log(`  ${k}: ${d.label || 'none'}${d.rank ? ` (position ${d.rank})` : ''}${d.exhausted ? ', all rejected' : ''}`));
      log(`  ${s.markings.dashes} dashes placed, cycle ${s.markings.cycle_px} px, lines ${s.markings.width_px} px wide${
        s.markings.width_mode === 'learned' ? ` (${(s.markings.width_ratio*100).toFixed(1)}% of each street's width, learned)`
          : ' (fixed width)'}.`);
      if($('#lineMode').value === 'learned' && s.markings.width_mode !== 'learned')
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
    }catch(e){ log(e.message, 'bad'); $('#genNote').textContent = e.message; }
    status('Ready.'); $('#btnGenerate').disabled = false; renderMemory();
  });

  $('#btn3d').addEventListener('click', async () => {
    if(!S.gen.result) return;
    $('#btn3d').disabled = true; status('Building 3D model…');
    log('Building the 3D model: tracing the road outline, triangulating, applying the texture…');
    try{
      const res = await api('/api/export3d', { method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ generation: S.gen.result.id, mesh: $('#meshMode').value,
          straightness: +$('#straight').value, spacing_m: +$('#rowSpacing').value,
          variation: +$('#variation').value, seed: +$('#seed').value,
          sidewalks: $('#swOn').checked, sw_height_cm: +$('#swHeight').value, sw_share: +$('#swShare').value,
          sw_min_m: +$('#swMin').value, sw_max_m: +$('#swMax').value,
          sw_skip_interchanges: $('#swSkip').checked,
          markings: $('#markMode').value,
          mesh_detail: $('#meshDetail').value,
          blocks: $('#blocksOn').checked,
          scatter: S.gen.placements,
          bridges: S.gen.bridges, bridge_height_m: +$('#brHeight').value,
          bridge_ramp_m: +$('#brRamp').value, bridge_deck_m: +$('#brDeck').value }) });
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
            + `${res.marked_textures.length} marked texture(s) (${res.marked_textures.map(m => m.width_cm + ' cm lines').join(', ')}).`
            + (res.strip_dashes ? ` ${res.strip_dashes} dashes on bridge crossings stay as strips.` : ''));
        else log(`  Markings: ${res.dashes} dashes as separate strips.`);
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
      log('  Units are metres, Y up. Opens directly in Blender (File > Import > glTF) and Unreal (drag into the Content Browser).');
      const link = document.createElement('a');
      link.href = API + res.url; link.download = '';
      document.body.appendChild(link); link.click(); link.remove();
    }catch(e){ log('3D export failed: ' + e.message, 'bad'); }
    status('Ready.'); $('#btn3d').disabled = false; renderMemory();
  });

  async function showLayer(name){
    const res = S.gen.result;
    if(!res) return;
    S.gen.layer = name;
    const img = await loadImage(API + res.urls[name]);
    genVp.show(img, `${name}, ${img.width} × ${img.height}`);
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
    const map = { material:'material', noise:'wear', markings:'markings' };
    if(!S.gen.result) return;
    const name = map[b.dataset.layer];
    showLayer(S.gen.layer === name ? 'result' : name);
  }));

  /* ------------------------------------------------ review */
  genVp.canvas.addEventListener('click', async e => {
    if(!S.gen.result || genVp.wasDrag && genVp.wasDrag()) return;
    const c = e.currentTarget, r = c.getBoundingClientRect();
    const x = Math.floor((e.clientX - r.left) / r.width * c.width);
    const y = Math.floor((e.clientY - r.top) / r.height * c.height);
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
  $('#lineMode').addEventListener('change', () => { $('#markW').disabled = $('#lineMode').value !== 'fixed'; });

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

  function drawBridges(){
    const f = shown();
    layer.setAttribute('width', genVp.canvas.width); layer.setAttribute('height', genVp.canvas.height);
    layer.innerHTML = '';
    if(!S.gen.mask) return;
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
      return { cx, cy, a, u, v, pts, onRoad, w: w / m, d: d / m, slot: slot || null, id };
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
      if((g0.curve || g0.pkg) && ed !== 'objects' && !p.foliage){
        // a small arrow on each copy's front: Blender's +Y, turned with the curve and the object
        // (not on plants, whose random turns would make them point every way)
        g0.copies.forEach(c => {
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
    if(S.gen.edit && S.gen.edit.paint){ S.gen.edit.paint = null; saveScatter(); }
  });
  // Single objects: click to pick one, shift-click to add or remove more
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
    Object.values(KIND).forEach(K => { $(K.dup).disabled = $(K.flip).disabled = !p || kindOf(p) !== K; });
    if(!p){ box.hidden = true; S.gen.edit = null; return; }
    box.hidden = false;
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
    const g0 = placementGeom(p);
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
    const onRoad = g0 ? g0.copies.filter(c => c.onRoad).length : 0;
    const curve = isCurve(p);
    $('#plTitle').textContent = `${p.foliage ? 'Foliage placement' : 'Placement'} ${S.gen.plSel + 1}: `
      + (p.package ? (pk ? `package ${pk.name}` : 'missing package') : (o ? o.name : 'missing object')) + (curve ? ', curved' : '');
    $('#btnShuffle').hidden = !p.package;
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
    const road = onRoad ? `, ${onRoad} on the road (left out)` : '';
    if(p.package){
      // how many of each object the mix holds
      const n = {};
      (g0 ? g0.copies : []).forEach(c => { n[c.slot.o.name] = (n[c.slot.o.name] || 0) + 1; });
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
    p.seed = (Math.random() * 4294967296) >>> 0; log('New mix.'); saveScatter();
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
    S.gen.placements.splice(S.gen.plSel, 1); S.gen.plSel = -1;
    log('Placement removed.'); saveScatter();
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
          renderObjList(); drawBridges();
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
  }

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
        </div>`;
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
            Object.assign(pk, res); drawBridges();
          }catch(err){ log(err.message, 'bad'); }
        });
        $('.sc', row).addEventListener('change', async ev => {
          try{
            Object.assign(o, await jsonPost('/api/objects/' + o.id, { scale: Math.max(0.0001, +ev.target.value || 1) }));
            renderPkgList(); drawBridges();
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
            log(`Removed ${o.name} from ${pk.name}.`); await loadObjects();
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
          for(const f of inp.files){
            status('Importing object…'); log(`Importing ${f.name} into ${pk.name}…`);
            try{
              const fd = new FormData(); fd.append('file', f); fd.append('package', pk.id);
              const r = await fetch(API + '/api/objects/import', { method:'POST', body: fd });
              const res = await r.json();
              if(!r.ok) throw new Error(res.detail || 'import failed');
              log(`Imported ${res.name} into ${pk.name}: ${res.width_m} × ${res.depth_m} × ${res.height_m} m.`, 'ok');
              if(Math.max(res.width_m, res.depth_m, res.height_m) > 60)
                log('  That is very large for an object: the file\'s units were probably off. Set its scale (for example 0.01).', 'bad');
            }catch(e){ log('Import failed: ' + e.message, 'bad'); }
          }
          status('Ready.'); await loadObjects();
        });
        inp.click();
      });
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
  [['#btnImportObj', KIND.objects], ['#btnImportFol', KIND.foliage]].forEach(([id, K]) => $(id).addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file'; inp.accept = '.glb,.obj,.fbx';
    inp.addEventListener('change', async () => {
      const f = inp.files[0]; if(!f) return;
      status(`Importing ${K.word}…`); log(`Importing ${f.name}…`);
      try{
        const fd = new FormData(); fd.append('file', f);
        if(K.foliage) fd.append('foliage', '1');
        const r = await fetch(API + '/api/objects/import', { method:'POST', body: fd });
        const res = await r.json();
        if(!r.ok) throw new Error(res.detail || 'import failed');
        S.gen.objects.push(res); S.gen[K.sel] = res.id;
        log(`Imported ${res.name}: ${res.width_m} × ${res.depth_m} × ${res.height_m} m, ${res.triangles.toLocaleString()} triangles.`, 'ok');
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
      const res = await api('/api/tiles', { method:'POST', headers:{'Content-Type':'application/json'},
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
