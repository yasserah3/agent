const UI_VERSION = '2026.10.01-objects8';   // must match VERSION in server.py
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

  /* ------------------------------------------------ tabs */
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    S.tab = t.dataset.tab;
    $$('.tab').forEach(x => x.setAttribute('aria-selected', x === t));
    $('#pane-train').hidden = S.tab !== 'train';
    $('#pane-generate').hidden = S.tab !== 'generate';
    $('#pane-memory').hidden = S.tab !== 'memory';
    if(S.tab === 'memory') renderPairs2();
    $('#clickHint').hidden = true;
    renderInfo();
    if(S.tab !== 'memory') requestAnimationFrame(() => (S.tab === 'train' ? trainVp : genVp).fit());
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
  S.gen.objects = []; S.gen.objSel = null; S.gen.placements = []; S.gen.plSel = -1;
  let maskPixels = null;

  function objById(id){ return S.gen.objects.find(o => o.id === id); }
  function objSize(o){
    const k = o.scale || 1, w = o.width_m * k, d = o.depth_m * k;
    return (o.turn || 0) % 2 ? [d, w, o.height_m * k] : [w, d, o.height_m * k];   // a quarter turn swaps them
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

  function placementGeom(p){
    const o = objById(p.object); if(!o) return null;
    const m = mppMask(), [w, d] = objSize(o);
    const a = p.angle * Math.PI / 180, u = [Math.cos(a), Math.sin(a)], v = [-u[1], u[0]];
    const Lx = (p.nx * w + (p.nx - 1) * p.gap_x) / m, Ly = (p.ny * d + (p.ny - 1) * p.gap_y) / m;
    const copies = [];
    for(let i = 0; i < p.nx; i++) for(let k = 0; k < p.ny; k++){
      const ox = (-Lx/2 + (w/2 + i*(w + p.gap_x)) / m), oy = (-Ly/2 + (d/2 + k*(d + p.gap_y)) / m);
      const cx = p.cx + u[0]*ox + v[0]*oy, cy = p.cy + u[1]*ox + v[1]*oy;
      const hw = w / m / 2, hd = d / m / 2;
      const pts = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([sx, sy]) => [cx + u[0]*sx*hw + v[0]*sy*hd, cy + u[1]*sx*hw + v[1]*sy*hd]);
      const onRoad = pts.concat([[cx, cy]]).some(q => roadAt(q[0], q[1]));
      copies.push({ cx, cy, pts, onRoad });
    }
    const outer = [[-1,-1],[1,-1],[1,1],[-1,1]].map(([sx, sy]) => [p.cx + u[0]*sx*Lx/2 + v[0]*sy*Ly/2, p.cy + u[1]*sx*Lx/2 + v[1]*sy*Ly/2]);
    return { u, v, Lx, Ly, w: w/m, d: d/m, copies, outer };
  }

  function drawScatter(f, hr){
    S.gen.placements.forEach((p, i) => {
      const g0 = placementGeom(p); if(!g0) return;
      const sel = i === S.gen.plSel;
      const g = el('g', {}, layer);
      const P = q => `${q[0]*f},${q[1]*f}`;
      // the whole area in blue: what shows between the copies is the gap
      const area = el('polygon', { points: g0.outer.map(P).join(' '), class:'grab',
        fill:'rgba(70,130,220,.35)', stroke:'#4682DC', 'stroke-width': sel ? hr*0.6 : hr*0.35 }, g);
      area.addEventListener('pointerdown', e => startScatterDrag(e, i, 'move'));
      g0.copies.forEach(c => {
        el('polygon', { points: c.pts.map(P).join(' '), fill: c.onRoad ? 'rgba(120,120,120,.55)' : 'rgba(216,96,76,.55)',
          stroke: c.onRoad ? '#999' : '#D8604C', 'stroke-width': hr*0.25, 'pointer-events':'none' }, g);
        if(c.onRoad){
          el('line', { x1:c.pts[0][0]*f, y1:c.pts[0][1]*f, x2:c.pts[2][0]*f, y2:c.pts[2][1]*f, stroke:'#ddd', 'stroke-width': hr*0.25, 'pointer-events':'none' }, g);
          el('line', { x1:c.pts[1][0]*f, y1:c.pts[1][1]*f, x2:c.pts[3][0]*f, y2:c.pts[3][1]*f, stroke:'#ddd', 'stroke-width': hr*0.25, 'pointer-events':'none' }, g);
        }
      });
      // the front: Blender's +Y, the top edge at rotation 0, turned with the object
      const ob = objById(p.object), th = (p.angle + 90 * ((ob && ob.turn) || 0)) * Math.PI / 180;
      const fd = [Math.sin(th), -Math.cos(th)], sd = [Math.cos(th), Math.sin(th)];
      const reach = Math.abs(fd[0]*g0.u[0] + fd[1]*g0.u[1]) * g0.Lx/2 + Math.abs(fd[0]*g0.v[0] + fd[1]*g0.v[1]) * g0.Ly/2;
      const fx = p.cx + fd[0]*reach, fy = p.cy + fd[1]*reach;
      const tip = [fx + fd[0]*hr*2.2/f, fy + fd[1]*hr*2.2/f];
      const l = [fx - sd[0]*hr*1.2/f, fy - sd[1]*hr*1.2/f], r = [fx + sd[0]*hr*1.2/f, fy + sd[1]*hr*1.2/f];
      el('polygon', { points: [tip, l, r].map(P).join(' '), fill:'#fff', 'pointer-events':'none' }, g);
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

  let sdrag = null;
  function startScatterDrag(e, i, mode){
    e.stopPropagation(); e.preventDefault();
    S.gen.plSel = i;
    sdrag = { i, mode, start: toMask(e), orig: { ...S.gen.placements[i] } };
    drawBridges();
  }
  addEventListener('pointermove', e => {
    if(!sdrag) return;
    const p = S.gen.placements[sdrag.i], o = sdrag.orig, q = toMask(e);
    if(sdrag.mode === 'move'){
      p.cx = o.cx + q[0] - sdrag.start[0]; p.cy = o.cy + q[1] - sdrag.start[1];
    } else if(sdrag.mode === 'size'){
      // stretch adds whole copies; the rectangle never goes below one object
      const ob = objById(p.object); if(!ob) return;
      const [w, d] = objSize(ob), m = mppMask();
      const a = o.angle * Math.PI / 180, dx = q[0] - o.cx, dy = q[1] - o.cy;
      const wantX = 2 * Math.abs(dx*Math.cos(a) + dy*Math.sin(a)) * m, wantY = 2 * Math.abs(-dx*Math.sin(a) + dy*Math.cos(a)) * m;
      p.nx = Math.max(1, Math.round((wantX + p.gap_x) / (w + p.gap_x)));
      p.ny = Math.max(1, Math.round((wantY + p.gap_y) / (d + p.gap_y)));
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
    if(!p){ box.hidden = true; return; }
    box.hidden = false;
    const o = objById(p.object);
    const g0 = placementGeom(p);
    const onRoad = g0 ? g0.copies.filter(c => c.onRoad).length : 0;
    $('#plTitle').textContent = `Placement ${S.gen.plSel + 1}: ${o ? o.name : 'missing object'}`;
    if(document.activeElement !== $('#gapX')) $('#gapX').value = p.gap_x;
    if(document.activeElement !== $('#gapY')) $('#gapY').value = p.gap_y;
    $('#plInfo').textContent = `${p.nx} × ${p.ny} = ${p.nx * p.ny} copies` + (onRoad ? `, ${onRoad} on the road (left out)` : '')
      + (g0 ? `, area ${(g0.Lx * mppMask()).toFixed(1)} × ${(g0.Ly * mppMask()).toFixed(1)} m` : '');
  }
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
    $('#objCount').textContent = S.gen.objects.length ? `${S.gen.objects.length}` : '';
    const list = $('#objList'); list.innerHTML = '';
    S.gen.objects.forEach(o => {
      const [w, d, hgt] = objSize(o);
      const big = Math.max(w, d, hgt) > 60;
      const row = document.createElement('div');
      row.className = 'bridge-item' + (o.id === S.gen.objSel ? ' sel' : '');
      row.style.flexWrap = 'wrap';
      row.innerHTML = `<span style="flex:1">${o.name} <span style="color:var(--muted)">· ${w.toFixed(2)} × ${d.toFixed(2)} × ${hgt.toFixed(2)} m</span>
          ${big ? '<span style="color:var(--amber)"> · very large: check the scale</span>' : ''}</span>
        <span style="display:flex;align-items:center;gap:4px"><span style="color:var(--muted);font-size:11.5px">scale</span>
          <input type="number" value="${o.scale || 1}" step="0.01" min="0.0001" style="width:62px;background:var(--field);border:1px solid var(--line);border-radius:4px;padding:2px 4px" title="Scale">
          <button class="x turn" title="Turn the object a quarter turn within its rectangle" aria-label="Turn ${o.name}" style="font-size:13px">↻ ${90 * ((o.turn || 0) % 4)}°</button>
          <button class="x" title="Delete object" aria-label="Delete ${o.name}">×</button></span>`;
      row.addEventListener('click', e => {
        if(e.target.tagName === 'INPUT' || e.target.classList.contains('x')) return;
        S.gen.objSel = o.id; $('#btnAddObj').disabled = !S.gen.mask; renderObjList();
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
          if(S.gen.objSel === o.id) S.gen.objSel = null;
          S.gen.plSel = -1; log(`Deleted ${o.name} and its placements.`);
          renderObjList(); saveScatter();
        }catch(err){ log(err.message, 'bad'); }
      });
      list.appendChild(row);
    });
    $('#btnAddObj').disabled = !(S.gen.objSel && S.gen.mask);
  }

  async function loadObjects(){
    try{ S.gen.objects = (await api('/api/objects')).objects; renderObjList(); }catch(_){}
  }
  loadObjects();

  $('#btnImportObj').addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file'; inp.accept = '.glb,.obj,.fbx';
    inp.addEventListener('change', async () => {
      const f = inp.files[0]; if(!f) return;
      status('Importing object…'); log(`Importing ${f.name}…`);
      try{
        const fd = new FormData(); fd.append('file', f);
        const r = await fetch(API + '/api/objects/import', { method:'POST', body: fd });
        const res = await r.json();
        if(!r.ok) throw new Error(res.detail || 'import failed');
        S.gen.objects.push(res); S.gen.objSel = res.id;
        log(`Imported ${res.name}: ${res.width_m} × ${res.depth_m} × ${res.height_m} m, ${res.triangles.toLocaleString()} triangles.`, 'ok');
        if(res.frame) log(`  Kept ${res.frame}: at rotation 0 its +Y faces the top of the map.`);
        if(res.straightened_deg) log(`  Straightened by ${res.straightened_deg}° so its footprint lines up with the axes. If the wrong side is now the front, use ↻.`);
        if(Math.max(res.width_m, res.depth_m, res.height_m) > 60)
          log('  That is very large for an object: the file\'s units were probably off. Set its scale (for example 0.01).', 'bad');
        renderObjList();
      }catch(e){ log('Import failed: ' + e.message, 'bad'); }
      status('Ready.');
    });
    inp.click();
  });

  $('#btnAddObj').addEventListener('click', () => {
    if(!S.gen.objSel || !S.gen.mask) return;
    const r = $('#genVp').getBoundingClientRect(), c = genVp.canvas.getBoundingClientRect(), f = shown();
    const cx = ((r.left + r.width/2) - c.left) / c.width * genVp.canvas.width / f;
    const cy = ((r.top + r.height/2) - c.top) / c.height * genVp.canvas.height / f;
    S.gen.placements.push({ object: S.gen.objSel, cx: Math.min(Math.max(cx, 0), S.gen.mask.width),
                            cy: Math.min(Math.max(cy, 0), S.gen.mask.height), angle: 0, nx: 1, ny: 1, gap_x: 0, gap_y: 0 });
    S.gen.plSel = S.gen.placements.length - 1;
    log(`Placed ${objById(S.gen.objSel).name}. Drag a corner to add copies; set the gaps in the panel.`);
    saveScatter();
  });

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
