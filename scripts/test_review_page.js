'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(id='') { this.id=id; this.value=''; this.checked=false; this.disabled=false; this.innerHTML=''; this.className=''; this.dataset={}; this.listeners=new Map(); }
  addEventListener(type, fn) { if (!this.listeners.has(type)) this.listeners.set(type, []); this.listeners.get(type).push(fn); }
  dispatch(type, event={}) { for (const fn of this.listeners.get(type) || []) fn(Object.assign({type,target:this,currentTarget:this,preventDefault(){},stopPropagation(){}}, event)); }
  appendChild(child) { (this.children ||= []).push(child); }
  setAttribute(name, value) { this[name] = value; }
  getBoundingClientRect() { return {width:600}; }
  setPointerCapture() {}
  releasePointerCapture() {}
  getContext() { return {clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}}; }
  querySelectorAll() { return []; }
  querySelector() { return null; }
  closest() { return null; }
  blur() {}
  removeAttribute(name) { if (name === 'src') this.src=''; }
}

class AudioMock {
  static instances=[];
  static rejectNext=false;
  constructor(url='') { this.src=url; this.readyState=1; this.currentTime=0; this.paused=true; this.listeners=new Map(); this.playCalls=[]; AudioMock.instances.push(this); }
  addEventListener(type, fn) { if (!this.listeners.has(type)) this.listeners.set(type, []); this.listeners.get(type).push(fn); }
  removeEventListener(type, fn) { this.listeners.set(type, (this.listeners.get(type)||[]).filter(item => item !== fn)); }
  dispatch(type, event={}) { for (const fn of [...(this.listeners.get(type)||[])]) fn(Object.assign({type,target:this}, event)); }
  load() {}
  pause() { this.paused=true; }
  play() { this.playCalls.push(this.currentTime); if (AudioMock.rejectNext) { AudioMock.rejectNext=false; return Promise.reject(new Error('blocked')); } this.paused=false; return Promise.resolve(); }
}

const timers = new Map();
let timerId=0;
function fakeSetTimeout(fn, ms) { const id=++timerId; timers.set(id,{fn,ms}); return id; }
function fakeClearTimeout(id) { timers.delete(id); }
function runNextTimer() { const next=[...timers.entries()].sort((a,b)=>a[1].ms-b[1].ms)[0]; assert.ok(next, 'expected a scheduled playback timer'); timers.delete(next[0]); next[1].fn(); }
async function flush() { for (let i=0;i<8;i++) await Promise.resolve(); await new Promise(resolve => setImmediate(resolve)); }

const ids = new Map(['queue','status','save','server-mode','untrusted','mode'].map(id => [id,new Element(id)]));
const documentMock = {
  getElementById(id) { if (!ids.has(id)) ids.set(id,new Element(id)); return ids.get(id); },
  createElement() { return new Element(); },
  addEventListener(type, fn) { (this.listeners ||= new Map()).set(type, fn); },
};
const row = {
  entry:1, current_cs:100, auto_cs:100, review:'REVIEW', severity_score:1, severity_rank:1,
  lyrics:['line'], candidates:[{cs:200, sources:['ctc-current'], diagnostic_only:false, rejected:false}],
};
const song={index:0,song:'demo',has_flac:true,has_stem:true,read_only:false,detail:{has_flac:true,has_stem:true,base_sha256:'base',rows:[row]}};
let peakCalls = 0;
const jobsState = {import_enabled:true, jobs:[]};
const jobsPosts = [], inspectPosts = [], decisionPosts = [];
let loadCalls = 0;
async function fetchMock(url, options={}) {
  // the server answers 415 to any POST that is not declared as JSON
  if (options.method === 'POST' && (options.headers || {})['Content-Type'] !== 'application/json') return {ok:false,status:415,statusText:'Unsupported Media Type',json:async()=>{throw new Error('not json');}};
  if (url === '/api/songs') { loadCalls++; return {ok:true,json:async()=>[song]}; }
  if (url === '/api/version') return {ok:true,json:async()=>({generation:0})};
  if (url === '/api/jobs' && !options.method) return {ok:true,json:async()=>jobsState};
  if (url === '/api/jobs') { jobsPosts.push(JSON.parse(options.body)); return {ok:true,json:async()=>({job:'j1',items:1,rejected:[]})}; }
  if (url === '/api/import/inspect') { inspectPosts.push(JSON.parse(options.body)); return {ok:true,json:async()=>({items:[{audio:'D:\\m\\new.flac',name:'new',lyrics:'D:\\m\\new.lyrics.txt',existing_lrc:false},{audio:'D:\\m\\bare.flac',name:'bare',lyrics:null,existing_lrc:true}],skipped:[]})}; }
  if (url === '/api/song/0/decision') { decisionPosts.push(JSON.parse(options.body)); return {ok:true,json:async()=>({recorded:'x'})}; }
  if (url === '/api/song/0') return {ok:true,json:async()=>song.detail};
  if (url.startsWith('/api/song/0/peaks')) { peakCalls++; return {ok:true,json:async()=>({from_cs:0,to_cs:600,bins:120,sample_rate:16000,peaks:Array.from({length:120},()=>[0,0])})}; }
  if (url === '/api/song/0/save') return {ok:true,json:async()=>song.detail};
  throw new Error('unexpected fetch '+url);
}

let source = fs.readFileSync('scripts/review_page/index.html','utf8');
const match = source.match(/<script>\r?\n([\s\S]*?)\r?\n<\/script>/);
assert.ok(match, 'page script missing');
const script = match[1].replace(/  load\(\)\.then\(\(\) => \{ refreshJobs\(\); watchFiles\(\); \}\);\r?\n\}\)\(\);/, '  globalThis.__reviewTest = {state, render, playAround, playAB, stopPlayer, select, stage, save, rowElement, attachPeaks, accept, keep, openImport, inspectImports, startImport, refreshJobs, renderImport, volume, volumeGain, sliderToDb, dbToSlider, setVolumeDb, toggleMute, setShowAll};\n  load().then(() => { refreshJobs(); watchFiles(); });\n})();');
const storage = new Map();
const localStorageMock = {getItem(key) { return storage.has(key) ? storage.get(key) : null; }, setItem(key, value) { storage.set(key, String(value)); }, removeItem(key) { storage.delete(key); }};
const context = {console, document:documentMock, localStorage:localStorageMock, addEventListener(){}, URLSearchParams: class { constructor(){ } get(){return 'token';} }, location:{search:'',hash:''}, fetch:fetchMock, Audio:AudioMock, setTimeout:fakeSetTimeout, clearTimeout:fakeClearTimeout, alert(){}, Promise, encodeURIComponent, Map, Set, Number, String, Math, JSON, Object, Error, RegExp};
context.globalThis=context;
vm.runInNewContext(script, context, {filename:'review_page/index.html'});

(async () => {
  await flush();
  const test=context.__reviewTest;
  assert.ok(test, 'test hooks did not initialize');
  assert.equal(test.state.autoplay, true, 'autoplay on row change is on by default');
  assert.equal(test.state.batchSkip, true, 'the batch-keep button is on by default');
  test.state.autoplay = false; test.state.batchSkip = false;  // the earlier sections count plays and buttons without them
  const item=test.state.rows[0];
  assert.equal(test.state.player.audio instanceof Map, true, 'player audio cache is a Map');

  test.playAround(item, 100, 'flac'); await flush();
  const flac=AudioMock.instances[0];
  test.playAround(item, 200, 'flac'); await flush();
  assert.equal(AudioMock.instances.length, 1, 'same song/kind reuses one Audio');
  assert.equal(flac.paused, false, 'latest playback remains active');
  test.playAround(item, 300, 'stem'); await flush();
  assert.equal(AudioMock.instances.length, 2, 'flac and stem use separate Audio elements');
  flac.currentTime=99; flac.dispatch('timeupdate');
  assert.equal(test.state.error, '', 'late stale timeupdate has no effect');

  const seqBeforeReject=test.state.player.seq;
  AudioMock.rejectNext=true; test.playAround(item, 400, 'flac'); await flush(); await flush();
  assert.ok(test.state.player.seq > seqBeforeReject, 'play rejection invalidates sequence');
  assert.match(test.state.error, /blocked/, 'play rejection is surfaced');
  test.state.error=''; test.playAround(item, 500, 'stem'); await flush();
  const stem = AudioMock.instances.find(audio => audio.src.includes('/stem'));
  const seqBeforeError = test.state.player.seq; stem.dispatch('error'); await flush();
  assert.ok(test.state.player.seq > seqBeforeError, 'audio error invalidates sequence');
  assert.match(test.state.error, /could not be loaded/, 'audio error is surfaced');

  test.state.error=''; const abStart = (AudioMock.instances.find(audio => audio.src.includes('/flac')) || {playCalls:[]}).playCalls.length; test.playAB(item, {cs:200, diagnostic_only:false}); await flush();
  const abAudio=AudioMock.instances.find(audio => audio.src.includes('/flac'));
  assert.deepEqual(abAudio.playCalls.at(-1), 1, 'A starts at current timestamp');
  runNextTimer(); await flush();
  assert.deepEqual(abAudio.playCalls.at(-1), 2, 'B starts at selected candidate timestamp');
  for (let i=0;i<7;i++) { runNextTimer(); await flush(); }
  assert.equal(test.state.player.ab, null, 'A/B stops after eight segments');
  assert.equal(abAudio.playCalls.length - abStart, 8, 'A/B has eight segments maximum');

  test.playAround(item,100,'flac'); await flush(); const beforeRender=test.state.player.seq; test.render(); assert.ok(test.state.player.seq>beforeRender,'render invalidates playback');
  test.playAround(item,100,'flac'); await flush(); const beforeSelect=test.state.player.seq; test.select(item); assert.ok(test.state.player.seq>beforeSelect,'select invalidates playback');
  const second=Object.assign({}, item, {row:Object.assign({}, item.row, {entry:2, current_cs:300, candidates:[]})}); test.state.rows.push(second); test.render();
  test.playAround(item,100,'flac'); await flush(); const beforeOtherRow=test.state.player.seq; test.select(second); assert.ok(test.state.player.seq>beforeOtherRow,'switching rows invalidates playback');
  test.playAround(item,100,'flac'); await flush(); const beforeMode=test.state.player.seq; ids.get('mode').dispatch('click'); assert.ok(test.state.player.seq>beforeMode,'version switching invalidates playback');
  test.playAround(item,100,'flac'); await flush(); const beforeStage=test.state.player.seq; test.stage(item,120,'manual',0); assert.ok(test.state.player.seq>beforeStage,'stage invalidates playback');
  test.playAround(item,100,'flac'); await flush(); const beforeSave=test.state.player.seq; await test.save(); assert.ok(test.state.player.seq>beforeSave,'save invalidates playback');

  const initialRendered=test.rowElement(item).innerHTML;
  assert.match(initialRendered, /class="ab-play"/, 'row renders A/B control');
  assert.match(initialRendered, /class="ab-play"[^>]*disabled/, 'A/B is disabled without a selected candidate');
  test.state.edits.clear(); test.state.dirty=false;
  test.state.peaks.clear();
  const peaksBefore = peakCalls;
  const canvas = new Element(); test.attachPeaks(item, canvas); await flush();
  assert.equal(peakCalls, peaksBefore + 1, 'first peaks attach fetches the window');
  test.attachPeaks(item, new Element()); await flush();
  assert.equal(peakCalls, peaksBefore + 1, 'same peaks window uses page cache');
  canvas.dispatch('pointerdown',{button:0,clientX:100,pointerId:1});
  canvas.dispatch('pointermove',{clientX:400,pointerId:1});
  assert.equal(test.state.edits.size, 0, 'drag preview does not stage before pointerup');
  canvas.dispatch('pointerup',{clientX:200,pointerId:1});
  const dragEdit = [...test.state.edits.values()][0];
  assert.equal(dragEdit.cs, 200, 'drag maps pixels to centiseconds');
  assert.equal(dragEdit.source, 'drag', 'drag commits through stage');
  assert.ok(test.state.player.active, 'pointerup auto-previews the dropped timestamp');

  const rendered=test.rowElement(item).innerHTML;
  assert.match(rendered, /class="ab-play"/, 'row renders A/B control after drag');
  test.accept(item); await flush();
  assert.deepEqual(decisionPosts.at(-1), {entry:item.row.entry, decision:'change', cs:200, source:'drag'}, 'accepting a staged edit logs a change');
  test.keep(item); await flush();
  assert.deepEqual(decisionPosts.at(-1), {entry:item.row.entry, decision:'keep', cs:item.row.current_cs, source:'keep'}, 'keep logs a keep');
  test.state.edits.set(String(item.song.index) + ':' + item.row.entry, dragEdit); test.state.selected = 0;
  assert.ok(!/class="ab-play"[^>]*disabled/.test(rendered), 'A/B enables after selecting a candidate');

  // ---- import panel ----
  const rail = documentMock.getElementById('rail');
  test.state.importEnabled = false; rail.children = []; test.render();
  assert.ok(!(documentMock.getElementById('rail').children || []).some(child => child.className === 'add-song'), 'rail hides 新增歌曲 before /api/jobs answers');
  rail.children = []; documentMock.getElementById('import').hidden = true;
  await test.refreshJobs(); await flush();
  assert.equal(test.state.importEnabled, true, 'import enabled from /api/jobs');
  assert.ok((documentMock.getElementById('rail').children || []).some(child => child.className === 'add-song'), 'rail shows 新增歌曲');
  test.openImport();
  assert.equal(documentMock.getElementById('import').hidden, false, 'panel opens');
  documentMock.getElementById('import-paths').value = 'D:\\m\\new.flac\nD:\\m\\bare.flac';
  await test.inspectImports(); await flush();
  assert.equal(inspectPosts[0].paths.length, 2, 'inspect posts both paths');
  assert.equal(test.state.imports.length, 2, 'two inspected items');
  const rows = documentMock.getElementById('import-list').children || [];
  assert.match(rows[0].className, /ok/, 'song with lyrics is ready');
  assert.match(rows[1].className, /bad/, 'song without lyrics is flagged');
  assert.match(rows[1].innerHTML, /缺歌詞/, 'missing lyrics is spelled out');
  assert.equal(documentMock.getElementById('import-start').disabled, false, 'start enabled when one song is ready');
  await test.startImport(); await flush();
  assert.deepEqual(jobsPosts[0], {items:[{audio:'D:\\m\\new.flac', overwrite:false}]}, 'only ready songs are queued');
  assert.equal(test.state.imports.length, 1, 'queued song leaves the inspect list');
  jobsState.jobs = [{id:'j1', items:[{audio:'D:\\m\\new.flac', name:'new', status:'running', stage:'LRC tools: generating same-folder .lrc'}]}];
  await test.refreshJobs(); await flush();
  let queueRows = documentMock.getElementById('import-queue').children || [];
  assert.ok(queueRows.some(row => /對齊中/.test(row.innerHTML) && /LRC tools/.test(row.innerHTML)), 'running item shows its stage');
  const loadsBefore = loadCalls;
  jobsState.jobs[0].items[0].status = 'done'; jobsState.jobs[0].items[0].stage = '完成';
  await test.refreshJobs(); await flush();
  assert.equal(loadCalls, loadsBefore + 1, 'a finished song reloads the song list');
  queueRows = documentMock.getElementById('import-queue').children || [];
  assert.ok(queueRows.some(row => /完成 1 首/.test(row.innerHTML)), 'summary appears when the job is finished');
  await test.refreshJobs(); await flush();
  assert.equal(loadCalls, loadsBefore + 1, 'an already-seen finished song does not reload again');
  jobsState.jobs = [{id:'old', items:[{audio:'D:\\m\\old.flac', name:'old', status:'done'}]}, ...jobsState.jobs];
  documentMock.getElementById('import-queue').children = [];
  await test.refreshJobs(); await flush();
  queueRows = documentMock.getElementById('import-queue').children || [];
  assert.ok(!queueRows.some(row => /old/.test(row.innerHTML || '')), 'older finished jobs are not listed');
  assert.equal(queueRows.filter(row => row.className === 'summary').length, 1, 'one 開始審核 for the newest job');
  documentMock.listeners.get('keydown')({key:'Escape', target:{}, preventDefault(){}});
  assert.equal(documentMock.getElementById('import').hidden, true, 'Escape closes the panel');
  console.log('review page import tests: PASS');
  console.log('review page playback tests: PASS');

  // ---- volume slider in dB (0 to -100 dB, like foobar2000) ----
  const volText = () => documentMock.getElementById('vol-text').textContent;
  assert.equal(test.volume.db, 0, 'volume starts at 0 dB');
  assert.equal(test.volumeGain(), 1, '0 dB is full gain');
  assert.equal(volText(), '0.0 dB', 'readout shows 0.0 dB');
  assert.equal(documentMock.getElementById('vol').value, '1000', 'slider starts at the top');
  assert.equal(test.sliderToDb(0), -100, 'slider bottom is -100 dB');
  assert.equal(test.sliderToDb(500), -25, 'the curve gives the quiet end less travel');
  for (const pos of [0, 137, 500, 862, 1000]) assert.equal(test.dbToSlider(test.sliderToDb(pos)), pos, 'slider position round-trips through dB: ' + pos);
  test.setVolumeDb(-20);
  assert.ok(Math.abs(test.volumeGain() - 0.1) < 1e-9, '-20 dB is gain 0.1');
  assert.equal(volText(), '−20.0 dB', 'readout shows the negative dB value');
  assert.ok(AudioMock.instances.length > 0 && AudioMock.instances.every(audio => Math.abs(audio.volume - 0.1) < 1e-9), 'every cached Audio element takes the new gain');
  const before = AudioMock.instances.length;
  test.playAround(item, 100, 'flac'); await flush();
  test.state.player.audio.clear(); test.playAround(item, 100, 'stem'); await flush();
  assert.ok(AudioMock.instances.length > before, 'a fresh Audio element was created');
  assert.ok(Math.abs(AudioMock.instances.at(-1).volume - 0.1) < 1e-9, 'a fresh Audio element starts at the chosen gain');
  assert.deepEqual(JSON.parse(storage.get('lrc-review-volume')), {db:-20, muted:false}, 'the level is remembered in the browser');
  test.setVolumeDb(-123.46);
  assert.equal(test.volume.db, -100, 'dB is clamped at -100');
  assert.equal(volText(), '−∞ dB', '-100 dB reads as silence');
  assert.equal(test.volumeGain(), 0, '-100 dB is gain 0');
  assert.equal(documentMock.getElementById('vol-mute').textContent, '🔇', 'the mute button shows silence at -100 dB');
  test.setVolumeDb(7);
  assert.equal(test.volume.db, 0, 'dB is clamped at 0');
  test.setVolumeDb(-12.34);
  assert.equal(test.volume.db, -12.3, 'dB is kept to one decimal');
  documentMock.getElementById('vol').value = '500'; ids.get('vol').dispatch('input');
  assert.equal(test.volume.db, -25, 'dragging the slider sets dB through the curve');
  ids.get('vol-box').dispatch('wheel', {deltaY:-1});
  assert.equal(test.volume.db, -24, 'wheel up steps 1 dB louder');
  ids.get('vol-box').dispatch('wheel', {deltaY:1}); ids.get('vol-box').dispatch('wheel', {deltaY:1});
  assert.equal(test.volume.db, -26, 'wheel down steps 1 dB quieter');
  documentMock.listeners.get('keydown')({key:'m', target:{}, preventDefault(){}});
  assert.equal(test.volume.muted, true, 'M mutes');
  assert.equal(test.volumeGain(), 0, 'mute is gain 0');
  assert.equal(volText(), '靜音', 'readout says muted');
  assert.equal(documentMock.getElementById('vol-mute').title, '取消靜音', 'mute button offers to unmute');
  assert.equal(AudioMock.instances.at(-1).volume, 0, 'muting silences the cached Audio elements');
  assert.deepEqual(JSON.parse(storage.get('lrc-review-volume')), {db:-26, muted:true}, 'mute is remembered with the level');
  ids.get('vol-mute').dispatch('click');
  assert.equal(test.volume.muted, false, 'the button unmutes');
  assert.equal(test.volume.db, -26, 'unmuting restores the previous level');
  assert.equal(documentMock.getElementById('vol-mute').textContent, '🔊', 'the button shows sound again');
  test.toggleMute(); test.setVolumeDb(-10);
  assert.equal(test.volume.muted, false, 'setting a level unmutes');
  console.log('review page volume tests: PASS');

  // ---- song rail: pending songs first, confirmed songs folded ----
  const railChildren = () => documentMock.getElementById('rail').children || [];
  const tabs = () => railChildren().filter(child => /^song-tab( active)?( todo| done)$/.test(child.className));
  const finished = {index:1, song:'finished', has_flac:true, has_stem:true, read_only:false, detail:{has_flac:true, has_stem:true, base_sha256:'b', rows:[]}};
  test.state.songs.push(finished);
  test.state.done.clear(); test.state.edits.clear(); test.state.dirty = false;
  documentMock.getElementById('rail').children = []; test.render();
  assert.deepEqual(railChildren().filter(c => c.className === 'rail-sec').map(c => c.textContent), ['待確認 1'], 'the pending heading counts songs with lines left');
  const confirmedToggle = () => railChildren().find(c => c.className === 'song-tab group rail-sec');
  assert.match(confirmedToggle().innerHTML, /▾ 已確認 1/, 'finished songs sit under an open 已確認 group');
  assert.deepEqual(tabs().map(c => c.className.replace(' active', '')), ['song-tab todo', 'song-tab done'], 'pending song is listed before the finished one');
  assert.match(tabs()[1].innerHTML, /✓/, 'a finished song shows a check instead of a count');
  confirmedToggle().dispatch('click');
  documentMock.getElementById('rail').children = []; test.render();
  assert.equal(test.state.showDone, false, 'clicking the group folds it');
  assert.match(confirmedToggle().innerHTML, /▸ 已確認 1/, 'the folded group shows its count');
  assert.deepEqual(tabs().map(c => c.className.replace(' active', '')), ['song-tab todo'], 'folded finished songs are not listed');
  test.state.activeSong = 1; documentMock.getElementById('rail').children = []; test.render();
  assert.ok(tabs().some(c => c.className === 'song-tab active done'), 'the active song stays visible even when its group is folded');
  test.state.activeSong = 0; test.state.showDone = true; test.state.songs.pop();
  console.log('review page song rail tests: PASS');

  // ---- review lines grouped by risk: high first, low-risk lines folded and not counted ----
  const queueEl = documentMock.getElementById('queue');
  const listWrap = () => { const list = (queueEl.children || []).find(c => c.className === 'queue-list glass'); return (list && list.children && list.children[0] && list.children[0].children) || []; };
  const headings = () => listWrap().filter(c => c.className === 'q-sec').map(c => c.textContent);
  const lines = () => listWrap().filter(c => /^q( cur)?( done)?$/.test(c.className)).map(c => Number(c.innerHTML.match(/<span class="num">(\d+)<\/span>/)[1]));
  const foldButton = () => listWrap().find(c => c.className === 'q-fold');
  const mkRow = (entry, risk) => Object.assign({}, item, {row:Object.assign({}, item.row, {entry, current_cs:entry * 100, risk, candidates:[]})});
  test.state.rows = [Object.assign({}, item, {row:Object.assign({}, item.row, {risk:'normal'})}), mkRow(2, 'low'), mkRow(3, 'high'), mkRow(4, 'low')];
  test.state.done.clear(); test.state.edits.clear(); test.state.selected = 0; test.state.showLow = false;
  assert.equal(test.state.showAll, true, 'the whole-song view is the default');
  queueEl.children = []; test.render();
  const allLines = () => listWrap().filter(c => /^q( cur)?( done)?( other)?$/.test(c.className)).map(c => Number(c.innerHTML.match(/<span class="num">(\d+)<\/span>/)[1]));
  assert.deepEqual(allLines(), [1, 2, 3, 4], 'the whole-song view lists every line in lyric order');
  assert.deepEqual(headings(), [], 'the whole-song view has no risk headings');
  assert.equal(foldButton(), undefined, 'the whole-song view does not fold low-risk lines');
  test.state.rows[3] = Object.assign({}, test.state.rows[3], {row:Object.assign({}, test.state.rows[3].row, {review:'TRUSTED', risk:null})});
  queueEl.children = []; test.render();
  assert.deepEqual(allLines(), [1, 2, 3, 4], 'lines not marked for review are listed too');
  assert.match(listWrap().find(c => / other$/.test(c.className)).innerHTML, /未標記/, 'an unmarked line says so');
  assert.equal(documentMock.getElementById('left-song').textContent, 2, 'unmarked lines are not counted as left');
  test.state.rows[3] = mkRow(4, 'low');
  test.setShowAll(false);
  assert.equal(JSON.parse(storage.get('lrc-review-settings') || '{}').showAll, false, 'the view choice is remembered');
  queueEl.children = []; test.render();
  assert.deepEqual(lines(), [3, 1], 'the high-risk line is listed first; low-risk lines are folded');
  assert.deepEqual(headings(), ['高風險：先聽這些', '一般'], 'risk headings separate the groups');
  assert.match(foldButton().textContent, /▸ 展開可略過的 2 句/, 'the fold counts the low-risk lines');
  assert.equal(documentMock.getElementById('left-song').textContent, 2, 'low-risk lines are not counted as left');
  assert.equal(documentMock.getElementById('left-all').textContent, 2, 'overall count skips low-risk lines too');
  foldButton().dispatch('click'); queueEl.children = []; test.render();
  assert.equal(test.state.showLow, true, 'clicking the fold opens it');
  assert.deepEqual(lines(), [3, 1, 2, 4], 'opened low-risk lines follow the others in lyric order');
  assert.deepEqual(headings(), ['高風險：先聽這些', '一般', '可略過：多個偵測結果一致，或前後銜接合理'], 'the low-risk group gets its heading');
  assert.match(foldButton().textContent, /▾ 收起可略過的 2 句/, 'the fold offers to close again');
  assert.equal(documentMock.getElementById('left-song').textContent, 2, 'opening the fold does not change the count');
  test.keep(test.state.rows[2]); await flush();
  assert.equal(decisionPosts.at(-1).entry, 3, 'keeping the high-risk line logs that line');
  queueEl.children = []; test.render();
  assert.equal(documentMock.getElementById('left-song').textContent, 1, 'confirming the high-risk line reduces the count');
  console.log('review page risk grouping tests: PASS');

  // ---- reviewer settings: autoplay on row change, batch keep, one-key option accept ----
  const flacAudio = () => AudioMock.instances.filter(audio => audio.src.includes('/0/flac')).at(-1);  // newest element: the cache was cleared earlier
  const flacPlays = () => AudioMock.instances.filter(audio => audio.src.includes('/0/flac')).reduce((n, audio) => n + audio.playCalls.length, 0);
  test.state.rows = [item, second]; test.state.done.clear(); test.state.edits.clear(); test.state.dirty = false; test.state.selected = 0; test.state.showLow = true; test.state.autoplay = false; test.state.batchSkip = false; test.state.mode = 'flac';
  queueEl.children = []; test.render();
  let plays = flacPlays();
  test.keep(item); await flush();
  assert.equal(flacPlays(), plays, 'without autoplay, moving to the next row plays nothing');
  documentMock.getElementById('autoplay').checked = true; ids.get('autoplay').dispatch('change');
  assert.equal(test.state.autoplay, true, 'the autoplay checkbox turns the setting on');
  assert.deepEqual(JSON.parse(storage.get('lrc-review-settings')), {autoplay:true, batchSkip:false, showAll:false}, 'settings are remembered in the browser');
  test.state.done.clear(); test.state.selected = 0; queueEl.children = []; test.render();
  plays = flacPlays();
  test.keep(item); await flush();
  assert.equal(test.state.selected, 1, 'keep moves to the next row');
  assert.equal(flacPlays(), plays + 1, 'with autoplay, the next row starts playing by itself');
  assert.equal(flacAudio().playCalls.at(-1), 3, 'playback starts at the next row\'s working time');
  test.state.selected = 0; plays = flacPlays();
  documentMock.listeners.get('keydown')({key:'ArrowDown', target:{}, preventDefault(){}}); await flush();
  assert.equal(flacPlays(), plays + 1, 'arrow navigation autoplays too');
  documentMock.getElementById('autoplay').checked = false; ids.get('autoplay').dispatch('change');

  test.state.done.clear(); test.state.edits.clear(); test.state.selected = 0; queueEl.children = []; test.render();
  const postsBefore = decisionPosts.length;
  documentMock.listeners.get('keydown')({key:'A', shiftKey:true, target:{}, preventDefault(){}}); await flush();
  assert.equal(decisionPosts.length, postsBefore + 1, 'Shift+A records one decision');
  assert.deepEqual(decisionPosts.at(-1), {entry:1, decision:'change', cs:200, source:'option:ctc-current'}, 'Shift+A takes option A and logs the change');
  assert.ok(test.state.done.has('0:1'), 'Shift+A marks the row done');
  assert.equal(test.state.selected, 1, 'Shift+A moves on to the next row');
  documentMock.listeners.get('keydown')({key:'B', shiftKey:true, target:{}, preventDefault(){}}); await flush();
  assert.equal(decisionPosts.length, postsBefore + 1, 'Shift+B with no option B does nothing');

  test.state.rows = [mkRow(2, 'low'), mkRow(4, 'low'), Object.assign({}, item, {row:Object.assign({}, item.row, {risk:'normal'})})];
  test.state.done.clear(); test.state.edits.clear(); test.state.selected = 0; test.state.batchSkip = false; queueEl.children = []; test.render();
  const batchButton = () => listWrap().find(c => c.className === 'q-fold' && /^全部維持/.test(c.textContent || ''));
  assert.equal(batchButton(), undefined, 'no batch button unless the setting is on');
  documentMock.getElementById('batch-skip').checked = true; ids.get('batch-skip').dispatch('change');
  assert.deepEqual(JSON.parse(storage.get('lrc-review-settings')), {autoplay:false, batchSkip:true, showAll:false}, 'the batch setting is remembered');
  queueEl.children = []; test.render();
  assert.match(batchButton().textContent, /全部維持可略過的 2 句/, 'the batch button counts the unreviewed skip-tier lines');
  const batchBefore = decisionPosts.length;
  batchButton().dispatch('click'); await flush();
  assert.equal(decisionPosts.length, batchBefore + 2, 'one click records a keep for each skip-tier line');
  assert.deepEqual(decisionPosts.slice(-2).map(p => [p.entry, p.decision, p.source]), [[2, 'keep', 'batch-keep'], [4, 'keep', 'batch-keep']], 'batch keeps are logged with their own source');
  assert.ok(test.state.done.has('0:2') && test.state.done.has('0:4') && !test.state.done.has('0:1'), 'only skip-tier lines are marked done');
  assert.match(test.state.notice, /已維持 2 句/, 'the page says what it did');
  queueEl.children = []; test.render();
  assert.equal(batchButton(), undefined, 'nothing left to batch-keep');
  documentMock.getElementById('batch-skip').checked = false; ids.get('batch-skip').dispatch('change');
  console.log('review page settings tests: PASS');
})().catch(error => { console.error(error.stack || error); process.exitCode=1; });
