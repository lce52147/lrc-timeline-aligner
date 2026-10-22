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
const jobsPosts = [], inspectPosts = [];
let loadCalls = 0;
async function fetchMock(url, options={}) {
  if (url === '/api/songs') { loadCalls++; return {ok:true,json:async()=>[song]}; }
  if (url === '/api/version') return {ok:true,json:async()=>({generation:0})};
  if (url === '/api/jobs' && !options.method) return {ok:true,json:async()=>jobsState};
  if (url === '/api/jobs') { jobsPosts.push(JSON.parse(options.body)); return {ok:true,json:async()=>({job:'j1',items:1,rejected:[]})}; }
  if (url === '/api/import/inspect') { inspectPosts.push(JSON.parse(options.body)); return {ok:true,json:async()=>({items:[{audio:'D:\\m\\new.flac',name:'new',lyrics:'D:\\m\\new.lyrics.txt',existing_lrc:false},{audio:'D:\\m\\bare.flac',name:'bare',lyrics:null,existing_lrc:true}],skipped:[]})}; }
  if (url === '/api/song/0') return {ok:true,json:async()=>song.detail};
  if (url.startsWith('/api/song/0/peaks')) { peakCalls++; return {ok:true,json:async()=>({from_cs:0,to_cs:600,bins:120,sample_rate:16000,peaks:Array.from({length:120},()=>[0,0])})}; }
  if (url === '/api/song/0/save') return {ok:true,json:async()=>song.detail};
  throw new Error('unexpected fetch '+url);
}

let source = fs.readFileSync('scripts/review_page/index.html','utf8');
const match = source.match(/<script>\r?\n([\s\S]*?)\r?\n<\/script>/);
assert.ok(match, 'page script missing');
const script = match[1].replace(/  load\(\)\.then\(\(\) => \{ refreshJobs\(\); watchFiles\(\); \}\);\r?\n\}\)\(\);/, '  globalThis.__reviewTest = {state, render, playAround, playAB, stopPlayer, select, stage, save, rowElement, attachPeaks, openImport, inspectImports, startImport, refreshJobs, renderImport};\n  load().then(() => { refreshJobs(); watchFiles(); });\n})();');
const context = {console, document:documentMock, addEventListener(){}, URLSearchParams: class { constructor(){ } get(){return 'token';} }, location:{search:'',hash:''}, fetch:fetchMock, Audio:AudioMock, setTimeout:fakeSetTimeout, clearTimeout:fakeClearTimeout, alert(){}, Promise, encodeURIComponent, Map, Set, Number, String, Math, JSON, Object, Error, RegExp};
context.globalThis=context;
vm.runInNewContext(script, context, {filename:'review_page/index.html'});

(async () => {
  await flush();
  const test=context.__reviewTest;
  assert.ok(test, 'test hooks did not initialize');
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
  assert.ok(!/class="ab-play"[^>]*disabled/.test(rendered), 'A/B enables after selecting a candidate');

  // ---- import panel ----
  await test.refreshJobs(); await flush();
  assert.equal(test.state.importEnabled, true, 'import enabled from /api/jobs');
  test.render();
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
  documentMock.listeners.get('keydown')({key:'Escape', target:{}, preventDefault(){}});
  assert.equal(documentMock.getElementById('import').hidden, true, 'Escape closes the panel');
  console.log('review page import tests: PASS');
  console.log('review page playback tests: PASS');
})().catch(error => { console.error(error.stack || error); process.exitCode=1; });
