'use strict';
// Draft-autosave scenarios for scripts/review_page/index.html: real page script, getElementById-only DOM shim,
// several "tabs" sharing one localStorage Map. Run: node scripts/test_review_drafts.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const page = fs.readFileSync(path.join(__dirname, 'review_page', 'index.html'), 'utf8').replace(/\r\n/g, '\n');
let source = page.split('<script>', 2)[1].split('</script>')[0];
const hook = '  load().then(() => { refreshJobs(); watchFiles(); });\n})();';
assert.ok(source.includes(hook), 'page bootstrap hook moved');
source = source.replace(hook, '  globalThis.__t = {state, stage, load, refreshSong, flushDraft, watchFiles, settleExportGeneration, markDone, get seen() { return seenGeneration; }, get pending() { return refreshPending; }};\n  load().then(() => { refreshJobs(); watchFiles(); });\n})();');

const KEY = 'lrc-review-drafts';
function makeStorage(opts = {}) {
  const map = new Map();
  return {map, api: {getItem: k => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { if (opts.failWrites && k === KEY) throw new Error('QuotaExceededError'); map.set(k, String(v)); },
    removeItem: k => map.delete(k)}};
}
function element(id = '') {
  return {id, children: [], dataset: {}, className: '', textContent: '', disabled: false, hidden: id === 'draft-restored' || id === 'draft-stale', listeners: {},
    addEventListener(type, fn) { this.listeners[type] = () => fn({type, currentTarget: this, target: this, preventDefault() {}, stopPropagation() {}}); },
    appendChild(child) { this.children.push(child); }, querySelectorAll() { return []; }, blur() {},
    set innerHTML(value) { this._html = value; this.children = []; }, get innerHTML() { return this._html || ''; }};
}
const settle = async () => { for (let i = 0; i < 12; i++) await new Promise(resolve => setImmediate(resolve)); };
const response = (ok, status, data) => ({ok, status, statusText: 'x', json: async () => data});

// songs: [{name, report, sha, rows:[{entry,current_cs,candidates}], detailFails, readOnly}]
async function boot(storage, songs, opts = {}) {
  const elements = new Map(), docListeners = {}, winListeners = {}, timers = new Map(), saves = [];
  let timerId = 0;
  const doc = {getElementById(id) { if (!elements.has(id)) elements.set(id, element(id)); return elements.get(id); }, createElement() { return element(); },
    addEventListener(type, fn) { docListeners[type] = fn; }, visibilityState: 'visible'};
  const rowOf = (song, r) => Object.assign({review: 'REVIEW', auto_cs: r.current_cs, lyrics: ['l' + r.entry], candidates: [{cs: r.current_cs + 100, sources: ['ctc'], diagnostic_only: false}], severity_score: 1}, r);
  const context = {console, document: doc, localStorage: storage.api, location: {search: '?t=test', hash: '', origin: 'http://test'}, addEventListener(type, fn) { winListeners[type] = fn; },
    URLSearchParams, encodeURIComponent, Map, Set, Number, String, Math, JSON, Object, Error, RegExp, Date, Promise, Array,
    setTimeout(fn, ms) { const id = ++timerId; timers.set(id, {fn, ms}); return id; }, clearTimeout(id) { timers.delete(id); }, setInterval() { return 0; }, clearInterval() {},
    fetch: async (url, options = {}) => {
      if (url === '/api/songs') return response(true, 200, songs.map((s, index) => ({index, song: s.name, report: s.report, read_only: !!s.readOnly, has_flac: false, has_stem: false})));
      let m = /^\/api\/song\/(\d+)\/save$/.exec(url);
      if (m) {
        const s = songs[Number(m[1])], body = JSON.parse(options.body);
        saves.push({index: Number(m[1]), body});
        if (opts.save409) {
          if (opts.save409Changes) { s.sha = s.sha + '-ext'; s.rows.forEach(r => { r.current_cs += 1000; }); } // someone else changed the file
          return response(false, 409, {error: 'stale base_sha256'});
        }
        s.sha = s.sha + '+'; // the file changed, as after a real save
        body.edits.forEach(e => { s.rows.find(r => r.entry === e.entry).current_cs = e.cs; });
        return response(true, 200, Object.assign({base_sha256: s.sha, server_read_only: false, has_flac: true, has_stem: true, rows: s.rows.map(r => rowOf(s, r))}, opts.adjusted ? {adjusted_cs: opts.adjusted} : {}));
      }
      m = /^\/api\/song\/(\d+)$/.exec(url);
      if (m && !options.method) {
        const s = songs[Number(m[1])];
        if (s.detailFails) return response(false, 503, {error: 'starting'});
        return response(true, 200, {base_sha256: s.sha, server_read_only: false, has_flac: true, has_stem: true, rows: s.rows.map(r => rowOf(s, r))});
      }
      if (url === '/api/jobs') return response(true, 200, {import_enabled: false, jobs: []});
      if (url === '/api/version') return response(true, 200, {generation: opts.gen ? opts.gen.value : 0, port_notice: opts.portNotice || ''});
      throw Error('unexpected fetch ' + url);
    }};
  context.globalThis = context;
  vm.runInNewContext(source, context);
  await settle();
  const tab = {t: context.__t, el: id => doc.getElementById(id), docListeners, winListeners, saves, context,
    flush() { for (const [id, timer] of [...timers]) { if (timer.ms === 750) { timers.delete(id); timer.fn(); } } },
    pendingFlush: () => [...timers.values()].some(timer => timer.ms === 750),
    rowOf: (index, entry) => tab.t.state.rows.find(item => item.song.index === index && item.row.entry === entry),
    edit(index, entry, cs, nudge = 50) { tab.t.stage(tab.rowOf(index, entry), cs, 'manual', nudge); }};
  return tab;
}
const mk = (name, report, sha = 's', rows) => ({name, report, sha, rows: rows || [{entry: 1, current_cs: 100}, {entry: 2, current_cs: 500}]});
const drafts = storage => JSON.parse(storage.map.get(KEY) || '{}');
const tests = [];
const test = (name, fn) => tests.push([name, fn]);

test('first save of a freshly loaded song clears its draft and raises no stale banner (claude-2 F2, gemini F1)', async () => {
  const st = makeStorage(), songs = [mk('S', '/r/S.align-report.json')];
  const p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  assert.ok(drafts(st)['/r/S.align-report.json']);
  p.el('save').listeners.click(); await settle();
  assert.equal(p.saves.length, 1);
  assert.deepEqual(drafts(st), {});
  assert.equal(p.el('draft-stale').hidden, true);
  p.edit(0, 2, 560, 60); p.flush();
  assert.equal(drafts(st)['/r/S.align-report.json'].edits.length, 1, 'later edits on the same song are drafted again');
});

test('a 409 save keeps the draft and shows no stale banner for the same session (F2)', async () => {
  const st = makeStorage(), songs = [mk('S', '/r/S.json')];
  const p = await boot(st, songs, {save409: true});
  p.edit(0, 1, 150); p.flush();
  p.el('save').listeners.click(); await settle();
  assert.equal(drafts(st)['/r/S.json'].edits.length, 1);
  assert.equal(p.el('draft-stale').hidden, true);
});

test('a failed detail fetch or read-only load never deletes the draft; it is restored on the next load (claude-2 F3, gemini F2)', async () => {
  const st = makeStorage(), songs = [mk('A', '/r/A.json')];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  p.winListeners.beforeunload({preventDefault() {}}); // the first tab is closed
  songs[0].detailFails = true;
  p = await boot(st, songs);
  p.winListeners.beforeunload({preventDefault() {}}); p.flush();
  p.context.document.visibilityState = 'hidden'; p.docListeners.visibilitychange();
  assert.equal(drafts(st)['/r/A.json'].edits[0].cs, 150);
  songs[0].detailFails = false;
  await p.t.load();
  assert.equal(p.t.state.edits.size, 1, 'retried on the next load');
  assert.equal(p.t.state.edits.get('0:1').cs, 150);
});

test('same-named songs get separate drafts (claude-2 F4, gemini F3)', async () => {
  const st = makeStorage(), songs = [mk('Dup', '/cd1/Dup.json', 's1'), mk('Dup', '/cd2/Dup.json', 's2')];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  assert.deepEqual(Object.keys(drafts(st)), ['/cd1/Dup.json']);
  p.edit(1, 2, 560, 60); p.flush();
  assert.deepEqual(Object.keys(drafts(st)).sort(), ['/cd1/Dup.json', '/cd2/Dup.json']);
  p = await boot(st, songs);
  assert.equal(p.t.state.edits.get('0:1').cs, 150);
  assert.equal(p.t.state.edits.get('1:2').cs, 560);
  assert.equal(p.t.state.edits.size, 2);
});

test('a live refresh that renumbers songs keeps edits and drafts with the right song (claude-2 F8)', async () => {
  const st = makeStorage(), songs = [mk('B', '/r/B.json', 'b')];
  const p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  songs.unshift(mk('A', '/r/A.json', 'a', [{entry: 1, current_cs: 500}]));
  await p.t.load(); p.flush();
  assert.equal(p.t.state.edits.get('1:1').cs, 150, 'edit followed song B to its new index');
  assert.equal(p.t.state.edits.has('0:1'), false);
  assert.deepEqual(Object.keys(drafts(st)), ['/r/B.json']);
});

test('two tabs sharing one storage do not delete each other\'s drafts (claude-2 F9, gemini F4)', async () => {
  const st = makeStorage(), songs = [mk('A', '/r/A.json'), mk('B', '/r/B.json')];
  const a = await boot(st, songs), b = await boot(st, songs);
  a.edit(0, 1, 150); a.flush();
  b.edit(1, 2, 560, 60); b.flush();
  assert.deepEqual(Object.keys(drafts(st)).sort(), ['/r/A.json', '/r/B.json']);
  b.winListeners.beforeunload({preventDefault() {}});
  a.context.document.visibilityState = 'hidden'; a.docListeners.visibilitychange();
  assert.deepEqual(Object.keys(drafts(st)).sort(), ['/r/A.json', '/r/B.json']);
});

test('an edit made inside the debounce window survives a reload; undo is not undone (claude-2 F5)', async () => {
  const st = makeStorage(), songs = [mk('S', '/r/S.json')];
  const p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  p.edit(0, 1, 175, 75); // still inside the debounce
  await p.t.load();
  assert.equal(p.t.state.edits.get('0:1').cs, 175);
  assert.equal(p.el('draft-restored').hidden, true, 'nothing was restored');
  p.flush();
  assert.equal(drafts(st)['/r/S.json'].edits[0].cs, 175);
  p.t.state.edits.delete('0:1'); // undo
  await p.t.load();
  assert.equal(p.t.state.edits.size, 0);
});

test('捨棄 removes only what the restore applied (claude-2 F6, gemini F5)', async () => {
  const st = makeStorage(), songs = [mk('A', '/r/A.json'), mk('B', '/r/B.json')];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  p = await boot(st, songs);
  assert.equal(p.t.state.edits.size, 1);
  p.edit(1, 2, 560, 60); p.edit(0, 2, 560, 60); p.flush();
  p.el('draft-restored-discard').listeners.click();
  assert.equal(p.t.state.edits.has('0:1'), false);
  assert.equal(p.t.state.edits.get('1:2').cs, 560);
  assert.equal(p.t.state.edits.get('0:2').cs, 560);
  const d = drafts(st);
  assert.deepEqual(d['/r/B.json'].edits.map(e => e.entry), [2]);
  assert.deepEqual(d['/r/A.json'].edits.map(e => e.entry), [2], 'the new edit on the same song stays drafted');
});

test('entries that cannot be applied stay in storage and are counted (claude-2 F7)', async () => {
  const st = makeStorage(), songs = [mk('E', '/r/E.json', 's')];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.edit(0, 2, 560, 60); p.flush();
  songs[0].rows = [{entry: 1, current_cs: 100}]; // entry 2 is gone
  p = await boot(st, songs);
  assert.equal(p.t.state.edits.size, 1);
  assert.match(p.el('draft-restored-text').textContent, /另有 1 筆無法套用/);
  p.winListeners.beforeunload({preventDefault() {}});
  assert.deepEqual(drafts(st)['/r/E.json'].edits.map(e => e.entry).sort(), [1, 2]);
});

test('仍然套用 rebases nudges to current + nudge and skips picks outside the row (gemini F6)', async () => {
  const st = makeStorage(), songs = [mk('R', '/r/R.json', 'old', [{entry: 1, current_cs: 1000, line_key: '1:a'}, {entry: 2, current_cs: 1500, line_key: '2:b'}])];
  let p = await boot(st, songs);
  p.edit(0, 1, 1010, 10);
  p.t.stage(p.rowOf(0, 2), 1600, 'candidate', 0); // a candidate pick, 1600 is the +100 candidate
  p.flush();
  songs[0].sha = 'new';
  songs[0].rows = [{entry: 1, current_cs: 2000, line_key: '1:a'}, {entry: 2, current_cs: 3000, line_key: '2:b', candidates: [{cs: 3050, sources: ['ctc'], diagnostic_only: false}]}];
  p = await boot(st, songs);
  assert.equal(p.el('draft-stale').hidden, false);
  p.el('draft-stale-apply').listeners.click();
  assert.equal(p.t.state.edits.get('0:1').cs, 2010);
  assert.equal(p.t.state.edits.get('0:1').base_cs, 2000);
  assert.equal(p.t.state.edits.has('0:2'), false, 'a pick outside the row span is skipped');
  assert.match(p.t.state.notice, /已套用 1 筆.*1 筆已不適用/);
});

test('a null or junk draft entry does not break writing (gemini F7)', async () => {
  const st = makeStorage(), songs = [mk('S', '/r/S.json')];
  st.map.set(KEY, JSON.stringify({junk: null, other: {saved_at: 5, edits: []}}));
  const p = await boot(st, songs);
  p.edit(0, 1, 150);
  assert.doesNotThrow(() => p.flush());
  assert.equal(drafts(st)['/r/S.json'].edits.length, 1);
});

test('a storage write failure shows a visible notice once (claude-2 F10)', async () => {
  const st = makeStorage({failWrites: true}), songs = [mk('S', '/r/S.json')];
  const p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush();
  assert.equal(p.t.state.notice, '暫存失敗：瀏覽器儲存空間不可用');
  p.t.state.notice = '';
  p.edit(0, 2, 560, 60); p.flush();
  assert.equal(p.t.state.notice, '', 'only once');
});

test('flush happens before a song switch and before a load (item 9)', async () => {
  const st = makeStorage(), songs = [mk('A', '/r/A.json'), mk('B', '/r/B.json')];
  const p = await boot(st, songs);
  p.edit(0, 1, 150);
  assert.ok(p.pendingFlush() && !st.map.has(KEY));
  await p.t.load();
  assert.ok(drafts(st)['/r/A.json'], 'load() flushes first');
});


const close = tab => tab.winListeners.beforeunload({preventDefault() {}});
const SONG = '/r/S.json';

test('stale draft pending: new edits are still drafted beside it under the current sha and come back (claude-2 F6 / X2)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 'old')];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.flush(); close(p);
  songs[0].sha = 'new'; songs[0].rows = [{entry: 1, current_cs: 1000}, {entry: 2, current_cs: 500}];
  p = await boot(st, songs);
  assert.equal(p.el('draft-stale').hidden, false);
  p.edit(0, 2, 560, 60); p.flush(); close(p);
  const d = drafts(st)[SONG];
  assert.equal(d.base_sha256, 'old'); assert.deepEqual(d.edits.map(e => e.entry), [1]);
  assert.equal(d.more.length, 1); assert.equal(d.more[0].base_sha256, 'new'); assert.deepEqual(d.more[0].edits.map(e => e.entry), [2]);
  p = await boot(st, songs);
  assert.equal(p.t.state.edits.get('0:2').cs, 560, 'the new edit is restored');
  assert.equal(p.t.state.edits.has('0:1'), false, 'the stale one waits for an answer');
  assert.equal(p.el('draft-stale').hidden, false);
  p.el('draft-stale-discard').listeners.click();
  const after = drafts(st)[SONG];
  assert.equal(after.base_sha256, 'new'); assert.deepEqual(after.edits.map(e => e.entry), [2]);
});

test('409 keeps the old base sha: after the file changed, a reload offers 仍然套用 (gemini F2)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', [{entry: 1, current_cs: 100, line_key: '1:a'}, {entry: 2, current_cs: 500, line_key: '2:b'}])];
  let p = await boot(st, songs, {save409: true, save409Changes: true});
  p.edit(0, 1, 150); p.flush();
  p.el('save').listeners.click(); await settle();
  assert.equal(drafts(st)[SONG].base_sha256, 's', 'draft still names the sha its base_cs came from');
  assert.equal(p.el('draft-stale').hidden, true, 'same session: no stale banner');
  close(p);
  p = await boot(st, songs);
  assert.equal(p.el('draft-stale').hidden, false);
  assert.equal(p.t.state.edits.size, 0);
  p.el('draft-stale-apply').listeners.click();
  assert.equal(p.t.state.edits.get('0:1').cs, 1150); // the file's current 1100 + nudge 50
});

test('捨棄 leaves a row that was re-edited after the restore (gemini F3, claude-2 F4 / X3)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.edit(0, 2, 560, 60); p.flush(); close(p);
  p = await boot(st, songs);
  assert.equal(p.t.state.edits.size, 2);
  p.edit(0, 1, 175, 75); p.flush();
  p.el('draft-restored-discard').listeners.click();
  assert.equal(p.t.state.edits.get('0:1').cs, 175, 're-edited row stays in memory');
  assert.equal(p.t.state.edits.has('0:2'), false, 'untouched restored row is discarded');
  assert.deepEqual(drafts(st)[SONG].edits.map(e => [e.entry, e.cs]), [[1, 175]]);
});

test('two tabs editing different rows of one song keep both; undo-all removes only this tab\'s entries (X4)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a = await boot(st, songs), b = await boot(st, songs);
  a.edit(0, 1, 150); a.flush();
  b.edit(0, 2, 560, 60); b.flush();
  assert.deepEqual(drafts(st)[SONG].edits.map(e => e.entry), [1, 2]);
  b.t.state.edits.delete('0:2'); await b.t.load();
  assert.deepEqual(drafts(st)[SONG].edits.map(e => e.entry), [1], 'a\'s entry survives b undoing everything');
  a.t.state.edits.delete('0:1'); await a.t.load();
  assert.deepEqual(drafts(st), {});
});

test('a live refresh does not take over a draft another live tab wrote; a fresh load or a closed writer does (claude-2 F2 / X1)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a = await boot(st, songs), b = await boot(st, songs);
  a.edit(0, 1, 150); a.flush();
  await b.t.load();
  assert.equal(b.t.state.edits.size, 0, 'b leaves a\'s draft alone');
  const c = await boot(st, songs);
  assert.equal(c.t.state.edits.get('0:1').cs, 150, 'a fresh page load restores it');
  close(a);
  const e = await boot(st, songs);
  assert.equal(e.t.state.edits.size, 1);
  // a's tab entry is gone and c's is there; once c closes too, b's refresh may take the draft over
  close(c); close(e);
  await b.t.load();
  assert.equal(b.t.state.edits.get('0:1').cs, 150);
});

test('double nudge: another tab saves the edit, the refreshing tab never re-drafts it (claude-2 F3)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a = await boot(st, songs), b = await boot(st, songs);
  a.edit(0, 1, 150, 50); a.flush();
  await b.t.load();
  a.el('save').listeners.click(); await settle();
  assert.deepEqual(drafts(st), {});
  await b.t.load(); b.flush(); close(b);
  assert.deepEqual(drafts(st), {}, 'nothing resurrected');
  const c = await boot(st, songs);
  assert.equal(c.el('draft-stale').hidden, true);
  assert.equal(c.t.state.edits.size, 0);
});

test('saving a restored song hides the restored banner and clears its leftovers (gemini F1)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  let p = await boot(st, songs);
  p.edit(0, 1, 150); p.edit(0, 2, 560, 60); p.flush(); close(p);
  songs[0].rows = [{entry: 1, current_cs: 100}, {entry: 2, current_cs: 500}, {entry: 3, current_cs: 900}];
  const d = drafts(st); d[SONG].edits.push({entry: 3, cs: 950, source: 'manual', nudge_cs: 50, base_cs: 111}); // base differs: cannot be applied
  st.map.set(KEY, JSON.stringify(d));
  p = await boot(st, songs);
  assert.match(p.el('draft-restored-text').textContent, /已還原 2 筆.*另有 1 筆無法套用/);
  p.el('save').listeners.click(); await settle();
  assert.equal(p.el('draft-restored').hidden, true);
  assert.deepEqual(drafts(st), {});
});

test('a page notice carries the port-fallback line (gemini F6)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const p = await boot(st, songs, {portNotice: '58400 已被佔用（可能有另一個 Review LRC 還開著），改用 58401；此埠的暫存草稿不會跨埠號'});
  assert.match(p.t.state.portNotice, /58400 已被佔用.*改用 58401/);
  assert.equal(p.el('port-notice').hidden, false);
  assert.match(p.el('port-notice-text').textContent, /改用 58401/);
  p.t.state.activeSong = 0; p.t.state.notice = ''; p.el('save').listeners.click(); await settle();   // song click / save / scan progress only touch state.notice
  assert.match(p.t.state.portNotice, /改用 58401/);
  p.el('port-notice-close').listeners.click();
  assert.equal(p.el('port-notice').hidden, true);
});

test('AS4 F1: a save drops every drafted entry it contained whatever the group sha; a fresh load shows no stale banner (claude-2 F1)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const p = await boot(st, songs);
  p.edit(0, 1, 150, 50); p.flush();
  songs[0].sha = 's2'; songs[0].rows[1].current_cs = 520;   // another tab saved entry 2: sha changes, entry 1 untouched
  await p.t.load(); await settle();
  assert.equal(p.t.state.edits.size, 1);
  p.el('save').listeners.click(); await settle(); p.flush();
  assert.equal(p.saves.length, 1);
  assert.deepEqual(drafts(st), {}, 'the saved edit is gone from every group');
  p.winListeners.beforeunload({preventDefault() {}});
  const q = await boot(st, songs);
  assert.equal(q.el('draft-stale').hidden, true);
  assert.equal(q.t.state.edits.size, 0);
});

test('AS4 F1: a drafted entry whose cs already equals the row is never re-applied; other groups\' remnants stay', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's2', [{entry: 1, current_cs: 150, line_key: '1:a'}, {entry: 2, current_cs: 500, line_key: '2:b'}])];
  st.map.set(KEY, JSON.stringify({[SONG]: {saved_at: Date.now(), tabs: {}, base_sha256: 's', edits: [{entry: 1, cs: 150, source: 'manual', nudge_cs: 50, base_cs: 100, line_key: '1:a', edited_at: 1}], more: []}}));
  const q = await boot(st, songs);
  assert.equal(q.el('draft-stale').hidden, true, 'nothing live to offer');
  assert.equal(q.t.state.edits.size, 0);
  assert.deepEqual(drafts(st), {}, 'the obsolete group is dropped');
  // a stale group with one saved and one pending entry offers only the pending one
  st.map.set(KEY, JSON.stringify({[SONG]: {saved_at: Date.now(), tabs: {}, base_sha256: 's', edits: [{entry: 1, cs: 150, source: 'manual', nudge_cs: 50, base_cs: 100, line_key: '1:a', edited_at: 1}, {entry: 2, cs: 540, source: 'manual', nudge_cs: 40, base_cs: 500, line_key: '2:b', edited_at: 1}], more: []}}));
  const r = await boot(st, songs);
  assert.match(r.el('draft-stale-text').textContent, /有 1 筆/);
  r.el('draft-stale-apply').listeners.click(); await settle();
  assert.deepEqual([...r.t.state.edits.values()].map(e => [e.entry, e.cs]), [[2, 540]]);
});

test('AS4 F2: no staging on the exporting song; edits made before refreshSong are drafted and the notice says so (claude-2 F2)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG), mk('T', '/r/T.json')];
  const p = await boot(st, songs);
  p.t.state.exporting = true; p.t.state.exportIndex = 0;
  p.edit(0, 1, 150, 50);
  assert.equal(p.t.state.edits.size, 0);
  assert.match(p.t.state.notice, /匯出中，請等匯出完成再修改/);
  p.edit(1, 1, 150, 50);   // another song stays editable
  assert.equal(p.t.state.edits.size, 1);
  p.t.state.exporting = false; p.t.state.exportIndex = null;
  p.edit(0, 1, 150, 50);   // an edit that got in anyway (made just before the export began)
  assert.equal(p.pendingFlush(), true);
  songs[0].readOnly = true;   // the export archived the LRC
  await p.t.refreshSong(0); await settle();
  assert.equal([...p.t.state.edits.keys()].some(key => key.startsWith('0:')), false);
  assert.match(p.t.state.notice, /1 筆修改因匯出歸檔無法套用，已存成草稿/);
  assert.deepEqual(drafts(st)[SONG].edits.map(e => [e.entry, e.cs]), [[1, 150]]);
});

test('AS4 F3: a second tab\'s beat never overwrites a newer stored value or resurrects an undo (claude-2 F3)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a = await boot(st, songs);
  a.edit(0, 1, 150, 50); a.flush();
  const b = await boot(st, songs);
  assert.equal(b.t.state.edits.size, 1);
  a.edit(0, 1, 180, 80); a.flush();
  b.context.document.visibilityState = 'hidden'; b.docListeners.visibilitychange();
  assert.deepEqual(drafts(st)[SONG].edits.map(e => [e.entry, e.cs]), [[1, 180]], 'A\'s newer value stays');
  a.t.stage(a.rowOf(0, 1), 100, 'manual', 0); a.flush();
  assert.equal(drafts(st)[SONG], undefined, 'A\'s undo removed the entry');
  b.docListeners.visibilitychange();
  assert.equal(drafts(st)[SONG], undefined, 'B\'s beat does not bring it back');
});

test('AS4 F3: a stored value with a newer edited_at wins over this tab\'s stale write', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const p = await boot(st, songs);
  p.edit(0, 1, 150, 50);
  const d = drafts(st); d[SONG] = {saved_at: Date.now(), tabs: {other: Date.now()}, base_sha256: 's', edits: [{entry: 1, cs: 190, source: 'manual', nudge_cs: 90, base_cs: 100, edited_at: Date.now() + 5000}], more: []};
  st.map.set(KEY, JSON.stringify(d));
  p.flush();
  assert.equal(drafts(st)[SONG].edits[0].cs, 190);
});

test('AS4 F4: refreshSong after the server renumbered songs falls back to load() and overwrites no other song (gemini F3)', async () => {
  const st = makeStorage(), songs = [mk('A', '/r/A.json'), mk('B', '/r/B.json')];
  const p = await boot(st, songs);
  p.edit(1, 1, 150, 50);
  songs.unshift(mk('Z', '/r/Z.json'));   // a report appeared: B is now index 2
  await p.t.refreshSong(1); await settle();
  assert.deepEqual(p.t.state.songs.map(song => song.report), ['/r/Z.json', '/r/A.json', '/r/B.json']);
  assert.deepEqual([...p.t.state.edits.keys()], ['2:1'], 'the edit followed B');
  assert.equal(p.t.state.songs[1].song, 'A');
});

test('AS4 F9: the export\'s own generation bump does not queue a full reload', async () => {
  const gen = {value: 0}, st = makeStorage(), songs = [mk('S', SONG), mk('T', '/r/T.json')];
  const p = await boot(st, songs, {gen});
  p.t.state.exporting = true; gen.value = 1;
  await p.t.watchFiles();
  assert.equal(p.t.seen, 0, 'a bump during the export is judged later');
  p.t.state.exporting = false; p.t.settleExportGeneration(1);
  assert.equal(p.t.pending, false);
  assert.equal(p.t.seen, 1);
});

test('AS5 M3: an outside change during an export is refreshed after it (gemini F2 = claude-2 F5)', async () => {
  const gen = {value: 0}, st = makeStorage(), songs = [mk('S', SONG), mk('T', '/r/T.json')];
  const p = await boot(st, songs, {gen});
  p.t.state.exporting = true; p.t.state.exportIndex = 0;
  songs[1].rows[0].current_cs = 999; songs[1].sha = 't-new'; gen.value = 1;   // T changed on disk meanwhile
  await p.t.watchFiles();
  assert.equal(p.t.seen, 0, 'seenGeneration did not advance past the deferred bump');
  p.t.state.exporting = false; p.t.state.exportIndex = null;
  p.t.settleExportGeneration(2);   // the export's own bump came after the outside one
  assert.equal(p.t.pending, true);
  await p.t.watchFiles(); await settle();
  assert.equal(p.rowOf(1, 1).row.current_cs, 999, 'T was refreshed');
  assert.equal(p.t.pending, false);
});

test('AS5 M3: a failed export still applies a bump seen during it', async () => {
  const gen = {value: 0}, st = makeStorage(), songs = [mk('S', SONG)];
  const p = await boot(st, songs, {gen});
  p.t.state.exporting = true; gen.value = 3;
  await p.t.watchFiles();
  p.t.state.exporting = false; p.t.settleExportGeneration(null);
  assert.equal(p.t.pending, true);
});

const lk = (key, entry, cs) => ({entry, current_cs: cs, line_key: key});
const lines5 = () => [lk('3:one', 1, 100), lk('4:two', 2, 200), lk('6:three', 3, 320), lk('7:four', 4, 300), lk('8:five', 5, 6200)];
const lines4Merged = () => [lk('3:one', 1, 100), lk('4:two', 2, 200), lk('6:three+four', 3, 300), lk('8:five', 4, 6200)];

test('AS5 B1: a save drops only its own entries; another tab\'s unsaved edit survives and shows after a reload (claude-2 F1 = gemini F1)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', [{entry: 1, current_cs: 100, line_key: '1:a'}, {entry: 2, current_cs: 500, line_key: '2:b'}, {entry: 3, current_cs: 900, line_key: '3:c'}])];
  const a = await boot(st, songs), b = await boot(st, songs);
  b.edit(0, 3, 950, 50); b.flush();
  a.edit(0, 1, 150, 50); a.flush();
  a.el('save').listeners.click(); await settle(); a.flush();
  assert.deepEqual(a.saves.map(x => x.body.edits.map(e => e.entry)), [[1]]);
  assert.deepEqual(drafts(st)[SONG].edits.map(e => [e.entry, e.cs]), [[3, 950]], 'only entry 1 left the draft');
  b.flush(); b.context.document.visibilityState = 'hidden'; b.docListeners.visibilitychange(); close(b);
  assert.deepEqual(drafts(st)[SONG].edits.map(e => [e.entry, e.cs]), [[3, 950]], 'B\'s beat and unload keep it');
  const b2 = await boot(st, songs);
  assert.equal(b2.el('draft-stale').hidden, false, 'B\'s edit is offered after the reload');
  b2.el('draft-stale-apply').listeners.click(); await settle();
  assert.deepEqual([...b2.t.state.edits.values()].map(e => [e.entry, e.cs]), [[3, 950]]);
});

test('AS5 B1: tab B closing after tab A saved leaves B\'s edit for the next tab (test_draft_drop)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 'sha1')];
  const a = await boot(st, songs), b = await boot(st, songs);
  b.edit(0, 2, 550, 50); b.flush();
  a.edit(0, 1, 150, 50); a.flush();
  a.el('save').listeners.click(); await settle(); a.flush();
  close(b);
  const c = await boot(st, songs);
  assert.equal(c.el('draft-stale').hidden, false);
  assert.match(c.el('draft-stale-text').textContent, /有 1 筆/);
});

test('AS5 B1: a beat re-writes an entry that went missing from the draft, but not one a save settled (test_beat_wont_save)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 'sha1')];
  const a = await boot(st, songs), b = await boot(st, songs);
  b.edit(0, 2, 550, 50); b.flush();
  const gone = drafts(st); delete gone[SONG]; st.map.set(KEY, JSON.stringify(gone));   // some other writer lost the entry
  b.t.flushDraft();
  assert.deepEqual(drafts(st)[SONG].edits.map(e => e.entry), [2], 'restored on the next beat');
  a.edit(0, 2, 560, 60); a.flush();   // A saves the same entry: B holds a ghost of it and must not bring it back
  a.el('save').listeners.click(); await settle(); a.flush();
  assert.equal(drafts(st)[SONG], undefined);
  b.t.flushDraft();
  assert.equal(drafts(st)[SONG], undefined, 'a saved entry is not resurrected');
});

test('AS5 B1: an entry stored with a newer edit elsewhere is left out of the save and named', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a = await boot(st, songs);
  a.edit(0, 1, 150, 50); a.edit(0, 2, 560, 60);
  const d = {}; d[SONG] = {saved_at: Date.now(), tabs: {other: Date.now()}, base_sha256: 's', edits: [{entry: 1, cs: 190, source: 'manual', nudge_cs: 90, base_cs: 100, edited_at: Date.now() + 5000}], more: []};
  st.map.set(KEY, JSON.stringify(d));
  a.el('save').listeners.click(); await settle();
  assert.deepEqual(a.saves.map(x => x.body.edits.map(e => e.entry)), [[2]]);
  assert.match(a.t.state.notice, /第 1 句在另一個分頁有較新的修改，未儲存/);
  assert.equal(a.t.state.edits.has('0:1'), true, 'still staged here, not written');
});

test('AS5 B2c: a stale draft whose line changed is refused, never applied to the line now holding its entry number (claude-2 F3)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', lines5())];
  const b = await boot(st, songs);
  b.edit(0, 4, 350, 50); b.flush(); close(b);
  assert.equal(drafts(st)[SONG].edits[0].line_key, '7:four');
  songs[0].sha = 's2'; songs[0].rows = lines4Merged();   // another tab's save merged three+four
  const c = await boot(st, songs);
  assert.equal(c.el('draft-stale').hidden, false);
  c.el('draft-stale-apply').listeners.click(); await settle();
  assert.equal(c.t.state.edits.size, 0, 'nothing staged on "five"');
  assert.match(c.t.state.notice, /無法套用：對應的句子已變動/);
  c.el('save').listeners.click(); await settle();
  assert.equal(c.saves.length, 0);
});

test('AS5 B2c: a same-sha draft whose line key no longer matches is listed as unapplied', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', lines5())];
  const b = await boot(st, songs);
  b.edit(0, 4, 350, 50); b.flush(); close(b);
  songs[0].rows = lines5().map(r => r.entry === 4 ? Object.assign(r, {line_key: '7:other'}) : r);
  const c = await boot(st, songs);
  assert.equal(c.t.state.edits.size, 0);
  assert.match(c.el('draft-restored-text').textContent, /無法套用：對應的句子已變動/);
});

test('AS5 B2c: carried edits and done marks follow their line when entries renumber on a live refresh', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', lines5())];
  const p = await boot(st, songs);
  p.edit(0, 4, 350, 50);
  p.t.markDone(p.rowOf(0, 5));
  songs[0].rows = [lk('3:one', 1, 100), lk('4:two', 2, 200), lk('7:four', 3, 300), lk('8:five', 4, 6200)];   // "three" was removed outside
  await p.t.load(); await settle();
  assert.deepEqual([...p.t.state.edits.keys()], ['0:3'], 'the edit followed "four"');
  assert.equal(p.t.state.edits.get('0:3').entry, 3);
  assert.deepEqual([...p.t.state.done], ['0:4'], 'the done mark followed "five"');
  p.el('save').listeners.click(); await settle();
  assert.deepEqual(p.saves[0].body.edits.map(e => [e.entry, e.line_key]), [[3, '7:four']]);
});

test('AS5 B2c: an edit whose line vanished is dropped, not moved onto a neighbour (repro_renumber)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', lines5())];
  const p = await boot(st, songs);
  p.edit(0, 4, 350, 50);
  songs[0].rows = lines4Merged();
  await p.t.load(); await settle();
  assert.equal(p.t.state.edits.size, 0);
  assert.match(p.t.state.error, /無法套用：對應的句子已變動/);
});

test('AS5 B2a: the page announces a save that was offset to avoid merging with the neighbour', async () => {
  const st = makeStorage(), songs = [mk('S', SONG, 's', lines5())];
  const p = await boot(st, songs, {adjusted: [{entry: 3, requested_cs: 300, cs: 299}]});
  p.edit(0, 3, 300, -20);
  p.el('save').listeners.click(); await settle();
  assert.match(p.t.state.notice, /第 3 句已偏移 0\.01 秒以免與相鄰句合併/);
});

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const entriesOf = (st, report = SONG) => ((drafts(st)[report] || {}).edits || []).map(e => [e.entry, e.cs]);

test('AS6 F3: a keyless legacy draft is never re-aimed onto another line by 仍然套用 (repro R2)', async () => {
  const st = makeStorage();
  const d = {}; d[SONG] = {saved_at: Date.now(), tabs: {}, base_sha256: 'old', edits: [{entry: 4, cs: 350, source: 'nudge', nudge_cs: 50, base_cs: 300, edited_at: Date.now()}], more: []};
  st.map.set(KEY, JSON.stringify(d));
  const p = await boot(st, [mk('S', SONG, 's2', [lk('3:one', 1, 100), lk('4:two', 2, 200), lk('6:three+four', 3, 300), lk('8:five', 4, 6200)])]);
  assert.equal(p.el('draft-stale').hidden, false);
  p.el('draft-stale-apply').listeners.click(); await settle();
  assert.equal(p.t.state.edits.size, 0, 'nothing staged');
  assert.match(p.t.state.notice, /1 筆無法套用：對應的句子已變動/);
  assert.match(p.el('draft-stale-text').textContent, /舊版草稿 1 筆無法對應，請重聽該句/);
  assert.equal(p.el('draft-stale').hidden, false);
  assert.deepEqual(entriesOf(st), [[4, 350]], 'the draft stays stored');
  p.el('save').listeners.click(); await settle();
  assert.equal(p.saves.length, 0, 'no save body');
});

test('AS6 F1: an undo drops only the record this tab wrote (repro R1)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a = await boot(st, songs);
  a.edit(0, 1, 150, 50); a.flush();
  const b = await boot(st, songs);
  assert.deepEqual([...b.t.state.edits.values()].map(e => [e.entry, e.cs]), [[1, 150]]);
  await wait(5);
  a.edit(0, 1, 180, 80); a.flush();
  await wait(5);
  b.t.state.edits.delete('0:1'); b.t.flushDraft();
  assert.deepEqual(entriesOf(st), [[1, 180]], 'A\'s newer value survives B\'s undo');
  a.t.flushDraft();
  assert.deepEqual(entriesOf(st), [[1, 180]], 'and A\'s beat keeps it');
  assert.equal(JSON.stringify(JSON.parse(st.map.get('lrc-review-draft-settled') || '{}')), '{}', 'nothing was settled');
});

test('AS6 F2: 捨棄 settles only what it dropped, not the re-edited row it kept (repro R6)', async () => {
  const st = makeStorage(), songs = [mk('S', SONG)];
  const a0 = await boot(st, songs);
  a0.edit(0, 1, 150, 50); a0.edit(0, 2, 550, 50); a0.flush(); close(a0);
  const a = await boot(st, songs);
  await wait(5);
  a.edit(0, 2, 580, 80); a.flush();
  await wait(5);
  a.el('draft-restored-discard').listeners.click(); await settle();
  assert.deepEqual([...a.t.state.edits.values()].map(e => [e.entry, e.cs]), [[2, 580]]);
  assert.deepEqual(entriesOf(st), [[2, 580]]);
  const gone = drafts(st); delete gone[SONG]; st.map.set(KEY, JSON.stringify(gone));   // lost to another writer
  a.t.flushDraft();
  assert.deepEqual(entriesOf(st), [[2, 580]], 'rewritten by the beat');
});

test('AS6 F4: saving a line does not remove a stale draft of a different line (repro R4)', async () => {
  const st = makeStorage();
  const d = {}; d[SONG] = {saved_at: Date.now(), tabs: {}, base_sha256: 'old', edits: [{entry: 4, cs: 6300, source: 'manual', nudge_cs: 100, base_cs: 6200, line_key: '8:five', edited_at: Date.now()}], more: []};
  st.map.set(KEY, JSON.stringify(d));
  const p = await boot(st, [mk('S', SONG, 's2', [lk('3:one', 1, 100), lk('4:two', 2, 200), lk('6:three', 3, 300), lk('7:four', 4, 320), lk('8:five', 5, 6200)])]);
  assert.equal(p.el('draft-stale').hidden, false);
  p.edit(0, 4, 400, 80); p.flush();
  p.el('save').listeners.click(); await settle(); p.flush();
  assert.deepEqual(p.saves[0].body.edits.map(e => [e.entry, e.line_key]), [[4, '7:four']]);
  assert.deepEqual(drafts(st)[SONG].edits.map(e => [e.entry, e.line_key]), [[4, '8:five']], 'the other line\'s draft stays');
  assert.equal(p.el('draft-stale').hidden, false, 'banner still shown');
});

test('AS6 F7: an outside bump followed by the export\'s own bump is not swallowed (repro5 / swallow proof)', async () => {
  const gen = {value: 0}, st = makeStorage(), songs = [mk('S', SONG), mk('T', '/r/T.json')];
  const p = await boot(st, songs, {gen});
  p.t.state.exporting = true; p.t.state.exportIndex = 0;
  gen.value = 1; await p.t.watchFiles();   // outside change
  gen.value = 2; await p.t.watchFiles();   // then our own bump
  p.t.state.exporting = false; p.t.state.exportIndex = null;
  p.t.settleExportGeneration(2);
  assert.equal(p.t.pending, true);
});

test('AS6 F7: the export answer\'s other_changes forces a refresh even when no poll saw the outside bump', async () => {
  const gen = {value: 0}, st = makeStorage(), songs = [mk('S', SONG)];
  const p = await boot(st, songs, {gen});
  p.t.state.exporting = true;
  p.t.state.exporting = false; p.t.settleExportGeneration(1, false);
  assert.equal(p.t.pending, false);
  p.t.state.exporting = true;
  p.t.state.exporting = false; p.t.settleExportGeneration(2, true);
  assert.equal(p.t.pending, true);
});

(async () => {
  let failed = 0;
  for (const [name, fn] of tests) {
    try { await fn(); console.log('ok   ' + name); } catch (error) { failed++; console.log('FAIL ' + name + '\n' + error.stack); }
  }
  console.log(tests.length - failed + '/' + tests.length + ' passed');
  process.exitCode = failed ? 1 : 0;
})();
