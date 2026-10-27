// Render scenes to video (or stills) with headless Chromium.
//   node render.js --stills s03_ikeda 0,4.5,12          -> out/stills/s03_ikeda_4.5.png ...
//   node render.js --video s03_ikeda [s04 ...]          -> out/scenes/<id>.mp4 (silent, 30fps)
//   node render.js --video all  [--jobs 4]
//   node render.js --sfx                                 -> out/sfx_events.json
//   node render.js --contact s03_ikeda 12               -> out/stills/<id>_contact.png (grid of N frames)
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path'), { spawn } = require('child_process');
const ROOT = __dirname, OUT = path.join(ROOT, 'out'), FPS = 30;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.ttf': 'font/ttf', '.png': 'image/png', '.jpg': 'image/jpeg' };

function serve() {
  return new Promise(res => {
    const srv = http.createServer((req, rsp) => {
      const p = path.join(ROOT, decodeURIComponent(req.url.split('?')[0]));
      if (!p.startsWith(ROOT) || !fs.existsSync(p)) { rsp.writeHead(404); return rsp.end(); }
      rsp.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' });
      fs.createReadStream(p).pipe(rsp);
    }).listen(0, () => res(srv));
  });
}

async function openPage(browser, port) {
  const page = await browser.newPage({ viewport: { width: 540, height: 960 } });
  page.on('pageerror', e => console.error('PAGE ERROR:', e.message));
  page.on('console', m => { if (m.type() === 'error') console.error('CONSOLE:', m.text()); });
  await page.goto(`http://127.0.0.1:${port}/player.html`);
  const ids = await page.evaluate(() => window.setup());
  for (const f of fs.readdirSync(path.join(ROOT, 'scenes')).filter(f => f.endsWith('.js')).sort())
    await page.addScriptTag({ url: `/scenes/${f}` });
  return { page, ids };
}

const b64 = d => Buffer.from(d.split(',')[1], 'base64');

async function renderVideo(browser, port, id) {
  const { page } = await openPage(browser, port);
  const cues = JSON.parse(fs.readFileSync(path.join(ROOT, 'cues.json')));
  const D = cues[id].duration, n = Math.round(D * FPS);
  fs.mkdirSync(path.join(OUT, 'scenes'), { recursive: true });
  const outFile = path.join(OUT, 'scenes', id + '.mp4');
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(FPS), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-r', String(FPS), outFile], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let i = 0; i < n; i++) {
    const data = await page.evaluate(([id, t]) => { renderFrame(id, t); return grab('image/jpeg', 0.93); }, [id, i / FPS]);
    if (!ff.stdin.write(b64(data))) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 300 === 0) console.log(`${id}: frame ${i}/${n}`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await page.close();
  console.log(`${id}: done ${n} frames in ${((Date.now() - t0) / 1000).toFixed(1)}s -> ${outFile}`);
}

(async () => {
  const [mode, ...args] = process.argv.slice(2);
  const srv = await serve(), port = srv.address().port;
  const browser = await chromium.launch();
  try {
    if (mode === '--stills' || mode === '--contact') {
      const id = args[0];
      const { page } = await openPage(browser, port);
      fs.mkdirSync(path.join(OUT, 'stills'), { recursive: true });
      const cues = JSON.parse(fs.readFileSync(path.join(ROOT, 'cues.json')));
      if (mode === '--stills') {
        for (const t of args[1].split(',').map(Number)) {
          const f = path.join(OUT, 'stills', `${id}_${t}.png`);
          fs.writeFileSync(f, b64(await page.evaluate(([id, t]) => { renderFrame(id, t); return grab('image/png'); }, [id, t])));
          console.log(f);
        }
      } else {
        // contact sheet: N evenly spaced frames in a 4-wide grid at 270x480
        const N = Number(args[1] || 12), D = cues[id].duration;
        const times = Array.from({ length: N }, (_, i) => +(D * (i + .5) / N).toFixed(2));
        const data = await page.evaluate(async ([id, times]) => {
          const cols = 4, rows = Math.ceil(times.length / cols), sheet = document.createElement('canvas');
          sheet.width = cols * 270; sheet.height = rows * 480; const sc = sheet.getContext('2d');
          times.forEach((t, i) => {
            renderFrame(id, t); sc.drawImage(document.getElementById('c'), (i % cols) * 270, Math.floor(i / cols) * 480, 270, 480);
            sc.fillStyle = 'rgba(0,0,0,.6)'; sc.fillRect((i % cols) * 270, Math.floor(i / cols) * 480, 110, 34);
            sc.fillStyle = '#fff'; sc.font = '24px sans-serif'; sc.fillText('t=' + t, (i % cols) * 270 + 8, Math.floor(i / cols) * 480 + 25);
          });
          return sheet.toDataURL('image/png');
        }, [id, times]);
        const f = path.join(OUT, 'stills', `${id}_contact.png`);
        fs.writeFileSync(f, b64(data)); console.log(f, 'times:', times.join(','));
      }
    } else if (mode === '--video') {
      const jobsIdx = args.indexOf('--jobs'); let jobs = 4;
      if (jobsIdx >= 0) { jobs = Number(args[jobsIdx + 1]); args.splice(jobsIdx, 2); }
      let ids = args;
      if (ids[0] === 'all') ids = Object.keys(JSON.parse(fs.readFileSync(path.join(ROOT, 'cues.json'))));
      const queue = [...ids];
      await Promise.all(Array.from({ length: jobs }, async () => { while (queue.length) await renderVideo(browser, port, queue.shift()); }));
    } else if (mode === '--sfx') {
      const { page, ids } = await openPage(browser, port);
      const ev = {};
      for (const id of ids) ev[id] = await page.evaluate(id => sfxFor(id), id);
      fs.mkdirSync(OUT, { recursive: true });
      fs.writeFileSync(path.join(OUT, 'sfx_events.json'), JSON.stringify(ev, null, 1));
      console.log(path.join(OUT, 'sfx_events.json'), Object.fromEntries(Object.entries(ev).map(([k, v]) => [k, v.length])));
    } else console.log('usage: see header');
  } finally { await browser.close(); srv.close(); }
})();
