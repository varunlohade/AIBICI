// Render style frames for a design direction.
//   node design/render_styleframes.js design/frames/<slug>/frames.js [A,B,C] [t]
// frames.js must define:
//   window.STYLEFRAMES = { fonts: [{family:'Young Serif', url:'/assets/YoungSerif.ttf', weight:'400'}],
//                          frames: { A(ctx, t){...}, B(ctx, t){...}, C(ctx, t){...} } }
// t (default 1.0) is a time in seconds for any animated detail. Output: <dir of frames.js>/<key>.png (1920x1080).
// lib/kz.js is loaded too (KZ helpers available) but you are free to write your own drawing helpers.
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.ttf': 'font/ttf', '.otf': 'font/otf', '.png': 'image/png', '.woff2': 'font/woff2' };
(async () => {
  const [file, keysArg = 'A,B,C', tArg = '1'] = process.argv.slice(2);
  const abs = path.resolve(file), rel = '/' + path.relative(ROOT, abs);
  const srv = http.createServer((req, rsp) => {
    const u = decodeURIComponent(req.url.split('?')[0]);
    if (u === '/') { rsp.writeHead(200, { 'Content-Type': 'text/html' }); return rsp.end('<!doctype html><canvas id="c" width="1920" height="1080"></canvas>'); }
    const p = path.join(ROOT, u);
    if (!p.startsWith(ROOT) || !fs.existsSync(p)) { rsp.writeHead(404); return rsp.end(); }
    rsp.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream' }); fs.createReadStream(p).pipe(rsp);
  }).listen(0);
  await new Promise(r => srv.on('listening', r));
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('pageerror', e => console.error('PAGE ERROR:', e.message));
  page.on('console', m => { if (m.type() === 'error') console.error('CONSOLE:', m.text()); });
  await page.goto(`http://127.0.0.1:${srv.address().port}/`);
  await page.addScriptTag({ url: '/lib/kz.js' });
  await page.addScriptTag({ url: rel + '?v=' + Date.now() });
  await page.evaluate(async () => {
    for (const f of (window.STYLEFRAMES.fonts || [])) {
      const ff = new FontFace(f.family, `url(${f.url})`, { weight: f.weight || '100 900', style: f.style || 'normal' });
      document.fonts.add(await ff.load());
    }
  });
  for (const k of keysArg.split(',')) {
    const data = await page.evaluate(([k, t]) => {
      const c = document.getElementById('c'), ctx = c.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, 1920, 1080);
      window.STYLEFRAMES.frames[k](ctx, t); return c.toDataURL('image/png');
    }, [k, Number(tArg)]);
    const out = path.join(path.dirname(abs), `${k}.png`);
    fs.writeFileSync(out, Buffer.from(data.split(',')[1], 'base64')); console.log(out);
  }
  await browser.close(); srv.close();
})();
