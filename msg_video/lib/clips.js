// CLIPS: real interview footage for the cold open, as pre-extracted JPEG frames (see prep_clips.py).
// Deterministic: a clip's frame is a pure function of clip time.
//   await CLIPS.load()                      -> called once by player.html setup
//   CLIPS.info(id)                          -> {frames, fps, w, h, speaker, credit, quote} or null
//   CLIPS.draw(ctx, id, tc, x, y, w, h)     -> draws the frame at clip time tc, cover-fit into the box
// Clip time is scene time minus the start of the clip's line in cues.json (its audio starts there).
// Before the clip starts it holds the first frame; after it ends it holds the last.
(function () {
  const CLIPS = { index: {}, imgs: {} };
  CLIPS.load = async () => {
    const r = await fetch('clips/prepared/index.json');
    if (!r.ok) return;
    CLIPS.index = await r.json();
    await Promise.all(Object.entries(CLIPS.index).map(([id, c]) => {
      CLIPS.imgs[id] = [];
      return Promise.all(Array.from({ length: c.frames }, (_, i) => new Promise(res => {
        const im = new Image();
        im.onload = im.onerror = res;
        im.src = `clips/prepared/${id}/f${String(i + 1).padStart(4, '0')}.jpg`;
        CLIPS.imgs[id][i] = im;
      })));
    }));
  };
  CLIPS.info = id => CLIPS.index[id] || null;
  CLIPS.frame = (id, tc) => {
    const c = CLIPS.index[id];
    if (!c) return null;
    const i = Math.max(0, Math.min(c.frames - 1, Math.floor(tc * c.fps)));
    return CLIPS.imgs[id][i];
  };
  CLIPS.draw = (ctx, id, tc, x, y, w, h) => {
    const im = CLIPS.frame(id, tc);
    if (!im || !im.naturalWidth) return false;
    const s = Math.max(w / im.naturalWidth, h / im.naturalHeight);
    const sw = w / s, sh = h / s;
    ctx.drawImage(im, (im.naturalWidth - sw) / 2, (im.naturalHeight - sh) / 2, sw, sh, x, y, w, h);
    return true;
  };
  // start time (scene seconds) of the cue line that plays clip `id`, or null
  CLIPS.lineStart = (cues, id) => { const l = cues.find(l => l.clip === id); return l ? l.start : null; };
  window.CLIPS = CLIPS;
})();
