// Record the Isaac Sim desktop from a noVNC browser tab to a .webm video.
//
// How it works: noVNC draws the remote screen into a <canvas>. This copies
// the Isaac part of that canvas into a smaller canvas 15 times a second and
// records it with the browser's MediaRecorder. Nothing is sent anywhere; the
// video stays in the tab until you save it.
//
// Use (Chrome / Edge):
//   1. Open http://localhost:6080/vnc.html and connect.
//   2. F12 -> Console, paste this whole file, press Enter.
//   3. startRec()                         before starting the flight
//   4. stopAndSave("my_flight.webm")      after the [flight] summary prints
//      -> the browser downloads the file (usually to Downloads).
//   recStatus() at any time shows state / seconds / MB.
//
// Settings used for isaac_px4_flight_152804.webm (2026-10-23): the Isaac
// window filled the left 1440x1000 of the 1920x1080 VNC screen (the rest was
// black), recorded at half size (720x500), 15 fps, VP8 2.5 Mbps -> ~10 MB for
// 114 s. A full-size 30 fps VP9 version froze the noVNC tab twice; keep it light.

(() => {
  const SRC_W = 1440, SRC_H = 1000;   // part of the VNC screen to record (top-left origin)
  const SCALE = 0.5;                  // output size = source x SCALE
  const FPS = 15;
  const BITRATE = 2.5e6;

  const vncCanvas = () => document.querySelector('#noVNC_container canvas');
  const rec = { mr: null, chunks: [], timer: null, t0: null, blob: null };

  window.startRec = () => {
    const src = vncCanvas();
    if (!src) throw new Error('no noVNC canvas: is the page connected?');
    const out = document.createElement('canvas');
    out.width = Math.round(SRC_W * SCALE);
    out.height = Math.round(SRC_H * SCALE);
    const g = out.getContext('2d');
    rec.timer = setInterval(() => g.drawImage(src, 0, 0, SRC_W, SRC_H, 0, 0, out.width, out.height), 1000 / FPS);
    const mr = new MediaRecorder(out.captureStream(FPS),
                                 { mimeType: 'video/webm;codecs=vp8', videoBitsPerSecond: BITRATE });
    rec.chunks = []; rec.blob = null;
    mr.ondataavailable = e => { if (e.data && e.data.size) rec.chunks.push(e.data); };
    mr.onstop = () => { rec.blob = new Blob(rec.chunks, { type: 'video/webm' }); clearInterval(rec.timer); };
    mr.start(2000);
    rec.mr = mr; rec.t0 = Date.now();
    return `recording ${out.width}x${out.height} @ ${FPS} fps`;
  };

  window.recStatus = () => ({
    state: rec.mr ? rec.mr.state : 'not started',
    seconds: rec.t0 ? ((Date.now() - rec.t0) / 1000).toFixed(1) : 0,
    mb: (rec.chunks.reduce((a, b) => a + b.size, 0) / 1e6).toFixed(2),
  });

  window.stopAndSave = (name = 'isaac_flight.webm') => new Promise(resolve => {
    const save = () => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(rec.blob);
      a.download = name;
      document.body.appendChild(a);
      a.click();
      resolve(`saved ${name} (${(rec.blob.size / 1e6).toFixed(2)} MB)`);
    };
    if (rec.mr && rec.mr.state !== 'inactive') { rec.mr.addEventListener('stop', save, { once: true }); rec.mr.stop(); }
    else if (rec.blob) save();
    else resolve('nothing recorded');
  });

  return 'recorder loaded: startRec(), recStatus(), stopAndSave("name.webm")';
})();
