/* =====================================================================
   live.js - Chế độ Realtime: quay camera, lời giải vẽ thẳng lên hình

   Vòng gửi frame: luôn chỉ có 1 request đang bay (tự điều tiết theo tốc
   độ server); server dùng "latest wins" nên không bao giờ dồn độ trễ.
   Vòng vẽ: requestAnimationFrame, vẽ overlay mới nhất lên canvas.
   ===================================================================== */

import { $, toast, randomId, cameraSupported, openCamera, stopStream, cameraErrorText, drawToJpeg, squareToQuad } from './common.js';

const STATUS_TEXT = {
    searching: 'Đang tìm lưới…',
    unsolved: 'Đã thấy lưới · đang đọc đề…',
    solved: 'Đã giải ✓',
    tracking: 'Đang bám lời giải ✓',
    loading: 'Model đang tải…',
};
const OVERLAY_TTL = 900;   // ms - overlay mờ dần nếu không có cập nhật mới

const els = {};
const session = randomId();
let stream = null;
let facing = 'environment';
let running = false;
let overlay = null;        // { data, at }
let rafId = null;
let needReset = true;
let unsolvedSince = 0;
const metrics = { times: [], serverMs: 0, rtt: 0 };
const grabCanvas = document.createElement('canvas');

export function initLive() {
    Object.assign(els, {
        viewer: $('liveViewer'), video: $('liveVideo'), canvas: $('liveCanvas'),
        guide: $('liveGuide'), badge: $('liveBadge'), status: $('liveStatus'),
        metrics: $('liveMetrics'), empty: $('liveEmpty'), emptyText: $('liveEmptyText'),
        toggle: $('liveToggle'), flip: $('liveFlipBtn'), showGrid: $('liveShowGrid'),
        hiRes: $('liveHiRes'), note: $('liveNote'),
    });
    els.toggle.addEventListener('click', () => (running ? stop() : start()));
    els.flip.addEventListener('click', flip);
    document.addEventListener('visibilitychange', () => { if (document.hidden && running) stop(); });

    if (!cameraSupported()) {
        els.toggle.disabled = true;
        els.flip.hidden = true;
        els.emptyText.textContent = window.isSecureContext
            ? 'Thiết bị/trình duyệt không hỗ trợ camera trực tiếp.'
            : 'Chế độ realtime cần mở trang qua HTTPS (xem đường dẫn ở thông báo phía trên).';
    }
}

export function deactivateLive() {
    if (running) stop();
}

// ---------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------

async function start() {
    try {
        stream = await openCamera(facing, false);
    } catch (err) {
        toast(cameraErrorText(err), 'bad');
        return;
    }
    els.video.srcObject = stream;
    els.video.style.transform = facing === 'user' ? 'scaleX(-1)' : '';
    await els.video.play().catch(() => {});

    running = true;
    needReset = true;
    overlay = null;
    metrics.times = [];
    els.empty.hidden = true;
    els.guide.hidden = false;
    els.badge.hidden = false;
    els.metrics.hidden = false;
    setToggle(true);
    setStatus('searching');

    pump();
    rafId = requestAnimationFrame(render);
}

function stop() {
    running = false;
    stopStream(stream);
    stream = null;
    els.video.srcObject = null;
    cancelAnimationFrame(rafId);
    const ctx = els.canvas.getContext('2d');
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    els.empty.hidden = false;
    els.guide.hidden = true;
    els.badge.hidden = true;
    els.metrics.hidden = true;
    setToggle(false);
}

async function flip() {
    facing = facing === 'environment' ? 'user' : 'environment';
    if (running) {
        stop();
        await start();
    }
}

function setToggle(on) {
    els.toggle.classList.toggle('primary', !on);
    els.toggle.classList.toggle('danger', on);
    els.toggle.querySelector('span').textContent = on ? 'Dừng' : 'Bắt đầu';
    els.toggle.querySelector('svg').innerHTML = on ? '<rect x="6" y="6" width="12" height="12" rx="2"/>' : '<path d="M8 5v14l11-7z"/>';
}

function setStatus(status) {
    els.status.textContent = STATUS_TEXT[status] || status;
    els.badge.classList.toggle('on', status !== 'solved' && status !== 'tracking');
    els.badge.classList.toggle('solved', status === 'solved' || status === 'tracking');
}

// ---------------------------------------------------------------------
// Frame pump
// ---------------------------------------------------------------------

async function pump() {
    while (running) {
        const v = els.video;
        if (!v.videoWidth) { await sleep(50); continue; }

        const maxSide = els.hiRes.checked ? 1024 : 640;
        const frame = await drawToJpeg(v, v.videoWidth, v.videoHeight, maxSide, 0.72, grabCanvas);
        if (!frame.blob || !running) continue;
        URL.revokeObjectURL(frame.url);

        const form = new FormData();
        form.append('frame', frame.blob, 'frame.jpg');
        form.append('session', session);
        if (needReset) { form.append('reset', '1'); needReset = false; }

        const t0 = performance.now();
        try {
            const res = await fetch('/api/realtime', { method: 'POST', body: form });
            const data = await res.json();
            if (!running) break;
            if (res.status === 503) { setStatus('loading'); await sleep(1000); continue; }
            if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);

            metrics.rtt = performance.now() - t0;
            if (data.ok) {
                metrics.serverMs = data.ms;
                metrics.times.push(performance.now());
                metrics.times = metrics.times.filter((t) => performance.now() - t < 2000);
                handleResult(data, v.videoWidth / frame.width);
            }
        } catch (err) {
            if (!running) break;
            setStatus(`Lỗi kết nối: ${err.message}`);
            await sleep(800);
        }
    }
}

function handleResult(data, scaleToVideo) {
    setStatus(data.status);
    if (data.found) overlay = { data, scaleToVideo, at: performance.now() };

    if (data.status === 'unsolved') {
        unsolvedSince ||= performance.now();
        if (performance.now() - unsolvedSince > 4000) {
            els.note.textContent = 'Chưa đọc được đề: hãy đưa camera gần hơn, chụp thẳng, tránh loá sáng.';
        }
    } else {
        unsolvedSince = 0;
        if (data.status === 'tracking' || data.status === 'solved') els.note.textContent = '';
    }
    const fps = metrics.times.length / 2;
    els.metrics.textContent = `${fps.toFixed(1)} fps · server ${Math.round(metrics.serverMs)}ms · rtt ${Math.round(metrics.rtt)}ms`;
}

// ---------------------------------------------------------------------
// Render overlay
// ---------------------------------------------------------------------

function render() {
    if (!running) return;
    rafId = requestAnimationFrame(render);

    const canvas = els.canvas;
    const dpr = window.devicePixelRatio || 1;
    const cw = canvas.clientWidth, ch = canvas.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
        canvas.width = Math.round(cw * dpr);
        canvas.height = Math.round(ch * dpr);
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    const v = els.video;
    if (!overlay || !v.videoWidth) return;
    const age = performance.now() - overlay.at;
    if (age > OVERLAY_TTL) return;
    const alpha = age < OVERLAY_TTL / 2 ? 1 : 1 - (age - OVERLAY_TTL / 2) / (OVERLAY_TTL / 2);

    // object-fit: cover
    const s = Math.max(cw / v.videoWidth, ch / v.videoHeight);
    const ox = (cw - v.videoWidth * s) / 2;
    const oy = (ch - v.videoHeight * s) / 2;
    const k = overlay.scaleToVideo * s;
    const mirror = facing === 'user';
    const map = ([x, y]) => {
        let px = ox + x * k;
        if (mirror) px = cw - px;
        return [px, oy + y * k];
    };

    const { data } = overlay;
    const solved = data.status === 'solved' || data.status === 'tracking';
    const quad = data.corners.map(map);
    ctx.globalAlpha = alpha;

    // Khung lưới
    ctx.lineJoin = 'round';
    ctx.lineWidth = 3;
    ctx.strokeStyle = solved ? '#34d399' : '#22d3ee';
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    quad.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (els.showGrid.checked) {
        const m = squareToQuad(quad);
        ctx.strokeStyle = 'rgba(165, 243, 252, .35)';
        for (let i = 1; i < 9; i++) {
            ctx.lineWidth = i % 3 === 0 ? 2 : 1;
            ctx.beginPath();
            ctx.moveTo(...m(i / 9, 0)); ctx.lineTo(...m(i / 9, 1));
            ctx.moveTo(...m(0, i / 9)); ctx.lineTo(...m(1, i / 9));
            ctx.stroke();
        }
    }

    // Chữ số lời giải
    if (solved && data.cells) {
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (const cell of data.cells) {
            const [x, y] = map([cell.x, cell.y]);
            const size = cell.s * k * 0.72;
            if (size < 6) continue;
            ctx.save();
            ctx.translate(x, y);
            ctx.rotate(((mirror ? -cell.a : cell.a) * Math.PI) / 180);
            ctx.font = `800 ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
            ctx.lineWidth = Math.max(2, size * 0.14);
            ctx.strokeStyle = 'rgba(3, 20, 12, .85)';
            ctx.strokeText(String(cell.d), 0, size * 0.04);
            ctx.fillStyle = '#4ade80';
            ctx.fillText(String(cell.d), 0, size * 0.04);
            ctx.restore();
        }
    }
    ctx.globalAlpha = 1;
}

function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
}
