/* =====================================================================
   photo.js - Chế độ "Chụp & Giải" với hiệu ứng quét
   ===================================================================== */

import {
    $, api, watchJob, toast, renderGrid, fmtMs, cameraSupported, openCamera,
    stopStream, cameraErrorText, fileToJpeg, drawToJpeg, cellsFromQuad,
} from './common.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const STAGES = ['queued', 'detect', 'align', 'ocr', 'solve', 'render'];
const STAGE_TEXT = {
    queued: 'Đang xếp hàng xử lý…',
    detect: 'Đang dò khung lưới sudoku…',
    align: 'Căn chỉnh phối cảnh…',
    ocr: 'Nhận diện chữ số…',
    solve: 'Chạy thuật toán giải…',
    render: 'Dựng kết quả…',
};
const STAGE_PCT = { queued: 6, detect: 22, align: 40, ocr: 62, solve: 84, render: 94 };
const MIN_STAGE_MS = 380;
const MIN_SCAN_MS = 1500;

const els = {};
let stream = null;
let facing = 'environment';
let busy = false;
let active = false;
let current = null;      // { result, imageUrl, width, height }
let abortCtrl = null;

// ---------------------------------------------------------------------
// Stage ticker: hiển thị lần lượt từng bước, mỗi bước tối thiểu MIN_STAGE_MS
// (job chạy rất nhanh, nhưng người dùng vẫn thấy được quá trình "quét")
// ---------------------------------------------------------------------

class StageTicker {
    constructor(onShow) {
        this.onShow = onShow;
        this.queue = [];
        this.lastIndex = -1;
        this.timer = null;
    }
    push(stage, info) {
        const idx = STAGES.indexOf(stage);
        if (idx <= this.lastIndex) {
            if (stage === 'queued') this.onShow(stage, info); // cập nhật vị trí hàng đợi
            return;
        }
        for (let i = this.lastIndex + 1; i <= idx; i++) this.queue.push([STAGES[i], i === idx ? info : null]);
        this.lastIndex = idx;
        this.pump();
    }
    pump() {
        if (this.timer) return;
        const next = () => {
            const item = this.queue.shift();
            if (!item) { this.timer = null; return; }
            this.onShow(item[0], item[1]);
            this.timer = setTimeout(next, MIN_STAGE_MS);
        };
        next();
    }
    drained() {
        return new Promise((resolve) => {
            const check = () => (!this.queue.length && !this.timer ? resolve() : setTimeout(check, 40));
            check();
        });
    }
    stop() {
        clearTimeout(this.timer);
        this.timer = null;
        this.queue = [];
    }
}

// ---------------------------------------------------------------------
// Init / lifecycle
// ---------------------------------------------------------------------

export function initPhoto() {
    Object.assign(els, {
        viewer: $('photoViewer'), video: $('photoVideo'), image: $('photoImage'),
        guide: $('photoGuide'), overlay: $('photoOverlay'), empty: $('photoEmpty'),
        emptyText: $('photoEmptyText'), hud: $('photoHud'), hudStage: $('hudStage'),
        hudPct: $('hudPct'), hudBar: $('hudBar'), shutter: $('shutterBtn'),
        upload: $('uploadBtn'), flip: $('photoFlipBtn'), fileInput: $('fileInput'),
        captureInput: $('captureInput'), result: $('photoResult'), message: $('photoMessage'),
        viewToggle: $('viewToggle'),
    });

    els.shutter.addEventListener('click', onShutter);
    els.upload.addEventListener('click', () => els.fileInput.click());
    els.flip.addEventListener('click', flipCamera);
    els.fileInput.addEventListener('change', (e) => onFile(e.target.files[0], e.target));
    els.captureInput.addEventListener('change', (e) => onFile(e.target.files[0], e.target));
    $('retakeBtn').addEventListener('click', reset);
    $('downloadBtn').addEventListener('click', download);
    els.viewToggle.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-view]');
        if (btn) setView(btn.dataset.view);
    });

    // Kéo-thả & dán ảnh
    els.viewer.addEventListener('dragover', (e) => e.preventDefault());
    els.viewer.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = [...(e.dataTransfer?.files || [])].find((f) => f.type.startsWith('image/'));
        if (file) onFile(file);
    });
    document.addEventListener('paste', (e) => {
        if (!active) return;
        const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
        if (item) onFile(item.getAsFile());
    });

    if (!cameraSupported()) els.flip.hidden = true;
}

export function activatePhoto() {
    active = true;
    if (!busy && !current) startPreview();
}

export function deactivatePhoto() {
    active = false;
    stopPreview();
}

// ---------------------------------------------------------------------
// Camera preview
// ---------------------------------------------------------------------

// Mỗi lần mở/tắt preview tăng token → kết quả mở camera về muộn sẽ bị bỏ qua
let previewToken = 0;
const previewValid = (token) => token === previewToken && active && !busy && !current;

async function startPreview() {
    const token = ++previewToken;
    showMode('empty');
    if (!cameraSupported()) {
        els.emptyText.textContent = window.isSecureContext
            ? 'Thiết bị không hỗ trợ camera trực tiếp. Bấm nút tròn để chụp bằng ứng dụng camera, hoặc "Tải ảnh".'
            : 'Bấm nút tròn để chụp bằng ứng dụng camera của điện thoại, hoặc "Tải ảnh". (Xem trước camera trực tiếp cần HTTPS.)';
        return;
    }
    try {
        stopStream(stream);
        stream = null;
        const s = await openCamera(facing, true);
        if (!previewValid(token)) { stopStream(s); return; }
        stream = s;
        els.video.srcObject = s;
        await els.video.play().catch(() => {});
        if (!previewValid(token)) return;
        els.video.style.transform = facing === 'user' ? 'scaleX(-1)' : '';
        showMode('camera');
    } catch (err) {
        if (!previewValid(token)) return;
        stream = null;
        els.emptyText.textContent = `${cameraErrorText(err)} Bạn vẫn có thể "Tải ảnh".`;
        showMode('empty');
    }
}

function stopPreview() {
    previewToken++;
    stopStream(stream);
    stream = null;
    els.video.srcObject = null;
}

async function flipCamera() {
    facing = facing === 'environment' ? 'user' : 'environment';
    if (!busy && !current) await startPreview();
}

function showMode(mode) {
    els.video.hidden = mode !== 'camera';
    els.guide.hidden = mode !== 'camera';
    els.image.hidden = mode !== 'image';
    els.overlay.style.display = mode === 'image' ? '' : 'none';
    els.empty.hidden = mode !== 'empty';
}

// ---------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------

async function onShutter() {
    if (busy) return;
    if (stream && els.video.videoWidth) {
        const v = els.video;
        flash();
        const jpeg = await drawToJpeg(v, v.videoWidth, v.videoHeight, 1600, 0.92);
        return scan(jpeg);
    }
    // Không có xem trước trực tiếp → dùng ứng dụng camera của hệ điều hành
    els.captureInput.click();
}

async function onFile(file, input) {
    if (input) input.value = '';
    if (!file || busy) return;
    try {
        scan(await fileToJpeg(file, 1600));
    } catch (err) {
        toast(err.message, 'bad');
    }
}

function flash() {
    els.viewer.animate([{ filter: 'brightness(2.2)' }, { filter: 'brightness(1)' }], { duration: 220 });
}

// ---------------------------------------------------------------------
// Scan flow
// ---------------------------------------------------------------------

async function scan(jpeg) {
    busy = true;
    current = null;
    setControls(false);
    stopPreview();
    els.result.hidden = true;

    els.image.src = jpeg.url;
    els.overlay.setAttribute('viewBox', `0 0 ${jpeg.width} ${jpeg.height}`);
    clearOverlay();
    showMode('image');
    els.viewer.classList.add('scanning');
    els.hud.className = 'hud';
    setHud('Khởi động máy quét…', 2);

    const fx = new ScanFx(jpeg.width, jpeg.height);
    const ticker = new StageTicker((stage, info) => {
        if (stage === 'queued') {
            const pos = info?.position;
            setHud(pos > 0 ? `Đang xếp hàng (trước bạn: ${pos})…` : STAGE_TEXT.queued, STAGE_PCT.queued);
            return;
        }
        setHud(STAGE_TEXT[stage], STAGE_PCT[stage]);
        if (STAGES.indexOf(stage) >= STAGES.indexOf('align') && fx.pending) fx.setQuad(fx.pending);
        if (stage === 'ocr') fx.startRain();
        if (stage === 'solve') fx.stopRain();
    });

    const started = performance.now();
    abortCtrl = new AbortController();
    try {
        const form = new FormData();
        form.append('image', jpeg.blob, 'capture.jpg');
        const job = await api('/api/jobs', { method: 'POST', body: form, signal: abortCtrl.signal });
        ticker.push(job.stage, job);

        const final = await watchJob(job.id, (j) => {
            ticker.push(j.stage, j);
            if (j.stage_info?.corners) fx.pending = j.stage_info.corners;
        }, abortCtrl.signal);

        if (final.result?.corners) fx.pending = final.result.corners;
        if (final.status === 'done') ticker.push('render');
        await ticker.drained();
        if (fx.pending) fx.setQuad(fx.pending);
        const remain = MIN_SCAN_MS - (performance.now() - started);
        if (remain > 0) await new Promise((r) => setTimeout(r, remain));

        fx.stopRain();
        els.viewer.classList.remove('scanning');
        current = { result: final.result, imageUrl: jpeg.url, width: jpeg.width, height: jpeg.height };

        if (final.status === 'done' && final.result?.success) {
            fx.reveal(final.result);
            els.hud.className = 'hud ok';
            setHud(`Đã giải xong trong ${fmtMs(final.run_ms)}`, 100);
            showResult(final);
        } else {
            const msg = final.result?.message || final.error || 'Không giải được';
            fx.fail(final.result);
            els.hud.className = 'hud fail';
            setHud(msg, 100);
            showResult(final);
        }
        els.viewer.classList.add('show-hud');
        setTimeout(() => els.viewer.classList.remove('show-hud'), 3500);
    } catch (err) {
        ticker.stop();
        fx.stopRain();
        els.viewer.classList.remove('scanning');
        if (err.name !== 'AbortError') {
            toast(err.message, 'bad');
            els.hud.className = 'hud fail';
            setHud(err.message, 100);
            els.viewer.classList.add('show-hud');
        }
        current = { result: null, imageUrl: jpeg.url };
    } finally {
        busy = false;
        setControls(true);
    }
}

function setControls(enabled) {
    els.shutter.disabled = !enabled;
    els.upload.disabled = !enabled;
    els.flip.disabled = !enabled;
}

function setHud(text, pct) {
    els.hudStage.textContent = text;
    els.hudPct.textContent = `${Math.round(pct)}%`;
    els.hudBar.style.width = `${pct}%`;
}

// ---------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------

function showResult(job) {
    const r = job.result || {};
    els.result.hidden = false;
    els.message.className = `message ${r.success ? 'ok' : 'bad'}`;
    let msg = r.message || job.error || 'Không giải được';
    if (!r.success && r.stage === 'detect') msg += '. Hãy chụp thẳng, đủ sáng và để cả 4 góc lưới trong ảnh.';
    if (!r.success && (r.conflicts?.length || r.stage === 'solve')) msg += ' Ô đỏ là các số bị trùng.';
    els.message.textContent = msg;

    const status = $('statStatus');
    status.textContent = r.success ? 'ĐÃ GIẢI' : 'THẤT BẠI';
    status.className = `v ${r.success ? 'ok' : 'bad'}`;
    $('statGivens').textContent = r.givens != null ? `${r.givens}/81` : '-';
    $('statTime').textContent = fmtMs(job.run_ms);
    $('statMethod').textContent = r.method ? r.method.toUpperCase() : '-';

    renderGrid($('detectedGrid'), r.detected_grid, null, r);
    renderGrid($('solvedGrid'), r.solved_grid || null, r.detected_grid);

    const btns = els.viewToggle.querySelectorAll('button');
    btns.forEach((b) => { b.disabled = b.dataset.view === 'aligned' && !r.aligned_image; });
    setView('solution');
    $('downloadBtn').disabled = !r.solved_image;

    if (window.matchMedia('(max-width: 899px)').matches) {
        setTimeout(() => els.result.scrollIntoView({ behavior: 'smooth', block: 'start' }), 600);
    }
}

function setView(view) {
    if (!current) return;
    els.viewToggle.querySelectorAll('button').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
    const r = current.result || {};
    if (view === 'aligned' && r.aligned_image) {
        els.image.src = r.aligned_image;
        els.overlay.style.display = 'none';
    } else {
        els.image.src = current.imageUrl;
        els.overlay.style.display = view === 'solution' ? '' : 'none';
    }
}

function download() {
    const src = current?.result?.solved_image;
    if (!src) return;
    const a = document.createElement('a');
    a.href = src;
    a.download = `sudoku-solved-${Date.now()}.jpg`;
    a.click();
}

function reset() {
    abortCtrl?.abort();
    current = null;
    els.result.hidden = true;
    els.viewer.classList.remove('show-hud', 'scanning');
    clearOverlay();
    if (active) startPreview();
    els.viewer.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ---------------------------------------------------------------------
// Overlay effects (SVG theo toạ độ pixel của ảnh)
// ---------------------------------------------------------------------

function clearOverlay() {
    els.overlay.innerHTML = '';
}

function svg(tag, attrs = {}, parent = els.overlay) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    parent.appendChild(el);
    return el;
}

class ScanFx {
    constructor(w, h) {
        this.w = w;
        this.h = h;
        this.quad = null;
        this.pending = null;   // góc nhận từ server, vẽ khi HUD tới bước "align"
        this.geo = null;
        this.rainTimer = null;
        this.layer = svg('g');
        this.rainLayer = svg('g');
    }

    setQuad(normCorners) {
        const quad = normCorners.map(([x, y]) => [x * this.w, y * this.h]);
        if (this.quad && JSON.stringify(quad) === JSON.stringify(this.quad)) return;
        this.quad = quad;
        this.geo = cellsFromQuad(quad);
        this.layer.innerHTML = '';
        const { map } = this.geo;

        this.poly = svg('polygon', { points: quad.map((p) => p.join(',')).join(' '), class: 'quad-line' }, this.layer);
        for (let i = 1; i < 9; i++) {
            const cls = `grid-line${i % 3 === 0 ? ' major' : ''}`;
            const [ax, ay] = map(i / 9, 0), [bx, by] = map(i / 9, 1);
            const [cx, cy] = map(0, i / 9), [dx, dy] = map(1, i / 9);
            svg('line', { x1: ax, y1: ay, x2: bx, y2: by, class: cls }, this.layer);
            svg('line', { x1: cx, y1: cy, x2: dx, y2: dy, class: cls }, this.layer);
        }
        const r = Math.max(4, Math.min(this.w, this.h) / 90);
        quad.forEach(([x, y]) => svg('circle', { cx: x, cy: y, r, class: 'corner-dot' }, this.layer));
    }

    startRain() {
        if (!this.geo || this.rainTimer) return;
        const texts = this.geo.cells.map((c) => {
            const t = svg('text', { x: c.x, y: c.y, class: 'rain', 'font-size': c.s * 0.5 }, this.rainLayer);
            t.textContent = '';
            return t;
        });
        this.rainTimer = setInterval(() => {
            for (let k = 0; k < 14; k++) {
                const t = texts[(Math.random() * 81) | 0];
                t.textContent = Math.random() < 0.3 ? '' : String(1 + ((Math.random() * 9) | 0));
                t.setAttribute('opacity', (0.3 + Math.random() * 0.7).toFixed(2));
            }
        }, 60);
    }

    stopRain() {
        clearInterval(this.rainTimer);
        this.rainTimer = null;
        this.rainLayer.innerHTML = '';
    }

    cellPolygon(r, c) {
        const { map } = this.geo;
        return [map(c / 9, r / 9), map((c + 1) / 9, r / 9), map((c + 1) / 9, (r + 1) / 9), map(c / 9, (r + 1) / 9)]
            .map((p) => p.join(',')).join(' ');
    }

    reveal(result) {
        if (!this.geo) return;
        this.poly?.classList.add('locked');
        this.layer.querySelectorAll('.grid-line, .corner-dot').forEach((el) => el.remove());
        const { detected_grid: det, solved_grid: sol } = result;
        const digits = svg('g');
        this.geo.cells.forEach((cell, i) => {
            const r = Math.floor(i / 9), c = i % 9;
            if (det[r][c]) return;
            const g = svg('g', { transform: `rotate(${cell.a.toFixed(2)} ${cell.x} ${cell.y})` }, digits);
            const t = svg('text', { x: cell.x, y: cell.y, class: 'sol-digit', 'font-size': cell.s * 0.72 }, g);
            t.style.strokeWidth = `${Math.max(2, cell.s * 0.08)}px`;
            t.style.animationDelay = `${(r + c) * 35}ms`;
            t.textContent = sol[r][c];
        });
    }

    fail(result) {
        this.poly?.classList.add('failed');
        if (!this.geo || !result) return;
        (result.conflicts || []).forEach(([r, c]) => svg('polygon', { points: this.cellPolygon(r, c), class: 'conflict-box' }, this.layer));
    }
}
