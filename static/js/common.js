/* =====================================================================
   common.js - tiện ích dùng chung: API, camera, hình học, UI
   ===================================================================== */

export const $ = (id) => document.getElementById(id);

export const state = {
    health: null,
};

// ---------------------------------------------------------------------
// API
// ---------------------------------------------------------------------

export async function api(path, options = {}) {
    const res = await fetch(path, options);
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok && !(res.status === 202)) {
        const msg = (data && (data.message || data.error)) || `HTTP ${res.status}`;
        const err = new Error(msg);
        err.status = res.status;
        err.data = data;
        throw err;
    }
    return data;
}

/** Theo dõi job tới khi kết thúc; onUpdate(job) được gọi mỗi khi có thay đổi */
export async function watchJob(jobId, onUpdate, signal) {
    let job = null;
    while (!signal?.aborted) {
        job = await api(`/api/jobs/${jobId}?wait=1500`, { signal });
        onUpdate?.(job);
        if (['done', 'failed', 'cancelled', 'dropped'].includes(job.status)) return job;
    }
    throw new DOMException('Aborted', 'AbortError');
}

// ---------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------

let toastTimer = null;
export function toast(message, type = '') {
    const el = $('toast');
    el.textContent = message;
    el.className = `toast show ${type}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 3500);
}

export function renderGrid(el, grid, base = null, marks = {}) {
    el.innerHTML = '';
    const conflicts = new Set((marks.conflicts || []).map(([r, c]) => r * 9 + c));
    const uncertain = new Set((marks.uncertain || []).map(([r, c]) => r * 9 + c));
    for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
            const v = grid ? grid[r][c] : 0;
            const cell = document.createElement('div');
            const i = r * 9 + c;
            if (!v) {
                cell.className = 'empty';
                cell.textContent = '·';
            } else {
                cell.textContent = v;
                const isSolved = base && !base[r][c];
                cell.className = isSolved ? 'solved' : 'given';
                if (isSolved) cell.style.animationDelay = `${(r + c) * 25}ms`;
            }
            if (conflicts.has(i)) cell.classList.add('conflict');
            if (uncertain.has(i)) cell.classList.add('uncertain');
            el.appendChild(cell);
        }
    }
}

export function fmtMs(ms) {
    if (ms == null) return '-';
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)}s` : `${Math.round(ms)}ms`;
}

export function randomId() {
    return Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
}

// ---------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------

export function cameraSupported() {
    return !!(window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

export async function openCamera(facing = 'environment', hiRes = true) {
    const video = hiRes
        ? { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } }
        : { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } };
    return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

export function stopStream(stream) {
    stream?.getTracks().forEach((t) => t.stop());
}

export function cameraErrorText(err) {
    if (!window.isSecureContext) return 'Trình duyệt chỉ cho mở camera qua HTTPS.';
    if (err?.name === 'NotAllowedError') return 'Bạn chưa cấp quyền camera cho trang này.';
    if (err?.name === 'NotFoundError') return 'Không tìm thấy camera trên thiết bị.';
    if (err?.name === 'NotReadableError') return 'Camera đang được ứng dụng khác sử dụng.';
    return `Không mở được camera: ${err?.message || err}`;
}

/** Chuyển file/ảnh bất kỳ (kể cả HEIC trên iPhone) thành JPEG ≤ maxSide */
export async function fileToJpeg(file, maxSide = 1600) {
    const url = URL.createObjectURL(file);
    try {
        const img = await new Promise((resolve, reject) => {
            const im = new Image();
            im.onload = () => resolve(im);
            im.onerror = () => reject(new Error('Không đọc được ảnh'));
            im.src = url;
        });
        return drawToJpeg(img, img.naturalWidth, img.naturalHeight, maxSide, 0.9);
    } finally {
        URL.revokeObjectURL(url);
    }
}

export function drawToJpeg(source, w, h, maxSide, quality = 0.85, canvas = null) {
    const scale = Math.min(1, maxSide / Math.max(w, h));
    const cw = Math.round(w * scale);
    const ch = Math.round(h * scale);
    const cv = canvas || document.createElement('canvas');
    cv.width = cw;
    cv.height = ch;
    cv.getContext('2d').drawImage(source, 0, 0, cw, ch);
    return new Promise((resolve) => {
        cv.toBlob((blob) => resolve({ blob, width: cw, height: ch, url: blob ? URL.createObjectURL(blob) : null }),
            'image/jpeg', quality);
    });
}

// ---------------------------------------------------------------------
// Geometry: homography từ hình vuông đơn vị → tứ giác (TL, TR, BR, BL)
// ---------------------------------------------------------------------

export function squareToQuad(q) {
    const [[x0, y0], [x1, y1], [x2, y2], [x3, y3]] = q;
    const sx = x0 - x1 + x2 - x3;
    const sy = y0 - y1 + y2 - y3;
    let a, b, c, d, e, f, g, h;
    if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {
        a = x1 - x0; b = x2 - x1; c = x0;
        d = y1 - y0; e = y2 - y1; f = y0;
        g = 0; h = 0;
    } else {
        const dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2;
        const det = dx1 * dy2 - dx2 * dy1;
        g = (sx * dy2 - dx2 * sy) / det;
        h = (dx1 * sy - sx * dy1) / det;
        a = x1 - x0 + g * x1; b = x3 - x0 + h * x3; c = x0;
        d = y1 - y0 + g * y1; e = y3 - y0 + h * y3; f = y0;
    }
    return (u, v) => {
        const w = g * u + h * v + 1;
        return [(a * u + b * v + c) / w, (d * u + e * v + f) / w];
    };
}

/** Tâm + kích thước + góc xoay của 81 ô trong tứ giác */
export function cellsFromQuad(quad) {
    const map = squareToQuad(quad);
    const cells = [];
    const k = 1 / 18;
    for (let r = 0; r < 9; r++) {
        for (let c = 0; c < 9; c++) {
            const u = (c + 0.5) / 9, v = (r + 0.5) / 9;
            const [x, y] = map(u, v);
            const [lx, ly] = map(u - k, v);
            const [rx, ry] = map(u + k, v);
            const [tx, ty] = map(u, v - k);
            const [bx, by] = map(u, v + k);
            cells.push({
                x, y,
                s: Math.min(Math.hypot(rx - lx, ry - ly), Math.hypot(bx - tx, by - ty)),
                a: Math.atan2(ry - ly, rx - lx) * 180 / Math.PI,
            });
        }
    }
    return { cells, map };
}
