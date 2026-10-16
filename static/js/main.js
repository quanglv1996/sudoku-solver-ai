/* =====================================================================
   main.js - Khởi động ứng dụng: tab, trạng thái engine, cảnh báo HTTPS
   ===================================================================== */

import { $, api, state } from './common.js';
import { initPhoto, activatePhoto, deactivatePhoto } from './photo.js';
import { initLive, deactivateLive } from './live.js';
import { initScreen, activateScreen } from './screen.js';

const MODES = {
    photo: { activate: activatePhoto, deactivate: deactivatePhoto },
    live: { activate: () => {}, deactivate: deactivateLive },
    screen: { activate: activateScreen, deactivate: () => {} },
};

let mode = null;

function setMode(next) {
    if (next === mode || !MODES[next]) return;
    if (mode) MODES[mode].deactivate();
    mode = next;
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t.dataset.mode === next));
    document.querySelectorAll('.panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${next}`));
    MODES[next].activate();
    try { localStorage.setItem('sudoku-mode', next); } catch { /* private mode */ }
    if (location.hash !== `#${next}`) history.replaceState(null, '', `#${next}`);
}

// ---------------------------------------------------------------------
// Engine status
// ---------------------------------------------------------------------

async function pollHealth() {
    const pill = $('engineStatus');
    const text = pill.querySelector('.text');
    let delay = 2000;
    try {
        const h = await api('/api/health');
        state.health = h;
        pill.className = `status-pill ${h.engine === 'ready' ? 'ready' : h.engine === 'error' ? 'error' : 'loading'}`;
        if (h.engine === 'ready') {
            text.textContent = `Sẵn sàng · ${(h.device || '').toUpperCase()}`;
            delay = 10000;
        } else if (h.engine === 'error') {
            text.textContent = 'Lỗi model';
            pill.title = h.engine_error || '';
            delay = 15000;
        } else {
            text.textContent = 'Đang tải model…';
        }
        const q = h.queue || {};
        $('queueInfo').textContent = `${q.queued ?? 0} chờ${q.running ? ` · đang chạy: ${q.running}` : ''}`;
        updateSecureBanner(h);
    } catch {
        pill.className = 'status-pill error';
        text.textContent = 'Mất kết nối';
        delay = 3000;
    }
    setTimeout(pollHealth, delay);
}

function updateSecureBanner(h) {
    const banner = $('secureBanner');
    if (window.isSecureContext) { banner.hidden = true; return; }
    const https = h?.urls?.https?.[0];
    banner.hidden = false;
    banner.innerHTML = https
        ? `Camera trực tiếp &amp; Realtime cần HTTPS. Trên điện thoại hãy mở <a href="${https}">${https}</a> `
          + '(lần đầu chọn "Nâng cao → Tiếp tục" vì chứng chỉ tự ký). Chế độ chụp ảnh vẫn dùng được qua ứng dụng camera.'
        : 'Camera trực tiếp &amp; Realtime cần HTTPS (hoặc mở bằng localhost trên máy tính).';
}

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

initPhoto();
initLive();
initScreen();

document.querySelectorAll('.tab').forEach((t) => t.addEventListener('click', () => setMode(t.dataset.mode)));
window.addEventListener('hashchange', () => setMode(location.hash.slice(1)));

let initial = location.hash.slice(1);
if (!MODES[initial]) {
    try { initial = localStorage.getItem('sudoku-mode'); } catch { initial = null; }
}
setMode(MODES[initial] ? initial : 'photo');
updateSecureBanner(null);
pollHealth();
