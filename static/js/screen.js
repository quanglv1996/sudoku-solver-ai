/* =====================================================================
   screen.js - Tự giải & tự điền sudoku trên màn hình máy chạy server
   ===================================================================== */

import { $, api, watchJob, toast, fmtMs, state } from './common.js';

const STAGE_TEXT = {
    queued: 'Đang xếp hàng…',
    capture: 'Đang chụp màn hình…',
    detect: 'Đang tìm lưới sudoku trên màn hình…',
    align: 'Căn chỉnh…',
    ocr: 'Đọc đề…',
    solve: 'Đang giải…',
    fill: 'Đang tự điền…',
};

const els = {};
let running = false;
let countdownTimer = null;
let abortCtrl = null;
let monitorsLoaded = false;

export function initScreen() {
    Object.assign(els, {
        start: $('screenStart'), monitor: $('monitorSelect'), countdownSel: $('countdownSelect'),
        speed: $('speedSelect'), autoFill: $('autoFill'), note: $('screenNote'),
        result: $('screenResult'), message: $('screenMessage'), preview: $('screenPreview'),
        overlay: $('countdown'), num: $('countdownNum'), cancel: $('countdownCancel'),
    });
    els.start.addEventListener('click', () => (running ? cancel() : begin()));
    els.cancel.addEventListener('click', cancel);
}

export async function activateScreen() {
    const s = state.health?.screen;
    if (s && !s.enabled) {
        els.note.textContent = 'Tính năng đang tắt trên server (SCREEN_CONTROL=false).';
        els.start.disabled = true;
    } else if (s && !s.allowed) {
        els.note.textContent = 'Server chỉ cho phép kích hoạt từ chính máy chủ (SCREEN_LOCAL_ONLY=true).';
        els.start.disabled = true;
    } else if (s && !s.is_server_machine) {
        els.note.textContent = 'Bạn đang điều khiển từ thiết bị khác: màn hình của MÁY CHỦ sẽ được chụp và điền. Không cần đếm ngược.';
        els.countdownSel.value = '0';
    }
    if (!monitorsLoaded) loadMonitors();
}

async function loadMonitors() {
    try {
        const { monitors } = await api('/api/screen/monitors');
        els.monitor.innerHTML = monitors
            .map((m) => `<option value="${m.index}">#${m.index} · ${m.width}×${m.height}</option>`)
            .join('');
        monitorsLoaded = true;
    } catch { /* giữ mặc định */ }
}

function setRunning(on) {
    running = on;
    els.start.classList.toggle('primary', !on);
    els.start.classList.toggle('danger', on);
    els.start.lastChild.textContent = on ? ' Huỷ' : ' Bắt đầu';
}

async function begin() {
    setRunning(true);
    abortCtrl = new AbortController();
    const seconds = Number(els.countdownSel.value);
    try {
        if (seconds > 0) await countdown(seconds);
        await run();
    } catch (err) {
        if (err.name !== 'AbortError') {
            toast(err.message, 'bad');
            showMessage(err.message, 'bad');
        }
    } finally {
        hideCountdown();
        document.title = 'Sudoku Vision';
        setRunning(false);
    }
}

function countdown(seconds) {
    return new Promise((resolve, reject) => {
        let n = seconds;
        els.overlay.hidden = false;
        const tick = () => {
            if (abortCtrl.signal.aborted) return reject(new DOMException('Aborted', 'AbortError'));
            if (n <= 0) { hideCountdown(); return resolve(); }
            els.num.textContent = n;
            document.title = `⏱ ${n}… Sudoku Vision`;
            n -= 1;
            countdownTimer = setTimeout(tick, 1000);
        };
        tick();
    });
}

function hideCountdown() {
    clearTimeout(countdownTimer);
    els.overlay.hidden = true;
}

async function run() {
    const body = {
        monitor: Number(els.monitor.value || 1),
        fill: els.autoFill.checked,
        type_delay: Number(els.speed.value),
    };
    const job = await api('/api/screen', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: abortCtrl.signal,
    });
    abortCtrl.jobId = job.id;
    showMessage(STAGE_TEXT.queued, '');

    const final = await watchJob(job.id, (j) => {
        let text = STAGE_TEXT[j.stage] || j.stage;
        const info = j.stage_info || {};
        if (j.stage === 'fill' && info.total) text += ` ${info.filled || 0}/${info.total}`;
        showMessage(text, '');
        document.title = `${text} - Sudoku Vision`;
    }, abortCtrl.signal);

    const r = final.result || {};
    els.result.hidden = false;
    $('scrStatus').textContent = r.success ? 'ĐÃ GIẢI' : 'THẤT BẠI';
    $('scrStatus').className = `v ${r.success ? 'ok' : 'bad'}`;
    $('scrGivens').textContent = r.givens != null ? `${r.givens}/81` : '-';
    $('scrFilled').textContent = r.filled != null ? `${r.filled}` : '-';
    $('scrTime').textContent = fmtMs(final.run_ms);
    if (r.preview_image) els.preview.src = r.preview_image;
    els.preview.hidden = !r.preview_image;

    if (final.status === 'cancelled') {
        showMessage('Đã huỷ.', 'warn');
    } else if (!r.success) {
        let msg = r.message || final.error || 'Không giải được';
        if (r.stage === 'detect') msg += '. Đảm bảo lưới sudoku hiển thị đầy đủ, không bị che, trên đúng màn hình đã chọn.';
        showMessage(msg, 'bad');
    } else if (r.fill_error) {
        showMessage(`Đã giải nhưng dừng điền sau ${r.filled} ô: ${r.fill_error}`, 'warn');
    } else if (body.fill) {
        showMessage(`Đã giải và tự điền ${r.filled} ô 🎉`, 'ok');
        toast('Đã điền xong!', 'ok');
    } else {
        showMessage('Đã giải (không tự điền). Xem lời giải trong ảnh.', 'ok');
    }
}

function cancel() {
    abortCtrl?.abort();
    if (abortCtrl?.jobId) api(`/api/jobs/${abortCtrl.jobId}`, { method: 'DELETE' }).catch(() => {});
    hideCountdown();
}

function showMessage(text, type) {
    els.result.hidden = false;
    els.message.textContent = text;
    els.message.className = `message ${type}`;
}
