"""
WEB_UI_GUIDE.md

Quick Guide to Web UI
Hướng dẫn nhanh về giao diện web
"""

# 🌐 Web UI Quick Guide

## 🚀 Bắt Đầu Nhanh

### 1. Chạy ứng dụng

```bash
python run_web.py
```

### 2. Mở trình duyệt

```
http://localhost:5000
```

### 3. Tải ảnh lên

- Kéo và thả (drag & drop)
- Hoặc click để chọn file

### 4. Nhấn "Solve Sudoku"

- Chờ xử lý (5-30 giây)
- Xem kết quả

## 🎨 UI Layout

```
┌─────────────────────────────────────────────┐
│              Header / Title                 │
│    Sudoku Solver AI - Powered by AI         │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           Upload Section                    │
│  ┌──────────────────────┐  ┌──────────────┐ │
│  │ Upload Box           │  │ Solve Button │ │
│  │ 📸 Drag & Drop       │  │ ⚡ Solve     │ │
│  └──────────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           Results Section                   │
│  ┌──────────────────┐  ┌──────────────────┐ │
│  │ Original Image   │  │ Solved Image     │ │
│  │                  │  │                  │ │
│  │ [Image Display]  │  │ [Image Display]  │ │
│  └──────────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           Statistics                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Difficulty   │  │ Status       │  │ Steps        │  │
│  │ 42/81 (Med)  │  │ ✓ SOLVED     │  │ 5/5          │  │
│  └──────────┘  └──────────┘  └──────────┘  │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           Grid Display                      │
│  ┌────────────────────┐  ┌─────────────────┐│
│  │ Detected Grid      │  │ Solved Grid     ││
│  │                    │  │                 ││
│  │ [9x9 Sudoku]       │  │ [9x9 Sudoku]    ││
│  └────────────────────┘  └─────────────────┘│
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              Footer                         │
│       Sudoku Solver AI • 2026               │
└─────────────────────────────────────────────┘
```

## 🎯 Main Sections

### 1. Upload Box

```
📸
Upload Sudoku Image
Drag and drop or click to select
```

**Hiệu ứng:**
- Hover: Đổi màu, nâng lên
- Drag over: Sáng lên
- Click: Chọn file

**Hỗ trợ:**
- PNG, JPG, GIF, BMP
- Max 10MB

### 2. Solve Button

```
⚡ Solve Sudoku
```

**Trạng thái:**
- Disabled: Trước khi chọn ảnh
- Active: Sau khi tải ảnh
- Loading: Đang xử lý

**Hiệu ứng:**
- Hover: Nâng lên + tăng shadow
- Click: Flash animation

### 3. Image Display

Hai cột:
- **Trái**: Ảnh gốc
- **Phải**: Ảnh kết quả (vẽ chữ số giải được)

**Badge:**
- Input (xanh)
- ✓ Solved (xanh lá)

**Hiệu ứng:**
- Fade-in animation
- Hover: Nâng lên + tăng border glow

### 4. Statistics

3 stat cards:
- **Difficulty**: Số ô đã điền (42/81)
- **Status**: Trạng thái giải (✓ SOLVED)
- **Steps**: Bước hoàn thành (5/5)

**Hiệu ứng:**
- Hover: Nâng lên + tăng glow

### 5. Grid Display

Hai bảng sudoku:
- **Trái**: Lưới detect được
- **Phải**: Lưới đã giải

**Kiểu ô:**
- Trống (mờ): ·
- Detect (trắng): Số
- Giải được (xanh): Số (có background)

**Hiệu ứng:**
- Cell appear animation (delay theo vị trí)
- Stagger effect

### 6. Loading State

**Spinner:**
- Vòng tròn quay
- Có glow effect

**Progress bar:**
- Animation chạy qua lại

**Text:**
- "Solving your sudoku..."
- Pulse effect

## ✨ Color & Effects

### Colors

| Element | Color | Hex Code |
|---------|-------|----------|
| Primary | Cyan | #00d4ff |
| Secondary | Bright Green | #00ff88 |
| Accent | Hot Pink | #ff006e |
| Background | Dark | #0a0e27 |
| Secondary BG | Darker | #1a1f3a |

### Effects

| Effect | Where | Animation |
|--------|-------|-----------|
| Glitch | Title | Text shadow pulse |
| Float | Background | Particles moving up |
| Shimmer | Upload box | Gradient sweep |
| Bounce | Upload icon | Up-down movement |
| Spin | Loading | Full rotation |
| Pulse | Loading text | Opacity change |
| Cell Appear | Grid cells | Scale + fade-in |
| Fade-in | Images | Opacity transition |

## 🔄 User Flow

### Happy Path (Success)

```
1. Load page
2. Drag/drop image
   → Upload box updates
   → Button enables
3. Click "Solve"
   → Loading animation starts
4. Processing complete (5-30s)
   → Results displayed
   → Images shown
   → Stats shown
   → Grids shown
5. Can try another image
```

### Error Path

```
1. Invalid file
   → Error message appears
   → Input cleared
2. Processing fails
   → Error message shown
   → Original image still visible
3. Server error
   → Error message shown
   → Button re-enabled
```

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+V` | Paste image from clipboard |
| `Enter` | Solve (khi đã chọn ảnh) |

## 🖥️ Responsive Design

| Screen Size | Layout |
|-------------|--------|
| Desktop (1200px+) | 2 columns (side by side) |
| Tablet (768-1200px) | 1 column (stacked) |
| Mobile (<768px) | Full width, adjusted fonts |

## 🎨 Dark Mode (Already Dark!)

- Primary background: Very dark navy (#0a0e27)
- Secondary background: Slightly lighter (#1a1f3a)
- Text: White (#ffffff)
- Accents: Neon colors (cyan, green, pink)

**Why dark mode?**
- Reduces eye strain
- Trendy tech aesthetic
- Better contrast for neon colors
- Professional look

## 📊 Performance Indicators

### Frontend Load Time

- HTML: ~100ms
- CSS: ~50ms
- JS: ~50ms
- Images: On demand
- **Total: ~200ms**

### Backend Processing

- Point detection: ~2-5s
- Alignment: ~0.5s
- Preprocessing: ~1s
- OCR: ~5-15s
- Solver: ~0.5-2s
- **Total: ~8-23s** (CPU)
- **Total: ~3-8s** (GPU)

## 🔧 Customization

### Change Colors

Edit `static/css/style.css`:

```css
:root {
    --primary-color: #00d4ff;      /* Cyan */
    --secondary-color: #00ff88;    /* Green */
    --accent-color: #ff006e;       /* Pink */
    --dark-bg: #0a0e27;            /* Dark navy */
}
```

### Add Custom Title

Edit `templates/index.html`:

```html
<h1 class="title">
    <span class="title-icon">🧩</span>
    Your Title Here
</h1>
```

### Change Port

Edit `run_web.py` hoặc `app.py`:

```python
app.run(port=5001)
```

## 🐛 Common Issues

### Upload Box không phản ứng

- Check console (F12)
- Verify JavaScript loaded
- Try browser refresh

### Buttons disabled

- Ensure image selected
- Check file size < 10MB
- Try different image format

### Images not loading

- Check API response (Network tab)
- Verify backend running
- Check file permissions

### Grid looks wrong

- Grid is read-only (for display)
- Check original image quality
- Verify OCR detection

## 📞 Get Help

1. **Console errors** (F12)
   - Check JavaScript errors
   - Check network requests

2. **Network tab** (F12)
   - Check API response
   - Check file upload

3. **Backend logs**
   - Check terminal running Flask
   - Look for error messages

## 🎉 Tips & Tricks

1. **Paste from clipboard**
   - Press Ctrl+V to paste image
   - Works with screenshots

2. **Download results**
   - Right-click image → Save As
   - Or use keyboard shortcut

3. **Test with different images**
   - Printed sudoku
   - Digital displays
   - Handwritten (limited)

4. **Monitor processing**
   - Watch loading bar
   - Check browser console
   - Look at backend logs

## 📱 Mobile Support

- Responsive layout
- Touch-friendly buttons
- Optimized for small screens
- Drag & drop on mobile (limited)

**Recommended:** Use desktop for best experience

---

**Enjoy solving sudoku! 🧩✨**

Version: 1.0
Last Updated: 2026-08-10
