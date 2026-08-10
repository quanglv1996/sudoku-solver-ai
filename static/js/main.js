/* =====================================================================
   main.js - Frontend Logic for Sudoku Solver
   ===================================================================== */

// ===== Global State =====
let selectedImage = null;
let lastResult = null;

// ===== DOM Elements =====
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const solveBtn = document.getElementById('solveBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingContainer = document.getElementById('loadingContainer');
const messageContainer = document.getElementById('messageContainer');
const messageBox = document.getElementById('messageBox');

// ===== Event Listeners =====
document.addEventListener('DOMContentLoaded', init);
uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', handleDragOver);
uploadBox.addEventListener('dragleave', handleDragLeave);
uploadBox.addEventListener('drop', handleDrop);
imageInput.addEventListener('change', handleImageSelect);
solveBtn.addEventListener('click', solveSudoku);

// ===== Initialization =====
function init() {
    console.log('Sudoku Solver App Initialized');
}

// ===== File Upload Handling =====
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadBox.style.borderColor = '#00ff88';
    uploadBox.style.background = 'linear-gradient(135deg, rgba(0, 212, 255, 0.15), rgba(0, 255, 136, 0.15))';
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadBox.style.borderColor = '#00d4ff';
    uploadBox.style.background = 'linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 255, 136, 0.05))';
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadBox.style.borderColor = '#00d4ff';
    uploadBox.style.background = 'linear-gradient(135deg, rgba(0, 212, 255, 0.05), rgba(0, 255, 136, 0.05))';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleImageSelect();
    }
}

function handleImageSelect(e) {
    const file = imageInput.files[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showMessage('Please select a valid image file', 'error');
        return;
    }
    
    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        showMessage('File size exceeds 10MB limit', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (event) => {
        selectedImage = event.target.result;
        updateUploadBox(file.name);
        solveBtn.disabled = false;
        hideMessage();
        hideResults();
    };
    reader.onerror = () => {
        showMessage('Error reading file', 'error');
    };
    reader.readAsDataURL(file);
}

function updateUploadBox(filename) {
    uploadBox.innerHTML = `
        <div class="upload-icon">✅</div>
        <h3>Image Selected</h3>
        <p>${filename}</p>
    `;
}

// ===== Sudoku Solving =====
async function solveSudoku() {
    if (!selectedImage) {
        showMessage('Please select an image first', 'error');
        return;
    }
    
    try {
        // Show loading
        showLoading(true);
        hideMessage();
        hideResults();
        
        // Prepare form data
        const formData = new FormData();
        const blob = await fetch(selectedImage).then(r => r.blob());
        formData.append('image', blob, 'sudoku.jpg');
        
        // Send to backend
        console.log('Sending image to backend...');
        const response = await fetch('/api/solve', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            showMessage('✓ Sudoku solved successfully!', 'success');
            displayResults(result);
            lastResult = result;
        } else {
            showMessage(`✗ ${result.message}`, 'error');
            // Still display original image if available
            if (result.original_image) {
                displayPartialResults(result);
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        showLoading(false);
    }
}

// ===== Display Results =====
function displayResults(result) {
    // Display images
    document.getElementById('originalImage').src = result.original_image;
    document.getElementById('solvedImage').src = result.solved_image;
    
    // Display stats
    const difficulty = result.difficulty;
    let difficultyLabel = 'Easy';
    if (difficulty < 20) difficultyLabel = 'Hard';
    else if (difficulty < 35) difficultyLabel = 'Medium';
    
    document.getElementById('difficultyValue').textContent = `${difficulty}/81 (${difficultyLabel})`;
    document.getElementById('statusValue').textContent = result.success ? '✓ SOLVED' : '✗ FAILED';
    document.getElementById('stepsValue').textContent = Object.values(result.steps).filter(s => s === 'Success').length + '/5';
    
    // Display grids
    displayGrid('detectedGridDisplay', result.detected_grid);
    displayGrid('solvedGridDisplay', result.solved_grid, result.detected_grid);
    
    // Show results section
    showResults(true);
}

function displayPartialResults(result) {
    if (result.original_image) {
        document.getElementById('originalImage').src = result.original_image;
    }
    showResults(true);
}

function displayGrid(elementId, grid, baseGrid = null) {
    const container = document.getElementById(elementId);
    container.innerHTML = '';
    
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            const cell = document.createElement('div');
            const value = grid[i][j];
            const baseValue = baseGrid ? baseGrid[i][j] : value;
            
            cell.className = 'grid-cell';
            if (value === 0) {
                cell.textContent = '·';
                cell.classList.add('empty');
            } else {
                cell.textContent = value;
                if (baseValue === 0) {
                    cell.classList.add('solved');
                }
            }
            
            // Stagger animation
            cell.style.animationDelay = `${(i + j) * 0.02}s`;
            
            container.appendChild(cell);
        }
    }
}

// ===== UI State Management =====
function showLoading(show) {
    loadingContainer.style.display = show ? 'flex' : 'none';
    if (show) {
        // Animate progress bar
        const progressFill = document.querySelector('.progress-fill');
        progressFill.style.animation = 'none';
        setTimeout(() => {
            progressFill.style.animation = '';
        }, 10);
    }
}

function showResults(show) {
    resultsSection.style.display = show ? 'block' : 'none';
    if (show) {
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
}

function hideResults() {
    showResults(false);
}

function showMessage(message, type = 'info') {
    messageBox.textContent = message;
    messageBox.className = `message ${type}`;
    messageContainer.style.display = 'flex';
}

function hideMessage() {
    messageContainer.style.display = 'none';
}

// ===== Utility Functions =====
async function downloadResult(type = 'solved') {
    if (!lastResult) {
        showMessage('No result to download', 'error');
        return;
    }
    
    const imageData = type === 'solved' ? lastResult.solved_image : lastResult.original_image;
    const link = document.createElement('a');
    link.href = imageData;
    link.download = `sudoku-${type}-${new Date().getTime()}.jpg`;
    link.click();
}

// ===== Advanced Features =====
function analyzeImage() {
    if (!lastResult) {
        showMessage('No result available', 'error');
        return;
    }
    
    const analysis = {
        filled_cells: lastResult.difficulty,
        empty_cells: 81 - lastResult.difficulty,
        status: lastResult.success ? 'Solved' : 'Failed',
        steps_completed: Object.values(lastResult.steps).filter(s => s === 'Success').length
    };
    
    console.table(analysis);
    alert(JSON.stringify(analysis, null, 2));
}

// ===== Keyboard Shortcuts =====
document.addEventListener('keydown', (e) => {
    // Ctrl+V: Paste image
    if (e.ctrlKey && e.key === 'v') {
        navigator.clipboard.read().then(items => {
            for (let item of items) {
                if (item.type.indexOf('image') > -1) {
                    item.getType('image/png').then(blob => {
                        const dataTransfer = new DataTransfer();
                        dataTransfer.items.add(new File([blob], 'pasted.png', { type: 'image/png' }));
                        imageInput.files = dataTransfer.files;
                        handleImageSelect();
                    });
                }
            }
        }).catch(err => console.log('Paste error:', err));
    }
    
    // Enter: Solve
    if (e.key === 'Enter' && solveBtn.disabled === false) {
        solveSudoku();
    }
});

// ===== Theme Toggle (Optional) =====
function toggleTheme() {
    const root = document.documentElement;
    const isDark = root.style.getPropertyValue('--dark-bg') === '#0a0e27';
    
    if (isDark) {
        root.style.setProperty('--dark-bg', '#ffffff');
        root.style.setProperty('--dark-bg-secondary', '#f5f5f5');
        root.style.setProperty('--text-primary', '#000000');
        root.style.setProperty('--text-secondary', '#666666');
    } else {
        root.style.setProperty('--dark-bg', '#0a0e27');
        root.style.setProperty('--dark-bg-secondary', '#1a1f3a');
        root.style.setProperty('--text-primary', '#ffffff');
        root.style.setProperty('--text-secondary', '#b0b9c1');
    }
}

// ===== API Health Check =====
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('Health check:', data);
        return data.status === 'ok';
    } catch (error) {
        console.error('Health check failed:', error);
        return false;
    }
}

// ===== Performance Monitoring =====
window.addEventListener('load', () => {
    const perfData = window.performance.timing;
    const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
    console.log('Page load time:', pageLoadTime + 'ms');
    
    // Log performance metrics
    if (window.performance && window.performance.measure) {
        console.log('Performance metrics available');
    }
});

// ===== Error Handling =====
window.addEventListener('error', (e) => {
    console.error('Unhandled error:', e.error);
    showMessage('An unexpected error occurred', 'error');
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
    showMessage('An unexpected error occurred', 'error');
});

// ===== Console Logging =====
console.log('%c🧩 Sudoku Solver AI - Frontend', 
    'color: #00d4ff; font-size: 14px; font-weight: bold;');
console.log('%cReady to solve!', 
    'color: #00ff88; font-size: 12px;');
