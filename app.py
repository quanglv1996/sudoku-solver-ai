"""
app.py

Flask Backend API for Sudoku Solver Web Application
Kết nối frontend với sudoku solver pipeline
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "src"))

from pipeline import SudokuSolverPipeline

# =====================================================================
# Flask Setup
# =====================================================================

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# Global Variables
# =====================================================================

pipeline = None
pipeline_loaded = False
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# =====================================================================
# Initialization
# =====================================================================

def get_pipeline():
    """Lazy load pipeline - only init when first needed"""
    global pipeline, pipeline_loaded
    
    if pipeline is not None:
        return pipeline
    
    if pipeline_loaded:
        return None  # Already tried and failed
    
    return init_pipeline()

def init_pipeline():
    """Initialize sudoku solver pipeline"""
    global pipeline, pipeline_loaded
    try:
        # Convert to Path object for cross-platform compatibility
        point_model = Path(__file__).parent / "backend" / "weights" / "best.pt"
        
        # Check if model file exists
        if not point_model.exists():
            logger.error(f"❌ Model file not found at: {point_model}")
            logger.error("Please download the YOLO model and place it at: backend/weights/best.pt")
            pipeline_loaded = True
            return None
        
        logger.info(f"📍 Loading model from: {point_model}")
        logger.info("⏳ Initializing pipeline (this may take a minute on first run)...")
        logger.info("   - Loading YOLO point detection model...")
        
        pipeline = SudokuSolverPipeline(
            point_detection_model=str(point_model),
            cell_size=64,
            device="cpu",  # Sử dụng CPU để tương thích tốt hơn
            use_easyocr=True,
            conf_threshold=0.15,  # 🔍 Giảm threshold để nhận diện được 4, 1, 7
            save_debug=True,  # Bật lưu ảnh debug
            debug_dir="debug_output"
        )
        pipeline.set_verbose(True)
        logger.info("✅ Pipeline initialized successfully!")
        pipeline_loaded = True
        return pipeline
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline!")
        logger.error(f"   Error type: {type(e).__name__}")
        logger.error(f"   Error message: {str(e)}")
        logger.error(f"   Full traceback:", exc_info=True)
        pipeline_loaded = True
        return None

# =====================================================================
# Utility Functions
# =====================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_image_from_file(file):
    """Load image from Flask file object"""
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    
    nparr = np.frombuffer(in_memory_file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

def draw_solved_grid(image, solved_grid, detected_grid=None, transform_matrix=None):
    """
    Vẽ các chữ số giải được lên ảnh gốc
    Chỉ vẽ những chữ số mới được giải (detected = 0, solved > 0)
    
    Args:
        image: Ảnh gốc
        solved_grid: Ma trận sudoku đã giải (9x9)
        detected_grid: Ma trận sudoku detect được (9x9), optional
        transform_matrix: Perspective transform matrix (3x3) từ ảnh gốc -> align
    """
    image_copy = image.copy()
    h, w = image_copy.shape[:2]
    
    # Nếu không có detected_grid, vẽ tất cả chữ số
    if detected_grid is None:
        detected_grid = np.zeros_like(solved_grid)
    
    # Kích thước ảnh align (576x576)
    aligned_size = 576
    
    # Kiểm tra transform_matrix hợp lệ
    use_transform = False
    transform_inv = None
    if transform_matrix is not None:
        try:
            # Đảm bảo là numpy array shape (3, 3)
            if isinstance(transform_matrix, np.ndarray) and transform_matrix.shape == (3, 3):
                # Inverse ma trận perspective (3x3)
                transform_inv = np.linalg.inv(transform_matrix)
                use_transform = True
            else:
                print(f"⚠️  Transform matrix shape {transform_matrix.shape if isinstance(transform_matrix, np.ndarray) else 'unknown'} != (3, 3), using scale fallback")
        except Exception as e:
            print(f"⚠️  Error inverting transform matrix: {e}, using scale fallback")
    
    # Vẽ các chữ số giải được
    for row in range(9):
        for col in range(9):
            detected_digit = int(detected_grid[row, col]) if detected_grid is not None else 0
            solved_digit = int(solved_grid[row, col])
            
            # Chỉ vẽ những chữ số được giải (chưa có trong detected)
            if solved_digit > 0 and detected_digit == 0:
                # Tọa độ trong ảnh align (64x64 mỗi ô)
                cell_size_aligned = aligned_size / 9
                x_aligned = (col + 0.5) * cell_size_aligned
                y_aligned = (row + 0.5) * cell_size_aligned
                
                # Map về tọa độ ảnh gốc
                if use_transform and transform_inv is not None:
                    # Dùng inverse perspective transformation
                    try:
                        # Tạo point trong homogeneous coordinates
                        pt_aligned = np.array([[x_aligned, y_aligned, 1.0]], dtype=np.float32).T
                        
                        # Apply inverse transform: pt_original = inv(M) @ pt_aligned
                        pt_original_homo = transform_inv @ pt_aligned
                        
                        # Chia cho coordinate thứ 3 để normalize (perspective division)
                        x = int(pt_original_homo[0, 0] / pt_original_homo[2, 0])
                        y = int(pt_original_homo[1, 0] / pt_original_homo[2, 0])
                    except Exception as e:
                        print(f"⚠️  Error applying transform: {e}")
                        # Fallback: scale theo tỉ lệ
                        scale_x = w / aligned_size
                        scale_y = h / aligned_size
                        x = int(x_aligned * scale_x)
                        y = int(y_aligned * scale_y)
                else:
                    # Fallback: scale theo tỉ lệ kích thước
                    scale_x = w / aligned_size
                    scale_y = h / aligned_size
                    x = int(x_aligned * scale_x)
                    y = int(y_aligned * scale_y)
                
                # Kiểm tra tọa độ hợp lệ
                if 0 <= x < w and 0 <= y < h:
                    # Tính font size dựa trên ảnh gốc
                    cell_w = w / 9
                    cell_h = h / 9
                    
                    text = str(solved_digit)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = max(0.6, (cell_h / 64) * 1.2)
                    thickness = max(2, int((cell_h / 64) * 2))
                    
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                    
                    # Căn giữa
                    text_x = x - text_size[0] // 2
                    text_y = y + text_size[1] // 2
                    
                    # Vẽ background tối để chữ số rõ
                    padding = 4
                    cv2.rectangle(
                        image_copy,
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (30, 30, 30),  # Background xám đen
                        -1
                    )
                    
                    # Vẽ chữ số màu xanh lá
                    cv2.putText(
                        image_copy, text, (text_x, text_y),
                        font, font_scale, (0, 255, 100), thickness
                    )
    
    return image_copy

def image_to_base64(image):
    """Convert image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64

def resize_image(image, max_size=800):
    """Resize image nếu quá lớn"""
    h, w = image.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

# =====================================================================
# Routes
# =====================================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/solve', methods=['POST'])
def solve_sudoku():
    """
    Solve sudoku from uploaded image
    
    Request: multipart/form-data with 'image' file
    Response: JSON with:
    - success: bool
    - message: string
    - original_image: base64
    - solved_image: base64
    - detected_grid: 2D array
    - solved_grid: 2D array
    - steps: dict
    """
    try:
        # Check if request has file part
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            }), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Check file size
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({
                'success': False,
                'message': f'File too large. Maximum size: 10MB'
            }), 400
        
        # Load image
        logger.info(f"Loading image: {file.filename}")
        image = load_image_from_file(file)
        
        if image is None:
            return jsonify({
                'success': False,
                'message': 'Failed to load image'
            }), 400
        
        # Resize if too large
        image = resize_image(image, max_size=800)
        
        # Save original
        original_image_base64 = image_to_base64(image)
        
        # Run pipeline
        logger.info("Running sudoku solver pipeline...")
        
        # Get/initialize pipeline on first use
        proc_pipeline = get_pipeline()
        if proc_pipeline is None:
            logger.error("❌ Pipeline initialization failed!")
            logger.error("Check the error messages above for details")
            return jsonify({
                'success': False,
                'message': 'Pipeline not initialized. Check server logs for details.',
                'original_image': f'data:image/jpeg;base64,{original_image_base64}'
            }), 500
        
        logger.info("🔍 Running sudoku solver pipeline...")
        result = proc_pipeline.run(image)
        
        if not result.get('success'):
            return jsonify({
                'success': False,
                'message': 'Failed to solve sudoku',
                'original_image': f'data:image/jpeg;base64,{original_image_base64}'
            }), 200
        
        # Draw results on image
        transform_matrix = result.get('transform_matrix', None)
        solved_image = draw_solved_grid(
            image, 
            result['solved_grid'], 
            result['detected_grid'],
            transform_matrix
        )
        solved_image_base64 = image_to_base64(solved_image)
        
        # Prepare response
        response = {
            'success': True,
            'message': 'Sudoku solved successfully!',
            'original_image': f'data:image/jpeg;base64,{original_image_base64}',
            'solved_image': f'data:image/jpeg;base64,{solved_image_base64}',
            'detected_grid': result['detected_grid'].tolist(),
            'solved_grid': result['solved_grid'].tolist(),
            'steps': result['steps'],
            'difficulty': int(np.sum(result['detected_grid'] != 0))
        }
        
        logger.info(f"Successfully solved sudoku. Difficulty: {response['difficulty']}")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Error in /api/solve: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pipeline_loaded': pipeline is not None,
        'pipeline_initialized_attempt': pipeline_loaded,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

# =====================================================================
# Error Handlers
# =====================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# =====================================================================
# Main
# =====================================================================

if __name__ == '__main__':
    # Pipeline will be loaded on first use (lazy loading)
    # This allows Flask to start quickly without waiting for model downloads
    logger.info("Starting Flask server...")
    logger.info("Pipeline will be loaded on first API request")
    logger.info("Opening http://localhost:5000")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
