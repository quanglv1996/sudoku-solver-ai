import sys
sys.path.append("./..")
import logging
from pathlib import Path

import cv2
import numpy as np

from src.pipeline.sudoku_pipeline import SudokuPipeline


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/sudoku_solver.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def process_single_image(
    pipeline: SudokuPipeline,
    image_path: str,
    output_dir: str = "output"
):
    """Process single image and save results."""
    logger = logging.getLogger(__name__)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    logger.info(f"Processing: {image_path}")
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return False
        
        success, results = pipeline.process(image)
        
        if not success:
            logger.error(f"Pipeline failed: {results.get('error')}")
            return False
        
        # Save results
        base_name = Path(image_path).stem
        
        if results['viz_original'] is not None:
            cv2.imwrite(
                f"{output_dir}/{base_name}_original.jpg",
                results['viz_original']
            )
        
        if results['viz_aligned'] is not None:
            cv2.imwrite(
                f"{output_dir}/{base_name}_aligned.jpg",
                results['viz_aligned']
            )
        
        if results['input_grid'] is not None:
            logger.info(f"Input Grid:\n{results['input_grid']}")
        
        if results['solution_grid'] is not None:
            logger.info(f"Solution:\n{results['solution_grid']}")
        
        logger.info(f"✓ Completed: {image_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}", exc_info=True)
        return False


def process_directory(
    pipeline: SudokuPipeline,
    input_dir: str,
    output_dir: str = "output"
):
    """Process all images in directory."""
    logger = logging.getLogger(__name__)
    
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return False
    
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    if not image_files:
        logger.warning(f"No images found in: {input_dir}")
        return False
    
    logger.info(f"Found {len(image_files)} images")
    
    success_count = 0
    for image_file in image_files:
        if process_single_image(pipeline, str(image_file), output_dir):
            success_count += 1
    
    logger.info(f"Processed {success_count}/{len(image_files)} images successfully")
    return success_count > 0


def main():
    """Main entry point."""
    logger = setup_logging()
    
    # Configuration
    model_path = r"D:\Projects-Persional\sudoku-solver-ai\src\weights\best.pt"
    device = "cuda"
    input_dir = r"D:\Projects-Persional\sudoku-solver-ai\input"
    output_dir = "output"
    
    # Check model exists
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    logger.info("Initializing pipeline...")
    pipeline = SudokuPipeline(
        point_model_path=model_path,
        device=device,
        use_gpu=True
    )
    
    logger.info("Pipeline ready")
    
    # Process directory
    if not process_directory(pipeline, input_dir, output_dir):
        return 1
    
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
