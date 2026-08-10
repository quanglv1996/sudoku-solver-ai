"""
sudoku_pipeline.py

Complete sudoku solving pipeline
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from ..ai.point_detection import PointDetection
from ..ai.preprocessing import PreProcessing, PreProcessingResult
from ..ai.ocr_digist import OCRDigist
from ..ai.sudoku_solve import SudokuSolver


class SudokuPipeline:
    """
    Complete sudoku solving pipeline:
    Image → Point Detection → Align → OCR → Solve → Visualize
    """

    def __init__(
        self,
        point_model_path: str,
        device: str = "cuda",
        use_gpu: bool = True
    ):
        """
        Initialize pipeline.
        
        Args:
            point_model_path: Path to YOLO point detection model
            device: Device ("cuda" or "cpu")
            use_gpu: Use GPU for OCR
        """
        self.point_detector = PointDetection(
            model_path=point_model_path,
            device=device,
            conf=0.25
        )
        
        self.preprocessor = PreProcessing(cell_size=64)
        self.ocr = OCRDigist(lang="en", use_gpu=use_gpu)
        self.solver = SudokuSolver()
        
        self.last_image = None
        self.last_aligned_image = None
        self.last_grid = None
        self.last_solution = None

    def step1_detect_points(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Step 1: Detect 4 corner points.
        
        Args:
            image: Input image
            
        Returns:
            Sorted corner points (4, 2) [TL, TR, BR, BL] or None if detection fails
        """
        self.last_image = image.copy()
        
        try:
            # Returns 4 sorted points directly: [TL, TR, BR, BL]
            points = self.point_detector.predict(image)
            return points
        except ValueError as e:
            print(f"Point detection error: {e}")
            return None

    def step2_preprocess(
        self,
        image: np.ndarray,
        corner_points: np.ndarray
    ) -> Optional[PreProcessingResult]:
        """
        Step 2: Align image and extract 9x9 grid.
        
        Args:
            image: Input image
            corner_points: 4 corner points (4, 2) [TL, TR, BR, BL]
            
        Returns:
            PreProcessingResult or None if fails
        """
        try:
            result = self.preprocessor.preprocess(image, corner_points)
            self.last_aligned_image = result.aligned_image.copy()
            return result
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

    def step3_ocr(
        self,
        aligned_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Step 3: OCR on aligned sudoku image.
        
        Args:
            aligned_image: Aligned sudoku image (450x450)
            
        Returns:
            9x9 grid with recognized digits
        """
        try:
            grid = self.ocr.recognize_grid(aligned_image)
            self.last_grid = grid.copy()
            return grid
        except Exception as e:
            print(f"OCR error: {e}")
            return None

    def step4_solve(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Step 4: Solve sudoku.
        
        Args:
            grid: Input 9x9 grid
            
        Returns:
            Solved grid
        """
        if not self.solver.validate_input(grid):
            print("Invalid sudoku grid")
            return None
        
        solution = self.solver.solve_sudoku(grid)
        self.last_solution = solution.copy()
        return solution

    def step5_visualize(
        self,
        original_image: np.ndarray,
        aligned_image: np.ndarray,
        input_grid: np.ndarray,
        solution_grid: np.ndarray,
        detection_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 5: Visualize results.
        
        Args:
            original_image: Original input image
            aligned_image: Aligned sudoku image
            input_grid: OCR recognized grid
            solution_grid: Solved grid
            detection_points: Detected corner points
            
        Returns:
            (visualization_image, aligned_with_solution)
        """
        # Visualization 1: Aligned image with solution overlaid
        viz_aligned = self.preprocessor.draw_grid(aligned_image)
        
        # Add solution to aligned image
        cell_size = self.preprocessor.cell_size
        for row in range(9):
            for col in range(9):
                val = solution_grid[row, col]
                
                # Only show solved cells (where input was 0)
                if input_grid[row, col] == 0 and val != 0:
                    x = col * cell_size + cell_size // 2
                    y = row * cell_size + cell_size // 2
                    
                    cv2.putText(
                        viz_aligned,
                        str(val),
                        (x - 10, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),  # Red for solved
                        2
                    )
        
        # Visualization 2: Original image with corner points
        viz_original = original_image.copy()
        
        if detection_points is not None:
            points = self.preprocessor.sort_points(detection_points)
            
            for i, (x, y) in enumerate(points):
                cv2.circle(viz_original, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(
                    viz_original,
                    str(i),
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
        
        return viz_original, viz_aligned

    def process(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Full pipeline processing.
        
        Args:
            image: Input image
            
        Returns:
            (success: bool, results: dict)
        """
        results = {
            'detection_points': None,
            'input_grid': None,
            'solution_grid': None,
            'viz_original': None,
            'viz_aligned': None,
            'error': None
        }
        
        try:
            # Step 1: Detect points
            detection = self.step1_detect_points(image)
            if detection is None:
                results['error'] = "Point detection failed"
                return False, results
            
            results['detection_points'] = detection.points
            
            # Step 2: Preprocess
            preproc = self.step2_preprocess(image, detection.points)
            if preproc is None:
                results['error'] = "Preprocessing failed"
                return False, results
            
            # Step 3: OCR
            grid = self.step3_ocr(preproc.aligned_image)
            if grid is None:
                results['error'] = "OCR failed"
                return False, results
            
            results['input_grid'] = grid
            
            # Step 4: Solve
            solution = self.step4_solve(grid)
            if solution is None:
                results['error'] = "Solver failed"
                return False, results
            
            results['solution_grid'] = solution
            
            # Step 5: Visualize
            viz_orig, viz_align = self.step5_visualize(
                image,
                preproc.aligned_image,
                grid,
                solution,
                detection.points
            )
            
            results['viz_original'] = viz_orig
            results['viz_aligned'] = viz_align
            
            return True, results
            
        except Exception as e:
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
            return False, results

    def display_results(self, results: dict):
        """
        Display results using OpenCV.
        
        Args:
            results: Results dictionary from process()
        """
        if results['viz_original'] is not None:
            cv2.imshow("Original with Points", results['viz_original'])
        
        if results['viz_aligned'] is not None:
            cv2.imshow("Aligned with Solution", results['viz_aligned'])
        
        if results['input_grid'] is not None:
            print("\nInput Grid (OCR):")
            self.solver.print_grid(results['input_grid'])
        
        if results['solution_grid'] is not None:
            print("\nSolved Grid:")
            self.solver.print_grid(results['solution_grid'])
