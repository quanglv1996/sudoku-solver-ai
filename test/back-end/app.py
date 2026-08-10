import sys
from pathlib import Path
import base64
import logging

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.pipeline.sudoku_pipeline import SudokuPipeline


app = FastAPI(title="Sudoku Solver API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

pipeline = None


def init_pipeline():
    global pipeline
    model_path = "src/weights/testttt_1_best.pt"
    
    if not Path(model_path).exists():
        raise RuntimeError(f"Model not found: {model_path}")
    
    pipeline = SudokuPipeline(
        point_model_path=model_path,
        device="cuda",
        use_gpu=True
    )
    logger.info("Pipeline initialized")


@app.on_event("startup")
async def startup_event():
    init_pipeline()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/info")
async def info():
    return {
        "name": "Sudoku Solver API",
        "version": "1.0.0",
        "description": "Solve sudoku puzzles from images using AI"
    }


@app.post("/solve")
async def solve_sudoku(file: UploadFile = File(...)):
    """
    Upload sudoku image and get solution.
    
    Returns:
        - input_grid: Recognized sudoku grid
        - solution: Solved sudoku
        - visualization_original: Image with detected corners
        - visualization_aligned: Aligned grid with solution
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image"
            )
        
        logger.info(f"Processing image: {file.filename}")
        
        success, results = pipeline.process(image)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Processing failed: {results.get('error')}"
            )
        
        # Convert results to response format
        response = {
            "success": True,
            "input_grid": results["input_grid"].tolist() if results["input_grid"] is not None else None,
            "solution": results["solution_grid"].tolist() if results["solution_grid"] is not None else None,
        }
        
        # Encode visualizations as base64
        if results["viz_original"] is not None:
            _, buffer = cv2.imencode(".jpg", results["viz_original"])
            response["visualization_original"] = base64.b64encode(buffer).decode()
        
        if results["viz_aligned"] is not None:
            _, buffer = cv2.imencode(".jpg", results["viz_aligned"])
            response["visualization_aligned"] = base64.b64encode(buffer).decode()
        
        logger.info("Sudoku solved successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/solve-base64")
async def solve_sudoku_base64(data: dict):
    """
    Solve sudoku from base64 encoded image.
    
    Request:
        {
            "image": "base64_encoded_image"
        }
    
    Returns:
        Same as /solve endpoint
    """
    try:
        if "image" not in data:
            raise HTTPException(
                status_code=400,
                detail="Missing 'image' field"
            )
        
        image_data = base64.b64decode(data["image"])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image"
            )
        
        logger.info("Processing base64 image")
        
        success, results = pipeline.process(image)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Processing failed: {results.get('error')}"
            )
        
        response = {
            "success": True,
            "input_grid": results["input_grid"].tolist() if results["input_grid"] is not None else None,
            "solution": results["solution_grid"].tolist() if results["solution_grid"] is not None else None,
        }
        
        if results["viz_original"] is not None:
            _, buffer = cv2.imencode(".jpg", results["viz_original"])
            response["visualization_original"] = base64.b64encode(buffer).decode()
        
        if results["viz_aligned"] is not None:
            _, buffer = cv2.imencode(".jpg", results["viz_aligned"])
            response["visualization_aligned"] = base64.b64encode(buffer).decode()
        
        logger.info("Sudoku solved successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
