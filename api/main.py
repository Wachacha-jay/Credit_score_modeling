"""
FastAPI Application for Loan Approval Prediction
"""
import sys
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    LoanApplicationRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.utils.logger import logger
from src.utils.exception import PredictionException


# Initialize FastAPI app
app = FastAPI(
    title="Loan Approval Prediction API",
    description="API for predicting loan approval status using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction pipeline globally
prediction_pipeline = None


@app.on_event("startup")
async def startup_event():
    """Load model and artifacts on startup"""
    global prediction_pipeline
    try:
        logger.info("Loading model and artifacts...")
        prediction_pipeline = PredictionPipeline()
        prediction_pipeline.load_artifacts()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        prediction_pipeline = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Loan Approval Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if prediction_pipeline is not None else "unhealthy",
        "model_loaded": prediction_pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
async def predict_loan_approval(request: LoanApplicationRequest):
    """
    Predict loan approval status for a single application
    
    Args:
        request: Loan application details
        
    Returns:
        Prediction result with probability
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        logger.info("Received prediction request")
        
        # Convert request to dictionary
        input_data = request.dict()
        
        # Make prediction
        result = prediction_pipeline.predict(input_data)
        
        response = {
            "success": True,
            "prediction": result['prediction'],
            "prediction_code": result['prediction_code'],
            "probability": result.get('probability'),
            "message": "Prediction completed successfully"
        }
        
        logger.info(f"Prediction successful: {result['prediction']}")
        return response
        
    except PredictionException as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict loan approval status for multiple applications
    
    Args:
        request: List of loan applications
        
    Returns:
        Batch prediction results
    """
    try:
        if prediction_pipeline is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please try again later."
            )
        
        logger.info(f"Received batch prediction request for {len(request.applications)} applications")
        
        # Convert requests to dictionaries
        input_data_list = [app.dict() for app in request.applications]
        
        # Make batch predictions
        results = prediction_pipeline.predict_batch(input_data_list)
        
        response = {
            "success": True,
            "total_applications": len(results),
            "predictions": results,
            "message": f"Batch prediction completed for {len(results)} applications"
        }
        
        logger.info(f"Batch prediction successful for {len(results)} applications")
        return response
        
    except PredictionException as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception handler caught: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )