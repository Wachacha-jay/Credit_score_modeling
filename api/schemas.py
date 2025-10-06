"""
API Schemas
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict


class LoanApplicationRequest(BaseModel):
    """Schema for loan application request"""
    
    no_of_dependents: int = Field(
        ...,
        ge=0,
        le=10,
        description="Number of dependents"
    )
    education: str = Field(
        ...,
        description="Education level (Graduate/Not Graduate)"
    )
    self_employed: str = Field(
        ...,
        description="Self employment status (Yes/No)"
    )
    income_annum: float = Field(
        ...,
        gt=0,
        description="Annual income"
    )
    loan_amount: float = Field(
        ...,
        gt=0,
        description="Requested loan amount"
    )
    loan_term: int = Field(
        ...,
        gt=0,
        le=360,
        description="Loan term in months"
    )
    credit_score: int = Field(
        ...,
        ge=300,
        le=900,
        description="Credit score"
    )
    residential_assets_value: float = Field(
        ...,
        ge=0,
        description="Value of residential assets"
    )
    commercial_assets_value: float = Field(
        ...,
        ge=0,
        description="Value of commercial assets"
    )
    luxury_assets_value: float = Field(
        ...,
        ge=0,
        description="Value of luxury assets"
    )
    bank_asset_value: float = Field(
        ...,
        ge=0,
        description="Value of bank assets"
    )
    
    @validator('education')
    def validate_education(cls, v):
        """Validate education field"""
        allowed = ['Graduate', 'Not Graduate']
        if v not in allowed:
            raise ValueError(f"Education must be one of {allowed}")
        return v
    
    @validator('self_employed')
    def validate_self_employed(cls, v):
        """Validate self_employed field"""
        allowed = ['Yes', 'No']
        if v not in allowed:
            raise ValueError(f"Self employed must be one of {allowed}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "no_of_dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "income_annum": 5000000,
                "loan_amount": 1500000,
                "loan_term": 12,
                "credit_score": 750,
                "residential_assets_value": 8000000,
                "commercial_assets_value": 0,
                "luxury_assets_value": 1000000,
                "bank_asset_value": 500000
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    
    success: bool
    prediction: str
    prediction_code: int
    probability: Optional[Dict[str, float]] = None
    message: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "prediction": "Approved",
                "prediction_code": 1,
                "probability": {
                    "rejected": 0.25,
                    "approved": 0.75
                },
                "message": "Prediction completed successfully"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    
    applications: list[LoanApplicationRequest]
    
    class Config:
        schema_extra = {
            "example": {
                "applications": [
                    {
                        "no_of_dependents": 2,
                        "education": "Graduate",
                        "self_employed": "No",
                        "income_annum": 5000000,
                        "loan_amount": 1500000,
                        "loan_term": 12,
                        "credit_score": 750,
                        "residential_assets_value": 8000000,
                        "commercial_assets_value": 0,
                        "luxury_assets_value": 1000000,
                        "bank_asset_value": 500000
                    }
                ]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    
    success: bool
    total_applications: int
    predictions: list[Dict]
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for health check response"""
    
    status: str
    model_loaded: bool
    timestamp: str


class ErrorResponse(BaseModel):
    """Schema for error response"""
    
    success: bool = False
    error: str
    detail: Optional[str] = None