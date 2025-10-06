"""
Prediction Pipeline
Handles inference for new loan applications
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

from src.utils.logger import logger
from src.utils.exception import PredictionException
from src.utils.common import load_object


class PredictionPipeline:
    """
    Pipeline for making predictions on new data
    """
    
    def __init__(self, model_path: str = "data/artifacts/models/best_model.pkl"):
        """
        Initialize prediction pipeline
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = Path(model_path)
        self.scaler_path = Path("data/artifacts/scaler.pkl")
        self.encoder_path = Path("data/artifacts/encoder.pkl")
        self.preprocessor_path = Path("data/artifacts/preprocessor.pkl")
        
        self.model = None
        self.scaler = None
        self.encoders = None
        self.preprocessor_info = None
        
        logger.info("PredictionPipeline initialized")
    
    def load_artifacts(self):
        """Load all required artifacts"""
        try:
            logger.info("Loading artifacts...")
            
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at: {self.model_path}")
            self.model = load_object(self.model_path)
            logger.info("Model loaded")
            
            # Load scaler
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at: {self.scaler_path}")
            self.scaler = load_object(self.scaler_path)
            logger.info("Scaler loaded")
            
            # Load encoders
            if not self.encoder_path.exists():
                raise FileNotFoundError(f"Encoders not found at: {self.encoder_path}")
            self.encoders = load_object(self.encoder_path)
            logger.info("Encoders loaded")
            
            # Load preprocessor info
            if not self.preprocessor_path.exists():
                raise FileNotFoundError(
                    f"Preprocessor info not found at: {self.preprocessor_path}"
                )
            self.preprocessor_info = load_object(self.preprocessor_path)
            logger.info("Preprocessor info loaded")
            
            logger.info("All artifacts loaded successfully")
            
        except Exception as e:
            raise PredictionException(e, sys)
    
    def preprocess_input(self, input_data: Dict) -> np.ndarray:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Preprocessed numpy array
        """
        try:
            logger.info("Preprocessing input data...")
            
            # Create DataFrame from input
            df = pd.DataFrame([input_data])
            
            # Encode categorical features
            for col in self.preprocessor_info['categorical_features']:
                if col in df.columns and col in self.encoders:
                    encoder = self.encoders[col]
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except ValueError as e:
                        logger.warning(
                            f"Unknown category for {col}, using default encoding"
                        )
                        df[col] = 0
            
            # Ensure correct feature order
            feature_names = self.preprocessor_info['feature_names']
            df = df[feature_names]
            
            # Scale features
            scaled_data = self.scaler.transform(df.values)
            
            logger.info("Input data preprocessed successfully")
            return scaled_data
            
        except Exception as e:
            raise PredictionException(e, sys)
    
    def predict(self, input_data: Dict) -> Dict:
        """
        Make prediction for single input
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load artifacts if not already loaded
            if self.model is None:
                self.load_artifacts()
            
            logger.info("Making prediction...")
            
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Get probability if available
            probability = None
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(processed_data)[0]
                probability = {
                    'rejected': float(proba[0]),
                    'approved': float(proba[1])
                }
            
            # Decode prediction
            target_encoder = self.encoders.get(
                self.preprocessor_info['target_column']
            )
            if target_encoder:
                prediction_label = target_encoder.inverse_transform([prediction])[0]
            else:
                prediction_label = "Approved" if prediction == 1 else "Rejected"
            
            result = {
                'prediction': prediction_label,
                'prediction_code': int(prediction),
                'probability': probability,
                'input_data': input_data
            }
            
            logger.info(f"Prediction: {prediction_label}")
            return result
            
        except Exception as e:
            raise PredictionException(e, sys)
    
    def predict_batch(self, input_data_list: List[Dict]) -> List[Dict]:
        """
        Make predictions for multiple inputs
        
        Args:
            input_data_list: List of input dictionaries
            
        Returns:
            List of prediction results
        """
        try:
            logger.info(f"Making batch predictions for {len(input_data_list)} samples...")
            
            results = []
            for input_data in input_data_list:
                result = self.predict(input_data)
                results.append(result)
            
            logger.info("Batch predictions completed")
            return results
            
        except Exception as e:
            raise PredictionException(e, sys)


class CustomData:
    """
    Class for handling custom input data
    """
    
    def __init__(
        self,
        no_of_dependents: int,
        education: str,
        self_employed: str,
        income_annum: float,
        loan_amount: float,
        loan_term: int,
        credit_score: int,
        residential_assets_value: float,
        commercial_assets_value: float,
        luxury_assets_value: float,
        bank_asset_value: float
    ):
        """
        Initialize custom data
        
        Args:
            All loan application features
        """
        self.no_of_dependents = no_of_dependents
        self.education = education
        self.self_employed = self_employed
        self.income_annum = income_annum
        self.loan_amount = loan_amount
        self.loan_term = loan_term
        self.credit_score = credit_score
        self.residential_assets_value = residential_assets_value
        self.commercial_assets_value = commercial_assets_value
        self.luxury_assets_value = luxury_assets_value
        self.bank_asset_value = bank_asset_value
    
    def get_data_as_dict(self) -> Dict:
        """
        Convert data to dictionary
        
        Returns:
            Dictionary with all features
        """
        return {
            'no_of_dependents': self.no_of_dependents,
            'education': self.education,
            'self_employed': self.self_employed,
            'income_annum': self.income_annum,
            'loan_amount': self.loan_amount,
            'loan_term': self.loan_term,
            'credit_score': self.credit_score,
            'residential_assets_value': self.residential_assets_value,
            'commercial_assets_value': self.commercial_assets_value,
            'luxury_assets_value': self.luxury_assets_value,
            'bank_asset_value': self.bank_asset_value
        }


if __name__ == "__main__":
    # Example usage
    try:
        # Create sample data
        sample_data = CustomData(
            no_of_dependents=2,
            education="Graduate",
            self_employed="No",
            income_annum=5000000,
            loan_amount=1500000,
            loan_term=12,
            credit_score=750,
            residential_assets_value=8000000,
            commercial_assets_value=0,
            luxury_assets_value=1000000,
            bank_asset_value=500000
        )
        
        # Make prediction
        pipeline = PredictionPipeline()
        result = pipeline.predict(sample_data.get_data_as_dict())
        
        logger.info(f"Prediction Result: {result}")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise e