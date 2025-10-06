"""
Unit tests for prediction pipeline
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData
from src.utils.exception import PredictionException


class TestCustomData:
    """Test CustomData class"""
    
    def test_custom_data_initialization(self):
        """Test CustomData initialization"""
        data = CustomData(
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
        
        assert data.no_of_dependents == 2
        assert data.education == "Graduate"
        assert data.credit_score == 750
    
    def test_get_data_as_dict(self):
        """Test conversion to dictionary"""
        data = CustomData(
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
        
        data_dict = data.get_data_as_dict()
        
        assert isinstance(data_dict, dict)
        assert data_dict['no_of_dependents'] == 2
        assert data_dict['education'] == "Graduate"
        assert len(data_dict) == 11


class TestPredictionPipeline:
    """Test PredictionPipeline class"""
    
    @pytest.fixture
    def mock_artifacts(self, tmp_path):
        """Create mock artifacts"""
        # Mock model
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.array([1]))
        mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
        
        # Mock scaler
        mock_scaler = Mock()
        mock_scaler.transform = Mock(return_value=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]))
        
        # Mock encoders
        mock_encoder = Mock()
        mock_encoder.transform = Mock(return_value=np.array([0]))
        mock_encoder.inverse_transform = Mock(return_value=np.array(["Approved"]))
        
        mock_encoders = {
            'education': mock_encoder,
            'self_employed': mock_encoder,
            'loan_status': mock_encoder
        }
        
        # Mock preprocessor info
        mock_preprocessor_info = {
            'feature_names': [
                'no_of_dependents', 'education', 'self_employed',
                'income_annum', 'loan_amount', 'loan_term',
                'credit_score', 'residential_assets_value',
                'commercial_assets_value', 'luxury_assets_value',
                'bank_asset_value'
            ],
            'numerical_features': ['income_annum', 'loan_amount'],
            'categorical_features': ['education', 'self_employed'],
            'target_column': 'loan_status'
        }
        
        return {
            'model': mock_model,
            'scaler': mock_scaler,
            'encoders': mock_encoders,
            'preprocessor_info': mock_preprocessor_info
        }
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        pipeline = PredictionPipeline()
        
        assert pipeline.model is None
        assert pipeline.scaler is None
        assert pipeline.encoders is None
        assert isinstance(pipeline.model_path, Path)
    
    @patch('src.pipelines.prediction_pipeline.load_object')
    def test_load_artifacts(self, mock_load, mock_artifacts):
        """Test loading artifacts"""
        # Configure mock to return artifacts
        mock_load.side_effect = [
            mock_artifacts['model'],
            mock_artifacts['scaler'],
            mock_artifacts['encoders'],
            mock_artifacts['preprocessor_info']
        ]
        
        pipeline = PredictionPipeline()
        
        # Mock file existence
        with patch.object(Path, 'exists', return_value=True):
            pipeline.load_artifacts()
        
        assert pipeline.model is not None
        assert pipeline.scaler is not None
        assert pipeline.encoders is not None
        assert pipeline.preprocessor_info is not None
    
    def test_preprocess_input(self, mock_artifacts):
        """Test input preprocessing"""
        pipeline = PredictionPipeline()
        pipeline.scaler = mock_artifacts['scaler']
        pipeline.encoders = mock_artifacts['encoders']
        pipeline.preprocessor_info = mock_artifacts['preprocessor_info']
        
        input_data = {
            'no_of_dependents': 2,
            'education': 'Graduate',
            'self_employed': 'No',
            'income_annum': 5000000,
            'loan_amount': 1500000,
            'loan_term': 12,
            'credit_score': 750,
            'residential_assets_value': 8000000,
            'commercial_assets_value': 0,
            'luxury_assets_value': 1000000,
            'bank_asset_value': 500000
        }
        
        result = pipeline.preprocess_input(input_data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
    
    def test_predict(self, mock_artifacts):
        """Test prediction"""
        pipeline = PredictionPipeline()
        pipeline.model = mock_artifacts['model']
        pipeline.scaler = mock_artifacts['scaler']
        pipeline.encoders = mock_artifacts['encoders']
        pipeline.preprocessor_info = mock_artifacts['preprocessor_info']
        
        input_data = {
            'no_of_dependents': 2,
            'education': 'Graduate',
            'self_employed': 'No',
            'income_annum': 5000000,
            'loan_amount': 1500000,
            'loan_term': 12,
            'credit_score': 750,
            'residential_assets_value': 8000000,
            'commercial_assets_value': 0,
            'luxury_assets_value': 1000000,
            'bank_asset_value': 500000
        }
        
        result = pipeline.predict(input_data)
        
        assert isinstance(result, dict)
        assert 'prediction' in result
        assert 'prediction_code' in result
        assert 'probability' in result
        assert result['prediction'] == "Approved"
    
    def test_predict_batch(self, mock_artifacts):
        """Test batch prediction"""
        pipeline = PredictionPipeline()
        pipeline.model = mock_artifacts['model']
        pipeline.scaler = mock_artifacts['scaler']
        pipeline.encoders = mock_artifacts['encoders']
        pipeline.preprocessor_info = mock_artifacts['preprocessor_info']
        
        input_data_list = [
            {
                'no_of_dependents': 2,
                'education': 'Graduate',
                'self_employed': 'No',
                'income_annum': 5000000,
                'loan_amount': 1500000,
                'loan_term': 12,
                'credit_score': 750,
                'residential_assets_value': 8000000,
                'commercial_assets_value': 0,
                'luxury_assets_value': 1000000,
                'bank_asset_value': 500000
            },
            {
                'no_of_dependents': 1,
                'education': 'Not Graduate',
                'self_employed': 'Yes',
                'income_annum': 3000000,
                'loan_amount': 1000000,
                'loan_term': 6,
                'credit_score': 650,
                'residential_assets_value': 5000000,
                'commercial_assets_value': 1000000,
                'luxury_assets_value': 500000,
                'bank_asset_value': 300000
            }
        ]
        
        results = pipeline.predict_batch(input_data_list)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert all('prediction' in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])