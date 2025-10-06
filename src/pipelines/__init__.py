"""
Training and prediction pipelines
"""
from src.pipelines.training_pipeline import TrainingPipeline
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

__all__ = [
    'TrainingPipeline',
    'PredictionPipeline',
    'CustomData'
]