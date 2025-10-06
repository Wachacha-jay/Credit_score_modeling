"""
Model Evaluation Component
Handles model evaluation and metrics calculation
"""
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from typing import Dict

from src.utils.logger import logger
from src.utils.exception import ModelEvaluationException
from src.config.configuration import ModelEvaluationConfig


class ModelEvaluation:
    """
    Class for evaluating machine learning models
    """
    
    def __init__(self, config: ModelEvaluationConfig):
        """
        Initialize ModelEvaluation component
        
        Args:
            config: ModelEvaluationConfig object
        """
        self.config = config
        logger.info("ModelEvaluation component initialized")
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        try:
            metrics = {}
            
            # Calculate basic metrics
            if 'accuracy' in self.config.metrics:
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            if 'precision' in self.config.metrics:
                metrics['precision'] = precision_score(
                    y_true, y_pred, average='binary'
                )
            
            if 'recall' in self.config.metrics:
                metrics['recall'] = recall_score(
                    y_true, y_pred, average='binary'
                )
            
            if 'f1_score' in self.config.metrics:
                metrics['f1_score'] = f1_score(
                    y_true, y_pred, average='binary'
                )
            
            # Calculate ROC AUC if probabilities provided
            if 'roc_auc' in self.config.metrics and y_pred_proba is not None:
                if len(y_pred_proba.shape) > 1:
                    # Multi-class probabilities, use positive class
                    y_pred_proba = y_pred_proba[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            logger.info("Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            raise ModelEvaluationException(e, sys)
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            logger.info(f"Confusion Matrix:\n{cm}")
            return cm
            
        except Exception as e:
            raise ModelEvaluationException(e, sys)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        try:
            report = classification_report(y_true, y_pred)
            logger.info(f"Classification Report:\n{report}")
            return report
            
        except Exception as e:
            raise ModelEvaluationException(e, sys)
    
    def evaluate_model(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """
        Evaluate a single model
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            logger.info(f"Evaluating {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            # Get confusion matrix
            cm = self.get_confusion_matrix(y_test, y_pred)
            
            # Get classification report
            report = self.get_classification_report(y_test, y_pred)
            
            # Compile results
            results = {
                'model_name': model_name,
                'metrics': metrics,
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'predictions': y_pred
            }
            
            # Log results
            logger.info(f"\n{model_name} Evaluation Results:")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")
            
            # Check if model meets threshold
            if metrics.get('accuracy', 0) >= self.config.threshold:
                logger.info(
                    f"{model_name} meets accuracy threshold "
                    f"({self.config.threshold})"
                )
            else:
                logger.warning(
                    f"{model_name} does not meet accuracy threshold "
                    f"({self.config.threshold})"
                )
            
            return results
            
        except Exception as e:
            raise ModelEvaluationException(e, sys)
    
    def evaluate_multiple_models(
        self,
        models: Dict,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of {model_name: model_object}
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with results for all models
        """
        try:
            logger.info("=" * 50)
            logger.info("Model Evaluation Started")
            logger.info("=" * 50)
            
            all_results = {}
            
            for model_name, model in models.items():
                results = self.evaluate_model(model, X_test, y_test, model_name)
                all_results[model_name] = results
            
            # Find best model
            best_model_name = max(
                all_results.keys(),
                key=lambda x: all_results[x]['metrics'].get('accuracy', 0)
            )
            
            logger.info("=" * 50)
            logger.info(f"Best Model on Test Set: {best_model_name}")
            logger.info(
                f"Test Accuracy: "
                f"{all_results[best_model_name]['metrics']['accuracy']:.4f}"
            )
            logger.info("=" * 50)
            
            return all_results
            
        except Exception as e:
            raise ModelEvaluationException(e, sys)