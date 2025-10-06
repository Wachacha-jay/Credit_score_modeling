"""
MLflow Tracking Component
Handles experiment tracking and model registry
"""
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
from typing import Dict, Any

from src.utils.logger import logger
from src.utils.exception import LoanApprovalException
from src.config.configuration import MLflowConfig


class MLflowTracker:
    """
    Class for MLflow experiment tracking and model management
    """
    
    def __init__(self, config: MLflowConfig):
        """
        Initialize MLflowTracker
        
        Args:
            config: MLflowConfig object
        """
        self.config = config
        self.active_run = None
        
        if self.config.enabled:
            self._setup_mlflow()
        
        logger.info("MLflowTracker initialized")
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.config.tracking_uri}")
            
            # Set experiment
            mlflow.set_experiment(self.config.experiment_name)
            logger.info(f"MLflow experiment set to: {self.config.experiment_name}")
            
        except Exception as e:
            logger.error(f"Error setting up MLflow: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def start_run(self, run_name: str = None):
        """
        Start a new MLflow run
        
        Args:
            run_name: Custom name for the run
        """
        try:
            if not self.config.enabled:
                logger.info("MLflow tracking is disabled")
                return
            
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{self.config.run_name_prefix}_{timestamp}"
            
            self.active_run = mlflow.start_run(run_name=run_name)
            logger.info(f"Started MLflow run: {run_name}")
            logger.info(f"Run ID: {self.active_run.info.run_id}")
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to MLflow
        
        Args:
            params: Dictionary of parameters to log
        """
        try:
            if not self.config.enabled or self.active_run is None:
                return
            
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            
            logger.info(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging parameters: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLflow
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for metric
        """
        try:
            if not self.config.enabled or self.active_run is None:
                return
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value, step=step)
            
            logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging metrics: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def log_model(
        self,
        model,
        artifact_path: str,
        model_name: str = None,
        registered_model_name: str = None
    ):
        """
        Log model to MLflow
        
        Args:
            model: Trained model object
            artifact_path: Path to store the model artifact
            model_name: Name of the model type
            registered_model_name: Name for model registry
        """
        try:
            if not self.config.enabled or self.active_run is None:
                return
            
            # Determine model framework and log accordingly
            if model_name and 'xgboost' in model_name.lower():
                mlflow.xgboost.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    artifact_path=artifact_path,
                    registered_model_name=registered_model_name
                )
            
            logger.info(f"Model logged to MLflow at: {artifact_path}")
            
            if registered_model_name:
                logger.info(f"Model registered as: {registered_model_name}")
            
        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def log_artifact(self, artifact_path: str):
        """
        Log artifact file to MLflow
        
        Args:
            artifact_path: Path to the artifact file
        """
        try:
            if not self.config.enabled or self.active_run is None:
                return
            
            mlflow.log_artifact(artifact_path)
            logger.info(f"Artifact logged to MLflow: {artifact_path}")
            
        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run
        
        Args:
            tags: Dictionary of tags to set
        """
        try:
            if not self.config.enabled or self.active_run is None:
                return
            
            for tag_name, tag_value in tags.items():
                mlflow.set_tag(tag_name, tag_value)
            
            logger.info(f"Set {len(tags)} tags in MLflow")
            
        except Exception as e:
            logger.error(f"Error setting tags: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def end_run(self):
        """End the current MLflow run"""
        try:
            if not self.config.enabled or self.active_run is None:
                return
            
            mlflow.end_run()
            logger.info("MLflow run ended")
            self.active_run = None
            
        except Exception as e:
            logger.error(f"Error ending run: {str(e)}")
            raise LoanApprovalException(e, sys)
    
    def log_training_session(
        self,
        model_name: str,
        model,
        params: Dict,
        metrics: Dict,
        train_results: Dict = None
    ):
        """
        Log a complete training session
        
        Args:
            model_name: Name of the model
            model: Trained model object
            params: Training parameters
            metrics: Evaluation metrics
            train_results: Additional training results
        """
        try:
            run_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.start_run(run_name)
            
            # Log tags
            self.set_tags({
                'model_type': model_name,
                'framework': 'sklearn' if 'xgboost' not in model_name.lower() else 'xgboost',
                'task': 'classification'
            })
            
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log additional training results
            if train_results:
                for key, value in train_results.items():
                    if isinstance(value, (int, float)):
                        self.log_metrics({key: value})
            
            # Log model
            registered_name = f"{self.config.registered_model_name}_{model_name}"
            self.log_model(
                model,
                artifact_path=model_name,
                model_name=model_name,
                registered_model_name=registered_name
            )
            
            self.end_run()
            logger.info(f"Training session logged for {model_name}")
            
        except Exception as e:
            self.end_run()
            raise LoanApprovalException(e, sys)