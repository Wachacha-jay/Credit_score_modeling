"""
Configuration manager for the loan approval system
"""
import os
import sys
from pathlib import Path
from dataclasses import dataclass

from src.utils.common import read_yaml, create_directories
from src.utils.logger import logger
from src.utils.exception import LoanApprovalException


# Configuration file path
CONFIG_FILE_PATH = Path("config.yaml")


@dataclass
class DataIngestionConfig:
    """Data ingestion configuration"""
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float
    random_state: int
    stratify: bool


@dataclass
class DataTransformationConfig:
    """Data transformation configuration"""
    numerical_features: list
    categorical_features: list
    target_column: str
    drop_columns: list
    outlier_method: str
    outlier_multiplier: float
    scaling_method: str
    preprocessor_path: Path
    scaler_path: Path
    encoder_path: Path


@dataclass
class ModelTrainerConfig:
    """Model training configuration"""
    models: dict
    cv_folds: int
    scoring: str
    model_dir: Path
    random_state: int


@dataclass
class ModelEvaluationConfig:
    """Model evaluation configuration"""
    metrics: list
    threshold: float


@dataclass
class MLflowConfig:
    """MLflow configuration"""
    enabled: bool
    tracking_uri: str
    experiment_name: str
    run_name_prefix: str
    registered_model_name: str


@dataclass
class ArtifactsConfig:
    """Artifacts configuration"""
    root_dir: Path
    model_dir: Path


class ConfigurationManager:
    """
    Configuration manager to handle all configurations
    """
    
    def __init__(self, config_filepath: Path = CONFIG_FILE_PATH):
        """
        Initialize configuration manager
        
        Args:
            config_filepath: Path to configuration YAML file
        """
        try:
            self.config = read_yaml(config_filepath)
            
            # Create necessary directories
            create_directories([
                self.config.artifacts.root_dir,
                self.config.artifacts.model_dir,
                Path(self.config.data.raw_data_path).parent,
                Path(self.config.data.train_data_path).parent,
            ])
            
            logger.info("Configuration manager initialized successfully")
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get data ingestion configuration
        
        Returns:
            DataIngestionConfig object
        """
        try:
            config = self.config.data
            
            data_ingestion_config = DataIngestionConfig(
                raw_data_path=Path(config.raw_data_path),
                train_data_path=Path(config.train_data_path),
                test_data_path=Path(config.test_data_path),
                test_size=config.test_size,
                random_state=config.random_state,
                stratify=config.stratify
            )
            
            logger.info("Data ingestion config retrieved")
            return data_ingestion_config
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Get data transformation configuration
        
        Returns:
            DataTransformationConfig object
        """
        try:
            config = self.config.transformation
            artifacts = self.config.artifacts
            
            data_transformation_config = DataTransformationConfig(
                numerical_features=config.numerical_features,
                categorical_features=config.categorical_features,
                target_column=config.target_column,
                drop_columns=config.drop_columns,
                outlier_method=config.outlier_removal.method,
                outlier_multiplier=config.outlier_removal.multiplier,
                scaling_method=config.scaling_method,
                preprocessor_path=Path(artifacts.preprocessor_path),
                scaler_path=Path(artifacts.scaler_path),
                encoder_path=Path(artifacts.encoder_path)
            )
            
            logger.info("Data transformation config retrieved")
            return data_transformation_config
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Get model training configuration
        
        Returns:
            ModelTrainerConfig object
        """
        try:
            config = self.config.model_training
            artifacts = self.config.artifacts
            
            model_trainer_config = ModelTrainerConfig(
                models=dict(config.models),
                cv_folds=config.cv_folds,
                scoring=config.scoring,
                model_dir=Path(artifacts.model_dir),
                random_state=self.config.data.random_state
            )
            
            logger.info("Model trainer config retrieved")
            return model_trainer_config
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Get model evaluation configuration
        
        Returns:
            ModelEvaluationConfig object
        """
        try:
            config = self.config.evaluation
            
            model_evaluation_config = ModelEvaluationConfig(
                metrics=config.metrics,
                threshold=config.threshold
            )
            
            logger.info("Model evaluation config retrieved")
            return model_evaluation_config
            
        except Exception as e:
            raise LoanApprovalException(e, sys)
    
    def get_mlflow_config(self) -> MLflowConfig:
        """
        Get MLflow configuration
        
        Returns:
            MLflowConfig object
        """
        try:
            config = self.config.mlflow
            
            mlflow_config = MLflowConfig(
                enabled=config.enabled,
                tracking_uri=config.tracking_uri,
                experiment_name=config.experiment_name,
                run_name_prefix=config.run_name_prefix,
                registered_model_name=config.registered_model_name
            )
            
            logger.info("MLflow config retrieved")
            return mlflow_config
            
        except Exception as e:
            raise LoanApprovalException(e, sys)