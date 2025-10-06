"""
Model Training Component
Handles model training with hyperparameter tuning
"""
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Tuple

from src.utils.logger import logger
from src.utils.exception import ModelTrainingException
from src.utils.common import save_object
from src.config.configuration import ModelTrainerConfig


class ModelTrainer:
    """
    Class for training and tuning machine learning models
    """
    
    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize ModelTrainer component
        
        Args:
            config: ModelTrainerConfig object
        """
        self.config = config
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        logger.info("ModelTrainer component initialized")
    
    def get_model_object(self, model_name: str):
        """
        Get model object based on model name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model object
        """
        try:
            models = {
                'logistic_regression': LogisticRegression(
                    max_iter=self.config.models.logistic_regression.params.max_iter
                ),
                'random_forest': RandomForestClassifier(
                    random_state=self.config.random_state
                ),
                'xgboost': XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=self.config.random_state
                )
            }
            
            return models.get(model_name)
            
        except Exception as e:
            raise ModelTrainingException(e, sys)
    
    def get_param_grid(self, model_name: str) -> dict:
        """
        Get parameter grid for model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Parameter grid dictionary
        """
        try:
            param_grids = {
                'logistic_regression': {
                    'C': self.config.models.logistic_regression.params.C
                },
                'random_forest': {
                    'n_estimators': self.config.models.random_forest.params.n_estimators,
                    'max_depth': self.config.models.random_forest.params.max_depth,
                    'min_samples_split': self.config.models.random_forest.params.min_samples_split
                },
                'xgboost': {
                    'n_estimators': self.config.models.xgboost.params.n_estimators,
                    'learning_rate': self.config.models.xgboost.params.learning_rate,
                    'max_depth': self.config.models.xgboost.params.max_depth
                }
            }
            
            return param_grids.get(model_name, {})
            
        except Exception as e:
            raise ModelTrainingException(e, sys)
    
    def train_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[object, dict]:
        """
        Train a single model with GridSearchCV
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (trained_model, results_dict)
        """
        try:
            logger.info(f"Training {model_name}...")
            
            # Get model and parameter grid
            model = self.get_model_object(model_name)
            param_grid = self.get_param_grid(model_name)
            
            if model is None:
                raise ValueError(f"Unknown model: {model_name}")
            
            # Perform grid search
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.config.cv_folds,
                scoring=self.config.scoring,
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            results = {
                'model_name': model_name,
                'best_params': best_params,
                'best_cv_score': best_score,
                'grid_search': grid_search
            }
            
            logger.info(f"{model_name} - Best CV Score: {best_score:.4f}")
            logger.info(f"{model_name} - Best Params: {best_params}")
            
            return best_model, results
            
        except Exception as e:
            raise ModelTrainingException(e, sys)
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict:
        """
        Train all enabled models
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with all trained models and their results
        """
        try:
            logger.info("=" * 50)
            logger.info("Model Training Started")
            logger.info("=" * 50)
            
            all_results = {}
            
            # Train each enabled model
            for model_name, model_config in self.config.models.items():
                if model_config.enabled:
                    trained_model, results = self.train_model(
                        model_name, X_train, y_train
                    )
                    
                    self.trained_models[model_name] = trained_model
                    all_results[model_name] = results
                    
                    # Track best model
                    if results['best_cv_score'] > self.best_score:
                        self.best_score = results['best_cv_score']
                        self.best_model = trained_model
                        self.best_model_name = model_name
                else:
                    logger.info(f"Skipping {model_name} (disabled in config)")
            
            logger.info("=" * 50)
            logger.info(f"Best Model: {self.best_model_name}")
            logger.info(f"Best CV Score: {self.best_score:.4f}")
            logger.info("=" * 50)
            
            return all_results
            
        except Exception as e:
            raise ModelTrainingException(e, sys)
    
    def save_models(self):
        """
        Save all trained models
        """
        try:
            logger.info("Saving trained models...")
            
            for model_name, model in self.trained_models.items():
                model_path = self.config.model_dir / f"{model_name}.pkl"
                save_object(model_path, model)
                logger.info(f"Saved {model_name} at: {model_path}")
            
            # Save best model separately
            if self.best_model:
                best_model_path = self.config.model_dir / "best_model.pkl"
                save_object(best_model_path, self.best_model)
                logger.info(f"Saved best model at: {best_model_path}")
            
            logger.info("All models saved successfully")
            
        except Exception as e:
            raise ModelTrainingException(e, sys)
    
    def initiate_model_training(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[object, str, Dict]:
        """
        Main method to execute model training pipeline
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (best_model, best_model_name, all_results)
        """
        try:
            # Train all models
            all_results = self.train_all_models(X_train, y_train)
            
            # Save models
            self.save_models()
            
            logger.info("=" * 50)
            logger.info("Model Training Completed Successfully")
            logger.info("=" * 50)
            
            return self.best_model, self.best_model_name, all_results
            
        except Exception as e:
            raise ModelTrainingException(e, sys)