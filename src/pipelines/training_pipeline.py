"""
Training Pipeline
Orchestrates the complete training workflow
"""
import sys
from pathlib import Path

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.model_explainer import ModelExplainer
from src.monitoring.mlflow_tracking import MLflowTracker
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger
from src.utils.exception import LoanApprovalException


class TrainingPipeline:
    """
    Complete training pipeline orchestrator
    """
    
    def __init__(self):
        """Initialize training pipeline"""
        self.config_manager = ConfigurationManager()
        logger.info("Training Pipeline initialized")
    
    def run_pipeline(self):
        """
        Execute complete training pipeline
        """
        try:
            logger.info("\n" + "=" * 70)
            logger.info("STARTING LOAN APPROVAL TRAINING PIPELINE")
            logger.info("=" * 70 + "\n")
            
            # ========== DATA INGESTION ==========
            logger.info("\n>>> STAGE 1: Data Ingestion")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            
            # ========== DATA TRANSFORMATION ==========
            logger.info("\n>>> STAGE 2: Data Transformation")
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            (
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
                feature_names
            ) = data_transformation.transform_data(train_data_path, test_data_path)
            
            # ========== MODEL TRAINING ==========
            logger.info("\n>>> STAGE 3: Model Training")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            best_model, best_model_name, training_results = model_trainer.initiate_model_training(
                X_train_scaled, y_train
            )
            
            # ========== MODEL EVALUATION ==========
            logger.info("\n>>> STAGE 4: Model Evaluation")
            model_evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            
            # Evaluate all trained models
            evaluation_results = model_evaluation.evaluate_multiple_models(
                model_trainer.trained_models,
                X_test_scaled,
                y_test
            )
            
            # ========== MLFLOW TRACKING ==========
            logger.info("\n>>> STAGE 5: MLflow Tracking")
            mlflow_config = self.config_manager.get_mlflow_config()
            mlflow_tracker = MLflowTracker(config=mlflow_config)
            
            # Log each model to MLflow
            for model_name, model in model_trainer.trained_models.items():
                train_result = training_results.get(model_name, {})
                eval_result = evaluation_results.get(model_name, {})
                
                # Combine parameters and metrics
                params = train_result.get('best_params', {})
                params['model_type'] = model_name
                
                metrics = eval_result.get('metrics', {})
                metrics['cv_score'] = train_result.get('best_cv_score', 0)
                
                # Log to MLflow
                mlflow_tracker.log_training_session(
                    model_name=model_name,
                    model=model,
                    params=params,
                    metrics=metrics,
                    train_results=train_result
                )
            
            # ========== MODEL EXPLAINABILITY ==========
            logger.info("\n>>> STAGE 6: Model Explainability")
            explainer = ModelExplainer(
                model=best_model,
                feature_names=feature_names
            )
            
            # Create explainer and calculate SHAP values
            explainer.create_explainer()
            explainer.calculate_shap_values(X_test_scaled, max_samples=100)
            
            # Get feature importance
            feature_importance = explainer.get_feature_importance()
            logger.info("\nTop 10 Most Important Features:")
            logger.info(feature_importance.head(10))
            
            # Save visualizations
            artifacts_dir = Path("data/artifacts/plots")
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            
            explainer.plot_summary(
                X_test_scaled[:100],
                save_path=str(artifacts_dir / "shap_summary.png")
            )
            
            explainer.plot_waterfall(
                X_test_scaled[:100],
                sample_idx=0,
                save_path=str(artifacts_dir / "shap_waterfall_sample0.png")
            )
            
            logger.info("\n" + "=" * 70)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70 + "\n")
            
            # Return results
            return {
                'best_model': best_model,
                'best_model_name': best_model_name,
                'training_results': training_results,
                'evaluation_results': evaluation_results,
                'feature_names': feature_names,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise LoanApprovalException(e, sys)


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        results = pipeline.run_pipeline()
        logger.info("Pipeline executed successfully!")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise e