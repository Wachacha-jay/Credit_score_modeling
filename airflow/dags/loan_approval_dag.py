"""
Airflow DAG for Loan Approval Model Training Pipeline
"""
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Add project root to path
sys.path.insert(0, '/opt/airflow/dags/loan_approval_system')

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.monitoring.mlflow_tracking import MLflowTracker
from src.config.configuration import ConfigurationManager
from src.utils.logger import logger


# Default arguments for the DAG
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}


def task_data_ingestion(**context):
    """Task for data ingestion"""
    try:
        logger.info("Starting data ingestion task...")
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        
        # Push paths to XCom for next tasks
        context['ti'].xcom_push(key='train_data_path', value=train_path)
        context['ti'].xcom_push(key='test_data_path', value=test_path)
        
        logger.info("Data ingestion task completed successfully")
        return {"status": "success", "train_path": train_path, "test_path": test_path}
        
    except Exception as e:
        logger.error(f"Data ingestion task failed: {str(e)}")
        raise


def task_data_transformation(**context):
    """Task for data transformation"""
    try:
        logger.info("Starting data transformation task...")
        
        # Pull paths from XCom
        ti = context['ti']
        train_path = ti.xcom_pull(key='train_data_path', task_ids='data_ingestion')
        test_path = ti.xcom_pull(key='test_data_path', task_ids='data_ingestion')
        
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        
        X_train, X_test, y_train, y_test, features = data_transformation.transform_data(
            train_path, test_path
        )
        
        # Store shapes for validation
        ti.xcom_push(key='X_train_shape', value=X_train.shape)
        ti.xcom_push(key='X_test_shape', value=X_test.shape)
        ti.xcom_push(key='feature_names', value=features)
        
        logger.info("Data transformation task completed successfully")
        return {
            "status": "success",
            "train_shape": X_train.shape,
            "test_shape": X_test.shape
        }
        
    except Exception as e:
        logger.error(f"Data transformation task failed: {str(e)}")
        raise


def task_model_training(**context):
    """Task for model training"""
    try:
        logger.info("Starting model training task...")
        
        ti = context['ti']
        train_path = ti.xcom_pull(key='train_data_path', task_ids='data_ingestion')
        test_path = ti.xcom_pull(key='test_data_path', task_ids='data_ingestion')
        
        # Re-transform data (in production, you might want to save transformed data)
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        X_train, X_test, y_train, y_test, features = data_transformation.transform_data(
            train_path, test_path
        )
        
        # Train models
        model_trainer_config = config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        best_model, best_model_name, training_results = model_trainer.initiate_model_training(
            X_train, y_train
        )
        
        # Store results
        ti.xcom_push(key='best_model_name', value=best_model_name)
        ti.xcom_push(key='best_cv_score', value=model_trainer.best_score)
        
        logger.info("Model training task completed successfully")
        return {
            "status": "success",
            "best_model": best_model_name,
            "cv_score": model_trainer.best_score
        }
        
    except Exception as e:
        logger.error(f"Model training task failed: {str(e)}")
        raise


def task_model_evaluation(**context):
    """Task for model evaluation"""
    try:
        logger.info("Starting model evaluation task...")
        
        ti = context['ti']
        train_path = ti.xcom_pull(key='train_data_path', task_ids='data_ingestion')
        test_path = ti.xcom_pull(key='test_data_path', task_ids='data_ingestion')
        
        # Re-transform and load models
        config_manager = ConfigurationManager()
        data_transformation_config = config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        X_train, X_test, y_train, y_test, features = data_transformation.transform_data(
            train_path, test_path
        )
        
        # Load trained models
        from src.utils.common import load_object
        from pathlib import Path
        
        model_dir = Path("data/artifacts/models")
        trained_models = {}
        
        for model_file in model_dir.glob("*.pkl"):
            if model_file.stem not in ['best_model']:
                model_name = model_file.stem
                trained_models[model_name] = load_object(model_file)
        
        # Evaluate models
        model_evaluation_config = config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        evaluation_results = model_evaluation.evaluate_multiple_models(
            trained_models, X_test, y_test
        )
        
        # Store best test accuracy
        best_accuracy = max(
            [res['metrics']['accuracy'] for res in evaluation_results.values()]
        )
        ti.xcom_push(key='best_test_accuracy', value=best_accuracy)
        
        logger.info("Model evaluation task completed successfully")
        return {"status": "success", "best_test_accuracy": best_accuracy}
        
    except Exception as e:
        logger.error(f"Model evaluation task failed: {str(e)}")
        raise


def task_mlflow_logging(**context):
    """Task for MLflow logging"""
    try:
        logger.info("Starting MLflow logging task...")
        
        ti = context['ti']
        best_model_name = ti.xcom_pull(key='best_model_name', task_ids='model_training')
        best_cv_score = ti.xcom_pull(key='best_cv_score', task_ids='model_training')
        best_test_accuracy = ti.xcom_pull(
            key='best_test_accuracy', task_ids='model_evaluation'
        )
        
        config_manager = ConfigurationManager()
        mlflow_config = config_manager.get_mlflow_config()
        mlflow_tracker = MLflowTracker(config=mlflow_config)
        
        # Log summary
        run_name = f"airflow_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_tracker.start_run(run_name)
        
        mlflow_tracker.log_params({
            'pipeline': 'airflow',
            'best_model': best_model_name,
            'execution_date': context['ds']
        })
        
        mlflow_tracker.log_metrics({
            'best_cv_score': best_cv_score,
            'best_test_accuracy': best_test_accuracy
        })
        
        mlflow_tracker.set_tags({
            'pipeline': 'airflow',
            'scheduled': 'true',
            'dag_id': context['dag'].dag_id
        })
        
        mlflow_tracker.end_run()
        
        logger.info("MLflow logging task completed successfully")
        return {"status": "success"}
        
    except Exception as e:
        logger.error(f"MLflow logging task failed: {str(e)}")
        raise


def task_send_notification(**context):
    """Task to send notification on completion"""
    try:
        ti = context['ti']
        best_model = ti.xcom_pull(key='best_model_name', task_ids='model_training')
        best_accuracy = ti.xcom_pull(
            key='best_test_accuracy', task_ids='model_evaluation'
        )
        
        message = f"""
        Loan Approval Model Training Pipeline Completed Successfully!
        
        Execution Date: {context['ds']}
        Best Model: {best_model}
        Test Accuracy: {best_accuracy:.4f}
        
        The new model has been trained and is ready for deployment.
        """
        
        logger.info(message)
        print(message)
        
        # Here you can add email/Slack notification logic
        
        return {"status": "success", "message": message}
        
    except Exception as e:
        logger.error(f"Notification task failed: {str(e)}")
        raise


# Define the DAG
with DAG(
    dag_id='loan_approval_training_pipeline',
    default_args=default_args,
    description='End-to-end ML pipeline for loan approval prediction',
    schedule_interval='@weekly',  # Run weekly
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'loan_approval', 'training'],
) as dag:
    
    # Define tasks
    start_task = BashOperator(
        task_id='start',
        bash_command='echo "Starting Loan Approval Training Pipeline..."'
    )
    
    data_ingestion_task = PythonOperator(
        task_id='data_ingestion',
        python_callable=task_data_ingestion,
        provide_context=True,
    )
    
    data_transformation_task = PythonOperator(
        task_id='data_transformation',
        python_callable=task_data_transformation,
        provide_context=True,
    )
    
    model_training_task = PythonOperator(
        task_id='model_training',
        python_callable=task_model_training,
        provide_context=True,
    )
    
    model_evaluation_task = PythonOperator(
        task_id='model_evaluation',
        python_callable=task_model_evaluation,
        provide_context=True,
    )
    
    mlflow_logging_task = PythonOperator(
        task_id='mlflow_logging',
        python_callable=task_mlflow_logging,
        provide_context=True,
    )
    
    notification_task = PythonOperator(
        task_id='send_notification',
        python_callable=task_send_notification,
        provide_context=True,
    )
    
    end_task = BashOperator(
        task_id='end',
        bash_command='echo "Loan Approval Training Pipeline Completed Successfully!"'
    )
    
    # Define task dependencies
    start_task >> data_ingestion_task >> data_transformation_task
    data_transformation_task >> model_training_task >> model_evaluation_task
    model_evaluation_task >> mlflow_logging_task >> notification_task >> end_task