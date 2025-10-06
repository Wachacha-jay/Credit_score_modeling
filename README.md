# Loan Approval Prediction System

A complete MLOps solution for predicting loan approval status, Credit Scoring using machine learning, featuring automated training pipelines, model monitoring with MLflow, and a production-ready REST API. For different credit scoring models all you have to do is choose the correct model according to the problem either regression or classification, define the correct model hyperparameters, metrics to be evaluated and add all that in your configurations.

## 🚀 Features

- **Modular Architecture**: Clean OOP design with separation of concerns
- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning
- **Model Interpretability**: SHAP-based explanations for predictions
- **MLflow Integration**: Experiment tracking and model registry
- **REST API**: FastAPI-based API for real-time predictions
- **Airflow Pipeline**: Automated training workflows
- **Comprehensive Logging**: Structured logging and exception handling
- **Data Validation**: Robust input validation and preprocessing

## 📁 Project Structure

```
loan_approval_system/
├── src/
│   ├── components/          # Core ML components
│   ├── pipelines/           # Training and prediction pipelines
│   ├── config/              # Configuration management
│   ├── utils/               # Utilities (logging, exceptions)
│   └── monitoring/          # MLflow tracking
├── api/                     # FastAPI application
├── airflow/dags/            # Airflow DAGs
├── data/                    # Data storage
├── logs/                    # Application logs
├── config.yaml              # Configuration file
└── requirements.txt         # Dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Wachacha-jay/Credit_score_modelling.git
cd Credit_score_modelling
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install package in development mode**
```bash
pip install -e .
```

5. **Set up directories**
```bash
mkdir -p data/raw data/processed data/artifacts/models logs
```

6. **Add your data**
```bash
# Place your loan_approval_dataset.csv in data/raw/
cp your_data.csv data/raw/loan_approval_dataset.csv
```

## 🎯 Usage

### Training Pipeline

Run the complete training pipeline:

```bash
python src/pipelines/training_pipeline.py
```

This will:
1. Ingest and split data
2. Transform and preprocess features
3. Train multiple models with hyperparameter tuning
4. Evaluate models on test data
5. Log experiments to MLflow
6. Generate SHAP explainability plots

### Making Predictions

```python
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

# Create sample data
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

# Make prediction
pipeline = PredictionPipeline()
result = pipeline.predict(data.get_data_as_dict())
print(result)
```

### API Server

Start the FastAPI server:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at `http://localhost:8000/docs`

#### API Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Single Prediction**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Batch Prediction**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"applications": [...]}'
```

### Airflow Pipeline

1. **Set up Airflow**
```bash
export AIRFLOW_HOME=~/airflow
airflow db init
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

2. **Copy DAG to Airflow directory**
```bash
cp airflow/dags/loan_approval_dag.py $AIRFLOW_HOME/dags/
```

3. **Start Airflow**
```bash
# Start webserver
airflow webserver --port 8080

# Start scheduler (in another terminal)
airflow scheduler
```

4. **Access Airflow UI**
Navigate to `http://localhost:8080` and enable the `loan_approval_training_pipeline` DAG

### MLflow Tracking

View experiment tracking:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access MLflow UI at `http://localhost:5000`

Or use DagsHub:
```python
import dagshub
dagshub.init(repo_owner='your-username', repo_name='your-repo', mlflow=True)
```

## 📊 Model Performance

The system trains and compares three models:
- **Logistic Regression**: Fast baseline model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting for best performance

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🧠 Model Interpretability

SHAP (SHapley Additive exPlanations) is used for model interpretability:

- **Summary plots**: Global feature importance
- **Waterfall plots**: Individual prediction explanations
- **Dependence plots**: Feature interaction analysis

Plots are automatically saved to `data/artifacts/plots/`

## ⚙️ Configuration

All configuration is managed through `config.yaml`:

```yaml
# Example configuration
data:
  raw_data_path: "data/raw/loan_approval_dataset.csv"
  test_size: 0.2
  random_state: 42

model_training:
  models:
    random_forest:
      enabled: True
      params:
        n_estimators: [100, 200, 500]
        max_depth: [5, 10, 20]

mlflow:
  enabled: True
  experiment_name: "Loan_Approval_System"
```

## 🔍 Logging

Logs are automatically generated in the `logs/` directory with:
- Timestamp
- Log level (INFO, WARNING, ERROR)
- Source file and line number
- Detailed error messages

Example log format:
```
[2025-10-05 14:30:45] 123 ModelTrainer - INFO - Training Random Forest...
```

## 🛡️ Exception Handling

Custom exception classes provide detailed error information:
- `DataIngestionException`
- `DataTransformationException`
- `ModelTrainingException`
- `ModelEvaluationException`
- `PredictionException`

Each exception includes:
- Error message
- File name and line number
- Stack trace

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v --cov=src
```

## 📈 Monitoring

### MLflow Tracking
- Experiment comparison
- Parameter logging
- Metric tracking
- Model versioning
- Model registry

### API Monitoring
- Health check endpoint
- Request/response logging
- Error tracking
- Performance metrics

## 🚀 Deployment

### Docker Deployment

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. **Build and run**
```bash
docker build -t loan-approval-api .
docker run -p 8000:8000 loan-approval-api
```

### Cloud Deployment Options

- **AWS**: Deploy using ECS, Lambda, or SageMaker
- **GCP**: Deploy using Cloud Run or AI Platform
- **Azure**: Deploy using Azure ML or App Service

## 📝 API Documentation

### Request Schema
```json
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
```

### Response Schema
```json
{
  "success": true,
  "prediction": "Approved",
  "prediction_code": 1,
  "probability": {
    "rejected": 0.25,
    "approved": 0.75
  },
  "message": "Prediction completed successfully"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- James w. Ngaruiya - Initial work

## 🙏 Acknowledgments

- Scikit-learn team for ML algorithms
- SHAP library for model interpretability
- FastAPI for the excellent web framework
- MLflow for experiment tracking
- Apache Airflow for workflow orchestration

## 📞 Support

For support, jameswachacha@gmail.com or open an issue on GitHub.

## 🗺️ Roadmap

- [ ] Add more ML models (LightGBM, CatBoost)
- [ ] Implement A/B testing framework
- [ ] Add model drift detection
- [ ] Implement real-time monitoring dashboard
- [ ] Add automated retraining triggers
- [ ] Implement CI/CD pipeline
- [ ] Add comprehensive unit tests
- [ ] Create mobile app interface
- [ ] Add multi-language support
- [ ] Implement explainable AI dashboard

## 📊 Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score | Training Time |
|-------|----------|-----------|--------|----------|---------------|
| Logistic Regression | 0.85 | 0.83 | 0.87 | 0.85 | 2s |
| Random Forest | 0.92 | 0.91 | 0.93 | 0.92 | 45s |
| XGBoost | 0.93 | 0.92 | 0.94 | 0.93 | 38s |

*Results may vary based on data and hyperparameters*