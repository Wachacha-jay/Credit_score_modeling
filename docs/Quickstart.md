# Quick Start Guide

Get your loan approval prediction system up and running in minutes!

## 🚀 5-Minute Setup

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd loan_approval_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Step 2: Prepare Data

```bash
# Create directories
make setup

# Add your dataset
cp /path/to/loan_approval_dataset.csv data/raw/
```

### Step 3: Train Model

```bash
# Run training pipeline
make train

# Or manually
python src/pipelines/training_pipeline.py
```

### Step 4: Start API

```bash
# Start the API server
make api

# Or manually
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5: Test API

Open http://localhost:8000/docs in your browser and test the API!

Or use curl:

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

## 🐳 Docker Quick Start

```bash
# Build and run with Docker
make docker-build
make docker-run

# Or use docker-compose for full stack
make docker-up

# Access services:
# API: http://localhost:8000
# MLflow: http://localhost:5000
# Airflow: http://localhost:8080
```

## 📊 View Experiments

```bash
# Start MLflow UI
make mlflow

# Visit http://localhost:5000
```

## 🔄 Airflow Pipeline

```bash
# Initialize Airflow
make airflow-init

# Start Airflow
make airflow-start

# Visit http://localhost:8080
# Username: admin, Password: admin
```

## 🧪 Run Tests

```bash
make test
```

## 🎯 Common Use Cases

### Use Case 1: Single Prediction

```python
from src.pipelines.prediction_pipeline import PredictionPipeline, CustomData

# Create input
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

# Predict
pipeline = PredictionPipeline()
result = pipeline.predict(data.get_data_as_dict())
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability']['approved']:.2%}")
```

### Use Case 2: Batch Predictions

```python
import pandas as pd
from src.pipelines.prediction_pipeline import PredictionPipeline

# Load data
df = pd.read_csv('new_applications.csv')
applications = df.to_dict('records')

# Predict
pipeline = PredictionPipeline()
results = pipeline.predict_batch(applications)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('predictions.csv', index=False)
```

### Use Case 3: Model Retraining

```python
from src.pipelines.training_pipeline import TrainingPipeline

# Retrain with new data
pipeline = TrainingPipeline()
results = pipeline.run_pipeline()

print(f"Best Model: {results['best_model_name']}")
print(f"Accuracy: {results['evaluation_results'][results['best_model_name']]['metrics']['accuracy']:.4f}")
```

### Use Case 4: Model Explanation

```python
from src.components.model_explainer import ModelExplainer
from src.utils.common import load_object
import numpy as np

# Load model
model = load_object('data/artifacts/models/best_model.pkl')

# Create explainer
explainer = ModelExplainer(model, feature_names)
explainer.create_explainer()

# Explain predictions
X_sample = np.array([[...]])  # Your data
explainer.calculate_shap_values(X_sample)

# Get feature importance
importance = explainer.get_feature_importance()
print(importance.head(10))

# Generate plots
explainer.plot_summary(X_sample, save_path='shap_summary.png')
explainer.plot_waterfall(X_sample, sample_idx=0, save_path='shap_waterfall.png')
```

## 🔧 Configuration

Edit `config.yaml` to customize:

- Data paths
- Model hyperparameters
- Training settings
- MLflow configuration
- API settings

Example:

```yaml
model_training:
  models:
    random_forest:
      enabled: True
      params:
        n_estimators: [100, 200, 500]
        max_depth: [5, 10, 20]
```

## 📝 Project Structure Overview

```
loan_approval_system/
├── src/
│   ├── components/       # Data ingestion, transformation, training, evaluation
│   ├── pipelines/        # Training and prediction pipelines
│   ├── config/           # Configuration management
│   ├── utils/            # Logging, exceptions, utilities
│   └── monitoring/       # MLflow tracking
├── api/                  # FastAPI application
├── airflow/dags/         # Airflow DAGs
├── data/                 # Data storage
├── tests/                # Unit tests
├── config.yaml           # Main configuration
└── Makefile             # Convenience commands
```

## 🐛 Troubleshooting

### Issue: Model files not found

**Solution:**
```bash
# Make sure you've trained the model first
python src/pipelines/training_pipeline.py
```

### Issue: Import errors

**Solution:**
```bash
# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: API not loading model

**Solution:**
Check that all artifacts exist:
```bash
ls data/artifacts/models/best_model.pkl
ls data/artifacts/scaler.pkl
ls data/artifacts/encoder.pkl
ls data/artifacts/preprocessor.pkl
```

### Issue: Docker container won't start

**Solution:**
```bash
# Check logs
docker-compose logs -f api

# Rebuild without cache
docker-compose build --no-cache
```

## 📚 Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **View Experiments**: Check MLflow UI at http://localhost:5000
3. **Schedule Training**: Set up Airflow DAG at http://localhost:8080
4. **Customize Models**: Edit hyperparameters in `config.yaml`
5. **Add Tests**: Write tests in `tests/` directory
6. **Deploy**: Use Docker or cloud deployment options

## 💡 Tips

- Use `make help` to see all available commands
- Check `logs/` directory for detailed logs
- Use MLflow to compare model performance
- Enable DagsHub for remote experiment tracking
- Run tests before deployment: `make test`
- Format code before committing: `make format`

## 🆘 Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Look at example notebooks in `notebooks/`
- Review test cases in `tests/`
- Check logs in `logs/` directory
- Open an issue on GitHub

## 🎉 Success!

You're now ready to:
- ✅ Train loan approval models
- ✅ Make predictions via API
- ✅ Track experiments with MLflow
- ✅ Automate with Airflow
- ✅ Deploy to production

Happy predicting! 🚀