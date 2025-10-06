.PHONY: help install setup train predict api docker clean test lint format

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make setup        - Setup project directories"
	@echo "  make train        - Run training pipeline"
	@echo "  make predict      - Run prediction example"
	@echo "  make api          - Start API server"
	@echo "  make mlflow       - Start MLflow UI"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-up    - Start all services with docker-compose"
	@echo "  make docker-down  - Stop all services"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linting"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

setup:
	mkdir -p data/raw data/processed data/artifacts/models data/artifacts/plots logs mlruns
	@echo "Project directories created"

train:
	python src/pipelines/training_pipeline.py

predict:
	python src/pipelines/prediction_pipeline.py

api:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

docker-build:
	docker build -t loan-approval-api:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data loan-approval-api:latest

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ api/ --max-line-length=100
	mypy src/ api/

format:
	black src/ api/ tests/
	isort src/ api/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.orig" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage

requirements-freeze:
	pip freeze > requirements-frozen.txt

airflow-init:
	export AIRFLOW_HOME=$(PWD)/airflow && \
	airflow db init && \
	airflow users create \
	    --username admin \
	    --firstname Admin \
	    --lastname User \
	    --role Admin \
	    --email admin@example.com \
	    --password admin

airflow-start:
	export AIRFLOW_HOME=$(PWD)/airflow && \
	airflow webserver --port 8080 & \
	airflow scheduler

notebook:
	jupyter notebook notebooks/

all: install setup train api