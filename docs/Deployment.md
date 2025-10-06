# Deployment Guide

Complete guide for deploying the Loan Approval System to production.

## 📋 Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Model performance acceptable (accuracy > threshold)
- [ ] API endpoints tested
- [ ] Configuration reviewed
- [ ] Security measures implemented
- [ ] Monitoring setup
- [ ] Backup strategy defined
- [ ] Rollback plan ready

## 🚀 Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Local Docker

```bash
# Build image
docker build -t loan-approval-api:v1.0 .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --name loan-api \
  loan-approval-api:v1.0

# Check logs
docker logs -f loan-api

# Health check
curl http://localhost:8000/health
```

#### Docker Compose (Full Stack)

```bash
# Start all services
docker-compose up -d

# Scale API workers
docker-compose up -d --scale api=3

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### Option 2: AWS Deployment

#### AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 20.04, t2.medium or larger)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install dependencies
sudo apt update
sudo apt install -y python3-pip docker.io docker-compose
sudo usermod -aG docker $USER

# 4. Clone repository
git clone <your-repo-url>
cd loan_approval_system

# 5. Deploy with Docker
docker-compose up -d

# 6. Configure security group to allow port 8000
```

#### AWS ECS (Elastic Container Service)

```bash
# 1. Push image to ECR
aws ecr create-repository --repository-name loan-approval-api

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag
docker build -t loan-approval-api:v1.0 .
docker tag loan-approval-api:v1.0 <account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-approval-api:v1.0

# Push to ECR
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/loan-approval-api:v1.0

# 2. Create ECS task definition (see ecs-task-definition.json)
# 3. Create ECS service
# 4. Configure load balancer
```

#### AWS Lambda + API Gateway

```bash
# 1. Install AWS SAM CLI
pip install aws-sam-cli

# 2. Create SAM template (see template.yaml)

# 3. Build
sam build

# 4. Deploy
sam deploy --guided

# 5. Test endpoint
curl https://your-api-gateway-url/predict
```

### Option 3: Google Cloud Platform

#### GCP Cloud Run

```bash
# 1. Install gcloud CLI
# 2. Authenticate
gcloud auth login

# 3. Set project
gcloud config set project your-project-id

# 4. Build and push to Container Registry
gcloud builds submit --tag gcr.io/your-project-id/loan-approval-api

# 5. Deploy to Cloud Run
gcloud run deploy loan-approval-api \
  --image gcr.io/your-project-id/loan-approval-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2

# 6. Get service URL
gcloud run services describe loan-approval-api --region us-central1
```

#### GCP Kubernetes Engine (GKE)

```bash
# 1. Create cluster
gcloud container clusters create loan-approval-cluster \
  --num-nodes=3 \
  --machine-type=n1-standard-2

# 2. Get credentials
gcloud container clusters get-credentials loan-approval-cluster

# 3. Deploy to Kubernetes
kubectl apply -f k8s/

# 4. Expose service
kubectl expose deployment loan-approval-api --type=LoadBalancer --port=8000
```

### Option 4: Azure

#### Azure Container Instances

```bash
# 1. Login to Azure
az login

# 2. Create resource group
az group create --name loan-approval-rg --location eastus

# 3. Create container registry
az acr create --resource-group loan-approval-rg \
  --name loanapprovalacr --sku Basic

# 4. Build and push
az acr build --registry loanapprovalacr \
  --image loan-approval-api:v1.0 .

# 5. Deploy container
az container create \
  --resource-group loan-approval-rg \
  --name loan-approval-api \
  --image loanapprovalacr.azurecr.io/loan-approval-api:v1.0 \
  --dns-name-label loan-approval-unique \
  --ports 8000
```

### Option 5: Kubernetes (General)

Create Kubernetes manifests:

**deployment.yaml**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loan-approval-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: loan-approval-api
  template:
    metadata:
      labels:
        app: loan-approval-api
    spec:
      containers:
      - name: api
        image: loan-approval-api:v1.0
        ports:
        - containerPort: 8000
        env:
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**service.yaml**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: loan-approval-service
spec:
  selector:
    app: loan-approval-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy:
```bash
kubectl apply -f k8s/
kubectl get pods
kubectl get services
```

## 🔒 Security Best Practices

### 1. Environment Variables

Never commit secrets! Use environment variables:

```bash
# Create .env file
cat > .env << EOF
MLFLOW_TRACKING_URI=your-tracking-uri
DATABASE_URL=your-database-url
API_KEY=your-api-key
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
EOF

# Use with Docker
docker run --env-file .env loan-approval-api:v1.0
```

### 2. API Authentication

Add authentication to FastAPI:

```python
# api/auth.py
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    return api_key
```

### 3. HTTPS/SSL

Use reverse proxy (nginx) with SSL:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, data: LoanApplicationRequest):
    ...
```

## 📊 Monitoring & Logging

### 1. Application Monitoring

```python
# Add Prometheus metrics
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(app)
```

### 2. Log Aggregation

Use ELK Stack or CloudWatch:

```python
import logging
import watchtower

# CloudWatch handler
logger.addHandler(watchtower.CloudWatchLogHandler())
```

### 3. Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }
```

## 🔄 CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Run tests
      run: |
        pip install -r requirements.txt
        pytest tests/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build Docker image
      run: docker build -t loan-approval-api:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
        docker push loan-approval-api:${{ github.sha }}
    
    - name: Deploy to production
      run: |
        # Your deployment script
        ./deploy.sh
```

## 🎯 Performance Optimization

### 1. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_prediction_cached(input_hash):
    return pipeline.predict(input_data)
```

### 2. Async Processing

```python
from fastapi import BackgroundTasks

@app.post("/predict/async")
async def predict_async(data: LoanApplicationRequest, background_tasks: BackgroundTasks):
    task_id = generate_task_id()
    background_tasks.add_task(process_prediction, task_id, data)
    return {"task_id": task_id, "status": "processing"}
```

### 3. Load Balancing

Use nginx or cloud load balancer:

```nginx
upstream api_backend {
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    location / {
        proxy_pass http://api_backend;
    }
}
```

## 📈 Scaling Strategy

### Horizontal Scaling

```bash
# Docker Compose
docker-compose up --scale api=5

# Kubernetes
kubectl scale deployment loan-approval-api --replicas=5
```

### Vertical Scaling

Increase resources:
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
```

## 🔙 Rollback Strategy

```bash
# Kubernetes rollback
kubectl rollout undo deployment/loan-approval-api

# Docker with version tags
docker stop loan-api
docker run -d loan-approval-api:v0.9  # Previous version

# AWS ECS
aws ecs update-service --cluster default \
  --service loan-approval-service \
  --task-definition loan-approval:previous-version
```

## ✅ Post-Deployment

1. **Smoke Tests**
```bash
curl -X POST https://your-api.com/predict -d @test_payload.json
```

2. **Monitor Logs**
```bash
kubectl logs -f deployment/loan-approval-api
```

3. **Check Metrics**
- Response times
- Error rates
- Prediction accuracy
- Resource usage

4. **Set Up Alerts**
- High error rate
- Slow response time
- Resource exhaustion
- Model drift

## 🆘 Troubleshooting

### Issue: High Memory Usage
**Solution**: Reduce batch size, optimize model, add memory limits

### Issue: Slow Predictions
**Solution**: Enable caching, use model optimization, scale horizontally

### Issue: Connection Timeouts
**Solution**: Increase timeout settings, optimize database queries

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [AWS ECS Guide](https://docs.aws.amazon.com/ecs/)
- [MLflow Deployment](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)

---

**Ready for Production!** 🎉