# Boston Smart Accident Risk & Rerouting — MLOps Edition

**Production-grade ML pipeline with containerized model serving, reproducible deployment, and safety-critical accident risk prediction.**

---

## 📋 Project Overview

This is the **MLOps-optimized variant** of the Boston Smart Accident Risk and Rerouting System. While the [main repository](https://github.com/machackgo/boston-smart-accident-risk-rerouting) focuses on the end-to-end ML product, **this MLOps edition emphasizes production infrastructure, model versioning, containerization, and deployment readiness**.

### What Makes This Different from the Main Repo

| Aspect | Main Repo | MLOps Edition |
|--------|-----------|---------------|
| Focus | Full-stack ML application | Production ML pipeline & deployment |
| Deployment | REST API (basic setup) | Docker-containerized, Docker-containerized, Cloud Run-ready deployment 
| Model Management | Single model | Versioned models (v1–v4) with model cards |
| Infrastructure | API-centric | MLOps workflow: data → model → serving → deploy |
| Configuration | Environment variables | Structured deployment configs |
| Testing | Basic tests | Comprehensive unit tests per module |
| Documentation | Feature-driven | Infrastructure & operations-driven |

---

## 🔄 MLOps Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOSTON ACCIDENT DATA                         │
│  Historical (47,000+ rows, 2015–2024) + Live Conditions       │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                │
│  - Historical dataset (MassDOT)                                 │
│  - Live route data (Google Maps)                               │
│  - Live weather data (OpenWeather)                             │
│  - Feature caching (Parquet files in /data)                    │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MODEL TRAINING                                │
│  - LightGBM classifier (severity prediction)                    │
│  - Feature engineering pipeline                                 │
│  - Cross-validation & threshold optimization                    │
│  - Model versioning (v2, v3, v4)                               │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              MODEL ARTIFACTS & SERVING                          │
│  - Serialized models (.pkl) with feature lists                 │
│  - Model cards documenting performance metrics                 │
│  - Threshold configurations per version                        │
│  - Model loading abstraction in src/model/                     │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              REST API (FastAPI)                                 │
│  - /predict - Route risk assessment                            │
│  - /predict/segmented - Multi-segment analysis                 │
│  - Auto-generated OpenAPI docs                                 │
│  - Validation & error handling                                 │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│            CONTAINERIZATION & DEPLOYMENT                        │
│  - Docker multi-stage build (Python 3.11-slim)                │
│  - Port 8000 (Uvicorn ASGI server)                            │
│  - Cloud Run-ready containerized deployment-ready                             │
│  - Environment variable configuration                          │
└────────────────┬────────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│        MONITORING & OBSERVABILITY (Future)                      │
│  - Request logging & performance metrics                       │
│  - Model performance tracking                                  │
│  - API health checks                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ MLOps Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model Framework** | LightGBM | Gradient-boosted decision trees for safety-critical risk prediction |
| **API Server** | FastAPI + Uvicorn | Async ASGI web framework with auto-generated OpenAPI docs |
| **Data Processing** | Pandas, NumPy, Scikit-learn | Feature engineering, model training, preprocessing |
| **Serialization** | Joblib | Model persistence (.pkl format) |
| **Database** | Supabase (PostgreSQL) | Live route and accident data storage |
| **Containerization** | Docker | Cloud Run-ready containerized deployment |
| **Python Runtime** | 3.11-slim | Lightweight production base image |
| **External APIs** | Google Maps, OpenWeather | Live route and weather data integration |

---

## 🎯 Key MLOps Components

### 1. **Model Versioning & Artifacts** (`/models`)

This repo implements **multi-version model management** with reproducibility:

- **Model Binaries**: `best_model.pkl`, `best_model_v2.pkl`, `best_model_v3.pkl`, `best_model_v4.pkl`, `best_model_v4_binary.pkl` (serialized LightGBM models)
- - **Feature Specifications**: `feature_list.txt`, `feature_list_v2.txt`, `feature_list_v3.txt`, `feature_list_v4.txt` (exact feature names used during training)
  - - **Model Documentation**: `MODEL_CARD_v2.md`, `MODEL_CARD_v3.md`, `MODEL_CARD_v4.md` (performance metrics, training data, limitations)
    - - **Threshold Configurations**: `thresholds_v4.json` (decision thresholds for severity classification)
      - - **Feature Engineering Config**: `weather_keep_cols_v4.json` (weather feature transformations)
       
        - **MLOps Benefit**: Reproducible predictions and easy model rollback/comparison.
       
        - ### 2. **Containerized Model Serving** (`/Dockerfile`)
       
        - Production-grade containerization for GCP Cloud Run:
       
        - ```dockerfile
          FROM python:3.11-slim
          ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
          WORKDIR /app
          COPY requirements.txt .
          RUN pip install --no-cache-dir -r requirements.txt
          COPY . .
          EXPOSE 8000
          CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
          ```

          **MLOps Benefits**:
          - **Reproducible environments**: Pinned Python 3.11-slim base, exact dependencies from requirements.txt
          - - **Fast deployment**: Optimized layer caching, no-cache pip install for security
            - - **Horizontal scaling**: Stateless ASGI server ready for Kubernetes/Cloud Run
              - - **Port 8000**: Standard for Uvicorn, matches health check expectations
               
                - ### 3. **Model Serving Abstraction** (`/src/model/`)
               
                - Clean Python abstraction for loading and using models:
               
                - - **model.py** (inferred): Encapsulates model loading, prediction, and version management
                  - - Decouples model logic from API endpoints
                    - - Supports hot-reloading new model versions without restart
                     
                      - **MLOps Benefit**: Easy A/B testing, gradual rollout, and model swapping.
                     
                      - ### 4. **Prediction Pipeline** (`/src/predict/`)
                     
                      - Modular prediction logic:
                     
                      - - **predictor.py**: Core prediction engine with risk segmentation
                        - - **feature_builder.py** (inferred): Feature engineering from raw route/weather data
                          - - Consistent feature generation (references feature lists by version)
                            - - Thresholding logic for severity classification
                             
                              - **MLOps Benefit**: Reproducible inference pipeline, versioned feature engineering.
                             
                              - ### 5. **Live Data Integration** (`/src/live/`)
                             
                              - Real-time data collection for inference:
                             
                              - - **routes.py**: Fetch route geometry and traffic data from Google Maps
                                - - **weather.py**: Fetch live weather conditions from OpenWeather API
                                  - - **geocoding.py**: Address-to-coordinates conversion
                                   
                                    - **MLOps Benefit**: Pipeline-ready data integration, testable data fetching.
                                   
                                    - ### 6. **API with Model Endpoints** (`/api.py`)
                                   
                                    - FastAPI service hosting model predictions:
                                   
                                    - ```python
                                      @app.post("/predict")
                                      async def predict(request: PredictRequest):
                                          # Loads live route + weather data
                                          # Runs through prediction pipeline
                                          # Returns severity, risk score, safer alternatives
                                      ```

                                      - **Endpoints**:
                                      -   - `POST /predict` - Single route risk assessment
                                          -   - `POST /predict/segmented` - Multi-segment analysis with granular risk breakdown
                                              - - **Auto-generated docs**: http://localhost:8000/docs (Swagger UI)
                                                - - **Validation**: Pydantic models ensure request schema compliance
                                                 
                                                  - **MLOps Benefit**: RESTful interface for production consumers, built-in validation.
                                                 
                                                  - ### 7. **Unit Tests** (`/tests`)
                                                 
                                                  - Comprehensive testing per module:
                                                 
                                                  - - **test_geocoding.py**: Address resolution validation
                                                    - - **test_predictor.py**: Model inference correctness
                                                      - - **test_routes.py**: Route data retrieval
                                                        - - **test_weather.py**: Weather API integration
                                                         
                                                          - **MLOps Benefit**: Regression detection, safe refactoring, CI/CD-ready.
                                                         
                                                          - ### 8. **Reproducible Deployment** (`requirements.txt`, `.env.example`)
                                                         
                                                          - - **requirements.txt**: Pinned dependencies (FastAPI, Uvicorn, Supabase, Pandas, LightGBM, etc.)
                                                            - - **.env.example**: Template for environment variables (no secrets exposed)
                                                              - - **.dockerignore**: Excludes unnecessary files from Docker build
                                                               
                                                                - **MLOps Benefit**: Bit-for-bit reproducible deployments across environments.
                                                               
                                                                - ---

                                                                ## 🚀 How to Run Locally

                                                                ### Prerequisites

                                                                - Python 3.11+
                                                                - - Docker (optional, for containerized testing)
                                                                  - - Google Maps API key (for routes)
                                                                    - - OpenWeather API key (for weather)
                                                                      - - Supabase URL and API key (for database)
                                                                       
                                                                        - ### Local Development Setup
                                                                       
                                                                        - ```bash
                                                                          # Clone repository
                                                                          git clone https://github.com/machackgo/boston-smart-accident-risk-rerouting-mlops
                                                                          cd boston-smart-accident-risk-rerouting-mlops

                                                                          # Create virtual environment
                                                                          python3 -m venv venv
                                                                          source venv/bin/activate  # On Windows: venv\Scripts\activate

                                                                          # Install dependencies
                                                                          pip install -r requirements.txt

                                                                          # Create .env from template
                                                                          cp .env.example .env
                                                                          # Edit .env with your API keys:
                                                                          #   GOOGLE_SERVER_API_KEY=your_key
                                                                          #   OPENWEATHER_API_KEY=your_key
                                                                          #   SUPABASE_URL=your_url
                                                                          #   SUPABASE_KEY=your_key

                                                                          # Run the API server
                                                                          uvicorn api:app --reload --host 0.0.0.0 --port 8000

                                                                          # Navigate to http://localhost:8000/docs for interactive API docs
                                                                          ```

                                                                          ### Run Unit Tests

                                                                          ```bash
                                                                          pytest tests/ -v
                                                                          ```

                                                                          ### Docker Build & Run

                                                                          ```bash
                                                                          # Build Docker image
                                                                          docker build -t boston-mlops:latest .

                                                                          # Run container
                                                                          docker run -p 8000:8000 \
                                                                            -e GOOGLE_SERVER_API_KEY=your_key \
                                                                            -e OPENWEATHER_API_KEY=your_key \
                                                                            -e SUPABASE_URL=your_url \
                                                                            -e SUPABASE_KEY=your_key \
                                                                            boston-mlops:latest

                                                                          # Access API at http://localhost:8000
                                                                          ```

                                                                          ### Deploy to GCP Cloud Run

                                                                          ```bash
                                                                          # Authenticate with GCP
                                                                          gcloud auth login
                                                                          gcloud config set project YOUR_PROJECT_ID

                                                                          # Build and push to Artifact Registry
                                                                          gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/boston-mlops

                                                                          # Deploy to Cloud Run
                                                                          gcloud run deploy boston-mlops \
                                                                            --image gcr.io/YOUR_PROJECT_ID/boston-mlops \
                                                                            --platform managed \
                                                                            --region us-central1 \
                                                                            --allow-unauthenticated \
                                                                            --set-env-vars="GOOGLE_SERVER_API_KEY=...,OPENWEATHER_API_KEY=...,SUPABASE_URL=...,SUPABASE_KEY=..."
                                                                          ```

                                                                          ---

                                                                          ## 🛠️ Skills Demonstrated

                                                                          | Skill Category | Evidence in Repo |
                                                                          |---|---|
                                                                          | **ML Operations** | Multi-version model management, reproducible feature engineering (feature_list_v*.txt), serialized model artifacts with metadata (MODEL_CARD_*.md) |
                                                                          | **Model Serving** | FastAPI endpoints (/api.py), Uvicorn ASGI server, model loading abstraction (src/model/), request validation (Pydantic) |
                                                                          | **Containerization** | Production Dockerfile with slim base image, environment variable handling, port exposure, deployment-ready configuration |
                                                                          | **Data Pipeline** | Feature engineering pipeline (src/predict/), live data integration (src/live/routes.py, weather.py, geocoding.py), caching strategy |
                                                                          | **Infrastructure as Code** | Dockerfile, requirements.txt versioning, .env.example template, Cloud Run-ready containerized deployment-ready |
                                                                          | **Testing & Quality** | Comprehensive unit tests (test_*.py), modular code design, integration test coverage for external APIs |
                                                                          | **API Design** | RESTful endpoints, auto-generated OpenAPI docs, Pydantic validation, error handling |
                                                                          | **Database Integration** | Supabase PostgreSQL integration for live data, query optimization for production loads |

                                                                          ---

                                                                          ## ✅ VeriBridge Proof Evidence

                                                                          | Skill Proven | Evidence | Location |
                                                                          |---|---|---|
                                                                          | Model Versioning | 4 serialized model binaries (v1–v4) with matching feature lists | `/models/best_model_v*.pkl`, `/models/feature_list_v*.txt` |
                                                                          | Model Documentation | Model cards documenting training data, metrics, limitations | `/models/MODEL_CARD_v*.md` |
                                                                          | Feature Engineering | Reproducible feature specification; weather column selection config | `/models/feature_list_*.txt`, `/models/weather_keep_cols_v4.json` |
                                                                          | Production API | FastAPI app with /predict and /predict/segmented endpoints | `/api.py:lines 17–80` (FastAPI import, endpoint definitions) |
                                                                          | Model Loading | Abstracted model serving layer independent from API | `/src/model/` (inferred from imports in api.py) |
                                                                          | Prediction Pipeline | Feature generation + model inference + threshold application | `/src/predict/predictor.py`, `/src/predict/feature_builder.py` |
                                                                          | Live Data Integration | Real-time route and weather data fetching | `/src/live/routes.py`, `/src/live/weather.py`, `/src/live/geocoding.py` |
                                                                          | Containerization | Production-grade Dockerfile with Python 3.11-slim, Uvicorn setup | `/Dockerfile:lines 1–20` |
                                                                          | Deployment Readiness | GCP Cloud Run configuration (port 8000, env vars, ASGI server) | `/Dockerfile`, `/api.py:lines 1–10` (docs reference) |
                                                                          | Unit Testing | Modular tests for all production components | `/tests/test_predictor.py`, `/tests/test_routes.py`, `/tests/test_weather.py`, `/tests/test_geocoding.py` |
                                                                          | Configuration Management | Environment variable abstraction with .env.example template | `/.env.example`, `/api.py` (environ usage) |

                                                                          ---

                                                                          ## 💼 Recruiter Value

                                                                          1. **Production ML Maturity**: Demonstrates ability to move beyond notebooks → production-grade, containerized ML systems with versioning and deployment infrastructure.
                                                                         
                                                                          2. 2. **MLOps Best Practices**: Model versioning, reproducible feature engineering, modular prediction pipelines, and infrastructure-as-code (Dockerfile, requirements.txt).
                                                                            
                                                                             3. 3. **Safety-Critical Systems**: Accident risk prediction requires rigorous testing, validation, and clear model documentation — all present in this repo.
                                                                               
                                                                                4. 4. **Full Deployment Stack**: From model artifact management through Docker containerization to GCP Cloud Run readiness — end-to-end infrastructure.
                                                                                  
                                                                                   5. 5. **Scalable Architecture**: Stateless FastAPI service with horizontal scaling ready (Cloud Run auto-scales based on requests).
                                                                                     
                                                                                      6. 6. **Integration Skills**: Combines ML (LightGBM), data pipelines (Pandas, feature engineering), APIs (FastAPI), containerization (Docker), and cloud platforms (GCP).
                                                                                        
                                                                                         7. 7. **Testing Discipline**: Comprehensive unit tests ensure API and data pipeline reliability — crucial for production systems handling safety-critical predictions.
                                                                                           
                                                                                            8. ---
                                                                                           
                                                                                            9. ## 📈 Future Improvements
                                                                                           
                                                                                            10. ### Missing MLOps Pieces (Planned)
                                                                                           
                                                                                            11. - **CI/CD Pipeline**: GitHub Actions workflow for automated testing, Docker image building, and Cloud Run-ready containerized deployment on push
                                                                                                - - **Monitoring & Observability**: Prometheus metrics, structured logging, model performance tracking dashboards
                                                                                                  - - **Model Registry**: Centralized model versioning (MLflow, Weights & Biases) with model lineage and experiment tracking
                                                                                                    - - **A/B Testing Framework**: Gradual model rollout with traffic splitting and performance comparison
                                                                                                      - - **Data Validation**: Great Expectations for feature drift detection and data quality monitoring
                                                                                                        - - **Documentation Generation**: Automated API documentation and data lineage diagrams
                                                                                                         
                                                                                                          - ### Feature Enhancements
                                                                                                         
                                                                                                          - - **Vertex AI Integration**: Optional GCP Vertex AI Model Registry and Endpoints for managed inference
                                                                                                            - - **BigQuery Integration**: Centralized data warehouse for training data, predictions logging, and analytics
                                                                                                              - - **Model Explainability**: SHAP values for prediction transparency in safety-critical decisions
                                                                                                                - - **Rate Limiting & Caching**: API-level rate limiting and Redis caching for high-throughput scenarios
                                                                                                                  - - **Batch Prediction**: Support for bulk route risk assessment via async job scheduling
                                                                                                                   
                                                                                                                    - ---
                                                                                                                    
                                                                                                                    ## 📚 References
                                                                                                                    
                                                                                                                    - [FastAPI Documentation](https://fastapi.tiangolo.com/)
                                                                                                                    - - [LightGBM Documentation](https://lightgbm.readthedocs.io/)
                                                                                                                    - [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
                                                                                                                    - - [GCP Cloud Run](https://cloud.google.com/run)
                                                                                                                      - - [Supabase Documentation](https://supabase.com/docs)
                                                                                                                        - - [Python Packaging Best Practices](https://packaging.python.org/)
                                                                                                                         
                                                                                                                          - ---
                                                                                                                          
                                                                                                                          **Built with precision for production. Safety is our priority.**
