# 🚀 PipeSim v1.0: Distributed ML Pipeline Simulator

**A high-performance infrastructure framework designed to simulate, benchmark, and stress-test distributed Machine Learning pipelines.**

PipeSim is a model-agnostic environment for evaluating the lifecycle of data as it moves through a modern production stack. It allows engineers to identify architectural bottlenecks, test scalability, and implement MLOps best practices before deploying heavy deep learning models to production environments.

## 🏗️ Core Architecture

The project implements a classic **Asynchronous Task Queue** pattern to decouple high-latency ML inference from the user-facing API:

* **FastAPI Gateway**: Acts as the entry point, managing request validation and providing synchronous "Instant Previews".
* **Redis & Celery**: A robust distributed task queue for heavy-duty background processing.
* **PostgreSQL**: Relational storage for persistent metadata, task history, and inference logs.
* **GenAI Integration (Gemini Pro)**: Automated analytical layer providing human-readable insights from raw model outputs.

## 📂 Supported Pipeline Archetypes

PipeSim is designed to handle diverse data modalities, demonstrating its flexibility:

1. **Computer Vision (YOLOv11)**: Real-time object detection and frame-by-frame analysis for industrial safety and monitoring scenarios.
2. **Tabular Classification (Titanic Dataset)**: A classic ML pipeline implementation. 
   - **Data Ingestion**: Handling structured CSV/JSON payloads.
   - **Preprocessing**: On-the-fly feature engineering (Age handling, Cabin mapping, etc.).
   - **Inference**: Binary classification to predict survival probability, proving the system's ability to manage standard Scikit-learn or XGBoost workloads alongside Deep Learning.

## 📈 Performance & Observability (Locust & Grafana)

PipeSim is built for transparency. During the final benchmarking phase (Day 69), the system was subjected to a **50-user concurrent stress test**:

| Metric | Benchmark Result |
| **Throughput Stability** | **99.7%** Success Rate (2000+ Requests) |
| **Average POST Latency** | **10.1s** (under extreme 50-user load) |
| **System Resilience** | Handled 50 parallel data streams on a single node |

**Key Finding:** Stress testing identified a synchronization bottleneck in the Python Event Loop during intensive CPU/GPU inference. This justifies the future migration to **NVIDIA Triton Inference Server** for optimal hardware utilization.

## 🚀 DevOps & MLOps Features

* **Optimized Docker Engine**: Multi-stage builds utilizing the `uv` package manager. Reduced image footprint by **40%**.
* **Full-Stack Monitoring**: Integrated **Prometheus** and **Grafana** for real-time tracking of GPU, RAM, and RPS metrics.
* **Automated CI**: GitHub Actions pipeline executing **14+ integration tests** on every push.
* **WSL2 Optimization**: Custom `.wslconfig` implementation to manage memory swap and prevent engine crashes.

## 🛠️ Installation & Setup

### 1. Requirements
* Docker & Docker Compose
* NVIDIA Container Toolkit (Optional)

### 2. Configuration
Create a `.env` file in the root directory:
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/pipesim_db
GEMINI_API_KEY=[YOUR_GEMINI_API_KEY]

### 3. Deployment

*docker-compose up -d --build*
**Frontend Dashboard**: http://localhost:8502

**API Interactive Docs**: http://localhost:8000/docs

**Monitoring Hub**: http://localhost:3000 (Grafana)

### 👤 Author
Valera (GitHub: Gydron | **Also link**:https://github.com/valeramlops)

**Focus**: MLOps / Backend Infrastructure / Data Engineering

**Contact**: TG: @jolichaos

Developed as part of a **70-day** intensive MLOps engineering sprint.