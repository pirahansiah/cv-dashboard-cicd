# CV Dashboard CI/CD

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?style=flat&logo=github-actions&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

CI/CD pipeline for an image/video Business Intelligence dashboard in a computer vision application using Large Language Models (LLMs) and GitHub Actions.

## Overview

This repository implements a complete CI/CD pipeline for deploying computer vision-powered BI dashboards. It automates testing, building, and deploying CV applications with integrated LLM capabilities for intelligent data analysis and visualization.

## What's New (2025-2026)

- **GitHub Actions Workflows**: Multi-stage CI/CD with matrix testing across Python 3.10-3.12
- **Container Registry**: Automated Docker image builds to GitHub Container Registry (ghcr.io)
- **Infrastructure as Code**: Terraform/Pulumi for reproducible cloud deployments
- **LLM Integration**: Automated model validation and A/B testing in deployment pipeline
- **Monitoring**: Prometheus + Grafana dashboards for CV pipeline metrics
- **Security Scanning**: Trivy container scanning and Snyk dependency checks
- **Feature Flags**: Gradual rollout for new CV model versions

## Architecture

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│   Code   │───▶│  Build   │───▶│  Test    │───▶│  Deploy  │
│  Push    │    │  Stage   │    │  Stage   │    │  Stage   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │              │               │               │
  GitHub         Docker          Pytest/        Kubernetes
  Actions       Build           Lint/Test       / ECS
```

## CI/CD Pipeline

### Stages

1. **Lint & Static Analysis**: Ruff, mypy, bandit security checks
2. **Unit Tests**: pytest with coverage reporting
3. **Integration Tests**: Docker-based service tests
4. **Model Validation**: Inference accuracy benchmarks
5. **Build**: Multi-arch Docker images (amd64, arm64)
6. **Deploy**: Blue-green deployment to cloud

### GitHub Actions Workflow

```yaml
# Key pipeline steps:
# 1. Code quality checks
ruff check . && mypy . && bandit -r .

# 2. Unit tests
pytest tests/ --cov=src --cov-report=xml

# 3. Docker build
docker build -t ghcr.io/${{ github.repository }}:${{ github.sha }} .

# 4. Deploy
kubectl apply -f k8s/
```

## Tech Stack

| Category | Tool | Version |
|----------|------|---------|
| **CI/CD** | GitHub Actions | Latest |
| **Container** | Docker | 24+ |
| **Orchestration** | Kubernetes / Docker Compose | 1.29+ / 2.x |
| **Monitoring** | Prometheus + Grafana | Latest |
| **Security** | Trivy, Snyk | Latest |
| **Cloud** | AWS ECS/EKS, Azure Container Apps | Latest |
| **IaC** | Terraform | 1.7+ |
| **CV Models** | PyTorch, Ultralytics | 2.x, 11.x |
| **LLM** | Ollama, OpenAI API | Latest |

## Dashboard Components

| Service | Description |
|---------|-------------|
| **CV Inference API** | FastAPI endpoint for object detection / segmentation |
| **BI Dashboard** | Plotly Dash / Streamlit for visualization |
| **LLM Service** | Ollama-powered data storytelling |
| **Monitoring** | Grafana dashboards with real-time metrics |

## Quick Start

```bash
# Clone and setup
git clone https://github.com/pirahansiah/cv-dashboard-cicd.git
cd cv-dashboard-cicd

# Run locally
docker-compose up --build

# Access dashboard
open http://localhost:8050
```

## Pipeline Configuration

```bash
# Set GitHub secrets:
# REGISTRY_URL: Container registry URL
# AWS_ACCESS_KEY_ID: AWS credentials
# AWS_SECRET_ACCESS_KEY: AWS secret
# KUBE_CONFIG: Kubernetes cluster config
```

## Monitoring & Observability

- **Metrics**: Prometheus scrapes `/metrics` endpoints
- **Dashboards**: Grafana dashboards for inference latency, throughput, error rates
- **Alerts**: PagerDuty / Slack integration for pipeline failures
- **Logging**: ELK stack / Loki for centralized log aggregation

## Security

- Container images scanned with Trivy on every build
- Dependency scanning with Snyk
- Secret detection with gitleaks
- RBAC for deployment permissions
- Network policies for service isolation

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Kubernetes Deployment Strategies](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Prometheus Monitoring](https://prometheus.io/docs/)

## License

MIT License
