# CV Dashboard CI/CD

![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088FF?style=flat&logo=github-actions&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

CI/CD pipeline for an image/video BI dashboard — ML model training, testing, and deployment automation.

## Overview

Automated ML training pipeline with CI/CD, Docker containerization, and comprehensive testing. Trains a linear regression model on synthetic data, evaluates performance, and packages the model for deployment.

## Project Structure

```
cv-dashboard-cicd/
├── main.py           # ML training pipeline (generate data, train, save model)
├── test_main.py      # Pytest unit/integration/e2e tests
├── requirements.txt  # Python dependencies
├── pyproject.toml    # Modern Python project config (ruff, pytest, mypy)
├── Dockerfile        # Multi-stage Docker build
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml    # GitHub Actions: lint + matrix test (3.10-3.12)
├── README.md
└── LICENSE
```

## Quick Start

### Local

```bash
pip install -r requirements.txt

# Train model
python main.py

# Run tests
pytest test_main.py -v
```

### Docker

```bash
docker compose up --build
```

### CI/CD

The GitHub Actions pipeline runs on push/PR to `main`:

1. **Lint** — ruff check + format verification
2. **Build & Test** — Matrix testing across Python 3.10, 3.11, 3.12
3. **Test** — Downloads the trained model and runs pytest

## Testing

Tests are self-contained — the training function is called in fixtures, so no pre-existing model artifact is needed.

```bash
# Run all tests
pytest test_main.py -v

# With coverage
pytest test_main.py -v --cov=main --cov-report=term-missing
```

## 12-Month Roadmap

| Phase | Timeline | Milestone |
|-------|----------|-----------|
| **Phase 1** | Month 1-2 | FastAPI inference endpoint with OpenAPI docs |
| **Phase 2** | Month 3-4 | Plotly Dash dashboard for model metrics visualization |
| **Phase 3** | Month 5-6 | Ollama LLM integration for natural language data insights |
| **Phase 4** | Month 7-8 | Prometheus metrics + Grafana dashboards for monitoring |
| **Phase 5** | Month 9-10 | Kubernetes manifests, Helm charts for orchestrated deployment |
| **Phase 6** | Month 11-12 | Terraform IaC, multi-environment deployment (dev/staging/prod) |

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [scikit-learn Documentation](https://scikit-learn.org/)

## License

MIT License — See [LICENSE](LICENSE) for details.

## Author

**Dr. Farshid Pirahansiah** — [LinkedIn](https://linkedin.com/in/pirahansiah) | [GitHub](https://github.com/pirahansiah)
