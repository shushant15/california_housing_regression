# End-to-End MLOps Pipeline

## Project Overview

This project implements a complete MLOps pipeline for the California Housing dataset using Linear Regression. The pipeline includes model training, Docker containerization, CI/CD automation with GitHub Actions, and model optimization through manual quantization.

## Objectives

* Train a Linear Regression model using sklearn on the California Housing dataset.
* Save the trained model as `model.joblib`.
* Containerize the application using Docker for reproducibility.
* Configure GitHub Actions workflows for:
    * Model Training
    * Docker Build and Verification
    * Push Docker Image to Docker Hub
* Perform manual quantization to reduce model size while maintaining accuracy.

## Directory Structure

```
.
├── .github/
│ └── workflows/
│ └── ci.yml
├── src/
│ ├── train.py
│ ├── predict.py
│ └── quantize.py
├── models/
│ ├── model.joblib
│ ├── unquant_params.joblib
│ └── quant_params.joblib
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup Instructions

Create and activate a virtual environment:

```bash
python -m venv venv  
source venv/bin/activate    # For Linux/Mac  
venv\Scripts\activate       # For Windows  
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training locally:

```bash
python train.py
```

Run prediction for verification:

```bash
python predict.py
```

Run quantization:

```bash
python quantize.py
```

## Docker Instructions

Build Docker Image:

```bash
docker build -t mlops-pipeline .
```

Run Docker Container:

```bash
docker run --rm mlops-pipeline
```

## GitHub Actions Workflow

`ci.yml`:

* Installs dependencies and trains the model.
* Builds Docker image and verifies container.
* Pushes image to Docker Hub using GitHub Secrets.

## Branching Strategy

* `main`: Initial setup with README and `.gitignore`
* `dev`: Implements model training logic
* `docker_ci`: Adds `Dockerfile` and GitHub Actions workflow
* `quantization`: Implements manual model quantization

## Results

| Metric     | Original Model | Quantized Model |
|------------|----------------|-----------------|
| R² Score   | 0.6053         | 0.6051          |
| Model Size | 0.40 KB        | 0.32 KB         |

Insight: Quantization reduced the model size by ~20% without significantly impacting accuracy.

## Author

Name: Shushant Kumar Tiwari
Roll No: G24AI1116


