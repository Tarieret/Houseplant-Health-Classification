# Houseplant Health Classification Using Amazon Rekognition

## Overview
An end-to-end computer vision pipeline that classifies houseplant health conditions (healthy vs unhealthy) using Amazon Rekognition Custom Labels. Built with a production-style architecture including automated inference via AWS Lambda.

## Architecture
```
Local Images → S3 Upload → Rekognition Custom Labels Training
                                        ↓
New Image → S3 (uploads/) → Lambda → Rekognition Inference → S3 (results/)
```

## Tech Stack
- **Python** — data preparation, augmentation, pipeline orchestration
- **Amazon Rekognition Custom Labels** — image classification model
- **Amazon S3** — image and results storage
- **AWS Lambda** — serverless inference pipeline
- **IAM** — permissions and security
- **scikit-learn** — train/test split and model evaluation
- **Albumentations** — image augmentation
- **boto3** — AWS SDK for Python

## Dataset
- 91 original images (67 healthy, 24 unhealthy) collected from personal houseplants
- Augmented to 201 training images to address class imbalance
- Available on Kaggle: [link]

## Pipeline
1. Images organized into healthy/unhealthy folders
2. Train/test split (80/20) with stratification
3. Augmentation applied to training set only
4. Images uploaded to S3 with manifest files for Rekognition
5. Rekognition Custom Labels model trained on 201 images
6. Model evaluated against 19 holdout test images
7. Lambda function deployed for automated inference

## Results
| Metric | Healthy | Unhealthy |
|--------|---------|-----------|
| Precision | 0.93 | 1.00 |
| Recall | 1.00 | 0.80 |
| F1 Score | 0.97 | 0.89 |
| **Overall Accuracy** | **0.95** | |

## Setup
### Prerequisites
- AWS account with appropriate permissions
- Python 3.11+
- AWS CLI configured

### Installation
```bash
pip install boto3 Pillow scikit-learn albumentations matplotlib pandas numpy tqdm torch torchvision
```

### Running the Project
1. Clone the repo
2. Add your images to `data/healthy/` and `data/unhealthy/`
3. Run the notebook top to bottom
4. To tear down AWS resources run the cleanup cell at the bottom

## Key Learnings
- Building a production-style ML pipeline on AWS
- Handling class imbalance with targeted augmentation
- Rekognition Custom Labels manifest file format
- Serverless inference with AWS Lambda
- boto3 for programmatic AWS resource management

## Author
Tarie Retimitimi
