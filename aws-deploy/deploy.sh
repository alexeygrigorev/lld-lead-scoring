#!/bin/bash
set -e

# Configuration
AWS_ACCOUNT_ID="387546586013"
AWS_REGION="eu-west-1"
FUNCTION_NAME="LDD-CMP-Score"
ECR_REPOSITORY_NAME="ldd-cmp-score"
ECR_REPOSITORY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY_NAME}"

echo "Building and deploying Docker image for ${FUNCTION_NAME}..."

# Authenticate Docker to ECR
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com


# Build Docker image
docker build -t ${ECR_REPOSITORY_NAME} .

# Tag and push the image
docker tag ${ECR_REPOSITORY_NAME}:latest ${ECR_REPOSITORY}:latest
docker push ${ECR_REPOSITORY}:latest

# Update Lambda function
aws lambda update-function-code \
    --function-name ${FUNCTION_NAME} \
    --image-uri ${ECR_REPOSITORY}:latest

echo "Deployment complete!"