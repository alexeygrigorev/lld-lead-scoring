#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Create a clean deployment directory
rm -rf deployment
mkdir -p deployment/package
cd deployment

echo "Installing dependencies..."
pip install --target ./package -r ../requirements.txt

echo "Cleaning up unnecessary files..."
cd package

rm -rf \
    matplotlib* \
    mlflow/server/ \
    PIL* \
    pillow* \
    fontTools* \
    databricks* \
    google* \
    opentelemetry* \
    alembic* \
    graphql* \
    werkzeug* \
    git* \
    docker* \
    jinja2* \
    flask* \
    gunicorn* \
    markdown* \
    yaml* \
    sqlalchemy* \
    mako* \
    cloudpickle* \
    greenlet* \
    contourpy* \
    protobuf* \
    sqlparse* \
    tzdata* \
    share/ \
    tests/ \
    *.dist-info/

rm -rf boto3 botocore

# Clean up test and example directories
find . -type d -name "tests" -exec rm -rf {} +
find . -type d -name "examples" -exec rm -rf {} +

# Clean up compiled Python files and source files, but keep .so
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete
find . -name "*.h" -delete
find . -name "*.c" -delete
find . -name "*.cpp" -delete

# Remove specific test directories
rm -rf pandas/tests/
rm -rf numpy/tests/
rm -rf sklearn/tests/
rm -rf */tests/
rm -rf */*/tests/
rm -rf scipy/stats/tests/

# Keep numpy's core files
touch numpy/__init__.py
touch numpy/core/__init__.py
touch numpy/random/__init__.py


echo "Copying source files..."
cp ../../*.py .

echo "Creating deployment package..."

zip -r ../deployment.zip .
cd ..

DEPLOYMENT_BUCKET="ldd-alexey-cms-data"
DEPLOYMENT_KEY="lambda/LDD-CMP-Score.zip"

aws s3 cp deployment.zip s3://${DEPLOYMENT_BUCKET}/${DEPLOYMENT_KEY}


echo "Updating Lambda function..."
# aws lambda update-function-code \
#     --function-name LDD-CMP-Score \
#     --zip-file fileb://deployment.zip

aws lambda update-function-code \
    --function-name LDD-CMP-Score \
    --s3-bucket ${DEPLOYMENT_BUCKET} \
    --s3-key ${DEPLOYMENT_KEY}

echo "Deployment complete!"