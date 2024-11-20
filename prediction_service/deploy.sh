#!/bin/bash

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
PROJECT_ID=$(gcloud config get-value project)

# Build the image using Google Cloud Build
gcloud builds submit --tag gcr.io/$PROJECT_ID/stock-predictor .

# Deploy to Cloud Run with increased memory
gcloud run deploy stock-predictor \
  --image gcr.io/$PROJECT_ID/stock-predictor \
  --platform managed \
  --region northamerica-northeast1 \
  --allow-unauthenticated \
  --memory 1Gi

echo "Deployment complete!" 