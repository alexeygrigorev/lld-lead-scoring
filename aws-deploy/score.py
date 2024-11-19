import argparse

from datetime import datetime

import pandas as pd
import mlflow

from common import create_log_features, prepare_features


def load_model_from_s3(model_bucket, run_id):
    """Load MLflow model directly from S3 using URI."""
    model_uri = f"s3://{model_bucket}/1/{run_id}/artifacts/final_model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def score_leads(
    users_s3_path,
    logs_s3_path,
    model_bucket,
    run_id,
):
    # Load model directly from S3
    model = load_model_from_s3(model_bucket, run_id)
    
    # Load data directly from S3
    df_users = pd.read_csv(users_s3_path)
    df_logs = pd.read_csv(logs_s3_path)
    
    df_users['signup_date'] = pd.to_datetime(df_users['signup_date'])
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    # Create features
    log_features = create_log_features(
        df_users, 
        df_logs, 
        df_logs['timestamp'].max()
    )
    
    df_scored = df_users.merge(log_features, on='user_id', how='left')
    feature_dicts = prepare_features(df_scored)
    
    # Generate predictions
    scores = model.predict_proba(feature_dicts)[:, 1]
    
    # Create output DataFrame
    results = pd.DataFrame({
        'user_id': df_users['user_id'],
        'conversion_probability': scores.round(4)
    })
    
    results['version'] = run_id
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Score leads using trained model from S3')
    parser.add_argument('--model-bucket', required=True,
                      help='S3 bucket containing the models')
    parser.add_argument('--run-id', required=True,
                      help='ID of the model run')
    parser.add_argument('--input-prefix', required=True,
                      help='S3 prefix for input data files')
    parser.add_argument('--output-prefix', required=True,
                      help='S3 prefix for output predictions')
    
    args = parser.parse_args()
    
    # Construct paths
    users_path = f"s3://{args.input_prefix}/prod-random-users.csv"
    logs_path = f"s3://{args.input_prefix}/prod-random-logs.csv"
    
    # Generate predictions
    results = score_leads(
        users_path,
        logs_path,
        args.model_bucket,
        args.run_id,
    )
    
    # Construct output path with current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_path = f"s3://{args.output_prefix}/score-{current_date}.csv"
    
    # Save results to S3
    results.to_csv(output_path, index=False)
    print(f"Scores saved to {output_path}")

if __name__ == "__main__":
    main()