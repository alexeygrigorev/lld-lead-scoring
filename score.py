import argparse
import mlflow
import pandas as pd
from datetime import datetime
from common import create_log_features, prepare_features


def load_model_with_run_id(run_id=None, mlflow_tracking_uri="http://localhost:5000"):
    """Load MLflow model and get its run ID."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    if run_id is None:
        try:
            model = mlflow.sklearn.load_model(f"models:/lead_scoring@prod")
            model_version = client.get_model_version_by_alias("lead_scoring", "prod")
            run_id = model_version.run_id
            return model, run_id
        except Exception as e:
            print(f"Error loading production model: {e}")
            raise
    else:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/final_model")
        return model, run_id

def score_leads(
    users_path,
    logs_path,
    run_id=None,
    mlflow_tracking_uri="http://localhost:5000",
):
    model, run_id = load_model_with_run_id(run_id, mlflow_tracking_uri)

    # Load and preprocess data
    df_users = pd.read_csv(users_path)
    df_logs = pd.read_csv(logs_path)
    
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
    parser = argparse.ArgumentParser(description='Score leads using trained model')
    parser.add_argument('--users-path', required=True, help='Path to users CSV file')
    parser.add_argument('--logs-path', required=True, help='Path to logs CSV file')
    parser.add_argument('--output-path', required=True, help='Path to save scores CSV')
    parser.add_argument('--mlflow-tracking-uri', default="http://localhost:5000",
                      help='MLflow tracking URI')
    parser.add_argument('--run-id', required=False, help='ID of the model run')
    
    args = parser.parse_args()
    
    results = score_leads(
        args.users_path,
        args.logs_path,
        args.run_id,
        args.mlflow_tracking_uri
    )
    
    results.to_csv(args.output_path, index=False)
    print(f"Scores saved to {args.output_path}")

if __name__ == "__main__":
    main()