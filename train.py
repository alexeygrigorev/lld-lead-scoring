#!/usr/bin/env python3

import argparse
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

from common import create_log_features, prepare_features

def prepare_lead_scoring_data(df_users, df_logs, train_end_date, val_end_date):
    df_users = df_users.copy()
    df_logs = df_logs.copy()
    
    df_users['signup_date'] = pd.to_datetime(df_users['signup_date'])
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    train_end_date = pd.to_datetime(train_end_date)
    val_end_date = pd.to_datetime(val_end_date)
    
    train_mask = df_users['signup_date'] < train_end_date
    val_mask = (df_users['signup_date'] >= train_end_date) & (df_users['signup_date'] < val_end_date)
    test_mask = df_users['signup_date'] >= val_end_date
    
    df_train = df_users[train_mask].copy()
    df_val = df_users[val_mask].copy()
    df_test = df_users[test_mask].copy()
    
    train_features = create_log_features(df_users, df_logs, train_end_date)
    val_features = create_log_features(df_users, df_logs, val_end_date)
    test_features = create_log_features(df_users, df_logs, df_logs['timestamp'].max())

    df_train = df_train.merge(train_features, on='user_id', how='left')
    df_val = df_val.merge(val_features, on='user_id', how='left')
    df_test = df_test.merge(test_features, on='user_id', how='left')

    return df_train, df_val, df_test

def train_model(users_path, logs_path, train_end_date, val_end_date, 
                mlflow_tracking_uri, mlflow_experiment_name):
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Load data
    df_users = pd.read_csv(users_path)
    df_logs = pd.read_csv(logs_path)
    
    # Prepare datasets
    df_train, df_val, df_test = prepare_lead_scoring_data(
        df_users,
        df_logs,
        train_end_date=train_end_date,
        val_end_date=val_end_date
    )
    
    # Prepare features
    train_dicts = prepare_features(df_train)
    val_dicts = prepare_features(df_val)
    test_dicts = prepare_features(df_test)
    
    y_train = df_train['converted'].values
    y_val = df_val['converted'].values
    y_test = df_test['converted'].values
    
    # Model parameters
    model_params = {
        'n_estimators': 80,
        'max_depth': 9,
        'min_samples_leaf': 43,
        'min_samples_split': 98,
        'class_weight': 'balanced',
        'random_state': 1
    }
    
    with mlflow.start_run():
        mlflow.log_params(model_params)
        
        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestClassifier(**model_params)
        )
        
        # Train on combined train+val data
        full_train = train_dicts + val_dicts
        y_full = np.concatenate([y_train, y_val])
        pipeline.fit(full_train, y_full)
        
        # Calculate and log test AUC
        y_test_pred = pipeline.predict_proba(test_dicts)[:, 1]
        test_auc = roc_auc_score(y_test, y_test_pred)
        mlflow.log_metric("test_auc", test_auc)
        print(f"Test AUC: {test_auc:.4f}")
        
        # Train final model on all data
        all_data = train_dicts + val_dicts + test_dicts
        y_all = np.concatenate([y_train, y_val, y_test])
        
        final_model = make_pipeline(
            DictVectorizer(),
            RandomForestClassifier(**model_params)
        )
        
        final_model.fit(all_data, y_all)
        mlflow.sklearn.log_model(
            final_model, 
            "final_model",
            registered_model_name="lead_scoring"
        )

def main():
    parser = argparse.ArgumentParser(description='Train lead scoring model')
    parser.add_argument('--users-path', required=True, help='Path to users CSV file')
    parser.add_argument('--logs-path', required=True, help='Path to logs CSV file')
    parser.add_argument('--train-end-date', required=True, 
                      help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--val-end-date', required=True,
                      help='End date for validation data (YYYY-MM-DD)')
    parser.add_argument('--mlflow-tracking-uri', default="http://localhost:5000",
                      help='MLflow tracking URI')
    parser.add_argument('--mlflow-experiment-name', default="model",
                      help='MLflow experiment name')
    
    args = parser.parse_args()
    
    train_model(
        args.users_path,
        args.logs_path,
        args.train_end_date,
        args.val_end_date,
        args.mlflow_tracking_uri,
        args.mlflow_experiment_name
    )

if __name__ == "__main__":
    main()