#!/usr/bin/env python3

import argparse
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

import mlflow


def preprocess_required_features(df):
    df = df.copy()

    def process_feature_string(feature_string):
        features = [f.strip() for f in feature_string.split(',')]
        return {f'required_feature_{feature}': 1 for feature in features}
    
    feature_dicts = df['required_features'].apply(process_feature_string)
    df = df.drop('required_features', axis=1)
    record_dicts = df.to_dict('records')
    
    for record, feature_dict in zip(record_dicts, feature_dicts):
        record.update(feature_dict)
    
    return record_dicts

def create_log_features(df_users, df_logs, cutoff_date):
    df_logs_filtered = df_logs[df_logs['timestamp'] < cutoff_date].copy()
    
    engagement_metrics = df_logs_filtered.groupby('user_id').agg({
        'timestamp': ['count', 'nunique'],  # Total actions and unique days
        'duration_seconds': ['sum', 'mean', 'std']  # Time spent metrics
    }).round(2)
    
    engagement_metrics.columns = [
        'total_actions',
        'active_days',
        'total_duration',
        'avg_duration',
        'std_duration'
    ]
    
    # Action category distribution
    category_counts = df_logs_filtered.groupby(['user_id', 'action_category']).size().unstack(
        fill_value=0
    ).add_prefix('category_')
    
    # Action type distribution (top 10 most common)
    top_actions = df_logs_filtered['action_type'].value_counts().nlargest(10).index
    action_counts = df_logs_filtered[df_logs_filtered['action_type'].isin(top_actions)]\
        .groupby(['user_id', 'action_type']).size().unstack(fill_value=0).add_prefix('action_')
    
    # Time-based features
    df_logs_filtered['hour'] = df_logs_filtered['timestamp'].dt.hour
    time_metrics = df_logs_filtered.groupby('user_id').agg({
        'hour': lambda x: len(x[x.between(9, 17)]) / len(x)  # Fraction of activity during business hours
    }).round(2)
    time_metrics.columns = ['business_hours_ratio']
    
    # Activity patterns
    df_logs_filtered['days_since_signup'] = (
        df_logs_filtered['timestamp'] - 
        df_logs_filtered['user_id'].map(df_users.set_index('user_id')['signup_date'])
    ).dt.days
    
    recency_metrics = df_logs_filtered.groupby('user_id').agg({
        'days_since_signup': ['min', 'max']
    }).round(2)
    recency_metrics.columns = ['days_to_first_action', 'days_to_last_action']
    
    # Advanced engagement metrics
    df_logs_filtered['prev_timestamp'] = df_logs_filtered.groupby('user_id')['timestamp'].shift(1)
    df_logs_filtered['time_between_actions'] = (
        df_logs_filtered['timestamp'] - df_logs_filtered['prev_timestamp']
    ).dt.total_seconds() / 3600  # Convert to hours
    
    engagement_patterns = df_logs_filtered.groupby('user_id').agg({
        'time_between_actions': ['mean', 'std']
    }).round(2)
    engagement_patterns.columns = ['avg_hours_between_actions', 'std_hours_between_actions']
    
    # Feature importance indicators
    feature_exploration = df_logs_filtered[
        df_logs_filtered['action_type'] == 'view_features'
    ].groupby('user_id').size().to_frame('feature_views')
    
    # Combine all features
    log_features = pd.concat([
        engagement_metrics,
        category_counts,
        action_counts,
        time_metrics,
        recency_metrics,
        engagement_patterns,
        feature_exploration
    ], axis=1).reset_index()
    
    # Fill NaN values with 0 for new users or users with missing metrics
    log_features = log_features.fillna(0)
    
    return log_features


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


def prepare_features(df):
    df = df.copy()
    
    date_columns = ['signup_date', 'conversion_date']
    exclude_columns = ['user_id', 'converted'] + date_columns

    df = df.drop(columns=exclude_columns)
    df = df.fillna(0)
    feature_dict = preprocess_required_features(df)

    return feature_dict


def train_model(users_path, logs_path, train_end_date, val_end_date, 
                mlflow_tracking_uri, mlflow_experiment_name):
    
    # Set up MLflow
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
        'n_estimators': 75,
        'max_depth': 9,
        'min_samples_leaf': 43,
        'min_samples_split': 98,
        'class_weight': 'balanced',
        'random_state': 1
    }
    
    with mlflow.start_run():
        mlflow.log_params(model_params)
        
        # Train on combined train+val data
        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestClassifier(**model_params)
        )
        
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
        mlflow.sklearn.log_model(final_model, "final_model")


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