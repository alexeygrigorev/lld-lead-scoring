import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime


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
        'timestamp': ['count', 'nunique'],
        'duration_seconds': ['sum', 'mean', 'std']
    }).round(2)
    
    engagement_metrics.columns = [
        'total_actions',
        'active_days',
        'total_duration',
        'avg_duration',
        'std_duration'
    ]
    
    category_counts = df_logs_filtered.groupby(['user_id', 'action_category']).size().unstack(
        fill_value=0
    ).add_prefix('category_')
    
    top_actions = df_logs_filtered['action_type'].value_counts().nlargest(10).index
    action_counts = df_logs_filtered[df_logs_filtered['action_type'].isin(top_actions)]\
        .groupby(['user_id', 'action_type']).size().unstack(fill_value=0).add_prefix('action_')
    
    df_logs_filtered['hour'] = df_logs_filtered['timestamp'].dt.hour
    time_metrics = df_logs_filtered.groupby('user_id').agg({
        'hour': lambda x: len(x[x.between(9, 17)]) / len(x)
    }).round(2)
    time_metrics.columns = ['business_hours_ratio']
    
    df_logs_filtered['days_since_signup'] = (
        df_logs_filtered['timestamp'] - 
        df_logs_filtered['user_id'].map(df_users.set_index('user_id')['signup_date'])
    ).dt.days
    
    recency_metrics = df_logs_filtered.groupby('user_id').agg({
        'days_since_signup': ['min', 'max']
    }).round(2)
    recency_metrics.columns = ['days_to_first_action', 'days_to_last_action']
    
    df_logs_filtered['prev_timestamp'] = df_logs_filtered.groupby('user_id')['timestamp'].shift(1)
    df_logs_filtered['time_between_actions'] = (
        df_logs_filtered['timestamp'] - df_logs_filtered['prev_timestamp']
    ).dt.total_seconds() / 3600
    
    engagement_patterns = df_logs_filtered.groupby('user_id').agg({
        'time_between_actions': ['mean', 'std']
    }).round(2)
    engagement_patterns.columns = ['avg_hours_between_actions', 'std_hours_between_actions']
    
    feature_exploration = df_logs_filtered[
        df_logs_filtered['action_type'] == 'view_features'
    ].groupby('user_id').size().to_frame('feature_views')
    
    log_features = pd.concat([
        engagement_metrics,
        category_counts,
        action_counts,
        time_metrics,
        recency_metrics,
        engagement_patterns,
        feature_exploration
    ], axis=1).reset_index()
    
    return log_features.fillna(0)

def prepare_features(df):
    df = df.copy()
    
    date_columns = ['signup_date', 'conversion_date']
    exclude_columns = ['user_id', 'converted'] + date_columns
    
    df = df.drop(columns=exclude_columns, errors='ignore')
    df = df.fillna(0)
    return preprocess_required_features(df)