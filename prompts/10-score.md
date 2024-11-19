**I have a script for generating data, now I want to make it more configurable**

**in particular, i want to specify via CLI:**

* **the start and end date**
* **the user id from which the generation starts (and later it increments)**
* **number of users**
* **the prefix for generated data**

**(+Attached the generate_data.py script)**


I'll help you modify the script to make it more configurable using command-line arguments. We'll use Python's `argparse` module for this purpose.

[generate_data_prod.py](../data/generate_data_prod.py)

I've modified the script to make it more configurable through command-line arguments. Here are the key changes:

1. Added CLI arguments for:
   - `--start-date`: Start date in YYYY-MM-DD format
   - `--end-date`: End date in YYYY-MM-DD format
   - `--start-user-id`: Starting user ID number
   - `--prefix`: Prefix for output files
   - `--num-users`: Number of users to generate
   - `--seed`: Random seed for reproducibility

2. Modified the DemoDataGenerator class to:
   - Accept start_date and end_date as parameters
   - Use start_user_id for user ID generation
   - Ensure all timestamps fall within the specified date range

3. Added better error handling and date parsing

You can now run the script with various configurations. Examples:

```bash
# Use default values
python script.py

# Specify custom date range
python script.py --start-date 2024-04-01 --end-date 2024-06-30

# Generate data for specific user ID range
python script.py --start-user-id 1001 --num-users 500

# Custom output files prefix
python script.py --prefix production-data

# Combine multiple options
python script.py --start-date 2024-01-01 --end-date 2024-12-31 --start-user-id 5000 --num-users 1000 --prefix annual-data
```

Would you like me to make any additional modifications to the configuration options?


--- 


**I have a model registered in Mlflow and now I want to create a script for scoring (score.py) the data. It loads the model from the registry and scores the provided csv files (via CLI). here's the script we used for training (train.py), so all the pre-processing for features should be the same**

**refactor the code in such a way that the common code between these two files in in common.py**

**the name of the model in mlflow is lead_scoring and version is 1. the url is http://localhost:5000/#/models/lead_scoring/versions/1**

**I also want to promote this model to production, but it seems it's done differently now in mlflow 2.17 - please let me know how I can do it, so instead of referring to the model version number, I can refer to something like "stage" or "prod" or something similar**

**this is how the model is loaded now and used for scoring:**

```python
logged_model = 'runs:/92db624fe3e149dba3d0b5f9166e2c23/final_model'
loaded_model = mlflow.sklearn.load_model(logged_model)
loaded_model.predict_proba(val_dicts)[:, 1]
```

**(+Attached train.py)**

I'll help create the scoring script and refactor the common code. Here's how to structure it:


```python
# common.py
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

# score.py
import argparse
import mlflow
import pandas as pd
from datetime import datetime
from common import create_log_features, prepare_features

def score_leads(users_path, logs_path, mlflow_tracking_uri="http://localhost:5000"):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load production model
    model = mlflow.sklearn.load_model("models:/lead_scoring/Production")
    
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
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Score leads using trained model')
    parser.add_argument('--users-path', required=True, help='Path to users CSV file')
    parser.add_argument('--logs-path', required=True, help='Path to logs CSV file')
    parser.add_argument('--output-path', required=True, help='Path to save scores CSV')
    parser.add_argument('--mlflow-tracking-uri', default="http://localhost:5000",
                      help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    results = score_leads(
        args.users_path,
        args.logs_path,
        args.mlflow_tracking_uri
    )
    
    results.to_csv(args.output_path, index=False)
    print(f"Scores saved to {args.output_path}")

if __name__ == "__main__":
    main()
```

To promote the model to production in MLflow 2.17, use:

```python
import mlflow
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="lead_scoring",
    version=1,
    stage="Production"
)
```

Then you can reference it as `models:/lead_scoring/Production` instead of using run IDs or version numbers.

The scoring script can be run like:
```bash
python score.py --users-path users.csv --logs-path logs.csv --output-path scores.csv
```

--- 

**I also need updated train.py**

```python
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
        'n_estimators': 75,
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

```

Key changes made to the training script:
1. Moved common functions to common.py
2. Added model registration with `registered_model_name="lead_scoring"`
3. Kept the training logic and data splitting in train.py

---

**what does this line do?**

```python
mlflow.sklearn.log_model(
    final_model, 
    "final_model",
    registered_model_name="lead_scoring"
)
```

This line does three things:
1. Saves the trained model artifacts to MLflow tracking server
2. Names the model "final_model" within the current run
3. Registers the model in MLflow's Model Registry under the name "lead_scoring", creating a new version if this name already exists

This registration allows you to later reference the model using its registered name (like we do in score.py with `models:/lead_scoring/Production`) rather than using run IDs.

---

**To promote the model to production in MLflow 2.17, use:**

To promote the model to production in MLflow 2.17, use:

```
import mlflow
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="lead_scoring",
    version=1,
    stage="Production"
)
```

**how can we do it without code? from UI?**

Yes, you can promote the model to production in MLflow UI:

1. Go to `http://localhost:5000/#/models/lead_scoring`
2. Click on Version 1 of the model
3. Click the "Stage" dropdown menu (it likely says "None")
4. Select "Production" from the dropdown
5. Confirm the transition when prompted

---

**here's this thing alias, I used "prod" how can I now refer to this model using this alias?**

**(+Attached screenshot)**

To use the alias in your code, replace `models:/lead_scoring/Production` with `models:/lead_scoring@prod`. You can update the scoring code to:

```python
model = mlflow.sklearn.load_model("models:/lead_scoring@prod")
```

This will load the model version that has the "prod" alias assigned to it.


---

**I want to also record run_id for the prediction so I can later use it for debugging. but also I want to be able to refer to the model as @prod to make it easier to maintain and not worry about forgetting to update my score.py script when there's a new version**

**so I want to use MLFLow API to get the run ID for the model with alias @prod and use it when recording the predictions**

**(+Attached the score.py file)**

Here's the modified score_leads function that retrieves and records the run ID for the @prod model:

```python
def score_leads(
    users_path,
    logs_path,
    run_id=None,
    mlflow_tracking_uri="http://localhost:5000",
):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load production model
    if run_id is None:
        model_uri = "models:/lead_scoring@prod"
        client = mlflow.tracking.MlflowClient()
        model_details = client.get_latest_versions("lead_scoring", stages=["Production"])[0]
        run_id = model_details.run_id
    else:
        model_uri = f"runs:/{run_id}/final_model"
    
    model = mlflow.sklearn.load_model(model_uri)

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
        'conversion_probability': scores.round(4),
        'model_version': run_id,
        'scoring_time': datetime.now().isoformat()
    })
    
    return results

```

Key changes:
1. Uses MLflow client to get run ID for @prod model
2. Added scoring_time to track when predictions were made
3. Renamed version column to model_version for clarity
4. Records full run ID instead of just "prod" string

---

```
/workspaces/lld-lead-scoring/score.py:19: FutureWarning: mlflow.tracking.client.MlflowClient.get_latest_versions is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages
  model_details = client.get_latest_versions("lead_scoring", stages=["Production"])[0]
Traceback (most recent call last):
  File "/workspaces/lld-lead-scoring/score.py", line 79, in <module>
    main()
  File "/workspaces/lld-lead-scoring/score.py", line 68, in main
    results = score_leads(
              ^^^^^^^^^^^^
  File "/workspaces/lld-lead-scoring/score.py", line 19, in score_leads
    model_details = client.get_latest_versions("lead_scoring", stages=["Production"])[0]
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^
IndexError: list index out of range
```

Let's fix the error by using MLflow's newer `search_model_versions` API and adding better error handling:

```python
def score_leads(
    users_path,
    logs_path,
    run_id=None,
    mlflow_tracking_uri="http://localhost:5000",
):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    # Load production model
    if run_id is None:
        try:
            model_versions = client.search_model_versions(f"name = 'lead_scoring'")
            prod_versions = [mv for mv in model_versions if mv.current_stage == "Production"]
            if not prod_versions:
                raise ValueError("No production model found for lead_scoring")
            model_details = prod_versions[0]
            run_id = model_details.run_id
            model_uri = f"models:/lead_scoring@prod"
        except Exception as e:
            print(f"Error finding production model: {e}")
            raise
    else:
        model_uri = f"runs:/{run_id}/final_model"
    
    model = mlflow.sklearn.load_model(model_uri)

    # Rest of the function remains the same
    df_users = pd.read_csv(users_path)
    df_logs = pd.read_csv(logs_path)
    
    df_users['signup_date'] = pd.to_datetime(df_users['signup_date'])
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    log_features = create_log_features(
        df_users, 
        df_logs, 
        df_logs['timestamp'].max()
    )
    
    df_scored = df_users.merge(log_features, on='user_id', how='left')
    feature_dicts = prepare_features(df_scored)
    
    scores = model.predict_proba(feature_dicts)[:, 1]
    
    results = pd.DataFrame({
        'user_id': df_users['user_id'],
        'conversion_probability': scores.round(4),
        'model_version': run_id,
        'scoring_time': datetime.now().isoformat()
    })
    
    return results

```

This version:
1. Uses newer `search_model_versions` API
2. Adds error handling for missing production model
3. Keeps original functionality but with better stability


---

**our code becomes more complex so let's move this logic to a separate function**

```python
# code
```

---

```
Error finding production model: No production model found for lead_scoring
Traceback (most recent call last):
  File "/workspaces/lld-lead-scoring/score.py", line 94, in <module>
    main()
  File "/workspaces/lld-lead-scoring/score.py", line 83, in main
    results = score_leads(
              ^^^^^^^^^^^^
  File "/workspaces/lld-lead-scoring/score.py", line 40, in score_leads
    model, run_id = load_model_with_run_id(run_id, mlflow_tracking_uri)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/workspaces/lld-lead-scoring/score.py", line 21, in load_model_with_run_id
    raise ValueError("No production model found for lead_scoring")
ValueError: No production model found for lead_scoring
```

It seems we need to handle both "Production" stage and "prod" alias. Let's modify the function to try both approaches:

```python
def load_model_with_run_id(run_id=None, mlflow_tracking_uri="http://localhost:5000"):
    """
    Load MLflow model and get its run ID. If run_id is None, loads production model.
    Returns tuple of (model, run_id).
    """
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    if run_id is None:
        try:
            # Try loading directly with prod alias first
            model = mlflow.sklearn.load_model("models:/lead_scoring@prod")
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name = 'lead_scoring'")
            latest_version = sorted(model_versions, key=lambda x: x.version, reverse=True)[0]
            run_id = latest_version.run_id
            return model, run_id
        except Exception as e:
            print(f"Warning: Could not load model with @prod alias: {e}")
            print("Trying to load latest model version...")
            
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"name = 'lead_scoring'")
            if not model_versions:
                raise ValueError("No model versions found for lead_scoring")
                
            latest_version = sorted(model_versions, key=lambda x: x.version, reverse=True)[0]
            run_id = latest_version.run_id
            model_uri = f"runs:/{run_id}/final_model"
            model = mlflow.sklearn.load_model(model_uri)
            return model, run_id
    else:
        model_uri = f"runs:/{run_id}/final_model"
        model = mlflow.sklearn.load_model(model_uri)
        return model, run_id

def score_leads(
    users_path,
    logs_path,
    run_id=None,
    mlflow_tracking_uri="http://localhost:5000",
):
    model, run_id = load_model_with_run_id(run_id, mlflow_tracking_uri)

    df_users = pd.read_csv(users_path)
    df_logs = pd.read_csv(logs_path)
    
    df_users['signup_date'] = pd.to_datetime(df_users['signup_date'])
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    log_features = create_log_features(
        df_users, 
        df_logs, 
        df_logs['timestamp'].max()
    )
    
    df_scored = df_users.merge(log_features, on='user_id', how='left')
    feature_dicts = prepare_features(df_scored)
    
    scores = model.predict_proba(feature_dicts)[:, 1]
    
    results = pd.DataFrame({
        'user_id': df_users['user_id'],
        'conversion_probability': scores.round(4),
        'model_version': run_id,
        'scoring_time': datetime.now().isoformat()
    })
    
    return results
```

Now it will:
1. First try loading with @prod alias
2. If that fails, load the latest model version
3. If no models exist, raise an error
---

**explain what this code is doing**

```python
model = mlflow.sklearn.load_model("models:/lead_scoring@prod")
client = mlflow.tracking.MlflowClient()
model_versions = client.search_model_versions(f"name = 'lead_scoring'")
latest_version = sorted(model_versions, key=lambda x: x.version, reverse=True)[0]
run_id = latest_version.run_id
```

This code snippet:
1. Loads the model tagged as "prod" in MLflow
2. Creates MLflow client to access model metadata
3. Gets all versions of the "lead_scoring" model
4. Finds the latest version by sorting on version number
5. Extracts the run ID that created this version

We need both the model and its run ID - model for making predictions and run ID for tracking/debugging. The run ID is not directly available when loading with "@prod" alias, so we have to look it up separately.

---

**im not sure if this is correct logic - if we have model v2 aliased @prod and v3 not aliased as @prod, then v3 run id will be used**

You're right - this logic has a bug. When loading with "@prod" alias, we should get the specific version that's aliased as prod, not just the latest version. Here's the corrected version:

```python
def load_model_with_run_id(run_id=None, mlflow_tracking_uri="http://localhost:5000"):
    """Load MLflow model and get its run ID."""
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    
    if run_id is None:
        try:
            # Get the model version with prod alias
            model_versions = client.search_model_versions("name = 'lead_scoring' and tags.prod = 'true'")
            if not model_versions:
                raise ValueError("No production model found for lead_scoring")
            model_version = model_versions[0]
            run_id = model_version.run_id
            model = mlflow.sklearn.load_model(f"models:/lead_scoring@prod")
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
    
    df_users = pd.read_csv(users_path)
    df_logs = pd.read_csv(logs_path)
    
    df_users['signup_date'] = pd.to_datetime(df_users['signup_date'])
    df_logs['timestamp'] = pd.to_datetime(df_logs['timestamp'])
    
    log_features = create_log_features(
        df_users, 
        df_logs, 
        df_logs['timestamp'].max()
    )
    
    df_scored = df_users.merge(log_features, on='user_id', how='left')
    feature_dicts = prepare_features(df_scored)
    
    scores = model.predict_proba(feature_dicts)[:, 1]
    
    results = pd.DataFrame({
        'user_id': df_users['user_id'],
        'conversion_probability': scores.round(4),
        'model_version': run_id,
        'scoring_time': datetime.now().isoformat()
    })
    
    return results
```

The key change is using `tags.prod = 'true'` to find the specific version marked as production, rather than just taking the latest version.


--- 

(Asked in [MLFlow docs](https://mlflow.org/docs/latest/python_api/mlflow.client.html))

**I want to get run_id for model registered with alias prod - something like that models:/lead_scoring@prod**

```python
from mlflow import MlflowClient
client = mlflow.tracking.MlflowClient()
model_version = client.get_model_version_by_alias("lead_scoring", "prod")
run_id = model_version.run_id
print(run_id)
```
