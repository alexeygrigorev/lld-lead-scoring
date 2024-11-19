import json
import os
import traceback
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from score import score_leads, prepare_features

DEFAULT_PARAMS = {
    'model_bucket': 'ldd-alexey-mlflow-models',
    'input_prefix': 'ldd-alexey-cms-data/input',
    'output_prefix': 'ldd-alexey-cms-data/predictions'
}

def test_s3_permissions(s3_path):
    """Test specific S3 path permissions"""
    try:
        print(f"\nTesting permissions for: {s3_path}")
        parts = s3_path.replace("s3://", "").split("/")
        bucket = parts[0]
        key = "/".join(parts[1:])
        
        s3 = boto3.client('s3')
        
        # Test bucket access
        try:
            s3.head_bucket(Bucket=bucket)
            print(f"✓ Can access bucket: {bucket}")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            print(f"✗ Cannot access bucket: {bucket}")
            print(f"  Error: {error_code} - {str(e)}")
            return
            
        # Test bucket listing
        try:
            s3.list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
            print(f"✓ Can list objects in: {bucket}/{key}")
        except ClientError as e:
            print(f"✗ Cannot list objects in: {bucket}/{key}")
            print(f"  Error: {str(e)}")
            
        # Test IAM role
        sts = boto3.client('sts')
        try:
            identity = sts.get_caller_identity()
            print("\nCurrent IAM Identity:")
            print(f"Account: {identity['Account']}")
            print(f"ARN: {identity['Arn']}")
            print(f"UserId: {identity['UserId']}")
        except Exception as e:
            print(f"✗ Cannot get IAM identity: {str(e)}")

    except Exception as e:
        print(f"Error testing S3 permissions: {str(e)}")
        print(traceback.format_exc())

def lambda_handler(event, context):
    try:
        print("\n=== Starting lambda execution ===")
        
        print("\n2. Testing S3 Paths:")
        # Test input paths
        users_path = f"s3://{DEFAULT_PARAMS['input_prefix']}/prod-random-users.csv"
        logs_path = f"s3://{DEFAULT_PARAMS['input_prefix']}/prod-random-logs.csv"
        model_path = f"s3://{DEFAULT_PARAMS['model_bucket']}"
        output_base = f"s3://{DEFAULT_PARAMS['output_prefix']}"
        
        # for path in [users_path, logs_path, model_path, output_base]:
        #     test_s3_permissions(path)
            
        print("\n3. Processing Event:")
        print(json.dumps(event, indent=2))
        
        # Extract parameters
        params = event.get('queryStringParameters', {})
        if not params and 'body' in event and event['body']:
            params = json.loads(event['body'])
        params = {**DEFAULT_PARAMS, **params}
        
        if 'run_id' not in params:
            print("ERROR: Missing run_id")
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameter: run_id',
                    'error_type': 'ValidationError'
                })
            }
            
        print("\n4. Starting Prediction Generation")
        print(f"Parameters: {json.dumps(params, indent=2)}")

        try:
            results = score_leads(
                users_s3_path=users_path,
                logs_s3_path=logs_path,
                model_bucket=params['model_bucket'],
                run_id=params['run_id']
            )
            print(f"Successfully generated {len(results)} predictions")
            
            # Save results
            current_date = datetime.now().strftime('%Y-%m-%d')
            output_path = f"s3://{params['output_prefix']}/score-{current_date}.csv"
            results.to_csv(output_path, index=False)
            print(f"Successfully saved results to {output_path}")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': f'Scores saved to {output_path}',
                    'output_path': output_path,
                    'num_predictions': len(results)
                })
            }
            
        except Exception as e:
            print("\nError during prediction generation:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            print(traceback.format_exc())
            
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                })
            }
            
    except Exception as e:
        print("\n!!! Lambda execution failed !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            })
        }

if __name__ == "__main__":
    test_event = {
        "body": "{\"run_id\": \"7e7664049dad47a4b299a96c47fec50d\"}"
    }
    
    class TestContext:
        function_name = "local-test"
        memory_limit_in_mb = 128
        def get_remaining_time_in_millis(self):
            return 300000
    
    result = lambda_handler(test_event, TestContext())
    print("\nTest result:")
    print(json.dumps(result, indent=2))