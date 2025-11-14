import boto3
import json
import time
import uuid
import datetime

# Initialize the Bedrock clients
region = 'us-east-1'
bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
bedrock = boto3.client('bedrock', region_name=region)
print(f"Using AWS region: {region}")

def parse_converse_response(response):
    """Parse the response from the Converse API"""
    if 'output' in response and 'message' in response['output']:
        message = response['output']['message']
        if 'content' in message:
            for content in message['content']:
                if 'text' in content:
                    return content.get('text', '')
    return "No text response found"

def create_inference_profile(profile_name, model_arn, tags):
    """Create Inference Profile using base model ARN"""
    try:
        if not hasattr(bedrock, 'create_inference_profile'):
            raise Exception("create_inference_profile method not available. Please update boto3: pip install --upgrade boto3")
        
        response = bedrock.create_inference_profile(
            inferenceProfileName=profile_name,
            description="test",
            modelSource={'copyFrom': model_arn},
            tags=tags
        )
        print(f"Created profile with ARN: {response.get('inferenceProfileArn', 'Unknown')}")
        return response
    except Exception as e:
        print(f"Error creating inference profile: {str(e)}")
        raise

def converse(model_id, messages):
    """Use the Converse API to engage in a conversation with the specified model"""
    try:
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={'maxTokens': 300}
        )
        parsed = parse_converse_response(response)
        print(f"Response text (truncated): {parsed[:50]}...")
        return response
    except Exception as e:
        print(f"Error in converse: {str(e)}")
        return None

def create_shared_profile():
    """Create a single profile to be reused across all iterations"""
    profile_name = f"cost_demo_shared_profile_{uuid.uuid4().hex[:8]}"
    tags = [{'key': 'dept', 'value': 'Dev'}, {'key': 'project', 'value': 'cost-demo'}]
    base_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-micro-v1:0"
    
    print(f"Creating shared profile {profile_name}")
    profile_response = create_inference_profile(profile_name, base_model_arn, tags)
    return profile_response['inferenceProfileArn']

def run_test(iteration, model_id):
    """Run a single test iteration using the shared profile"""
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration} - {datetime.datetime.now()}")
    print(f"{'='*50}")
    
    print(f"Using shared inference profile: {model_id}")
    
    prompts = [
        "Write a comprehensive technical documentation for deploying a multi-region, highly available microservices architecture on AWS.",
        "Analyze the current state of cloud computing in enterprise environments.",
        "Compare and contrast AWS container services: ECS, EKS, App Runner, Lambda, and Fargate.",
        "Design a real-time analytics platform for IoT devices at scale."
    ]
    
    prompt = prompts[iteration % len(prompts)]
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    response = converse(model_id, messages)
    time.sleep(1)
    return response

def main():
    """Main function to run the tests"""
    print("Starting test runs with shared profile for all iterations...")
    
    # Create one shared profile for all iterations
    shared_model_id = create_shared_profile()
    
    successful_runs = 0
    failed_runs = 0
    start_time = time.time()
    
    num_iterations = 50
    for i in range(1, num_iterations + 1):
        try:
            response = run_test(i, shared_model_id)
            
            if response:
                successful_runs += 1
            else:
                failed_runs += 1
                
        except Exception as e:
            print(f"Error in iteration {i}: {str(e)}")
            failed_runs += 1
        
        if i % 5 == 0:
            print(f"\nProgress: {i}/{num_iterations} iterations completed")
            print(f"Success rate: {successful_runs}/{i} ({successful_runs/i*100:.1f}%)")
            
        if i < num_iterations:
            time.sleep(2)
    
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total runs: {num_iterations}")
    print(f"Successful runs: {successful_runs}")
    print(f"Failed runs: {failed_runs}")
    print(f"Success rate: {successful_runs/num_iterations*100:.1f}%")
    print(f"Total execution time: {total_time:.1f} seconds")
    print(f"Average time per iteration: {total_time/num_iterations:.1f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()
