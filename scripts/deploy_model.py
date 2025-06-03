# scripts/deploy_model.py
import os
import sys
import yaml
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment, 
    Model,
    Environment,
    CodeConfiguration
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

def load_azure_config(config_path: str = "config/azure_config.yaml") -> dict:
    """Load Azure configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_deployment_config(config_path: str = "config/deployment_config.yaml") -> dict:
    """Load deployment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_ml_client(config: dict) -> MLClient:
    """Create Azure ML client."""
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['azure']['subscription_id'],
        resource_group_name=config['azure']['resource_group'],
        workspace_name=config['azure']['workspace_name']
    )
    
    print(f"✅ Connected to Azure ML workspace: {config['azure']['workspace_name']}")
    return ml_client

def validate_model_artifacts(model_path: str) -> bool:
    """Validate that model artifacts exist and are complete."""
    print("🔍 Validating model artifacts...")
    
    model_dir = Path(model_path)
    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_path}")
        print("   Run training first: python scripts/submit_training.py")
        return False
    
    # Check for essential files
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]
    
    # Check for model files (either pytorch_model.bin or adapter files)
    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
    
    missing_files = []
    for file_name in required_files:
        if not (model_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    if not model_files:
        print("❌ No model weight files found (.bin or .safetensors)")
        return False
    
    print(f"✅ Model artifacts validated in {model_path}")
    print(f"   Found {len(model_files)} model weight files")
    
    return True

def create_inference_script():
    """Create the inference script for the endpoint."""
    inference_script = '''
import json
import torch
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def init():
    """Initialize the model and tokenizer."""
    global model, tokenizer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        model_path = os.environ.get("AZUREML_MODEL_DIR", "./model")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapter if present
        adapter_config = os.path.join(model_path, "adapter_config.json")
        if os.path.exists(adapter_config):
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()  # Merge LoRA weights
        
        model.eval()
        logging.info("Model initialized successfully")
        
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise e

def run(raw_data):
    """Run inference on the input data."""
    try:
        # Parse input
        data = json.loads(raw_data)
        
        # Handle both single requests and batch requests
        if isinstance(data, dict):
            inputs = [data]
        else:
            inputs = data
        
        responses = []
        
        for input_data in inputs:
            # Extract input text
            if "messages" in input_data:
                # Chat format
                messages = input_data["messages"]
                text = format_chat_messages(messages)
            elif "prompt" in input_data:
                # Direct prompt
                text = input_data["prompt"]
            elif "instruction" in input_data:
                # Instruction format
                system_msg = input_data.get("system", "You are a helpful corporate AI assistant.")
                instruction = input_data["instruction"]
                text = format_instruction(system_msg, instruction)
            else:
                raise ValueError("Input must contain 'messages', 'prompt', or 'instruction'")
            
            # Generate response
            response = generate_response(text, input_data)
            responses.append(response)
        
        # Return single response or list
        return responses[0] if len(responses) == 1 else responses
        
    except Exception as e:
        return {"error": str(e)}

def format_chat_messages(messages):
    """Format chat messages into Llama format."""
    formatted = "<|begin_of_text|>"
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        if role == "system":
            formatted += f"<|start_header_id|>system<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "user":
            formatted += f"<|start_header_id|>user<|end_header_id|>\\n\\n{content}<|eot_id|>"
        elif role == "assistant":
            formatted += f"<|start_header_id|>assistant<|end_header_id|>\\n\\n{content}<|eot_id|>"
    
    # Add assistant header for generation
    formatted += "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
    
    return formatted

def format_instruction(system_msg, instruction):
    """Format instruction into Llama format."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

def generate_response(prompt, input_data):
    """Generate response from the model."""
    # Get generation parameters
    max_new_tokens = input_data.get("max_new_tokens", 512)
    temperature = input_data.get("temperature", 0.7)
    top_p = input_data.get("top_p", 0.9)
    do_sample = input_data.get("do_sample", True)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|start_header_id|>assistant<|end_header_id|>" in response:
        assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        assistant_response = assistant_response.replace("<|eot_id|>", "").strip()
    else:
        assistant_response = response
    
    return {
        "response": assistant_response,
        "prompt_tokens": len(inputs["input_ids"][0]),
        "completion_tokens": len(outputs[0]) - len(inputs["input_ids"][0])
    }
'''
    
    # Save inference script
    os.makedirs("src/deployment", exist_ok=True)
    with open("src/deployment/score.py", "w") as f:
        f.write(inference_script)
    
    print("✅ Inference script created: src/deployment/score.py")

def register_model(ml_client: MLClient, model_path: str, model_name: str) -> Model:
    """Register the trained model in Azure ML."""
    print("📝 Registering model in Azure ML...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        model = Model(
            name=model_name,
            version=timestamp,
            description="Fine-tuned Llama-3.2-2B for corporate Q&A",
            type="custom_model",
            path=model_path,
            tags={
                "framework": "transformers",
                "task": "text-generation",
                "base_model": "llama-3.2-2b",
                "fine_tuning": "unsloth"
            }
        )
        
        registered_model = ml_client.models.create_or_update(model)
        
        print(f"✅ Model registered successfully")
        print(f"   Name: {registered_model.name}")
        print(f"   Version: {registered_model.version}")
        
        return registered_model
        
    except Exception as e:
        print(f"❌ Failed to register model: {e}")
        raise e

def create_or_get_endpoint(ml_client: MLClient, deployment_config: dict) -> ManagedOnlineEndpoint:
    """Create or get the managed online endpoint."""
    endpoint_name = deployment_config['deployment']['endpoint_name']
    
    print(f"🔗 Setting up endpoint: {endpoint_name}")
    
    try:
        # Try to get existing endpoint
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        print(f"✅ Found existing endpoint: {endpoint_name}")
        return endpoint
        
    except ResourceNotFoundError:
        print(f"📝 Creating new endpoint: {endpoint_name}")
        
        # Create new endpoint
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Endpoint for fine-tuned Llama corporate Q&A model",
            auth_mode="key",
            tags={
                "model": "llama-3.2-2b",
                "task": "corporate-qa",
                "framework": "transformers"
            }
        )
        
        try:
            endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            print(f"✅ Endpoint created successfully: {endpoint_name}")
            return endpoint
            
        except Exception as e:
            print(f"❌ Failed to create endpoint: {e}")
            raise e

def create_deployment(
    ml_client: MLClient, 
    endpoint: ManagedOnlineEndpoint,
    model: Model,
    deployment_config: dict
) -> ManagedOnlineDeployment:
    """Create the model deployment."""
    
    config = deployment_config['deployment']
    deployment_name = config['deployment_name']
    
    print(f"🚀 Creating deployment: {deployment_name}")
    
    # Create inference environment
    inference_env = Environment(
        name="llama-inference-env",
        description="Environment for Llama inference",
        conda_file="inference_environment.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu20.04"
    )
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint.name,
        model=model,
        environment=inference_env,
        code_configuration=CodeConfiguration(
            code="src/deployment",
            scoring_script="score.py"
        ),
        instance_type=config['instance_type'],
        instance_count=config['instance_count'],
        request_settings={
            "request_timeout_ms": config['request_timeout_ms'],
            "max_concurrent_requests_per_instance": config['max_concurrent_requests_per_instance'],
            "max_queue_wait_ms": config['max_queue_wait_ms']
        },
        liveness_probe={
            "initial_delay": config['liveness_probe']['initial_delay'],
            "period": config['liveness_probe']['period'],
            "timeout": config['liveness_probe']['timeout'],
            "success_threshold": config['liveness_probe']['success_threshold'],
            "failure_threshold": config['liveness_probe']['failure_threshold']
        },
        readiness_probe={
            "initial_delay": config['readiness_probe']['initial_delay'],
            "period": config['readiness_probe']['period'],
            "timeout": config['readiness_probe']['timeout'],
            "success_threshold": config['readiness_probe']['success_threshold'],
            "failure_threshold": config['readiness_probe']['failure_threshold']
        }
    )
    
    try:
        deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
        print(f"✅ Deployment created successfully: {deployment_name}")
        return deployment
        
    except Exception as e:
        print(f"❌ Failed to create deployment: {e}")
        raise e

def set_traffic_to_deployment(ml_client: MLClient, endpoint_name: str, deployment_name: str):
    """Set 100% traffic to the new deployment."""
    print("🔄 Setting traffic to new deployment...")
    
    try:
        # Get current endpoint
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        
        # Set traffic to 100% for our deployment
        endpoint.traffic = {deployment_name: 100}
        
        # Update endpoint
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        print(f"✅ Traffic set to 100% for deployment: {deployment_name}")
        
    except Exception as e:
        print(f"❌ Failed to set traffic: {e}")
        raise e

def test_endpoint(ml_client: MLClient, endpoint_name: str) -> bool:
    """Test the deployed endpoint with a sample request."""
    print("🧪 Testing endpoint...")
    
    # Sample test data
    test_data = {
        "instruction": "How to create a new project in Azure ML?",
        "system": "You are a helpful corporate AI assistant that helps employees with internal processes.",
        "max_new_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        # Test the endpoint
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=None,
            deployment_name=None,  # Use default traffic allocation
            request_body=json.dumps(test_data)
        )
        
        print("✅ Endpoint test successful!")
        print(f"   Sample response: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ Endpoint test failed: {e}")
        return False

def print_deployment_summary(endpoint, deployment, deployment_config: dict):
    """Print deployment summary."""
    print("\n" + "="*80)
    print("🎉 MODEL DEPLOYMENT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Endpoint Name: {endpoint.name}")
    print(f"Endpoint URL: {endpoint.scoring_uri}")
    print(f"Deployment Name: {deployment.name}")
    print(f"Instance Type: {deployment_config['deployment']['instance_type']}")
    print(f"Instance Count: {deployment_config['deployment']['instance_count']}")
    print(f"Authentication: Key-based")
    
    print("\n📋 Next Steps:")
    print("1. Test endpoint: python src/deployment/endpoint_test.py")
    print("2. Get endpoint keys from Azure ML Studio")
    print("3. Use endpoint for inference in your applications")
    
    print("\n🔑 Endpoint Details:")
    print(f"   Scoring URI: {endpoint.scoring_uri}")
    print(f"   Swagger URI: {endpoint.openapi_uri}")
    print("="*80)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Llama model to Azure ML endpoint")
    parser.add_argument("--azure_config", type=str, default="config/azure_config.yaml",
                       help="Path to Azure config file")
    parser.add_argument("--deployment_config", type=str, default="config/deployment_config.yaml",
                       help="Path to deployment config file")
    parser.add_argument("--model_path", type=str, default="output/models/final_model",
                       help="Path to trained model")
    parser.add_argument("--model_name", type=str, default="llama-corporate-qa",
                       help="Model name for registration")
    parser.add_argument("--skip_test", action="store_true",
                       help="Skip endpoint testing")
    
    args = parser.parse_args()
    
    try:
        # Load configurations
        print("📋 Loading configurations...")
        azure_config = load_azure_config(args.azure_config)
        deployment_config = load_deployment_config(args.deployment_config)
        
        print("="*80)
        print("🚀 DEPLOYING LLAMA MODEL TO AZURE ML")
        print("="*80)
        print(f"Model Path: {args.model_path}")
        print(f"Endpoint: {deployment_config['deployment']['endpoint_name']}")
        print(f"Instance Type: {deployment_config['deployment']['instance_type']}")
        
        # Create ML client
        ml_client = get_ml_client(azure_config)
        
        # Validate model artifacts
        if not validate_model_artifacts(args.model_path):
            print("❌ Model validation failed. Please check model artifacts.")
            return False
        
        # Create inference script
        create_inference_script()
        
        # Register model
        model = register_model(ml_client, args.model_path, args.model_name)
        
        # Create or get endpoint
        endpoint = create_or_get_endpoint(ml_client, deployment_config)
        
        # Create deployment
        deployment = create_deployment(ml_client, endpoint, model, deployment_config)
        
        # Set traffic to deployment
        set_traffic_to_deployment(
            ml_client, 
            endpoint.name, 
            deployment.name
        )
        
        # Test endpoint
        if not args.skip_test:
            test_endpoint(ml_client, endpoint.name)
        
        # Print summary
        print_deployment_summary(endpoint, deployment, deployment_config)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Deployment interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Verify model artifacts exist and are complete")
        print("2. Check Azure ML quota for inference instances")
        print("3. Verify endpoint name is unique in workspace")
        print("4. Check deployment configuration parameters")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)