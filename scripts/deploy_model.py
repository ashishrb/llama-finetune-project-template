# scripts/deploy_model.py
import os
import sys
import yaml
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from src.training.utils import GPUMemoryManager, log_system_resources

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
    
    # ADD THIS: Initialize memory manager and check system resources
    memory_manager = GPUMemoryManager()
    log_system_resources()
    
    # ADD THIS: Estimate model size and validate against available memory
    model_size_gb = _estimate_model_size(model_dir)
    print(f"📏 Estimated model size: {model_size_gb:.2f}GB")
    
    # Check if we have enough memory for model validation
    memory_info = memory_manager.get_gpu_memory_info()
    if memory_info['total_gb'] > 0:  # GPU available
        available_memory = memory_info['total_gb'] - memory_info['allocated_gb']
        if model_size_gb > available_memory * 0.8:  # Use 80% as safety margin
            print(f"⚠️  Warning: Model size ({model_size_gb:.2f}GB) may exceed available GPU memory ({available_memory:.2f}GB)")
            print("   Consider using CPU deployment or larger GPU instance")
    
    # Check for essential files
    required_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json"
    ]

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

def _estimate_model_size(model_dir: Path) -> float:
    """Estimate model size in GB based on files."""
    total_size = 0
    
    # Get sizes of all model files
    model_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
    
    for file_path in model_files:
        if file_path.exists():
            total_size += file_path.stat().st_size
    
    # If no model files found, estimate based on config
    if total_size == 0:
        config_file = model_dir / "config.json"
        if config_file.exists():
            try:
                import json
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Rough estimation based on model parameters
                vocab_size = config.get('vocab_size', 32000)
                hidden_size = config.get('hidden_size', 2048)
                num_layers = config.get('num_hidden_layers', 24)
                
                # Rough calculation: parameters * 2 bytes (float16)
                estimated_params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 4
                total_size = estimated_params * 2  # 2 bytes per parameter for float16
                
            except Exception as e:
                print(f"⚠️  Could not estimate model size from config: {e}")
                total_size = 4 * 1024**3  # Default 4GB estimate
    
    return total_size / (1024**3)  # Convert to GB

def create_inference_script():
    """Create the inference script for the endpoint."""
    inference_script = '''
import json
import torch
import logging
import os
import gc
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ADD THIS: Memory management utilities
class InferenceMemoryManager:
    """Lightweight memory manager for inference."""
    
    def __init__(self):
        self.peak_memory = 0
        
    def get_gpu_memory_info(self):
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return {'allocated_gb': 0, 'total_gb': 0, 'usage_percent': 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        self.peak_memory = max(self.peak_memory, allocated)
        
        return {
            'allocated_gb': allocated,
            'total_gb': total,
            'usage_percent': allocated / total if total > 0 else 0,
            'peak_gb': self.peak_memory
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Global memory manager
memory_manager = InferenceMemoryManager()

def init():
    """Initialize the model and tokenizer."""
    global model, tokenizer
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        model_path = os.environ.get("AZUREML_MODEL_DIR", "./model")
        
        # ADD THIS: Log initial memory state
        memory_info = memory_manager.get_gpu_memory_info()
        logging.info(f"Initial GPU memory: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # ADD THIS: Check memory before model loading
        if memory_info['total_gb'] > 0:  # GPU available
            available_memory = memory_info['total_gb'] - memory_info['allocated_gb']
            logging.info(f"Available GPU memory before model load: {available_memory:.2f}GB")
        
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
        
        # ADD THIS: Log memory after model loading
        final_memory = memory_manager.get_gpu_memory_info()
        logging.info(f"Model loaded successfully. GPU memory: {final_memory['allocated_gb']:.2f}GB / {final_memory['total_gb']:.2f}GB ({final_memory['usage_percent']:.1%})")
        
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        # ADD THIS: Cleanup on error
        memory_manager.cleanup_memory()
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
        
        # ADD THIS: Monitor memory for batch processing
        if len(inputs) > 10:
            logging.info(f"Processing large batch: {len(inputs)} requests")
            memory_info = memory_manager.get_gpu_memory_info()
            logging.info(f"Memory before batch: {memory_info['allocated_gb']:.2f}GB ({memory_info['usage_percent']:.1%})")
        
        for i, input_data in enumerate(inputs):
            # ADD THIS: Memory cleanup every 20 requests in large batches
            if i > 0 and i % 20 == 0:
                memory_manager.cleanup_memory()
            
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
        
        # ADD THIS: Log final memory state for large batches
        if len(inputs) > 10:
            final_memory = memory_manager.get_gpu_memory_info()
            logging.info(f"Batch completed. Final memory: {final_memory['allocated_gb']:.2f}GB, Peak: {final_memory['peak_gb']:.2f}GB")
        
        # Return single response or list
        return responses[0] if len(responses) == 1 else responses
        
    except Exception as e:
        # ADD THIS: Cleanup on error
        memory_manager.cleanup_memory()
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
    
    # ADD THIS: Memory and size validation
    memory_manager = GPUMemoryManager()
    model_size_gb = _estimate_model_size(Path(model_path))

    print(f"📊 Model registration info:")
    print(f"   Model size: {model_size_gb:.2f}GB")
    print(f"   Available memory: {memory_manager.get_gpu_memory_info()['total_gb']:.2f}GB")
    
    # Check if model size is reasonable for deployment
    if model_size_gb > 50:  # Very large model
        print(f"⚠️  Warning: Large model size ({model_size_gb:.2f}GB) may cause deployment issues")
        print("   Consider model compression or larger deployment instances")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
            model = Model(
                name=model_name,
                version=timestamp,
                description=f"Fine-tuned Llama-3.2-2B for corporate Q&A (Size: {model_size_gb:.1f}GB)",
                type="custom_model",
                path=model_path,
                tags={
                    "framework": "transformers",
                    "task": "text-generation",
                    "base_model": "llama-3.2-2b",
                    "fine_tuning": "unsloth",
                    "model_size_gb": f"{model_size_gb:.2f}",
                    "memory_optimized": "true"
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
    instance_type = config['instance_type']
    
    print(f"🚀 Creating deployment: {deployment_name}")
    
    instance_memory_gb = _get_instance_memory(instance_type)
    model_size_gb = float(model.tags.get('model_size_gb', '8.0'))
    
    print(f"💾 Memory validation:")
    print(f"   Instance type: {instance_type}")
    print(f"   Instance memory: {instance_memory_gb}GB")
    print(f"   Model size: {model_size_gb}GB")
    print(f"   Memory ratio: {(model_size_gb / instance_memory_gb * 100):.1f}%")
    
    if model_size_gb > instance_memory_gb * 0.7:  # Model uses >70% of instance memory
        print(f"⚠️  Warning: Model size ({model_size_gb:.1f}GB) may be too large for instance ({instance_memory_gb}GB)")
        print("   Consider using a larger instance type")
    
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

def _get_instance_memory(instance_type: str) -> float:
    """Get approximate memory in GB for Azure ML instance types."""
    # Common Azure ML instance memory mappings (approximate)
    memory_map = {
        'Standard_NC6s_v3': 112,      # 6 vCPU, 112 GB RAM
        'Standard_NC12s_v3': 224,     # 12 vCPU, 224 GB RAM
        'Standard_NC24s_v3': 448,     # 24 vCPU, 448 GB RAM
        'Standard_NC24ads_A100_v4': 220,  # 24 vCPU, 220 GB RAM
        'Standard_NC40ads_H100_v5': 440,  # 40 vCPU, 440 GB RAM
        'Standard_ND40rs_v2': 672,    # 40 vCPU, 672 GB RAM
    }
    
    return memory_map.get(instance_type, 64)  # Default 64GB if unknown

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
        
        # ADD THIS: Log initial system state
        log_system_resources()
        
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