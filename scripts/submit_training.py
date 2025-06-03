# scripts/submit_training.py
import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime

from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import CommandJob
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

def load_azure_config(config_path: str = "config/azure_config.yaml") -> dict:
    """Load Azure configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_training_config(config_path: str = "config/training_config.yaml") -> dict:
    """Load training configuration."""
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

def validate_prerequisites(ml_client: MLClient, config: dict) -> bool:
    """Validate that required resources exist."""
    print("🔍 Validating prerequisites...")
    
    # Check compute cluster
    compute_name = config['compute']['cluster_name']
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"✅ Compute cluster found: {compute_name} ({compute.size})")
    except ResourceNotFoundError:
        print(f"❌ Compute cluster not found: {compute_name}")
        print("   Run: python scripts/setup_cluster.py")
        return False
    
    # Check environment
    env_name = config['environment']['name']
    try:
        environment = ml_client.environments.get(env_name, label="latest")
        print(f"✅ Environment found: {env_name}")
    except ResourceNotFoundError:
        print(f"❌ Environment not found: {env_name}")
        print("   Run: python scripts/setup_cluster.py")
        return False
    
    # Check processed data
    data_files = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    data_dir = Path("data/processed")
    
    missing_files = []
    for file_name in data_files:
        if not (data_dir / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing data files: {missing_files}")
        print("   Run: python data/data_preprocessing.py")
        return False
    else:
        print(f"✅ All data files found in {data_dir}")
    
    return True

def upload_data_to_datastore(ml_client: MLClient, local_data_path: str = "data/processed") -> str:
    """Upload processed data to Azure ML datastore."""
    print("📤 Uploading data to Azure ML datastore...")
    
    # Create a unique path for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    remote_path = f"llama_training_data/{timestamp}"
    
    try:
        # Upload data folder
        data_asset = ml_client.data.create_or_update({
            "name": f"llama-training-data-{timestamp}",
            "description": "Processed training data for Llama fine-tuning",
            "type": AssetTypes.URI_FOLDER,
            "path": local_data_path
        })
        
        print(f"✅ Data uploaded successfully")
        print(f"   Asset name: {data_asset.name}")
        print(f"   Version: {data_asset.version}")
        
        return f"azureml:{data_asset.name}:{data_asset.version}"
        
    except Exception as e:
        print(f"❌ Failed to upload data: {e}")
        raise e

def create_training_job(
    ml_client: MLClient, 
    azure_config: dict, 
    training_config: dict,
    data_uri: str,
    job_name: str = None
) -> CommandJob:
    """Create the training job."""
    
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"llama_finetune_{timestamp}"
    
    print(f"📋 Creating training job: {job_name}")
    
    # Define the command to run
    command = """
python src/training/train.py \
    --config config/training_config.yaml \
    --model_config config/model_config.yaml \
    --data_dir ${{inputs.data}} \
    --output_dir ${{outputs.model}}
"""
    
    # Create the job
    job = CommandJob(
        # Job identification
        display_name=job_name,
        description="Fine-tune Llama-3.2-2B on corporate Q&A dataset using Unsloth",
        experiment_name=azure_config.get('experiment', {}).get('name', 'llama-finetune'),
        
        # Compute and environment
        compute=azure_config['compute']['cluster_name'],
        environment=f"{azure_config['environment']['name']}@latest",
        
        # Code and command
        code="./",  # Upload current directory
        command=command,
        
        # Inputs and outputs
        inputs={
            "data": Input(
                type=AssetTypes.URI_FOLDER,
                path=data_uri,
                mode=InputOutputModes.RO_MOUNT
            )
        },
        outputs={
            "model": Output(
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT
            )
        },
        
        # Resource configuration
        instance_count=1,  # Single node training
        
        # Tags for organization
        tags={
            "model": "llama-3.2-2b",
            "task": "instruction-tuning",
            "framework": "unsloth",
            "dataset": "corporate-qa"
        }
    )
    
    return job

def submit_and_monitor_job(ml_client: MLClient, job: CommandJob, stream_logs: bool = True):
    """Submit job and monitor progress."""
    print("🚀 Submitting training job...")
    
    try:
        # Submit the job
        submitted_job = ml_client.jobs.create_or_update(job)
        
        print(f"✅ Job submitted successfully!")
        print(f"   Job name: {submitted_job.name}")
        print(f"   Job ID: {submitted_job.id}")
        print(f"   Status: {submitted_job.status}")
        print(f"   Studio URL: {submitted_job.studio_url}")
        
        if stream_logs:
            print("\n📊 Streaming logs (Ctrl+C to stop log streaming, job will continue)...")
            print("="*80)
            
            try:
                # Stream logs
                ml_client.jobs.stream(submitted_job.name)
                
            except KeyboardInterrupt:
                print("\n⚠️  Log streaming stopped. Job continues running in background.")
                print(f"   Monitor at: {submitted_job.studio_url}")
        
        # Wait for completion (with periodic status updates)
        print("\n⏳ Waiting for job completion...")
        
        while True:
            job_status = ml_client.jobs.get(submitted_job.name)
            print(f"   Status: {job_status.status} - {datetime.now().strftime('%H:%M:%S')}")
            
            if job_status.status in ["Completed", "Failed", "Canceled"]:
                break
                
            time.sleep(60)  # Check every minute
        
        # Final status
        final_job = ml_client.jobs.get(submitted_job.name)
        
        if final_job.status == "Completed":
            print("🎉 Training job completed successfully!")
            return final_job, True
        else:
            print(f"❌ Training job failed with status: {final_job.status}")
            return final_job, False
            
    except Exception as e:
        print(f"❌ Failed to submit or monitor job: {e}")
        raise e

def download_model_artifacts(ml_client: MLClient, job, output_dir: str = "output/models"):
    """Download trained model artifacts."""
    print("📥 Downloading model artifacts...")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Download model output
        ml_client.jobs.download(
            name=job.name,
            download_path=output_dir,
            output_name="model"
        )
        
        print(f"✅ Model artifacts downloaded to: {output_dir}")
        
        # List downloaded files
        model_path = Path(output_dir) / "named-outputs" / "model"
        if model_path.exists():
            files = list(model_path.rglob("*"))
            print(f"   Downloaded {len(files)} files")
            
            # Show key files
            key_files = ['pytorch_model.bin', 'config.json', 'tokenizer.json', 'adapter_model.bin']
            for key_file in key_files:
                if any(key_file in f.name for f in files):
                    print(f"   ✅ Found: {key_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to download model artifacts: {e}")
        return False

def print_job_summary(job, success: bool, azure_config: dict):
    """Print job completion summary."""
    print("\n" + "="*80)
    if success:
        print("🎉 TRAINING JOB COMPLETED SUCCESSFULLY!")
    else:
        print("❌ TRAINING JOB FAILED!")
    print("="*80)
    
    print(f"Job Name: {job.name}")
    print(f"Status: {job.status}")
    print(f"Duration: {job.creation_context.created_at} to {datetime.now()}")
    print(f"Compute: {azure_config['compute']['cluster_name']}")
    print(f"Studio URL: {job.studio_url}")
    
    if success:
        print("\n📋 Next Steps:")
        print("1. Check model artifacts in: output/models/")
        print("2. Deploy model: python scripts/deploy_model.py")
        print("3. Test inference: python src/deployment/endpoint_test.py")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Check logs in Azure ML Studio")
        print("2. Verify compute cluster has sufficient quota")
        print("3. Check data format and model configuration")
    
    print("="*80)

def main():
    """Main submission function."""
    parser = argparse.ArgumentParser(description="Submit Llama training job to Azure ML")
    parser.add_argument("--azure_config", type=str, default="config/azure_config.yaml",
                       help="Path to Azure config file")
    parser.add_argument("--training_config", type=str, default="config/training_config.yaml",
                       help="Path to training config file")
    parser.add_argument("--job_name", type=str, default=None,
                       help="Custom job name")
    parser.add_argument("--no_logs", action="store_true",
                       help="Don't stream logs")
    parser.add_argument("--no_download", action="store_true",
                       help="Don't download model artifacts")
    
    args = parser.parse_args()
    
    try:
        # Load configurations
        print("📋 Loading configurations...")
        azure_config = load_azure_config(args.azure_config)
        training_config = load_training_config(args.training_config)
        
        print("="*80)
        print("🚀 SUBMITTING LLAMA TRAINING JOB")
        print("="*80)
        print(f"Workspace: {azure_config['azure']['workspace_name']}")
        print(f"Compute: {azure_config['compute']['cluster_name']}")
        print(f"Environment: {azure_config['environment']['name']}")
        
        # Create ML client
        ml_client = get_ml_client(azure_config)
        
        # Validate prerequisites
        if not validate_prerequisites(ml_client, azure_config):
            print("❌ Prerequisites not met. Please fix issues and try again.")
            return False
        
        # Upload data
        data_uri = upload_data_to_datastore(ml_client)
        
        # Create training job
        job = create_training_job(
            ml_client, 
            azure_config, 
            training_config, 
            data_uri,
            args.job_name
        )
        
        # Submit and monitor job
        completed_job, success = submit_and_monitor_job(
            ml_client, 
            job, 
            stream_logs=not args.no_logs
        )
        
        # Download model artifacts if successful
        if success and not args.no_download:
            download_model_artifacts(ml_client, completed_job)
        
        # Print summary
        print_job_summary(completed_job, success, azure_config)
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Job submission interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Job submission failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Verify Azure ML workspace access")
        print("2. Check compute cluster is running")
        print("3. Ensure data preprocessing completed")
        print("4. Verify environment setup")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)