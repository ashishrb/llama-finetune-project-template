# scripts/setup_cluster.py
import os
import sys
import yaml
import argparse
from pathlib import Path
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute, 
    Environment, 
    BuildContext,
    Workspace
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

def load_azure_config(config_path: str = "config/azure_config.yaml") -> dict:
    """Load Azure configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_ml_client(config: dict) -> MLClient:
    """Create Azure ML client."""
    try:
        credential = DefaultAzureCredential()
        
        ml_client = MLClient(
            credential=credential,
            subscription_id=config['azure']['subscription_id'],
            resource_group_name=config['azure']['resource_group'],
            workspace_name=config['azure']['workspace_name']
        )
        
        print(f"✅ Connected to Azure ML workspace: {config['azure']['workspace_name']}")
        return ml_client
        
    except Exception as e:
        print(f"❌ Failed to connect to Azure ML: {e}")
        print("\nTroubleshooting steps:")
        print("1. Ensure you're logged in: az login")
        print("2. Set correct subscription: az account set --subscription <subscription-id>")
        print("3. Verify workspace exists in the specified resource group")
        raise e

def check_workspace_exists(ml_client: MLClient, config: dict) -> bool:
    """Check if the workspace exists and is accessible."""
    try:
        workspace = ml_client.workspaces.get(config['azure']['workspace_name'])
        print(f"✅ Workspace '{workspace.name}' found in region: {workspace.location}")
        return True
    except ResourceNotFoundError:
        print(f"❌ Workspace '{config['azure']['workspace_name']}' not found")
        return False
    except Exception as e:
        print(f"❌ Error accessing workspace: {e}")
        return False

def create_or_get_compute_cluster(ml_client: MLClient, config: dict) -> AmlCompute:
    """Create or get existing compute cluster."""
    compute_config = config['compute']
    cluster_name = compute_config['cluster_name']
    
    print(f"\n🔧 Setting up compute cluster: {cluster_name}")
    
    try:
        # Try to get existing cluster
        compute_cluster = ml_client.compute.get(cluster_name)
        print(f"✅ Found existing compute cluster: {cluster_name}")
        print(f"   VM Size: {compute_cluster.size}")
        print(f"   Current nodes: {compute_cluster.provisioning_state}")
        
        return compute_cluster
        
    except ResourceNotFoundError:
        print(f"📝 Creating new compute cluster: {cluster_name}")
        
        # Create new compute cluster
        compute_cluster = AmlCompute(
            name=cluster_name,
            type="amlcompute",
            size=compute_config['vm_size'],
            min_instances=compute_config['min_nodes'],
            max_instances=compute_config['max_nodes'],
            idle_time_before_scale_down=compute_config['idle_seconds_before_scaledown'],
            tier="Dedicated",  # Use dedicated for H100
        )
        
        try:
            compute_cluster = ml_client.compute.begin_create_or_update(compute_cluster).result()
            print(f"✅ Successfully created compute cluster: {cluster_name}")
            print(f"   VM Size: {compute_cluster.size}")
            print(f"   Min nodes: {compute_cluster.scale_settings.min_node_count}")
            print(f"   Max nodes: {compute_cluster.scale_settings.max_node_count}")
            
            return compute_cluster
            
        except Exception as e:
            print(f"❌ Failed to create compute cluster: {e}")
            
            # Common error handling
            if "quota" in str(e).lower():
                print("\n💡 Quota Error Solutions:")
                print("1. Request quota increase for Standard_NC40ads_H100_v5 in Azure portal")
                print("2. Try a different VM size like Standard_NC24ads_A100_v4")
                print("3. Use a different region with available quota")
                
            elif "not available" in str(e).lower():
                print("\n💡 Availability Error Solutions:")
                print("1. Try different regions: eastus, westus2, southcentralus")
                print("2. Use alternative VM sizes: Standard_NC24ads_A100_v4, Standard_ND40rs_v2")
                
            raise e

def create_or_get_environment(ml_client: MLClient, config: dict) -> Environment:
    """Create or get the training environment."""
    env_config = config['environment']
    env_name = env_config['name']
    
    print(f"\n🐳 Setting up environment: {env_name}")
    
    try:
        # Try to get existing environment
        environment = ml_client.environments.get(env_name, label="latest")
        print(f"✅ Found existing environment: {env_name}")
        return environment
        
    except ResourceNotFoundError:
        print(f"📝 Creating new environment: {env_name}")
        
        # Create environment from conda file
        environment = Environment(
            name=env_name,
            description="Environment for Llama fine-tuning with Unsloth",
            conda_file="environment.yaml",
            image=env_config['docker']['base_image'],
        )
        
        try:
            environment = ml_client.environments.create_or_update(environment)
            print(f"✅ Successfully created environment: {env_name}")
            return environment
            
        except Exception as e:
            print(f"❌ Failed to create environment: {e}")
            raise e

def validate_h100_availability(ml_client: MLClient, config: dict):
    """Validate H100 VM availability in the region."""
    try:
        # Get workspace to check region
        workspace = ml_client.workspaces.get(config['azure']['workspace_name'])
        region = workspace.location
        vm_size = config['compute']['vm_size']
        
        print(f"\n🔍 Validating VM availability:")
        print(f"   Region: {region}")
        print(f"   VM Size: {vm_size}")
        
        # Check if it's H100 VM size
        if "H100" in vm_size:
            h100_regions = ["eastus", "southcentralus", "westus2", "northcentralus"]
            if region.lower() not in [r.lower() for r in h100_regions]:
                print(f"⚠️  Warning: {vm_size} may not be available in {region}")
                print(f"   H100 is typically available in: {', '.join(h100_regions)}")
                print(f"   Consider using a supported region")
        
        print(f"✅ Proceeding with {vm_size} in {region}")
        
    except Exception as e:
        print(f"⚠️  Could not validate VM availability: {e}")

def check_quotas(ml_client: MLClient, config: dict):
    """Check quota availability (basic check)."""
    vm_size = config['compute']['vm_size']
    max_nodes = config['compute']['max_nodes']
    
    print(f"\n📊 Quota requirements:")
    print(f"   VM Size: {vm_size}")
    print(f"   Max nodes: {max_nodes}")
    
    if "H100" in vm_size:
        cores_needed = max_nodes * 40  # H100 VMs typically have 40 cores
        print(f"   Estimated cores needed: {cores_needed}")
        print(f"   💡 Ensure you have sufficient quota for H100 VMs")
    
    print(f"   💡 Check quota in Azure Portal > Subscriptions > Usage + quotas")

def print_setup_summary(config: dict, compute_cluster, environment):
    """Print setup summary."""
    print("\n" + "="*60)
    print("🎉 AZURE ML SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Subscription: {config['azure']['subscription_id']}")
    print(f"Resource Group: {config['azure']['resource_group']}")
    print(f"Workspace: {config['azure']['workspace_name']}")
    print(f"Compute Cluster: {compute_cluster.name} ({compute_cluster.size})")
    print(f"Environment: {environment.name}")
    print("\n📋 Next Steps:")
    print("1. Run data preprocessing: python data/data_preprocessing.py")
    print("2. Submit training job: python scripts/submit_training.py")
    print("="*60)

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Azure ML cluster and environment")
    parser.add_argument("--config", type=str, default="config/azure_config.yaml",
                       help="Path to Azure config file")
    parser.add_argument("--force-recreate", action="store_true",
                       help="Force recreation of compute cluster")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate setup without creating resources")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print("📋 Loading Azure configuration...")
        config = load_azure_config(args.config)
        
        print("="*60)
        print("🚀 AZURE ML CLUSTER SETUP")
        print("="*60)
        print(f"Subscription: {config['azure']['subscription_id']}")
        print(f"Resource Group: {config['azure']['resource_group']}")
        print(f"Workspace: {config['azure']['workspace_name']}")
        print(f"Target VM Size: {config['compute']['vm_size']}")
        
        # Create ML client
        ml_client = get_ml_client(config)
        
        # Check workspace exists
        if not check_workspace_exists(ml_client, config):
            print("❌ Cannot proceed without valid workspace")
            return False
        
        # Validate H100 availability
        validate_h100_availability(ml_client, config)
        
        # Check quotas
        check_quotas(ml_client, config)
        
        if args.validate_only:
            print("\n✅ Validation completed. Use --force-recreate to create resources.")
            return True
        
        # Create or get compute cluster
        if args.force_recreate:
            # Delete existing cluster if it exists
            try:
                ml_client.compute.begin_delete(config['compute']['cluster_name'])
                print(f"🗑️  Deleted existing cluster: {config['compute']['cluster_name']}")
            except ResourceNotFoundError:
                pass
        
        compute_cluster = create_or_get_compute_cluster(ml_client, config)
        
        # Create or get environment
        environment = create_or_get_environment(ml_client, config)
        
        # Print summary
        print_setup_summary(config, compute_cluster, environment)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Verify Azure credentials: az login")
        print("2. Check subscription access: az account show")
        print("3. Verify resource group exists")
        print("4. Check quota availability for H100 VMs")
        print("5. Try a different region or VM size")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)