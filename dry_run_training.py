#!/usr/bin/env python3
"""
Dry run test for training components without actual training.
"""
import os
import sys
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_memory_management():
    """Test GPU memory management utilities."""
    print("🧪 Testing GPU Memory Management...")
    
    try:
        from src.training.utils import GPUMemoryManager, log_system_resources
        
        # Test memory manager
        memory_manager = GPUMemoryManager()
        memory_info = memory_manager.get_gpu_memory_info()
        
        print(f"   GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU Memory: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB")
        
        # Test system resources
        log_system_resources()
        
        print("✅ Memory management working")
        return True
        
    except Exception as e:
        print(f"❌ Memory management failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading with processed data."""
    print("🧪 Testing Dataset Loading...")
    
    try:
        from src.training.dataset import CorporateQADataset
        
        # Use small test dataset
        data_path = "data/test_processed/train.jsonl"
        
        if not Path(data_path).exists():
            print(f"❌ Test data not found: {data_path}")
            print("   Run data preprocessing first")
            return False
        
        # Test with a lightweight tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset
        dataset = CorporateQADataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=512,
            cache_tokenization=False  # Disable caching for test
        )
        
        print(f"   Dataset size: {len(dataset)}")
        
        # Test sample retrieval
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"   Sample keys: {list(sample.keys())}")
            print(f"   Input IDs shape: {sample['input_ids'].shape}")
        
        # Test cache functionality
        cache_info = dataset.get_cache_info()
        print(f"   Cache info: {cache_info}")
        
        print("✅ Dataset loading working")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_training_config():
    """Test training configuration loading."""
    print("🧪 Testing Training Configuration...")
    
    try:
        from src.training.utils import TrainingConfig
        
        config_manager = TrainingConfig("config/training_config.yaml")
        
        # Test training arguments
        training_args = config_manager.get_training_args("output/test")
        print(f"   Batch size: {training_args.per_device_train_batch_size}")
        print(f"   Learning rate: {training_args.learning_rate}")
        print(f"   Output dir: {training_args.output_dir}")
        
        # Test LoRA config
        lora_config = config_manager.get_lora_config()
        print(f"   LoRA rank: {lora_config['r']}")
        print(f"   LoRA alpha: {lora_config['alpha']}")
        
        print("✅ Training configuration working")
        return True
        
    except Exception as e:
        print(f"❌ Training configuration failed: {e}")
        return False

def test_model_loading():
    """Test model loading without full model download."""
    print("🧪 Testing Model Loading (config only)...")
    
    try:
        from transformers import AutoConfig, AutoTokenizer
        
        # Test loading model config (lightweight)
        model_name = "Unsloth/llama-3.2-3B-Instruct"  # Use small model for testing
        
        config = AutoConfig.from_pretrained(model_name)
        print(f"   Model type: {config.model_type}")
        print(f"   Vocab size: {config.vocab_size}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"   Tokenizer vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   Test tokenization shape: {tokens['input_ids'].shape}")
        
        print("✅ Model configuration loading working")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_evaluation_components():
    """Test evaluation components without full evaluation."""
    print("🧪 Testing Evaluation Components...")
    
    try:
        from src.evaluation.metrics import calculate_bleu_score, calculate_diversity_metrics
        
        # Test with dummy data
        predictions = ["This is a test response.", "Another test response."]
        references = ["This is a reference.", "Another reference."]
        
        # Test BLEU calculation
        bleu_scores = calculate_bleu_score(predictions, references)
        print(f"   BLEU scores: {bleu_scores}")
        
        # Test diversity metrics
        diversity_scores = calculate_diversity_metrics(predictions)
        print(f"   Diversity scores: {diversity_scores}")
        
        print("✅ Evaluation components working")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation components failed: {e}")
        return False

def test_azure_config():
    """Test Azure configuration without actual connection."""
    print("🧪 Testing Azure Configuration...")
    
    try:
        from scripts.setup_cluster import load_azure_config
        
        config = load_azure_config("config/azure_config.yaml")
        
        required_fields = ['subscription_id', 'resource_group', 'workspace_name']
        for field in required_fields:
            if field not in config['azure']:
                print(f"❌ Missing Azure config field: {field}")
                return False
            
            value = config['azure'][field]
            if value.startswith('${') and value.endswith('}'):
                print(f"   {field}: Environment variable {value}")
            else:
                print(f"   {field}: {value[:8]}...")
        
        print("✅ Azure configuration structure valid")
        return True
        
    except Exception as e:
        print(f"❌ Azure configuration failed: {e}")
        return False

def main():
    """Run all dry run tests."""
    print("="*60)
    print("🧪 RUNNING DRY RUN TESTS")
    print("="*60)
    
    tests = [
        ("Memory Management", test_memory_management),
        ("Azure Configuration", test_azure_config),
        ("Dataset Loading", test_dataset_loading),
        ("Training Configuration", test_training_config),
        ("Model Loading", test_model_loading),
        ("Evaluation Components", test_evaluation_components),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 DRY RUN RESULTS")
    print("="*60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Your pipeline is ready for real training.")
    else:
        print("⚠️  Some tests failed. Fix issues before proceeding.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)