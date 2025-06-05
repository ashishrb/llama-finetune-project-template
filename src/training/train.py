# src/training/train.py
import os
import sys
import json
import yaml
import torch
import mlflow
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Setup import paths
def setup_imports():
    """Setup import paths for both local and Azure ML environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    paths_to_add = [
        project_root,
        current_dir,
        os.path.join(project_root, "src"),
        os.path.join(project_root, "src", "training")
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

setup_imports()

# Import custom modules
try:
    from .utils import GPUMemoryManager, log_system_resources
except (ImportError, ValueError):
    try:
        from utils import GPUMemoryManager, log_system_resources
    except ImportError:
        from src.training.utils import GPUMemoryManager, log_system_resources

# Standard transformer imports (no Unsloth)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    HfArgumentParser
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from trl import SFTTrainer
from datasets import Dataset
import wandb

def setup_hf_auth():
    """Setup HuggingFace authentication."""
    hf_token = os.environ.get("HF_TOKEN", "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")
    if hf_token:
        from huggingface_hub import login
        try:
            login(token=hf_token)
            print("✅ HuggingFace authentication successful")
        except Exception as e:
            print(f"⚠️ HuggingFace authentication failed: {e}")
            print("Continuing without authentication...")

    # Add paths to sys.path if not already present
    paths_to_add = [
        project_root,
        current_dir,
        os.path.join(project_root, "src"),
        os.path.join(project_root, "src", "training")
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)

# Setup imports before trying to import custom modules
setup_imports()

# Now import custom modules with multiple fallback strategies
try:
    # Try relative import first
    from .utils import GPUMemoryManager, log_system_resources
except (ImportError, ValueError):
    try:
        # Try direct import
        from utils import GPUMemoryManager, log_system_resources
    except ImportError:
        try:
            # Try with src.training prefix
            from src.training.utils import GPUMemoryManager, log_system_resources
        except ImportError:
            # Final fallback - create minimal implementations
            print("Warning: Could not import utils, using fallback implementations")
            
            class GPUMemoryManager:
                def __init__(self): pass
                def get_gpu_memory_info(self): return {'allocated_gb': 0, 'total_gb': 0, 'usage_percent': 0}
                def clear_gpu_cache(self): 
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                def monitor_and_cleanup(self): pass
                def log_memory_summary(self): 
                    print("GPU memory monitoring not available")
                def check_available_memory(self, required_gb): return True
            
            def log_system_resources():
                print("System resource logging not available")

# Add project root to path
#sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# try:
#     from unsloth import FastLanguageModel, is_bfloat16_supported
# except ImportError:
#     raise ImportError("Unsloth not available. Install with: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'")

# from unsloth import is_bfloat16_supported

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth loaded successfully")
except ImportError as e:
    print(f"⚠️  Unsloth not available: {e}")
    print("Falling back to standard transformers training...")
    UNSLOTH_AVAILABLE = False
    
    # Fallback implementations
    def is_bfloat16_supported():
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    class FastLanguageModel:
        @staticmethod
        def from_pretrained(model_name, max_seq_length=2048, dtype=None, load_in_4bit=True):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype or torch.float16,
                device_map="auto",
                load_in_4bit=load_in_4bit
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        
        @staticmethod
        def get_peft_model(model, r=16, target_modules=None, lora_alpha=16, lora_dropout=0.1, bias="none", use_gradient_checkpointing=True, random_state=3407, use_rslora=False, loftq_config=None):
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias=bias,
                task_type="CAUSAL_LM"
            )
            return get_peft_model(model, lora_config)

from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
import wandb

def setup_logging(config: Dict):
    """Setup logging for training."""
    # Initialize MLflow
    mlflow.set_experiment(config.get('experiment_name', 'llama-finetune'))
    
    # Initialize Weights & Biases if available
    try:
        wandb.init(
            project="llama-corporate-qa",
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config
        )
    except Exception as e:
        print(f"Warning: Could not initialize wandb: {e}")

def load_config(config_path: str = "config/training_config.yaml") -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model_config(config_path: str = "config/model_config.yaml") -> Dict:
    """Load model configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    
    # Handle both local and Azure ML paths
    if not os.path.exists(file_path):
        # Try common Azure ML data locations
        alt_paths = [
            file_path,
            os.path.join("/tmp/data", os.path.basename(file_path)),
            os.path.join(os.getcwd(), file_path)
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                file_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def prepare_datasets(data_dir: str = "data/processed") -> tuple:
    """Load and prepare training datasets."""
    print("Loading datasets...")
    
    train_data = load_jsonl_data(f"{data_dir}/train.jsonl")
    val_data = load_jsonl_data(f"{data_dir}/val.jsonl")
    
    print(f"Loaded {len(train_data)} training samples")
    print(f"Loaded {len(val_data)} validation samples")
    
    # Format data for training
    def format_data(samples):
        """Format samples with the prompt template."""
        formatted_samples = []
        for sample in samples:
            # The data already has 'text' field from preprocessing
            if 'text' in sample:
                formatted_samples.append({'text': sample['text']})
            else:
                # Fallback formatting if 'text' is missing
                text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sample['system']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{sample['output']}<|eot_id|>"""
                formatted_samples.append({'text': text})
        return formatted_samples
    
    # Format datasets
    train_formatted = format_data(train_data)
    val_formatted = format_data(val_data)
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_formatted)
    val_dataset = Dataset.from_list(val_formatted)
    
    return train_dataset, val_dataset

def setup_model_and_tokenizer(model_config: Dict, training_config: Dict):
    """Initialize model and tokenizer with standard transformers."""
    print("Setting up model and tokenizer...")
    
    # Initialize memory manager
    memory_manager = GPUMemoryManager()
    log_system_resources()
    
    model_name = model_config['model']['base_model']
    max_seq_length = training_config['training']['max_seq_length']
    
    # Setup HuggingFace authentication
    setup_hf_auth()
    
    # Quantization configuration for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    
    # Check memory before model loading
    estimated_memory_gb = 12.0  # 3B model estimate
    if not memory_manager.check_available_memory(estimated_memory_gb):
        memory_manager.clear_gpu_cache()
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Setup tokenizer
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Log memory after model loading
        memory_manager.log_memory_summary()
        
        print("✅ Model and tokenizer loaded successfully")
        
    except Exception as e:
        if "out of memory" in str(e).lower():
            print("❌ GPU out of memory during model loading")
            memory_manager.clear_gpu_cache()
            raise RuntimeError("GPU out of memory. Try reducing batch size.")
        raise e
    
    return model, tokenizer

def setup_lora_model(model, training_config: Dict):
    """Configure model for LoRA fine-tuning using PEFT."""
    print("Setting up LoRA configuration...")
    
    lora_config_dict = training_config['training']['lora']
    
    # Create LoRA configuration
    lora_config = LoraConfig(
        r=lora_config_dict['r'],
        lora_alpha=lora_config_dict['alpha'],
        target_modules=lora_config_dict['target_modules'],
        lora_dropout=lora_config_dict['dropout'],
        bias=lora_config_dict['bias'],
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    trainable_percent = 100 * trainable_params / all_params
    
    print(f"LoRA configuration:")
    print(f"  Rank: {lora_config_dict['r']}")
    print(f"  Alpha: {lora_config_dict['alpha']}")
    print(f"  Target modules: {lora_config_dict['target_modules']}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")
    
    return model

def monitor_training_memory(memory_manager, step: int, log_interval: int = 100):
    """Monitor memory during training."""
    if step % log_interval == 0:
        memory_manager.monitor_and_cleanup()
        
        # Force cleanup every 500 steps
        if step % 500 == 0:
            memory_manager.clear_gpu_cache()
            memory_manager.log_memory_summary()

def setup_training_arguments(training_config: Dict, output_dir: str) -> TrainingArguments:
    """Setup training arguments."""
    config = training_config['training']
    
    training_args = TrainingArguments(
        # Output and logging
        output_dir=output_dir,
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['logging_steps'],
        
        # Training parameters
        num_train_epochs=config['num_train_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_ratio=config['warmup_ratio'],
        lr_scheduler_type=config['lr_scheduler_type'],
        
        # Optimization
        fp16=config.get('fp16', False),
        bf16=is_bfloat16_supported(),  # Use bfloat16 if supported
        gradient_checkpointing=config['gradient_checkpointing'],
        dataloader_num_workers=config['dataloader_num_workers'],
        remove_unused_columns=config['remove_unused_columns'],
        
        # Evaluation and saving
        evaluation_strategy=config['eval_strategy'],
        eval_steps=config['eval_steps'],
        save_strategy=config['save_strategy'],
        save_steps=config['save_steps'],
        save_total_limit=3,
        load_best_model_at_end=config['load_best_model_at_end'],
        metric_for_best_model=config['metric_for_best_model'],
        greater_is_better=config['greater_is_better'],
        
        # MLflow integration
        report_to=["mlflow", "tensorboard"],
        
        # Memory optimization
        dataloader_pin_memory=False,  # Can cause issues with some setups
        group_by_length=True,  # Group sequences by length for efficiency
        
        # Seed for reproducibility
        seed=42,
        data_seed=42,
    )
    
    return training_args

def format_prompts(examples):
    """Format prompts for training (this function is called by the trainer)."""
    return {"text": examples["text"]}

def train_model(
    model, 
    tokenizer, 
    train_dataset, 
    val_dataset, 
    training_args, 
    training_config: Dict
):
    """Train the model using SFTTrainer."""
    print("Setting up trainer...")
    
    # ADD THIS: Initialize memory manager for training
    #memory_manager = GPUMemoryManager()

    try:
        memory_manager = GPUMemoryManager()
    except Exception as e:
        print(f"Warning: Could not initialize memory manager: {e}")
        memory_manager = None
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        formatting_func=None,  # We already formatted the data
        max_seq_length=training_config['training']['max_seq_length'],
        dataset_num_proc=2,
        packing=False,  # Can be set to True for efficiency if sequences are short
        args=training_args,
    )
    
    # ADD THIS: Add memory monitoring callback
    class MemoryMonitoringCallback:
        def __init__(self, memory_manager):
            self.memory_manager = memory_manager
            self.step_count = 0
        
        def on_step_end(self, args, state, control, **kwargs):
            self.step_count += 1
            monitor_training_memory(self.memory_manager, self.step_count)
    
    trainer.add_callback(MemoryMonitoringCallback(memory_manager))
    
    print("Starting training...")
    
    # ADD THIS: Memory check before training
    memory_manager.log_memory_summary()
    
    try:
        trainer.train()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ GPU out of memory during training")
            memory_manager.clear_gpu_cache()
            raise RuntimeError("GPU out of memory during training. Try reducing batch size or sequence length.")
        raise e
    finally:
        # ADD THIS: Always cleanup after training
        memory_manager.clear_gpu_cache()
        memory_manager.log_memory_summary()
    
    return trainer

def save_model(model, tokenizer, trainer, output_dir: str, config: Dict):
    """Save the trained model and tokenizer."""
    print("Saving model...")
    
    #memory_manager = GPUMemoryManager()

    try:
        memory_manager = GPUMemoryManager()
    except Exception as e:
    print(f"Warning: Could not initialize memory manager: {e}")
        memory_manager = None
    
    # Check available disk space (rough estimate)
    import shutil
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free / 1024**3
    
    if free_gb < 10:  # Require at least 10GB free space
        print(f"⚠️  Warning: Low disk space ({free_gb:.1f}GB available)")
    
    # Clear memory before saving to prevent issues
    memory_manager.clear_gpu_cache()
    
    # Save using Unsloth's optimized saving
    model.save_pretrained(f"{output_dir}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/final_model")
    
    # Save training configuration
    with open(f"{output_dir}/training_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save training metrics
    if hasattr(trainer, 'state') and trainer.state.log_history:
        with open(f"{output_dir}/training_metrics.json", 'w') as f:
            json.dump(trainer.state.log_history, f, indent=2)
    
    # Log model to MLflow
    try:
        model.save_pretrained(f"{output_dir}/final_model")
        tokenizer.save_pretrained(f"{output_dir}/final_model")

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            registered_model_name=config['model']['model_name']
        )
        print("Model logged to MLflow successfully")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")

def main():
    if "HF_TOKEN" not in os.environ:
        os.environ["HF_TOKEN"] = "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"
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama model")
    parser = argparse.ArgumentParser(description="Fine-tune Llama model")
    parser.add_argument("--config", type=str, default="./config/training_config.yaml", 
                   help="Path to training config")
    parser.add_argument("--model_config", type=str, default="./config/model_config.yaml", 
                   help="Path to model config")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                       help="Directory containing processed data")
    parser.add_argument("--output_dir", type=str, default="output/models", 
                       help="Output directory for model")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 STARTING LLAMA FINE-TUNING")
    print("="*60)

    # Log initial system state
    log_system_resources()
    
    # Load configurations
    training_config = load_config(args.config)
    model_config = load_model_config(args.model_config)

    # Load configurations
    training_config = load_config(args.config)
    model_config = load_model_config(args.model_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_logging({**training_config, **model_config})
    
    print("="*60)
    print("🚀 STARTING LLAMA FINE-TUNING")
    print("="*60)
    print(f"Model: {model_config['model']['base_model']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")
    
    try:
        # Load datasets
        train_dataset, val_dataset = prepare_datasets(args.data_dir)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_config, training_config)
        
        # Setup LoRA
        model = setup_lora_model(model, training_config)
        
        # Setup training arguments
        training_args = setup_training_arguments(training_config, args.output_dir)
        
        # Train model
        trainer = train_model(model, tokenizer, train_dataset, val_dataset, 
                            training_args, training_config)
        
        # Save model
        save_model(model, tokenizer, trainer, args.output_dir, 
                  {**training_config, **model_config})
        
        print("="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Model saved to: {args.output_dir}/final_model")
        
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise e
    
    finally:
        # Cleanup
        try:
            wandb.finish()
        except:
            pass

if __name__ == "__main__":
    main()