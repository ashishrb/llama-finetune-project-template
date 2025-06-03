# src/training/train.py
import os
import sys
import json
import yaml
import torch
import mlflow
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from src.training.utils import GPUMemoryManager, log_system_resources

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
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
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def setup_model_and_tokenizer(model_config: Dict, training_config: Dict):
    """Initialize model and tokenizer with Unsloth optimizations."""
    print("Setting up model and tokenizer...")
    
    # ADD THIS: Initialize memory manager
    memory_manager = GPUMemoryManager()
    log_system_resources()
    
    model_name = model_config['model']['base_model']
    max_seq_length = training_config['training']['max_seq_length']
    
    # Determine dtype - use bfloat16 if supported, otherwise float16
    dtype = torch.bfloat16 if is_bfloat16_supported() else torch.float16
    
    print(f"Loading model: {model_name}")
    print(f"Max sequence length: {max_seq_length}")
    print(f"Using dtype: {dtype}")
    
    # ADD THIS: Check memory before model loading
    try:
        # Estimate model memory requirement (rough estimate for Llama-2B)
        estimated_memory_gb = 8.0  # Adjust based on your model size
        if not memory_manager.check_available_memory(estimated_memory_gb):
            memory_manager.clear_gpu_cache()
        
        # Load model and tokenizer with Unsloth optimizations
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=model_config['quantization']['load_in_4bit'],
            # trust_remote_code=True,  # Uncomment if needed
        )
        
        # ADD THIS: Log memory after model loading
        memory_manager.log_memory_summary()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ GPU out of memory during model loading")
            memory_manager.clear_gpu_cache()
            raise RuntimeError("GPU out of memory. Try reducing model size or using CPU.")
        raise e
    
    # Setup tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def setup_lora_model(model, training_config: Dict):
    """Configure model for LoRA fine-tuning."""
    print("Setting up LoRA configuration...")
    
    lora_config = training_config['training']['lora']
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['alpha'],
        lora_dropout=lora_config['dropout'],
        bias=lora_config['bias'],
        use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
        random_state=3407,
        use_rslora=False,  # Set to True for rank stabilized LoRA
        loftq_config=None,  # LoftQ quantization
    )
    
    print(f"LoRA rank: {lora_config['r']}")
    print(f"LoRA alpha: {lora_config['alpha']}")
    print(f"Target modules: {lora_config['target_modules']}")
    
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
    memory_manager = GPUMemoryManager()
    
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
    
    memory_manager = GPUMemoryManager()
    
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
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Llama model")
    parser = argparse.ArgumentParser(description="Fine-tune Llama model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", 
                       help="Path to training config")
    parser.add_argument("--model_config", type=str, default="config/model_config.yaml", 
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