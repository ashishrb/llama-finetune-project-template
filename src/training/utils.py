# src/training/utils.py
import os
import json
import torch
import shutil
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import yaml

from transformers import (
    TrainingArguments, 
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel
import mlflow
import wandb

logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration manager for training parameters."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
    
    def get_training_args(self, output_dir: str) -> TrainingArguments:
        """Get TrainingArguments from config."""
        config = self.config['training']
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=config.get('num_train_epochs', 3),
            per_device_train_batch_size=config.get('batch_size', 2),
            per_device_eval_batch_size=config.get('batch_size', 2),
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 8),
            
            # Optimization
            learning_rate=config.get('learning_rate', 2e-4),
            weight_decay=config.get('weight_decay', 0.01),
            warmup_ratio=config.get('warmup_ratio', 0.1),
            lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
            
            # Mixed precision and optimization
            fp16=config.get('fp16', True),
            bf16=config.get('bf16', False),
            gradient_checkpointing=config.get('gradient_checkpointing', True),
            dataloader_num_workers=config.get('dataloader_num_workers', 4),
            remove_unused_columns=config.get('remove_unused_columns', False),
            
            # Evaluation and saving
            evaluation_strategy=config.get('eval_strategy', 'steps'),
            eval_steps=config.get('eval_steps', 50),
            save_strategy=config.get('save_strategy', 'steps'),
            save_steps=config.get('save_steps', 100),
            save_total_limit=config.get('save_total_limit', 3),
            load_best_model_at_end=config.get('load_best_model_at_end', True),
            metric_for_best_model=config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=config.get('greater_is_better', False),
            
            # Logging
            logging_strategy='steps',
            logging_steps=config.get('logging_steps', 10),
            report_to=['mlflow', 'tensorboard'],
            
            # Reproducibility
            seed=42,
            data_seed=42,
            
            # Memory optimization
            dataloader_pin_memory=False,
            group_by_length=True,
            
            # Additional settings
            prediction_loss_only=True,
            ignore_data_skip=True,
        )
    
    def get_lora_config(self) -> Dict:
        """Get LoRA configuration."""
        return self.config['training']['lora']
    
    def get_model_config(self) -> Dict:
        """Get model configuration."""
        return self.config.get('model', {})


class CheckpointManager:
    """Manages model checkpoints and saves the best models."""
    
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = Path(output_dir)
        self.save_total_limit = save_total_limit
        self.checkpoints = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        step: int,
        eval_loss: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """Save a checkpoint with metadata."""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save metadata
        checkpoint_metadata = {
            'step': step,
            'eval_loss': eval_loss,
            'timestamp': datetime.now().isoformat(),
            'model_path': str(checkpoint_dir),
            **(metadata or {})
        }
        
        with open(checkpoint_dir / 'metadata.json', 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        # Track checkpoint
        self.checkpoints.append({
            'path': checkpoint_dir,
            'step': step,
            'eval_loss': eval_loss,
            'metadata': checkpoint_metadata
        })
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        logger.info(f"Saved checkpoint at step {step} with eval_loss {eval_loss:.4f}")
        return str(checkpoint_dir)
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save limit."""
        if len(self.checkpoints) > self.save_total_limit:
            # Sort by eval_loss (keep best) and step (keep recent)
            self.checkpoints.sort(key=lambda x: (x['eval_loss'], -x['step']))
            
            # Remove worst checkpoints
            to_remove = self.checkpoints[self.save_total_limit:]
            self.checkpoints = self.checkpoints[:self.save_total_limit]
            
            for checkpoint in to_remove:
                try:
                    shutil.rmtree(checkpoint['path'])
                    logger.info(f"Removed checkpoint {checkpoint['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to the best checkpoint."""
        if not self.checkpoints:
            return None
        
        best = min(self.checkpoints, key=lambda x: x['eval_loss'])
        return str(best['path'])
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to the latest checkpoint."""
        if not self.checkpoints:
            return None
        
        latest = max(self.checkpoints, key=lambda x: x['step'])
        return str(latest['path'])


class TrainingMonitor:
    """Monitors training progress and logs metrics."""
    
    def __init__(self, experiment_name: str = "llama-finetune"):
        self.experiment_name = experiment_name
        self.metrics_history = []
        self.start_time = None
        
        # Initialize tracking
        self._init_mlflow()
        self._init_wandb()
    
    def _init_mlflow(self):
        """Initialize MLflow tracking."""
        try:
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
            logger.info("MLflow tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking."""
        try:
            wandb.init(
                project="llama-corporate-qa",
                name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                reinit=True
            )
            logger.info("W&B tracking initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to all tracking systems."""
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        metrics['step'] = step
        
        # Store locally
        self.metrics_history.append(metrics.copy())
        
        # Log to MLflow
        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log to MLflow: {e}")
        
        # Log to W&B
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
    
    def log_model_info(self, model_info: Dict):
        """Log model information."""
        try:
            mlflow.log_params(model_info)
            if wandb.run:
                wandb.config.update(model_info)
        except Exception as e:
            logger.warning(f"Failed to log model info: {e}")
    
    def start_training(self):
        """Mark start of training."""
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time}")
    
    def end_training(self, final_metrics: Optional[Dict] = None):
        """Mark end of training and cleanup."""
        end_time = datetime.now()
        
        if self.start_time:
            duration = end_time - self.start_time
            logger.info(f"Training completed in {duration}")
            
            if final_metrics:
                final_metrics['training_duration_seconds'] = duration.total_seconds()
                self.log_metrics(final_metrics, step=-1)
        
        # Cleanup tracking
        try:
            mlflow.end_run()
        except:
            pass
        
        try:
            wandb.finish()
        except:
            pass
    
    def save_metrics_history(self, output_path: str):
        """Save metrics history to file."""
        with open(output_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        logger.info(f"Metrics history saved to {output_path}")


class ModelSaver:
    """Handles saving trained models in different formats."""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_final_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict,
        model_name: str = "final_model"
    ) -> str:
        """Save the final trained model."""
        model_dir = self.base_output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        if isinstance(model, PeftModel):
            # For LoRA models, save the adapter
            model.save_pretrained(model_dir)
            
            # Also save merged model
            merged_dir = self.base_output_dir / f"{model_name}_merged"
            merged_dir.mkdir(exist_ok=True)
            
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            
            logger.info(f"Saved LoRA adapter to {model_dir}")
            logger.info(f"Saved merged model to {merged_dir}")
        else:
            model.save_pretrained(model_dir)
            logger.info(f"Saved full model to {model_dir}")
        
        tokenizer.save_pretrained(model_dir)
        
        # Save training configuration
        config_path = model_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create model card
        self._create_model_card(model_dir, config)
        
        return str(model_dir)
    
    def _create_model_card(self, model_dir: Path, config: Dict):
        """Create a model card with training information."""
        model_card = f"""# Llama-3.2-2B Corporate Q&A Model

## Model Description
Fine-tuned Llama-3.2-2B model for corporate Q&A tasks using LoRA (Low-Rank Adaptation).

## Training Configuration
- Base Model: {config.get('model', {}).get('base_model', 'Unknown')}
- Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- LoRA Rank: {config.get('training', {}).get('lora', {}).get('r', 'Unknown')}
- Batch Size: {config.get('training', {}).get('batch_size', 'Unknown')}
- Learning Rate: {config.get('training', {}).get('learning_rate', 'Unknown')}
- Epochs: {config.get('training', {}).get('num_train_epochs', 'Unknown')}

## Usage
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("{model_dir}")
base_model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3.2-2b-instruct")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{model_dir}")

# Generate response
inputs = tokenizer("Your question here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training Metrics
See training_metrics.json for detailed training progress.
"""
        
        with open(model_dir / "README.md", 'w') as f:
            f.write(model_card)


class EarlyStoppingWithPatience(EarlyStoppingCallback):
    """Enhanced early stopping with additional logging."""
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.01):
        super().__init__(early_stopping_patience, early_stopping_threshold)
        self.best_metric = None
        self.patience_counter = 0
    
    def check_metric_value(self, logs, metric_value):
        """Override to add custom logging."""
        result = super().check_metric_value(logs, metric_value)
        
        if self.best_metric is None:
            self.best_metric = metric_value
        elif self.is_metric_better(metric_value, self.best_metric):
            self.best_metric = metric_value
            self.patience_counter = 0
            logger.info(f"New best metric: {metric_value:.6f}")
        else:
            self.patience_counter += 1
            logger.info(f"No improvement. Patience: {self.patience_counter}/{self.early_stopping_patience}")
        
        return result


def setup_training_environment(
    config_path: str,
    output_dir: str,
    experiment_name: str = "llama-finetune"
) -> tuple:
    """Set up complete training environment."""
    
    # Load configuration
    config_manager = TrainingConfig(config_path)
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup components
    checkpoint_manager = CheckpointManager(output_path / "checkpoints")
    monitor = TrainingMonitor(experiment_name)
    model_saver = ModelSaver(output_path)
    
    # Get training arguments
    training_args = config_manager.get_training_args(str(output_path))
    
    # Setup early stopping
    early_stopping = EarlyStoppingWithPatience(
        early_stopping_patience=config_manager.config['training'].get('early_stopping_patience', 3),
        early_stopping_threshold=config_manager.config['training'].get('early_stopping_threshold', 0.01)
    )
    
    return config_manager, training_args, checkpoint_manager, monitor, model_saver, early_stopping


def calculate_model_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """Calculate model parameter statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'trainable_percentage': (trainable_params / total_params) * 100 if total_params > 0 else 0
    }


def log_system_info():
    """Log system and environment information."""
    info = {
        'python_version': str(sys.version),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            info[f'gpu_{i}_name'] = gpu_props.name
            info[f'gpu_{i}_memory'] = gpu_props.total_memory
    
    logger.info(f"System info: {info}")
    return info


def validate_training_setup(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    val_dataset,
    config: Dict
) -> bool:
    """Validate training setup before starting."""
    
    logger.info("Validating training setup...")
    
    # Check model
    if model is None:
        logger.error("Model is None")
        return False
    
    # Check tokenizer
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad token, setting to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check datasets
    if len(train_dataset) == 0:
        logger.error("Training dataset is empty")
        return False
    
    if len(val_dataset) == 0:
        logger.error("Validation dataset is empty")
        return False
    
    # Check configuration
    training_config = config.get('training', {})
    if training_config.get('learning_rate', 0) <= 0:
        logger.error("Invalid learning rate")
        return False
    
    # Test sample processing
    try:
        sample = train_dataset[0]
        if 'input_ids' not in sample or 'labels' not in sample:
            logger.error("Dataset samples missing required fields")
            return False
    except Exception as e:
        logger.error(f"Error processing dataset sample: {e}")
        return False
    
    # Calculate parameter info
    param_info = calculate_model_parameters(model)
    logger.info(f"Model parameters: {param_info}")
    
    if param_info['trainable_parameters'] == 0:
        logger.error("No trainable parameters found")
        return False
    
    logger.info("Training setup validation passed")
    return True


# Utility functions
def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    
    memory_info = {}
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
        memory_info[f'gpu_{i}_allocated_gb'] = memory_allocated
        memory_info[f'gpu_{i}_reserved_gb'] = memory_reserved
    
    return memory_info


def cleanup_training_artifacts(output_dir: str, keep_best: bool = True):
    """Clean up training artifacts, optionally keeping the best model."""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return
    
    # Clean up temporary files
    temp_patterns = ['*.tmp', '*.lock', 'events.out.tfevents.*']
    for pattern in temp_patterns:
        for file in output_path.rglob(pattern):
            try:
                file.unlink()
                logger.info(f"Removed temporary file: {file}")
            except Exception as e:
                logger.warning(f"Failed to remove {file}: {e}")
    
    # Optionally clean up intermediate checkpoints
    if not keep_best:
        checkpoint_dirs = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
        for checkpoint_dir in checkpoint_dirs:
            try:
                shutil.rmtree(checkpoint_dir)
                logger.info(f"Removed checkpoint: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to remove {checkpoint_dir}: {e}")


if __name__ == "__main__":
    # Test the utilities
    import sys
    
    # Test configuration loading
    try:
        config_manager = TrainingConfig("config/training_config.yaml")
        print("Configuration loaded successfully")
        
        # Test training arguments
        training_args = config_manager.get_training_args("test_output")
        print(f"Training arguments created: {training_args.output_dir}")
        
    except Exception as e:
        print(f"Error testing utilities: {e}")
        sys.exit(1)
    
    print("Training utilities test completed successfully")