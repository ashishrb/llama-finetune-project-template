# tests/test_training.py
import unittest
import tempfile
import os
import yaml
import json
import sys
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.training.dataset import CorporateQADataset, DataCollator
from src.training.utils import TrainingConfig, calculate_model_parameters

class TestTrainingComponents(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test data
        self.test_data = [
            {
                "system": "You are a helpful corporate AI assistant.",
                "instruction": "What is PM job code?",
                "output": "PM job code indicates Full Time Employee.",
                "text": """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful corporate AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is PM job code?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

PM job code indicates Full Time Employee.<|eot_id|>"""
            }
        ]
        
        # Create test data file
        self.test_data_file = os.path.join(self.temp_dir, "test_data.jsonl")
        with open(self.test_data_file, 'w') as f:
            for sample in self.test_data:
                f.write(json.dumps(sample) + '\n')
        
        # Create test config
        self.test_config = {
            'training': {
                'batch_size': 1,
                'gradient_accumulation_steps': 1,
                'max_seq_length': 512,
                'num_train_epochs': 1,
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'warmup_ratio': 0.1,
                'lr_scheduler_type': 'cosine',
                'lora': {
                    'r': 16,
                    'alpha': 32,
                    'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj"],
                    'dropout': 0.1,
                    'bias': 'none',
                    'task_type': 'CAUSAL_LM'
                },
                'fp16': True,
                'gradient_checkpointing': True,
                'dataloader_num_workers': 0,
                'remove_unused_columns': False,
                'eval_strategy': 'steps',
                'eval_steps': 10,
                'save_strategy': 'steps',
                'save_steps': 10,
                'logging_steps': 5,
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'early_stopping_patience': 2,
                'early_stopping_threshold': 0.01
            },
            'data': {
                'max_length': 512,
                'prompt_template': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, "config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_training_config_loading(self):
        """Test training configuration loading."""
        config_manager = TrainingConfig(self.config_file)
        
        # Test config loading
        self.assertIsNotNone(config_manager.config)
        self.assertIn('training', config_manager.config)
        
        # Test training arguments creation
        training_args = config_manager.get_training_args(self.temp_dir)
        self.assertEqual(training_args.output_dir, self.temp_dir)
        self.assertEqual(training_args.per_device_train_batch_size, 1)
        
        # Test LoRA config
        lora_config = config_manager.get_lora_config()
        self.assertEqual(lora_config['r'], 16)
        self.assertEqual(lora_config['alpha'], 32)
    
    @unittest.skipIf(not torch.cuda.is_available(), "GPU not available")
    def test_dataset_creation(self):
        """Test dataset creation with mock tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            # Use a small tokenizer for testing
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            dataset = CorporateQADataset(
                data_path=self.test_data_file,
                tokenizer=tokenizer,
                max_length=512,
                cache_tokenization=False
            )
            
            # Test dataset basic functionality
            self.assertEqual(len(dataset), len(self.test_data))
            
            # Test sample retrieval
            sample = dataset[0]
            self.assertIn('input_ids', sample)
            self.assertIn('attention_mask', sample)
            self.assertIn('labels', sample)
            
            # Test text formatting
            text = dataset.get_sample_text(0)
            self.assertIn('<|begin_of_text|>', text)
            
        except ImportError:
            self.skipTest("Transformers not available for testing")
    
    def test_data_collator(self):
        """Test data collator functionality."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            
            collator = DataCollator(tokenizer)
            
            # Create mock features
            features = [
                {
                    'input_ids': torch.tensor([1, 2, 3]),
                    'attention_mask': torch.tensor([1, 1, 1]),
                    'labels': torch.tensor([1, 2, 3])
                },
                {
                    'input_ids': torch.tensor([1, 2]),
                    'attention_mask': torch.tensor([1, 1]),
                    'labels': torch.tensor([1, 2])
                }
            ]
            
            batch = collator(features)
            
            # Check batch structure
            self.assertIn('input_ids', batch)
            self.assertIn('attention_mask', batch)
            self.assertIn('labels', batch)
            
            # Check padding
            self.assertEqual(batch['input_ids'].shape[0], 2)  # Batch size
            self.assertEqual(batch['input_ids'].shape[1], 3)  # Max length
            
        except ImportError:
            self.skipTest("Transformers not available for testing")
    
    def test_model_parameters_calculation(self):
        """Test model parameter calculation utility."""
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = torch.nn.Linear(10, 5)
                self.layer2 = torch.nn.Linear(5, 2)
                
                # Freeze layer2 for testing
                for param in self.layer2.parameters():
                    param.requires_grad = False
        
        model = MockModel()
        params = calculate_model_parameters(model)
        
        self.assertIn('total_parameters', params)
        self.assertIn('trainable_parameters', params)
        self.assertIn('frozen_parameters', params)
        self.assertIn('trainable_percentage', params)
        
        # Check calculations
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertEqual(params['total_parameters'], total_params)
        self.assertEqual(params['trainable_parameters'], trainable_params)
        self.assertEqual(params['frozen_parameters'], total_params - trainable_params)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid config
        invalid_config = {
            'training': {
                'learning_rate': -1,  # Invalid
                'batch_size': 0,      # Invalid
            }
        }
        
        invalid_config_file = os.path.join(self.temp_dir, "invalid_config.yaml")
        with open(invalid_config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Should handle invalid config gracefully
        config_manager = TrainingConfig(invalid_config_file)
        training_args = config_manager.get_training_args(self.temp_dir)
        
        # Should use defaults for invalid values
        self.assertGreater(training_args.learning_rate, 0)
        self.assertGreater(training_args.per_device_train_batch_size, 0)
    
    def test_file_operations(self):
        """Test file operations and error handling."""
        # Test loading non-existent config
        with self.assertRaises(FileNotFoundError):
            TrainingConfig("/nonexistent/config.yaml")
        
        # Test loading non-existent dataset
        with self.assertRaises(FileNotFoundError):
            CorporateQADataset("/nonexistent/data.jsonl", None)

class TestTrainingIntegration(unittest.TestCase):
    """Integration tests for training components."""
    
    def test_end_to_end_validation(self):
        """Test that all training components work together."""
        # This test validates that the training pipeline components
        # are compatible without actually running training
        
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create minimal valid config
            config = {
                'training': {
                    'batch_size': 1,
                    'learning_rate': 1e-4,
                    'num_train_epochs': 1,
                    'lora': {
                        'r': 8,
                        'alpha': 16,
                        'target_modules': ["q_proj", "v_proj"],
                        'dropout': 0.1,
                        'bias': 'none'
                    }
                },
                'data': {
                    'max_length': 256
                }
            }
            
            config_file = os.path.join(temp_dir, "config.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(config, f)
            
            # Test config loading
            config_manager = TrainingConfig(config_file)
            training_args = config_manager.get_training_args(temp_dir)
            
            # Validate training arguments
            self.assertIsNotNone(training_args)
            self.assertEqual(training_args.per_device_train_batch_size, 1)
            
            # Test LoRA config
            lora_config = config_manager.get_lora_config()
            self.assertIn('r', lora_config)
            self.assertIn('target_modules', lora_config)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    # Run tests with different verbosity levels
    unittest.main(verbosity=2)