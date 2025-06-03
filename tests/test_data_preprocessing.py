# tests/test_data_preprocessing.py
import unittest
import tempfile
import os
import json
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data.data_preprocessing import (
    load_jsonl_data,
    validate_sample,
    clean_and_validate_data,
    format_for_training,
    split_data,
    save_jsonl_data,
    preprocess_data,
    validate_processed_data
)

class TestDataPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and temporary directories."""
        self.test_data = [
            {
                "system": "You are a helpful corporate AI assistant.",
                "instruction": "What is PM job code?",
                "output": "PM job code indicates Full Time Employee."
            },
            {
                "system": "You are a helpful corporate AI assistant.",
                "instruction": "How to convert CWR to FTE?",
                "output": "Raise New Demand and mark requirement type as CWR Conversion."
            },
            {
                "system": "You are a helpful corporate AI assistant.",
                "instruction": "What is FC job code?",
                "output": "FC job code indicates Full Time Contractor."
            }
        ]
        
        self.invalid_data = [
            {"system": "", "instruction": "Test", "output": "Test"},  # Empty system
            {"instruction": "Test", "output": "Test"},  # Missing system
            {"system": "Test", "instruction": "Test"},  # Missing output
        ]
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_validate_sample(self):
        """Test sample validation function."""
        # Valid sample
        valid_sample = self.test_data[0]
        self.assertTrue(validate_sample(valid_sample))
        
        # Invalid samples
        for invalid_sample in self.invalid_data:
            self.assertFalse(validate_sample(invalid_sample))
    
    def test_clean_and_validate_data(self):
        """Test data cleaning and validation."""
        mixed_data = self.test_data + self.invalid_data
        
        clean_data = clean_and_validate_data(mixed_data)
        
        # Should only keep valid samples
        self.assertEqual(len(clean_data), len(self.test_data))
        
        # All cleaned samples should be valid
        for sample in clean_data:
            self.assertTrue(validate_sample(sample))
    
    def test_format_for_training(self):
        """Test training format application."""
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
        
        sample = self.test_data[0]
        formatted = format_for_training(sample, template)
        
        # Should have original fields plus 'text'
        self.assertIn('text', formatted)
        self.assertIn('system', formatted)
        self.assertIn('instruction', formatted)
        self.assertIn('output', formatted)
        
        # Text should contain formatted content
        self.assertIn(sample['system'], formatted['text'])
        self.assertIn(sample['instruction'], formatted['text'])
        self.assertIn(sample['output'], formatted['text'])
    
    def test_split_data(self):
        """Test data splitting functionality."""
        data = self.test_data * 10  # 30 samples total
        
        train_data, val_data, test_data = split_data(data, 0.7, 0.2, 0.1)
        
        # Check proportions
        total = len(train_data) + len(val_data) + len(test_data)
        self.assertEqual(total, len(data))
        
        # Check approximate ratios (allowing for rounding)
        self.assertAlmostEqual(len(train_data) / total, 0.7, delta=0.1)
        self.assertAlmostEqual(len(val_data) / total, 0.2, delta=0.1)
        self.assertAlmostEqual(len(test_data) / total, 0.1, delta=0.1)
    
    def test_save_and_load_jsonl(self):
        """Test JSONL save and load functionality."""
        test_file = os.path.join(self.temp_dir, "test.jsonl")
        
        # Save data
        save_jsonl_data(self.test_data, test_file)
        
        # Check file exists
        self.assertTrue(os.path.exists(test_file))
        
        # Load and verify
        loaded_data = load_jsonl_data(test_file)
        self.assertEqual(len(loaded_data), len(self.test_data))
        self.assertEqual(loaded_data, self.test_data)
    
    def test_preprocess_data_integration(self):
        """Test end-to-end preprocessing."""
        # Create test input file
        input_file = os.path.join(self.temp_dir, "input.jsonl")
        save_jsonl_data(self.test_data, input_file)
        
        # Create test config
        config = {
            'data': {
                'train_split': 0.6,
                'val_split': 0.3,
                'test_split': 0.1,
                'prompt_template': """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
            }
        }
        
        config_file = os.path.join(self.temp_dir, "config.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        output_dir = os.path.join(self.temp_dir, "processed")
        
        # Run preprocessing
        train_data, val_data, test_data = preprocess_data(
            input_file, output_dir, config_file
        )
        
        # Check outputs exist
        self.assertTrue(os.path.exists(os.path.join(output_dir, "train.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "val.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "preprocessing_stats.json")))
        
        # Validate processed data
        validate_processed_data(output_dir)
        
        # Check that all samples have 'text' field
        for sample in train_data + val_data + test_data:
            self.assertIn('text', sample)
            self.assertIn('<|begin_of_text|>', sample['text'])
    
    def test_error_handling(self):
        """Test error handling for edge cases."""
        # Test with empty data
        with self.assertRaises(ValueError):
            preprocess_data("/nonexistent/file.jsonl", self.temp_dir)
        
        # Test with invalid ratios
        with self.assertRaises(ValueError):
            split_data(self.test_data, 0.5, 0.3, 0.3)  # Sum > 1.0

if __name__ == '__main__':
    unittest.main()