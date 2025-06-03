# data/data_preprocessing.py
import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_jsonl_data(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                data.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
    return data

def validate_sample(sample: Dict) -> bool:
    """Validate that a sample has required fields."""
    required_fields = ['system', 'instruction', 'output']
    
    for field in required_fields:
        if field not in sample:
            logger.warning(f"Missing field '{field}' in sample")
            return False
        if not isinstance(sample[field], str) or not sample[field].strip():
            logger.warning(f"Empty or invalid field '{field}' in sample")
            return False
    
    return True

def clean_and_validate_data(data: List[Dict]) -> List[Dict]:
    """Clean and validate the dataset."""
    clean_data = []
    skipped_count = 0
    
    for i, sample in enumerate(data):
        if validate_sample(sample):
            # Clean whitespace and normalize text
            cleaned_sample = {
                'system': sample['system'].strip(),
                'instruction': sample['instruction'].strip(),
                'output': sample['output'].strip()
            }
            
            # Add any additional metadata
            if 'category' in sample:
                cleaned_sample['category'] = sample['category']
            
            clean_data.append(cleaned_sample)
        else:
            skipped_count += 1
            logger.warning(f"Skipped invalid sample at index {i}")
    
    logger.info(f"Cleaned {len(clean_data)} samples, skipped {skipped_count} invalid samples")
    return clean_data

def format_for_training(sample: Dict, prompt_template: str) -> Dict:
    """Format a sample for training with the specified template."""
    formatted_text = prompt_template.format(
        system=sample['system'],
        instruction=sample['instruction'],
        output=sample['output']
    )
    
    return {
        **sample,
        'text': formatted_text
    }

def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.15, test_ratio: float = 0.05) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, validation, and test sets."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        data, 
        test_size=test_ratio, 
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_data, val_data = train_test_split(
        train_val_data,
        test_size=val_size_adjusted,
        random_state=42,
        shuffle=True
    )
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data

def save_jsonl_data(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} samples to {file_path}")

def preprocess_data(
    input_file: str,
    output_dir: str,
    config_file: str = "config/training_config.yaml"
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Main preprocessing function."""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    data_config = config.get('data', {})
    train_ratio = data_config.get('train_split', 0.8)
    val_ratio = data_config.get('val_split', 0.15)
    test_ratio = data_config.get('test_split', 0.05)
    prompt_template = data_config.get('prompt_template', """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>""")
    
    logger.info(f"Loading data from {input_file}")
    
    # Load raw data
    raw_data = load_jsonl_data(input_file)
    logger.info(f"Loaded {len(raw_data)} raw samples")
    
    # Clean and validate
    clean_data = clean_and_validate_data(raw_data)
    
    if not clean_data:
        raise ValueError("No valid samples found after cleaning")
    
    # Format for training
    formatted_data = []
    for sample in clean_data:
        formatted_sample = format_for_training(sample, prompt_template)
        formatted_data.append(formatted_sample)
    
    # Split data
    train_data, val_data, test_data = split_data(
        formatted_data, 
        train_ratio, 
        val_ratio, 
        test_ratio
    )
    
    # Save splits
    save_jsonl_data(train_data, f"{output_dir}/train.jsonl")
    save_jsonl_data(val_data, f"{output_dir}/val.jsonl")
    save_jsonl_data(test_data, f"{output_dir}/test.jsonl")
    
    # Save preprocessing stats
    stats = {
        'total_raw_samples': len(raw_data),
        'total_clean_samples': len(clean_data),
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio
    }
    
    with open(f"{output_dir}/preprocessing_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("Preprocessing completed successfully")
    return train_data, val_data, test_data

def validate_processed_data(output_dir: str):
    """Validate that processed data files exist and are correct."""
    required_files = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    
    for file_name in required_files:
        file_path = os.path.join(output_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing processed file: {file_path}")
        
        # Load and validate a few samples
        data = load_jsonl_data(file_path)
        if not data:
            raise ValueError(f"Empty file: {file_path}")
        
        # Check first sample has required fields
        sample = data[0]
        required_fields = ['system', 'instruction', 'output', 'text']
        for field in required_fields:
            if field not in sample:
                raise ValueError(f"Missing field '{field}' in {file_path}")
        
        logger.info(f"Validated {file_path}: {len(data)} samples")
    
    logger.info("All processed data files validated successfully")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Preprocess data for Llama fine-tuning")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--config_file", type=str, default="config/training_config.yaml", help="Training config file")
    parser.add_argument("--validate_only", action="store_true", help="Only validate existing processed data")
    
    args = parser.parse_args()
    
    try:
        if args.validate_only:
            validate_processed_data(args.output_dir)
        else:
            preprocess_data(args.input_file, args.output_dir, args.config_file)
            validate_processed_data(args.output_dir)
        
        print("✅ Data preprocessing completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()