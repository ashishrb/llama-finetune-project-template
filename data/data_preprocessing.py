# data/data_preprocessing.py
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from sklearn.model_selection import train_test_split

def load_config(config_path: str = "config/training_config.yaml") -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def format_sample(sample: Dict, prompt_template: str) -> Dict:
    """Format a single sample according to the template."""
    formatted_text = prompt_template.format(
        system=sample['system'],
        instruction=sample['instruction'],
        output=sample['output']
    )
    
    return {
        "text": formatted_text,
        "system": sample['system'],
        "instruction": sample['instruction'],
        "output": sample['output']
    }

def split_data(data: List[Dict], train_split: float, val_split: float, test_split: float, 
               random_seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, validation, and test sets."""
    random.seed(random_seed)
    
    # First split: separate test set
    train_val, test = train_test_split(
        data, 
        test_size=test_split, 
        random_state=random_seed
    )
    
    # Second split: separate train and validation
    val_size_adjusted = val_split / (train_split + val_split)
    train, val = train_test_split(
        train_val, 
        test_size=val_size_adjusted, 
        random_state=random_seed
    )
    
    return train, val, test

def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def preprocess_data(input_file: str = "data/raw/azure_instruction_dataset.jsonl",
                   output_dir: str = "data/processed",
                   config_file: str = "config/training_config.yaml"):
    """Main preprocessing function."""
    
    print("Loading configuration...")
    config = load_config(config_file)
    
    print(f"Loading data from {input_file}...")
    raw_data = load_jsonl(input_file)
    print(f"Loaded {len(raw_data)} samples")
    
    # Get data split configuration
    data_config = config['data']
    train_split = data_config['train_split']
    val_split = data_config['val_split'] 
    test_split = data_config['test_split']
    prompt_template = data_config['prompt_template']
    
    print(f"Splitting data: {train_split:.1%} train, {val_split:.1%} val, {test_split:.1%} test")
    train_data, val_data, test_data = split_data(raw_data, train_split, val_split, test_split)
    
    print(f"Train: {len(train_data)} samples")
    print(f"Validation: {len(val_data)} samples") 
    print(f"Test: {len(test_data)} samples")
    
    # Format data according to prompt template
    print("Formatting data with prompt template...")
    train_formatted = [format_sample(sample, prompt_template) for sample in train_data]
    val_formatted = [format_sample(sample, prompt_template) for sample in val_data]
    test_formatted = [format_sample(sample, prompt_template) for sample in test_data]
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving processed data to {output_dir}/...")
    save_jsonl(train_formatted, f"{output_dir}/train.jsonl")
    save_jsonl(val_formatted, f"{output_dir}/val.jsonl")
    save_jsonl(test_formatted, f"{output_dir}/test.jsonl")
    
    # Save original splits for reference
    save_jsonl(train_data, f"{output_dir}/train_raw.jsonl")
    save_jsonl(val_data, f"{output_dir}/val_raw.jsonl")
    save_jsonl(test_data, f"{output_dir}/test_raw.jsonl")
    
    # Print sample formatted output
    print("\n" + "="*50)
    print("SAMPLE FORMATTED OUTPUT:")
    print("="*50)
    print(train_formatted[0]['text'][:500] + "...")
    print("="*50)
    
    # Print statistics
    print(f"\nData preprocessing completed successfully!")
    print(f"Files saved in {output_dir}/")
    print(f"- train.jsonl: {len(train_formatted)} samples")
    print(f"- val.jsonl: {len(val_formatted)} samples") 
    print(f"- test.jsonl: {len(test_formatted)} samples")
    
    return train_formatted, val_formatted, test_formatted

def validate_processed_data(processed_dir: str = "data/processed"):
    """Validate the processed data files."""
    print("Validating processed data...")
    
    files_to_check = ['train.jsonl', 'val.jsonl', 'test.jsonl']
    
    for file_name in files_to_check:
        file_path = f"{processed_dir}/{file_name}"
        if not os.path.exists(file_path):
            print(f"❌ Missing file: {file_path}")
            continue
            
        data = load_jsonl(file_path)
        print(f"✅ {file_name}: {len(data)} samples")
        
        # Check required fields
        if data:
            sample = data[0]
            required_fields = ['text', 'system', 'instruction', 'output']
            missing_fields = [field for field in required_fields if field not in sample]
            
            if missing_fields:
                print(f"   ❌ Missing fields: {missing_fields}")
            else:
                print(f"   ✅ All required fields present")
                
            # Check text format
            if '<|begin_of_text|>' in sample['text'] and '<|eot_id|>' in sample['text']:
                print(f"   ✅ Proper Llama format detected")
            else:
                print(f"   ❌ Llama format not detected")

if __name__ == "__main__":
    # Run preprocessing
    train_data, val_data, test_data = preprocess_data()
    
    # Validate the processed data
    validate_processed_data()
    
    print("\n🎉 Data preprocessing pipeline completed successfully!")
