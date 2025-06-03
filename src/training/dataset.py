# src/training/dataset.py
import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer
import logging
import weakref
from collections import OrderedDict

logger = logging.getLogger(__name__)

class LimitedCache:
    """
    Memory-aware cache with size limits and automatic cleanup.
    Uses OrderedDict for LRU (Least Recently Used) eviction.
    """
    
    def __init__(self, max_size: int = 1000, memory_limit_mb: int = 2048):
        self.max_size = max_size
        self.memory_limit_mb = memory_limit_mb
        self.cache = OrderedDict()
        self._memory_usage_mb = 0
        
    def __contains__(self, key):
        return key in self.cache
    
    def __getitem__(self, key):
        if key in self.cache:
            # Move to end (mark as recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        # Remove old value if exists
        if key in self.cache:
            old_value = self.cache.pop(key)
            self._memory_usage_mb -= self._estimate_size_mb(old_value)
        
        # Add new value
        self.cache[key] = value
        self._memory_usage_mb += self._estimate_size_mb(value)
        
        # Cleanup if necessary
        self._cleanup_if_needed()
    
    def _estimate_size_mb(self, value) -> float:
        """Estimate memory usage of cached value in MB."""
        try:
            if isinstance(value, dict):
                size_bytes = 0
                for k, v in value.items():
                    if hasattr(v, 'numel'):  # PyTorch tensor
                        size_bytes += v.numel() * v.element_size()
                    else:
                        size_bytes += len(str(v).encode('utf-8'))
                return size_bytes / (1024 * 1024)  # Convert to MB
            return 0.1  # Default estimate
        except:
            return 0.1
    
    def _cleanup_if_needed(self):
        """Remove old entries if cache exceeds limits."""
        # Remove oldest entries if over size limit
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            oldest_value = self.cache.pop(oldest_key)
            self._memory_usage_mb -= self._estimate_size_mb(oldest_value)
        
        # Remove oldest entries if over memory limit
        while self._memory_usage_mb > self.memory_limit_mb and self.cache:
            oldest_key = next(iter(self.cache))
            oldest_value = self.cache.pop(oldest_key)
            self._memory_usage_mb -= self._estimate_size_mb(oldest_value)
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self._memory_usage_mb = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'memory_usage_mb': self._memory_usage_mb,
            'memory_limit_mb': self.memory_limit_mb
        }

class CorporateQADataset(Dataset):
    """
    Custom dataset class for corporate Q&A data with efficient tokenization and formatting.
    Supports both instruction-response format and chat format.
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        prompt_template: Optional[str] = None,
        use_chat_format: bool = True,
        cache_tokenization: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSONL file containing the data
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            prompt_template: Template for formatting prompts (if None, uses default)
            use_chat_format: Whether to use chat format or instruction format
            cache_tokenization: Whether to cache tokenized data in memory
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_format = use_chat_format
        self.cache_tokenization = cache_tokenization
        
        # Default prompt template for Llama-3.2
        self.prompt_template = prompt_template or """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""
        
        # Load and validate data
        self.data = self._load_data()
        
        # Cache for tokenized data with size limit
        self._tokenized_cache = LimitedCache(max_size=1000) if cache_tokenization else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Loaded {len(self.data)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """Load and validate data from JSONL file."""
        data = []
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        sample = json.loads(line)
                        
                        # Validate required fields
                        if self._validate_sample(sample):
                            data.append(sample)
                        else:
                            logger.warning(f"Skipping invalid sample at line {line_num}")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        continue
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if not data:
            raise ValueError(f"No valid data found in {self.data_path}")
        
        return data
    
    def _validate_sample(self, sample: Dict) -> bool:
        """Validate that a sample has required fields."""
        required_fields = ['system', 'instruction', 'output']
        
        for field in required_fields:
            if field not in sample:
                return False
            if not isinstance(sample[field], str) or not sample[field].strip():
                return False
        
        return True
    
    def _format_prompt(self, sample: Dict) -> str:
        """Format a sample into the appropriate prompt format."""
        if self.use_chat_format:
            return self._format_chat_prompt(sample)
        else:
            return self._format_instruction_prompt(sample)
    
    def _format_chat_prompt(self, sample: Dict) -> str:
        """Format sample as a chat conversation."""
        return self.prompt_template.format(
            system=sample['system'].strip(),
            instruction=sample['instruction'].strip(),
            output=sample['output'].strip()
        )
    
    def _format_instruction_prompt(self, sample: Dict) -> str:
        """Format sample as instruction-response pair."""
        # Alternative format without chat tokens
        return f"### System:\n{sample['system']}\n\n### Instruction:\n{sample['instruction']}\n\n### Response:\n{sample['output']}"
    
    def _tokenize_sample(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize a text sample."""
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # We'll pad in collate_fn
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for causal LM)
        labels = encoded["input_ids"].clone()
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0)
        }
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized sample."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        # Check cache first
        if self._tokenized_cache is not None and idx in self._tokenized_cache:
            self._cache_hits += 1
            return self._tokenized_cache[idx]
        
        self._cache_misses += 1
        
        # Get raw sample
        sample = self.data[idx]
        
        # Format prompt
        formatted_text = self._format_prompt(sample)
        
        # Tokenize
        tokenized = self._tokenize_sample(formatted_text)
        
        # Add metadata
        tokenized["text"] = formatted_text
        tokenized["original_sample"] = sample
        
        # Cache if enabled with memory monitoring
        if self._tokenized_cache is not None:
            try:
                self._tokenized_cache[idx] = tokenized
                
                # Log cache stats periodically
                if (self._cache_hits + self._cache_misses) % 100 == 0:
                    self._log_cache_stats()
                    
            except Exception as e:
                logger.warning(f"Failed to cache sample {idx}: {e}")
                # Continue without caching
                pass

        return tokenized
    
    def get_sample_text(self, idx: int) -> str:
        """Get the formatted text for a sample without tokenization."""
        sample = self.data[idx]
        return self._format_prompt(sample)
    
    def get_raw_sample(self, idx: int) -> Dict:
        """Get the raw sample data."""
        return self.data[idx]
    
    # def clear_cache(self):
    #     """Clear the tokenization cache to free memory."""
    #     if self._tokenized_cache is not None:
    #         self._tokenized_cache.clear()
    #         logger.info("Cleared tokenization cache")
    
    def _log_cache_stats(self):
        """Log cache statistics."""
        if self._tokenized_cache is not None:
            stats = self._tokenized_cache.get_stats()
            hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
            
            logger.info(f"Cache stats - Size: {stats['size']}/{stats['max_size']}, "
                    f"Memory: {stats['memory_usage_mb']:.1f}/{stats['memory_limit_mb']}MB, "
                    f"Hit rate: {hit_rate:.2%}")

    def get_cache_info(self) -> dict:
        """Get detailed cache information."""
        if self._tokenized_cache is None:
            return {'caching_enabled': False}
        
        stats = self._tokenized_cache.get_stats()
        hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses) if (self._cache_hits + self._cache_misses) > 0 else 0
        
        return {
            'caching_enabled': True,
            'cache_size': stats['size'],
            'max_cache_size': stats['max_size'],
            'memory_usage_mb': stats['memory_usage_mb'],
            'memory_limit_mb': stats['memory_limit_mb'],
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate
        }

    def optimize_cache(self):
        """Optimize cache by removing least recently used items."""
        if self._tokenized_cache is not None:
            old_size = len(self._tokenized_cache.cache)
            
            # Force cleanup to half the current size
            target_size = max(100, old_size // 2)
            while len(self._tokenized_cache.cache) > target_size:
                oldest_key = next(iter(self._tokenized_cache.cache))
                oldest_value = self._tokenized_cache.cache.pop(oldest_key)
                self._tokenized_cache._memory_usage_mb -= self._tokenized_cache._estimate_size_mb(oldest_value)
            
            new_size = len(self._tokenized_cache.cache)
            logger.info(f"Cache optimized: {old_size} -> {new_size} items")

    def clear_cache(self):
        """Clear the tokenization cache to free memory."""
        if self._tokenized_cache is not None:
            old_stats = self._tokenized_cache.get_stats()
            self._tokenized_cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info(f"Cleared tokenization cache (freed {old_stats['memory_usage_mb']:.1f}MB)")


class DataCollator:
    """
    Custom data collator for batching tokenized samples with proper padding.
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, padding: bool = True, max_batch_memory_mb: float = 512):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_batch_memory_mb = max_batch_memory_mb
        self._batch_count = 0
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate features into a batch."""
        self._batch_count += 1
        
        # Monitor memory usage every 50 batches
        if self._batch_count % 50 == 0:
            self._check_memory_usage(features)
        
        # Extract tensors
        input_ids = [f["input_ids"] for f in features]
        attention_masks = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences
        if self.padding:
            input_ids = self._pad_sequences(input_ids, self.tokenizer.pad_token_id)
            attention_masks = self._pad_sequences(attention_masks, 0)
            labels = self._pad_sequences(labels, -100)  # -100 is ignored by loss function
        else:
            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)
            labels = torch.stack(labels)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }
    
    def _check_memory_usage(self, features: List[Dict[str, torch.Tensor]]):
        """Check memory usage of current batch."""
        try:
            # Estimate batch memory usage
            total_elements = sum(f["input_ids"].numel() for f in features)
            estimated_mb = (total_elements * 4) / (1024 * 1024)  # 4 bytes per float32
            
            if estimated_mb > self.max_batch_memory_mb:
                logger.warning(f"Large batch detected: {estimated_mb:.1f}MB (limit: {self.max_batch_memory_mb}MB)")
                
        except Exception as e:
            logger.debug(f"Could not estimate batch memory: {e}")
    
    def _pad_sequences(self, sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        """Pad sequences to the same length."""
        max_length = max(len(seq) for seq in sequences)
        
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                padding = torch.full((max_length - len(seq),), pad_value, dtype=seq.dtype)
                padded_seq = torch.cat([seq, padding])
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        return torch.stack(padded)


def create_dataloaders(
    train_dataset: CorporateQADataset,
    val_dataset: CorporateQADataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> tuple:
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer for the data collator
        batch_size: Batch size for data loading
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create data collator
    collator = DataCollator(tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def load_dataset_from_config(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    config: Dict,
    split: str = "train"
) -> CorporateQADataset:
    """
    Load dataset from configuration.
    
    Args:
        data_path: Path to the data file
        tokenizer: Tokenizer to use
        config: Configuration dictionary
        split: Dataset split ("train", "val", "test")
    
    Returns:
        CorporateQADataset instance
    """
    data_config = config.get('data', {})
    
    dataset = CorporateQADataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=data_config.get('max_length', 2048),
        prompt_template=data_config.get('prompt_template'),
        use_chat_format=True,  # Default to chat format for Llama
        cache_tokenization=split == "train"  # Cache training data only
    )
    
    return dataset


# Utility functions for dataset analysis
def analyze_dataset(dataset: CorporateQADataset) -> Dict:
    """Analyze dataset statistics."""
    stats = {
        "total_samples": len(dataset),
        "avg_input_length": 0,
        "max_input_length": 0,
        "min_input_length": float('inf'),
        "system_prompt_variety": set(),
        "instruction_lengths": [],
        "output_lengths": []
    }
    
    for i in range(min(len(dataset), 1000)):  # Sample first 1000 for analysis
        sample = dataset.get_raw_sample(i)
        text = dataset.get_sample_text(i)
        
        # Text length stats
        text_length = len(text)
        stats["avg_input_length"] += text_length
        stats["max_input_length"] = max(stats["max_input_length"], text_length)
        stats["min_input_length"] = min(stats["min_input_length"], text_length)
        
        # Component length stats
        stats["system_prompt_variety"].add(sample["system"])
        stats["instruction_lengths"].append(len(sample["instruction"]))
        stats["output_lengths"].append(len(sample["output"]))
    
    # Calculate averages
    analyzed_samples = min(len(dataset), 1000)
    stats["avg_input_length"] /= analyzed_samples
    stats["avg_instruction_length"] = sum(stats["instruction_lengths"]) / len(stats["instruction_lengths"])
    stats["avg_output_length"] = sum(stats["output_lengths"]) / len(stats["output_lengths"])
    stats["unique_system_prompts"] = len(stats["system_prompt_variety"])
    
    # Remove large lists to keep stats manageable
    del stats["system_prompt_variety"]
    del stats["instruction_lengths"] 
    del stats["output_lengths"]
    
    return stats


if __name__ == "__main__":
    # Example usage and testing
    from transformers import AutoTokenizer
    
    # Load tokenizer (example)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3.2-1b-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test dataset loading
    try:
        dataset = CorporateQADataset(
            data_path="data/processed/train.jsonl",
            tokenizer=tokenizer,
            max_length=2048
        )
        
        print(f"Dataset loaded successfully with {len(dataset)} samples")
        
        # Print sample
        sample = dataset[0]
        print(f"Sample input shape: {sample['input_ids'].shape}")
        print(f"Sample text preview: {sample['text'][:200]}...")
        
        # Analyze dataset
        stats = analyze_dataset(dataset)
        print(f"Dataset statistics: {stats}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")