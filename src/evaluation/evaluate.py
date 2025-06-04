# src/evaluation/evaluate.py
import os
import sys
import json
import torch
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.training.utils import GPUMemoryManager, log_system_resources, memory_efficient_model_load

# Add project root to path
#sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import PeftModel
from datasets import Dataset
import mlflow

# Import custom modules
from src.training.dataset import CorporateQADataset, load_jsonl_data
from src.evaluation.metrics import (
    calculate_bleu_score,
    calculate_rouge_scores,
    calculate_bertscore,
    calculate_semantic_similarity,
    calculate_perplexity,
    calculate_diversity_metrics,
    calculate_corporate_qa_metrics
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation for corporate Q&A tasks."""
    
    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            device: str = "auto"
        ):
            self.model = model
            self.tokenizer = tokenizer
            self.device = self._setup_device(device)
            self.model.to(self.device)
            self.model.eval()
            
            # ADD THIS: Initialize memory manager
            self.memory_manager = GPUMemoryManager()
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
            
            # ADD THIS: Log initial memory state
            self.memory_manager.log_memory_summary()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup device for evaluation."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        device = torch.device(device)
        logger.info(f"Using device: {device}")
        return device
    
    def generate_response(
    self,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    num_return_sequences: int = 1
) -> List[str]:
        """Generate response for a given prompt."""
        
        # ADD THIS: Monitor memory before generation
        if not self.memory_manager.check_available_memory(2.0):  # Require 2GB free
            self.memory_manager.clear_gpu_cache()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False
        ).to(self.device)
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU OOM during generation")
                self.memory_manager.clear_gpu_cache()
                # Retry with smaller parameters
                return self._retry_generation_with_reduced_params(prompt, max_new_tokens // 2)
            raise e
        
        # Decode responses
        responses = []
        for output in outputs:
            # Remove input tokens from output
            response_tokens = output[len(inputs.input_ids[0]):]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response.strip())
        
        return responses
    
    def _retry_generation_with_reduced_params(self, prompt: str, max_new_tokens: int) -> List[str]:
        """Retry generation with reduced parameters after OOM."""
        logger.warning(f"Retrying generation with reduced max_new_tokens: {max_new_tokens}")
        
        try:
            return self.generate_response(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("Failed even with reduced parameters")
                return ["Error: Unable to generate response due to memory constraints"]
            raise e

    def evaluate_dataset(
    self,
    dataset: Union[CorporateQADataset, List[Dict]],
    batch_size: int = 8,
    max_samples: Optional[int] = None,
    generation_params: Optional[Dict] = None
) -> Dict:
        """Evaluate model on a dataset."""
        
        logger.info("Starting dataset evaluation...")
        
        # ADD THIS: Log initial memory state
        self.memory_manager.log_memory_summary()
        
        # Default generation parameters
        gen_params = {
            'max_new_tokens': 200,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True
        }
        if generation_params:
            gen_params.update(generation_params)
        
        # Prepare data
        if isinstance(dataset, CorporateQADataset):
            eval_data = [dataset.get_raw_sample(i) for i in range(len(dataset))]
        else:
            eval_data = dataset
        
        # Limit samples if specified
        if max_samples:
            eval_data = eval_data[:max_samples]
        
        predictions = []
        references = []
        prompts = []
        
        # ADD THIS: Memory-aware batch processing
        processed_samples = 0
        
        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(eval_data), batch_size), desc="Evaluating"):
            batch = eval_data[i:i+batch_size]
            
            # ADD THIS: Memory monitoring every 10 batches
            if i % (batch_size * 10) == 0:
                self.memory_manager.monitor_and_cleanup()
            
            for sample in batch:
                # Format prompt
                prompt = self._format_prompt_for_evaluation(sample)
                
                # Generate response
                try:
                    response = self.generate_response(prompt, **gen_params)[0]
                    
                    predictions.append(response)
                    references.append(sample['output'])
                    prompts.append(prompt)
                    processed_samples += 1
                    
                except Exception as e:
                    logger.warning(f"Error generating response for sample {processed_samples}: {e}")
                    predictions.append("")
                    references.append(sample['output'])
                    prompts.append(prompt)
                    processed_samples += 1
            
            # ADD THIS: Aggressive cleanup every 50 samples
            if processed_samples % 50 == 0:
                self.memory_manager.clear_gpu_cache()
                logger.info(f"Processed {processed_samples}/{len(eval_data)} samples")
        
        # ADD THIS: Final memory cleanup and summary
        self.memory_manager.clear_gpu_cache()
        self.memory_manager.log_memory_summary()
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics...")
        metrics = self._calculate_all_metrics(predictions, references, prompts)
        
        # Add sample analysis
        metrics['sample_analysis'] = self._analyze_samples(predictions, references, prompts)
        
        logger.info("Dataset evaluation completed")
        return metrics
    
    def _format_prompt_for_evaluation(self, sample: Dict) -> str:
        """Format sample for evaluation (input only, no reference output)."""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{sample['system']}<|eot_id|><|start_header_id|>user<|end_header_id|>

{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def _calculate_all_metrics(
        self, 
        predictions: List[str], 
        references: List[str], 
        prompts: List[str]
    ) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {}
        
        # Basic statistics
        metrics['num_samples'] = len(predictions)
        metrics['avg_prediction_length'] = np.mean([len(pred) for pred in predictions])
        metrics['avg_reference_length'] = np.mean([len(ref) for ref in references])
        
        # BLEU scores
        try:
            bleu_scores = calculate_bleu_score(predictions, references)
            metrics['bleu'] = bleu_scores
        except Exception as e:
            logger.warning(f"Error calculating BLEU: {e}")
            metrics['bleu'] = {'bleu_1': 0, 'bleu_2': 0, 'bleu_3': 0, 'bleu_4': 0}
        
        # ROUGE scores
        try:
            rouge_scores = calculate_rouge_scores(predictions, references)
            metrics['rouge'] = rouge_scores
        except Exception as e:
            logger.warning(f"Error calculating ROUGE: {e}")
            metrics['rouge'] = {}
        
        # BERTScore
        try:
            bert_scores = calculate_bertscore(predictions, references)
            metrics['bertscore'] = bert_scores
        except Exception as e:
            logger.warning(f"Error calculating BERTScore: {e}")
            metrics['bertscore'] = {}
        
        # Semantic similarity
        try:
            semantic_scores = calculate_semantic_similarity(predictions, references)
            metrics['semantic_similarity'] = semantic_scores
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            metrics['semantic_similarity'] = {}
        
        # Perplexity (on predictions)
        try:
            perplexity = calculate_perplexity(self.model, self.tokenizer, predictions)
            metrics['perplexity'] = perplexity
        except Exception as e:
            logger.warning(f"Error calculating perplexity: {e}")
            metrics['perplexity'] = float('inf')
        
        # Diversity metrics
        try:
            diversity_scores = calculate_diversity_metrics(predictions)
            metrics['diversity'] = diversity_scores
        except Exception as e:
            logger.warning(f"Error calculating diversity: {e}")
            metrics['diversity'] = {}
        
        # Corporate Q&A specific metrics
        try:
            corporate_metrics = calculate_corporate_qa_metrics(predictions, references)
            metrics['corporate_qa'] = corporate_metrics
        except Exception as e:
            logger.warning(f"Error calculating corporate metrics: {e}")
            metrics['corporate_qa'] = {}
        
        return metrics
    
    def _analyze_samples(
        self, 
        predictions: List[str], 
        references: List[str], 
        prompts: List[str]
    ) -> Dict:
        """Analyze sample quality and patterns."""
        
        analysis = {
            'empty_predictions': sum(1 for pred in predictions if not pred.strip()),
            'very_short_predictions': sum(1 for pred in predictions if len(pred.strip()) < 10),
            'very_long_predictions': sum(1 for pred in predictions if len(pred) > 1000),
            'avg_word_count': np.mean([len(pred.split()) for pred in predictions]),
            'repetitive_predictions': 0
        }
        
        # Check for repetitive responses
        unique_predictions = set(predictions)
        analysis['unique_predictions'] = len(unique_predictions)
        analysis['prediction_diversity'] = len(unique_predictions) / len(predictions) if predictions else 0
        
        # Find most common responses
        from collections import Counter
        pred_counts = Counter(predictions)
        analysis['most_common_responses'] = pred_counts.most_common(5)
        
        return analysis
    
    def evaluate_specific_categories(
        self,
        dataset: List[Dict],
        category_field: str = "category",
        **eval_kwargs
    ) -> Dict:
        """Evaluate performance on specific categories of questions."""
        
        # Group by category
        categories = {}
        for sample in dataset:
            category = sample.get(category_field, "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(sample)
        
        # Evaluate each category
        category_results = {}
        for category, samples in categories.items():
            logger.info(f"Evaluating category: {category} ({len(samples)} samples)")
            
            category_metrics = self.evaluate_dataset(samples, **eval_kwargs)
            category_results[category] = category_metrics
        
        return category_results
    
    def compare_with_baseline(
        self,
        baseline_responses: List[str],
        test_dataset: List[Dict],
        **eval_kwargs
    ) -> Dict:
        """Compare model performance with baseline responses."""
        
        # Evaluate current model
        current_metrics = self.evaluate_dataset(test_dataset, **eval_kwargs)
        
        # Evaluate baseline
        references = [sample['output'] for sample in test_dataset]
        baseline_metrics = self._calculate_all_metrics(baseline_responses, references, [])
        
        # Calculate improvements
        comparison = {
            'current_model': current_metrics,
            'baseline': baseline_metrics,
            'improvements': {}
        }
        
        # Compare key metrics
        key_metrics = ['bleu', 'rouge', 'bertscore', 'semantic_similarity']
        for metric_group in key_metrics:
            if metric_group in current_metrics and metric_group in baseline_metrics:
                current_scores = current_metrics[metric_group]
                baseline_scores = baseline_metrics[metric_group]
                
                improvement = {}
                for key in current_scores:
                    if key in baseline_scores:
                        current_val = current_scores[key]
                        baseline_val = baseline_scores[key]
                        if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                            improvement[key] = current_val - baseline_val
                
                comparison['improvements'][metric_group] = improvement
        
        return comparison


def load_model_for_evaluation(
    model_path: str,
    base_model_name: Optional[str] = None,
    device: str = "auto"
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer for evaluation."""
    
    logger.info(f"Loading model from {model_path}")
    
    # ADD THIS: Initialize memory manager and log system state
    memory_manager = GPUMemoryManager()
    log_system_resources()
    
    # Check if it's a LoRA adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora_adapter = adapter_config_path.exists()
    
    # ADD THIS: Estimate memory requirements
    estimated_memory_gb = 12.0 if is_lora_adapter else 8.0  # LoRA needs base model + adapter
    
    if not memory_manager.check_available_memory(estimated_memory_gb):
        memory_manager.clear_gpu_cache()
        if not memory_manager.check_available_memory(estimated_memory_gb):
            raise RuntimeError(f"Insufficient GPU memory for evaluation (need {estimated_memory_gb}GB)")
    
    try:
        if is_lora_adapter:
            # Load LoRA adapter
            if not base_model_name:
                # Try to read base model from adapter config
                try:
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model_name = adapter_config.get('base_model_name_or_path', 'unsloth/llama-3.2-2b-instruct')
                except:
                    base_model_name = 'unsloth/llama-3.2-2b-instruct'
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # ADD THIS: Memory monitoring during base model loading
            memory_manager.log_memory_summary()
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True
            )
            
            logger.info(f"Loading LoRA adapter from: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        else:
            # Load full model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto" if device == "auto" else None,
                trust_remote_code=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        
        # ADD THIS: Log memory after successful loading
        memory_manager.log_memory_summary()
        logger.info("Model and tokenizer loaded successfully")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU out of memory during model loading")
            memory_manager.clear_gpu_cache()
            raise RuntimeError("GPU OOM during model loading. Try using CPU or smaller model.")
        raise e
    
    return model, tokenizer


def run_comprehensive_evaluation(
    model_path: str,
    test_data_path: str,
    output_dir: str,
    base_model_name: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 8,
    generation_params: Optional[Dict] = None
) -> Dict:
    """Run comprehensive evaluation pipeline."""
    
    # CREATE OUTPUT DIRECTORY
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize memory manager and log system state
    memory_manager = GPUMemoryManager()
    log_system_resources()
    
    # Adjust batch size based on available memory
    memory_info = memory_manager.get_gpu_memory_info()
    if memory_info['total_gb'] < 16:  # Less than 16GB GPU
        batch_size = min(batch_size, 4)
        logger.info(f"Reduced batch size to {batch_size} due to limited GPU memory")
    
    try:
        # Load model
        model, tokenizer = load_model_for_evaluation(model_path, base_model_name)
        evaluator = ModelEvaluator(model, tokenizer)
        
        # Load test data
        if test_data_path.endswith('.jsonl'):
            test_data = load_jsonl_data(test_data_path)
        else:
            raise ValueError("Only JSONL format supported for test data")
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Limit samples based on memory if not specified
        if max_samples is None and memory_info['total_gb'] < 12:
            max_samples = min(len(test_data), 100)  # Limit to 100 for small GPUs
            logger.info(f"Limited evaluation to {max_samples} samples due to memory constraints")
        
        # Run evaluation
        logger.info("Starting comprehensive evaluation...")
        results = evaluator.evaluate_dataset(
            test_data,
            batch_size=batch_size,
            max_samples=max_samples,
            generation_params=generation_params
        )
        
        # Add metadata
        results['evaluation_metadata'] = {
            'model_path': model_path,
            'test_data_path': test_data_path,
            'timestamp': datetime.now().isoformat(),
            'num_test_samples': len(test_data),
            'max_samples_evaluated': max_samples or len(test_data),
            'generation_params': generation_params or {},
            'gpu_memory_used_gb': memory_manager.get_gpu_memory_info()['allocated_gb'],
            'peak_memory_gb': memory_manager.peak_memory
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        memory_manager.clear_gpu_cache()
        raise e
    
    finally:
        # Always cleanup memory at the end
        memory_manager.clear_gpu_cache()
        memory_manager.log_memory_summary()
    
    # ⚠️  IMPORTANT: These lines should be OUTSIDE the try/except/finally block
    # Save results
    results_file = output_path / "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Evaluation results saved to {results_file}")
    
    # Generate report
    generate_evaluation_report(results, output_path)
    
    # Log to MLflow if available
    try:
        log_evaluation_to_mlflow(results, model_path)
    except Exception as e:
        logger.warning(f"Could not log to MLflow: {e}")
    
    # Log final memory summary
    final_memory = memory_manager.get_gpu_memory_info()
    logger.info(f"Evaluation completed. Final GPU memory: {final_memory['allocated_gb']:.2f}GB")
    
    return results


def generate_evaluation_report(results: Dict, output_dir: Path):
    """Generate human-readable evaluation report."""
    
    report_file = output_dir / "evaluation_report.md"
    
    report = f"""# Model Evaluation Report

## Overview
- **Evaluation Date**: {results['evaluation_metadata']['timestamp']}
- **Model Path**: {results['evaluation_metadata']['model_path']}
- **Test Samples**: {results['evaluation_metadata']['num_test_samples']}
- **Samples Evaluated**: {results['evaluation_metadata']['max_samples_evaluated']}

## Key Metrics

### BLEU Scores
"""
    
    # Add BLEU scores
    if 'bleu' in results:
        for key, value in results['bleu'].items():
            report += f"- **{key.upper()}**: {value:.4f}\n"
    
    # Add ROUGE scores
    if 'rouge' in results:
        report += f"\n### ROUGE Scores\n"
        for key, value in results['rouge'].items():
            if isinstance(value, (int, float)):
                report += f"- **{key.upper()}**: {value:.4f}\n"
    
    # Add BERTScore
    if 'bertscore' in results:
        report += f"\n### BERTScore\n"
        for key, value in results['bertscore'].items():
            if isinstance(value, (int, float)):
                report += f"- **{key}**: {value:.4f}\n"
    
    # Add quality analysis
    if 'sample_analysis' in results:
        analysis = results['sample_analysis']
        report += f"""
### Sample Quality Analysis
- **Empty Predictions**: {analysis.get('empty_predictions', 0)}
- **Very Short Predictions**: {analysis.get('very_short_predictions', 0)}
- **Very Long Predictions**: {analysis.get('very_long_predictions', 0)}
- **Average Word Count**: {analysis.get('avg_word_count', 0):.1f}
- **Prediction Diversity**: {analysis.get('prediction_diversity', 0):.3f}
- **Unique Predictions**: {analysis.get('unique_predictions', 0)}

### Most Common Responses
"""
        
        for i, (response, count) in enumerate(analysis.get('most_common_responses', [])[:3]):
            report += f"{i+1}. \"{response[:100]}...\" (appeared {count} times)\n"
    
    # Add corporate-specific metrics
    if 'corporate_qa' in results:
        report += f"\n### Corporate Q&A Specific Metrics\n"
        for key, value in results['corporate_qa'].items():
            if isinstance(value, (int, float)):
                report += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
    
    # Add recommendations
    report += f"""
## Recommendations

### Strengths
- Model successfully generates responses for corporate Q&A tasks
- Maintains reasonable response length and structure

### Areas for Improvement
"""
    
    # Add specific recommendations based on metrics
    if results.get('sample_analysis', {}).get('empty_predictions', 0) > 0:
        report += "- Reduce empty predictions through better prompt engineering\n"
    
    if results.get('sample_analysis', {}).get('prediction_diversity', 1) < 0.8:
        report += "- Increase response diversity to avoid repetitive answers\n"
    
    if 'perplexity' in results and results['perplexity'] > 20:
        report += "- Consider additional training to reduce perplexity\n"
    
    report += """
### Next Steps
1. Analyze specific failure cases in detail
2. Consider fine-tuning on additional corporate domain data
3. Implement response filtering for quality assurance
4. Set up continuous evaluation pipeline
"""
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation report saved to {report_file}")


def log_evaluation_to_mlflow(results: Dict, model_path: str):
    """Log evaluation results to MLflow."""
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("num_samples", results['num_samples'])
        
        # Log metrics
        if 'bleu' in results:
            for key, value in results['bleu'].items():
                mlflow.log_metric(f"eval_{key}", value)
        
        if 'rouge' in results:
            for key, value in results['rouge'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_rouge_{key}", value)
        
        if 'bertscore' in results:
            for key, value in results['bertscore'].items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"eval_bertscore_{key}", value)
        
        if 'perplexity' in results:
            mlflow.log_metric("eval_perplexity", results['perplexity'])
        
        # Log sample analysis
        if 'sample_analysis' in results:
            analysis = results['sample_analysis']
            mlflow.log_metric("eval_prediction_diversity", analysis.get('prediction_diversity', 0))
            mlflow.log_metric("eval_avg_word_count", analysis.get('avg_word_count', 0))


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Llama model on corporate Q&A")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data (JSONL)")
    parser.add_argument("--output_dir", type=str, default="output/evaluation", help="Output directory")
    parser.add_argument("--base_model", type=str, default=None, help="Base model name for LoRA adapters")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generation parameters
    generation_params = {
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'top_p': 0.9,
        'do_sample': True
    }
    
    try:
        # Run evaluation
        results = run_comprehensive_evaluation(
            model_path=args.model_path,
            test_data_path=args.test_data,
            output_dir=args.output_dir,
            base_model_name=args.base_model,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            generation_params=generation_params
        )
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Model: {args.model_path}")
        print(f"Test samples: {results['num_samples']}")
        
        if 'bleu' in results:
            print(f"BLEU-4: {results['bleu'].get('bleu_4', 0):.4f}")
        
        if 'rouge' in results and 'rouge_l' in results['rouge']:
            print(f"ROUGE-L: {results['rouge']['rouge_l']:.4f}")
        
        if 'bertscore' in results and 'f1' in results['bertscore']:
            print(f"BERTScore F1: {results['bertscore']['f1']:.4f}")
        
        print(f"Results saved to: {args.output_dir}")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise e


if __name__ == "__main__":
    main()