# llama-finetune-project-template

Usage example of E2E Pipe Line
# Run complete pipeline
python scripts/end_to_end_pipeline.py

# Run specific stages only
python scripts/end_to_end_pipeline.py --stages training evaluation

# Resume from failure
python scripts/end_to_end_pipeline.py --resume_from_failure

# Validate configuration only
python scripts/end_to_end_pipeline.py --dry_run

# Force reprocessing
python scripts/end_to_end_pipeline.py --force_reprocess_data --force_recreate_infrastructure

# Limit evaluation samples for faster testing
python scripts/end_to_end_pipeline.py --max_eval_samples 100

# Skip deployment testing
python scripts/end_to_end_pipeline.py --skip_deployment_test

Data Preprocessing Usage:
# Process your data
python data/data_preprocessing.py --input_file data/raw/azure_instruction_dataset.jsonl --output_dir data/processed

# Or just validate existing processed data
python data/data_preprocessing.py --validate_only --output_dir data/processed

How to use test scripts:
# Test data preprocessing
python -m pytest tests/test_data_preprocessing.py -v

# Test training components  
python -m pytest tests/test_training.py -v

# Test deployment
python -m pytest tests/test_deployment.py -v

# Test inference
python -m pytest tests/test_inference.py -v



Test with your actual data:
# Process your real dataset
python data/data_preprocessing.py --input_file data/raw/your_actual_data.jsonl --output_dir data/processed

Azure ML setup (if desired):
# Set up your Azure ML environment
python scripts/setup_cluster.py

Full training run:
# When ready, run the full pipeline
python scripts/end_to_end_pipeline.py