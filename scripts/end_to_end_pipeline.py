# scripts/end_to_end_pipeline.py
import os
import sys
import json
import yaml
import time
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError

# Import project modules
from data.data_preprocessing import preprocess_data, validate_processed_data
from src.evaluation.evaluate import run_comprehensive_evaluation
from scripts.setup_cluster import main as setup_cluster_main
from scripts.submit_training import main as submit_training_main
from scripts.deploy_model import main as deploy_model_main

logger = logging.getLogger(__name__)

class LlamaFinetunePipeline:
    """End-to-end pipeline for Llama fine-tuning on Azure ML."""
    
    def __init__(
        self,
        config_dir: str = "config",
        output_dir: str = "output",
        data_dir: str = "data"
    ):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.data_dir = Path(data_dir)
        
        # Pipeline state tracking
        self.pipeline_state = {
            'start_time': None,
            'current_stage': None,
            'completed_stages': [],
            'failed_stages': [],
            'stage_timings': {},
            'artifacts': {}
        }
        
        # Load configurations
        self.configs = self._load_all_configs()
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_pipeline_logging()
    
    def _load_all_configs(self) -> Dict:
        """Load all configuration files."""
        configs = {}
        
        config_files = {
            'azure': 'azure_config.yaml',
            'training': 'training_config.yaml',
            'model': 'model_config.yaml',
            'deployment': 'deployment_config.yaml'
        }
        
        for config_name, config_file in config_files.items():
            config_path = self.config_dir / config_file
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    configs[config_name] = yaml.safe_load(f)
                logger.info(f"Loaded {config_name} configuration")
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        return configs
    
    def _setup_directories(self):
        """Create necessary directories."""
        directories = [
            self.output_dir,
            self.output_dir / "logs",
            self.output_dir / "models",
            self.output_dir / "evaluation",
            self.output_dir / "pipeline_state",
            self.data_dir / "processed"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_pipeline_logging(self):
        """Setup comprehensive logging for the pipeline."""
        log_file = self.output_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"Pipeline logging initialized: {log_file}")
    
    def _update_stage(self, stage_name: str, status: str = "running"):
        """Update pipeline stage tracking."""
        if status == "running":
            self.pipeline_state['current_stage'] = stage_name
            self.pipeline_state['stage_timings'][stage_name] = {'start': time.time()}
            logger.info(f"Starting stage: {stage_name}")
        
        elif status == "completed":
            if stage_name in self.pipeline_state['stage_timings']:
                self.pipeline_state['stage_timings'][stage_name]['end'] = time.time()
                duration = (
                    self.pipeline_state['stage_timings'][stage_name]['end'] - 
                    self.pipeline_state['stage_timings'][stage_name]['start']
                )
                self.pipeline_state['stage_timings'][stage_name]['duration'] = duration
                logger.info(f"Completed stage: {stage_name} (Duration: {duration:.2f}s)")
            
            self.pipeline_state['completed_stages'].append(stage_name)
            self.pipeline_state['current_stage'] = None
        
        elif status == "failed":
            self.pipeline_state['failed_stages'].append(stage_name)
            self.pipeline_state['current_stage'] = None
            logger.error(f"Failed stage: {stage_name}")
        
        # Save state
        self._save_pipeline_state()
    
    def _save_pipeline_state(self):
        """Save pipeline state to file."""
        state_file = self.output_dir / "pipeline_state" / "current_state.json"
        
        with open(state_file, 'w') as f:
            json.dump(self.pipeline_state, f, indent=2, default=str)
    
    def _load_pipeline_state(self) -> bool:
        """Load existing pipeline state if available."""
        state_file = self.output_dir / "pipeline_state" / "current_state.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    self.pipeline_state = json.load(f)
                logger.info("Loaded existing pipeline state")
                return True
            except Exception as e:
                logger.warning(f"Could not load pipeline state: {e}")
        
        return False
    
    def validate_prerequisites(self) -> bool:
        """Validate all prerequisites before starting pipeline."""
        logger.info("Validating prerequisites...")
        
        # Check raw data exists
        raw_data_file = self.data_dir / "raw" / "azure_instruction_dataset.jsonl"
        if not raw_data_file.exists():
            logger.error(f"Raw data file not found: {raw_data_file}")
            return False
        
        # Check Azure authentication
        try:
            credential = DefaultAzureCredential()
            ml_client = MLClient(
                credential=credential,
                subscription_id=self.configs['azure']['azure']['subscription_id'],
                resource_group_name=self.configs['azure']['azure']['resource_group'],
                workspace_name=self.configs['azure']['azure']['workspace_name']
            )
            # Test connection
            workspace = ml_client.workspaces.get(self.configs['azure']['azure']['workspace_name'])
            logger.info(f"Azure ML workspace accessible: {workspace.name}")
        except Exception as e:
            logger.error(f"Azure ML connection failed: {e}")
            return False
        
        # Check configuration consistency
        if not self._validate_configurations():
            return False
        
        logger.info("Prerequisites validation passed")
        return True
    
    def _validate_configurations(self) -> bool:
        """Validate configuration consistency."""
        try:
            # Check model consistency
            training_base_model = self.configs['training']['model']['base_model']
            model_base_model = self.configs['model']['model']['base_model']
            
            if training_base_model != model_base_model:
                logger.warning(f"Base model mismatch: training={training_base_model}, model={model_base_model}")
            
            # Check compute configuration
            vm_size = self.configs['azure']['compute']['vm_size']
            if "H100" in vm_size:
                logger.info(f"Using H100 VM: {vm_size}")
            
            # Check output paths consistency
            training_output = self.configs['training']['output']['model_dir']
            deployment_model_path = self.configs['deployment']['deployment']['model']['path']
            
            logger.info("Configuration validation passed")
            return True
            
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            return False
    
    def stage_data_preprocessing(self, force_reprocess: bool = False) -> bool:
        """Stage 1: Data preprocessing."""
        self._update_stage("data_preprocessing", "running")
        
        try:
            # Check if processed data already exists
            processed_files = [
                self.data_dir / "processed" / "train.jsonl",
                self.data_dir / "processed" / "val.jsonl",
                self.data_dir / "processed" / "test.jsonl"
            ]
            
            if all(f.exists() for f in processed_files) and not force_reprocess:
                logger.info("Processed data already exists, skipping preprocessing")
            else:
                logger.info("Starting data preprocessing...")
                
                # Run preprocessing
                raw_file = self.data_dir / "raw" / "azure_instruction_dataset.jsonl"
                output_dir = self.data_dir / "processed"
                config_file = self.config_dir / "training_config.yaml"
                
                train_data, val_data, test_data = preprocess_data(
                    input_file=str(raw_file),
                    output_dir=str(output_dir),
                    config_file=str(config_file)
                )
                
                # Validate processed data
                validate_processed_data(str(output_dir))
                
                # Store artifacts info
                self.pipeline_state['artifacts']['processed_data'] = {
                    'train_samples': len(train_data),
                    'val_samples': len(val_data),
                    'test_samples': len(test_data),
                    'output_dir': str(output_dir)
                }
            
            self._update_stage("data_preprocessing", "completed")
            return True
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {e}")
            self._update_stage("data_preprocessing", "failed")
            return False
    
    def stage_setup_infrastructure(self, force_recreate: bool = False) -> bool:
        """Stage 2: Setup Azure ML infrastructure."""
        self._update_stage("setup_infrastructure", "running")
        
        try:
            logger.info("Setting up Azure ML infrastructure...")
            
            # Prepare arguments for setup script
            setup_args = [
                '--config', str(self.config_dir / 'azure_config.yaml')
            ]
            
            if force_recreate:
                setup_args.append('--force-recreate')
            
            # Run setup cluster script
            success = self._run_script_module('scripts.setup_cluster', setup_args)
            
            if success:
                # Store infrastructure info
                self.pipeline_state['artifacts']['infrastructure'] = {
                    'compute_cluster': self.configs['azure']['compute']['cluster_name'],
                    'vm_size': self.configs['azure']['compute']['vm_size'],
                    'environment': self.configs['azure']['environment']['name']
                }
                
                self._update_stage("setup_infrastructure", "completed")
                return True
            else:
                self._update_stage("setup_infrastructure", "failed")
                return False
                
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            self._update_stage("setup_infrastructure", "failed")
            return False
    
    def stage_training(self, custom_job_name: Optional[str] = None) -> bool:
        """Stage 3: Model training."""
        self._update_stage("training", "running")
        
        try:
            logger.info("Starting model training...")
            
            # Prepare training arguments
            training_args = [
                '--azure_config', str(self.config_dir / 'azure_config.yaml'),
                '--training_config', str(self.config_dir / 'training_config.yaml')
            ]
            
            if custom_job_name:
                training_args.extend(['--job_name', custom_job_name])
            
            # Run training script
            success = self._run_script_module('scripts.submit_training', training_args)
            
            if success:
                # Store training info
                model_output_dir = self.output_dir / "models"
                self.pipeline_state['artifacts']['training'] = {
                    'job_name': custom_job_name or f"llama_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'model_output_dir': str(model_output_dir),
                    'base_model': self.configs['training']['model']['base_model']
                }
                
                self._update_stage("training", "completed")
                return True
            else:
                self._update_stage("training", "failed")
                return False
                
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._update_stage("training", "failed")
            return False
    
    def stage_evaluation(self, max_eval_samples: Optional[int] = None) -> bool:
        """Stage 4: Model evaluation."""
        self._update_stage("evaluation", "running")
        
        try:
            logger.info("Starting model evaluation...")
            
            # Check if model exists
            model_path = self.output_dir / "models" / "named-outputs" / "model"
            if not model_path.exists():
                # Try alternative path
                model_path = self.output_dir / "models" / "final_model"
                if not model_path.exists():
                    logger.error(f"Trained model not found in expected locations")
                    self._update_stage("evaluation", "failed")
                    return False
            
            # Prepare test data
            test_data_path = self.data_dir / "processed" / "test.jsonl"
            if not test_data_path.exists():
                logger.error(f"Test data not found: {test_data_path}")
                self._update_stage("evaluation", "failed")
                return False
            
            # Run evaluation
            eval_output_dir = self.output_dir / "evaluation"
            
            evaluation_results = run_comprehensive_evaluation(
                model_path=str(model_path),
                test_data_path=str(test_data_path),
                output_dir=str(eval_output_dir),
                base_model_name=self.configs['training']['model']['base_model'],
                max_samples=max_eval_samples,
                batch_size=4,  # Conservative for H100 memory
                generation_params={
                    'temperature': 0.7,
                    'max_new_tokens': 200,
                    'top_p': 0.9,
                    'do_sample': True
                }
            )
            
            # Store evaluation results
            self.pipeline_state['artifacts']['evaluation'] = {
                'results_file': str(eval_output_dir / "evaluation_results.json"),
                'report_file': str(eval_output_dir / "evaluation_report.md"),
                'test_samples': evaluation_results.get('num_samples', 0),
                'key_metrics': {
                    'bleu_4': evaluation_results.get('bleu', {}).get('bleu_4', 0),
                    'rouge_l': evaluation_results.get('rouge', {}).get('rougel', 0),
                    'bertscore_f1': evaluation_results.get('bertscore', {}).get('f1', 0)
                }
            }
            
            logger.info(f"Evaluation completed. Key metrics: {self.pipeline_state['artifacts']['evaluation']['key_metrics']}")
            
            self._update_stage("evaluation", "completed")
            return True
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self._update_stage("evaluation", "failed")
            return False
    
    def stage_deployment(self, skip_test: bool = False) -> bool:
        """Stage 5: Model deployment."""
        self._update_stage("deployment", "running")
        
        try:
            logger.info("Starting model deployment...")
            
            # Check if model exists
            model_path = self.output_dir / "models" / "final_model"
            if not model_path.exists():
                # Try named-outputs path
                model_path = self.output_dir / "models" / "named-outputs" / "model"
                if not model_path.exists():
                    logger.error(f"Trained model not found for deployment")
                    self._update_stage("deployment", "failed")
                    return False
            
            # Prepare deployment arguments
            deployment_args = [
                '--azure_config', str(self.config_dir / 'azure_config.yaml'),
                '--deployment_config', str(self.config_dir / 'deployment_config.yaml'),
                '--model_path', str(model_path),
                '--model_name', 'llama-corporate-qa'
            ]
            
            if skip_test:
                deployment_args.append('--skip_test')
            
            # Run deployment script
            success = self._run_script_module('scripts.deploy_model', deployment_args)
            
            if success:
                # Store deployment info
                self.pipeline_state['artifacts']['deployment'] = {
                    'endpoint_name': self.configs['deployment']['deployment']['endpoint_name'],
                    'deployment_name': self.configs['deployment']['deployment']['deployment_name'],
                    'instance_type': self.configs['deployment']['deployment']['instance_type'],
                    'model_path': str(model_path)
                }
                
                self._update_stage("deployment", "completed")
                return True
            else:
                self._update_stage("deployment", "failed")
                return False
                
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self._update_stage("deployment", "failed")
            return False
    
    def _run_script_module(self, module_name: str, args: List[str]) -> bool:
        """Run a script module with arguments."""
        try:
            # Import and run the module's main function
            if module_name == 'scripts.setup_cluster':
                # Modify sys.argv for the module
                original_argv = sys.argv.copy()
                sys.argv = ['setup_cluster.py'] + args
                
                try:
                    from scripts.setup_cluster import main
                    success = main()
                finally:
                    sys.argv = original_argv
                
                return success
            
            elif module_name == 'scripts.submit_training':
                original_argv = sys.argv.copy()
                sys.argv = ['submit_training.py'] + args
                
                try:
                    from scripts.submit_training import main
                    success = main()
                finally:
                    sys.argv = original_argv
                
                return success
            
            elif module_name == 'scripts.deploy_model':
                original_argv = sys.argv.copy()
                sys.argv = ['deploy_model.py'] + args
                
                try:
                    from scripts.deploy_model import main
                    success = main()
                finally:
                    sys.argv = original_argv
                
                return success
            
            else:
                logger.error(f"Unknown module: {module_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to run {module_name}: {e}")
            return False
    
    def run_full_pipeline(
        self,
        stages: Optional[List[str]] = None,
        force_reprocess_data: bool = False,
        force_recreate_infrastructure: bool = False,
        max_eval_samples: Optional[int] = None,
        skip_deployment_test: bool = False,
        resume_from_failure: bool = False
    ) -> bool:
        """Run the complete end-to-end pipeline."""
        
        # Default stages
        if stages is None:
            stages = [
                "data_preprocessing",
                "setup_infrastructure", 
                "training",
                "evaluation",
                "deployment"
            ]
        
        # Resume from previous state if requested
        if resume_from_failure:
            self._load_pipeline_state()
            # Skip completed stages
            stages = [stage for stage in stages if stage not in self.pipeline_state.get('completed_stages', [])]
            logger.info(f"Resuming pipeline from stages: {stages}")
        
        # Initialize pipeline
        self.pipeline_state['start_time'] = time.time()
        self._save_pipeline_state()
        
        logger.info("="*80)
        logger.info("🚀 STARTING LLAMA FINE-TUNING PIPELINE")
        logger.info("="*80)
        logger.info(f"Stages to run: {', '.join(stages)}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("Prerequisites validation failed")
            return False
        
        # Execute stages
        for stage in stages:
            logger.info(f"\n{'='*60}")
            logger.info(f"STAGE: {stage.upper().replace('_', ' ')}")
            logger.info(f"{'='*60}")
            
            success = False
            
            if stage == "data_preprocessing":
                success = self.stage_data_preprocessing(force_reprocess_data)
            
            elif stage == "setup_infrastructure":
                success = self.stage_setup_infrastructure(force_recreate_infrastructure)
            
            elif stage == "training":
                success = self.stage_training()
            
            elif stage == "evaluation":
                success = self.stage_evaluation(max_eval_samples)
            
            elif stage == "deployment":
                success = self.stage_deployment(skip_deployment_test)
            
            else:
                logger.error(f"Unknown stage: {stage}")
                success = False
            
            if not success:
                logger.error(f"Pipeline failed at stage: {stage}")
                self._generate_failure_report()
                return False
        
        # Pipeline completed successfully
        self._generate_success_report()
        return True
    
    def _generate_success_report(self):
        """Generate success report."""
        end_time = time.time()
        total_duration = end_time - self.pipeline_state['start_time']
        
        report = f"""
{'='*80}
🎉 LLAMA FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY!
{'='*80}

Pipeline Summary:
- Start Time: {datetime.fromtimestamp(self.pipeline_state['start_time']).strftime('%Y-%m-%d %H:%M:%S')}
- End Time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}
- Total Duration: {total_duration/3600:.2f} hours ({total_duration:.0f} seconds)
- Completed Stages: {len(self.pipeline_state['completed_stages'])}
- Failed Stages: {len(self.pipeline_state['failed_stages'])}

Stage Timings:
"""
        
        for stage, timing in self.pipeline_state['stage_timings'].items():
            if 'duration' in timing:
                duration_mins = timing['duration'] / 60
                report += f"- {stage.replace('_', ' ').title()}: {duration_mins:.1f} minutes\n"
        
        if 'evaluation' in self.pipeline_state['artifacts']:
            eval_metrics = self.pipeline_state['artifacts']['evaluation']['key_metrics']
            report += f"""
Model Performance:
- BLEU-4: {eval_metrics.get('bleu_4', 0):.4f}
- ROUGE-L: {eval_metrics.get('rouge_l', 0):.4f}  
- BERTScore F1: {eval_metrics.get('bertscore_f1', 0):.4f}
"""
        
        if 'deployment' in self.pipeline_state['artifacts']:
            deployment_info = self.pipeline_state['artifacts']['deployment']
            report += f"""
Deployment Info:
- Endpoint: {deployment_info['endpoint_name']}
- Instance Type: {deployment_info['instance_type']}
- Model Path: {deployment_info['model_path']}
"""
        
        report += f"""
Artifacts Generated:
- Model: {self.output_dir}/models/
- Evaluation: {self.output_dir}/evaluation/
- Logs: {self.output_dir}/logs/
- Pipeline State: {self.output_dir}/pipeline_state/

Next Steps:
1. Test the deployed endpoint
2. Monitor model performance
3. Set up continuous evaluation
4. Plan for model updates

{'='*80}
"""
        
        logger.info(report)
        
        # Save report to file
        report_file = self.output_dir / "pipeline_success_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Success report saved to: {report_file}")
    
    def _generate_failure_report(self):
        """Generate failure report."""
        end_time = time.time()
        duration = end_time - self.pipeline_state['start_time']
        
        report = f"""
{'='*80}
❌ LLAMA FINE-TUNING PIPELINE FAILED
{'='*80}

Pipeline Summary:
- Start Time: {datetime.fromtimestamp(self.pipeline_state['start_time']).strftime('%Y-%m-%d %H:%M:%S')}
- Failure Time: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}
- Duration Before Failure: {duration/60:.1f} minutes
- Completed Stages: {self.pipeline_state['completed_stages']}
- Failed Stages: {self.pipeline_state['failed_stages']}
- Current Stage: {self.pipeline_state['current_stage']}

Troubleshooting:
1. Check logs in: {self.output_dir}/logs/
2. Review pipeline state: {self.output_dir}/pipeline_state/current_state.json
3. Verify Azure ML quota and permissions
4. Check data file integrity
5. Validate configuration files

To Resume:
Use --resume_from_failure flag to continue from the last successful stage.

{'='*80}
"""
        
        logger.error(report)
        
        # Save report to file
        report_file = self.output_dir / "pipeline_failure_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.error(f"Failure report saved to: {report_file}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="End-to-end Llama fine-tuning pipeline")
    
    # Pipeline configuration
    parser.add_argument("--config_dir", type=str, default="config", help="Configuration directory")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    
    # Stage selection
    parser.add_argument("--stages", nargs='+', 
                       choices=["data_preprocessing", "setup_infrastructure", "training", "evaluation", "deployment"],
                       help="Specific stages to run (default: all)")
    
    # Stage-specific options
    parser.add_argument("--force_reprocess_data", action="store_true", help="Force data reprocessing")
    parser.add_argument("--force_recreate_infrastructure", action="store_true", help="Force infrastructure recreation")
    parser.add_argument("--max_eval_samples", type=int, help="Maximum samples for evaluation")
    parser.add_argument("--skip_deployment_test", action="store_true", help="Skip deployment testing")
    
    # Pipeline control
    parser.add_argument("--resume_from_failure", action="store_true", help="Resume from last failure")
    parser.add_argument("--dry_run", action="store_true", help="Validate configuration without running")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = LlamaFinetunePipeline(
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            data_dir=args.data_dir
        )
        
        # Dry run mode
        if args.dry_run:
            logger.info("Running in dry-run mode - validating configuration only")
            if pipeline.validate_prerequisites():
                logger.info("✅ Configuration validation passed")
                return True
            else:
                logger.error("❌ Configuration validation failed")
                return False
        
        # Run pipeline
        success = pipeline.run_full_pipeline(
            stages=args.stages,
            force_reprocess_data=args.force_reprocess_data,
            force_recreate_infrastructure=args.force_recreate_infrastructure,
            max_eval_samples=args.max_eval_samples,
            skip_deployment_test=args.skip_deployment_test,
            resume_from_failure=args.resume_from_failure
        )
        
        return success
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        raise e


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)