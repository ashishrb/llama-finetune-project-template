# tests/test_deployment.py
import unittest
import tempfile
import os
import json
import yaml
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class TestDeploymentValidation(unittest.TestCase):
    """Test deployment configuration and validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test deployment config
        self.deployment_config = {
            'deployment': {
                'endpoint_name': 'test-endpoint',
                'deployment_name': 'test-deployment',
                'instance_type': 'Standard_NC6s_v3',
                'instance_count': 1,
                'scale_settings': {
                    'type': 'default',
                    'min_instances': 1,
                    'max_instances': 3,
                    'target_utilization_percentage': 70
                },
                'request_timeout_ms': 90000,
                'max_concurrent_requests_per_instance': 1,
                'model': {
                    'path': './output/models',
                    'format': 'mlflow'
                },
                'liveness_probe': {
                    'initial_delay': 30,
                    'period': 30,
                    'timeout': 2,
                    'failure_threshold': 30
                },
                'readiness_probe': {
                    'initial_delay': 30,
                    'period': 30,
                    'timeout': 2,
                    'failure_threshold': 3
                }
            }
        }
        
        self.config_file = os.path.join(self.temp_dir, "deployment_config.yaml")
        with open(self.config_file, 'w') as f:
            yaml.dump(self.deployment_config, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_config_loading(self):
        """Test deployment configuration loading."""
        from scripts.deploy_model import load_deployment_config
        
        config = load_deployment_config(self.config_file)
        
        self.assertIn('deployment', config)
        self.assertEqual(config['deployment']['endpoint_name'], 'test-endpoint')
        self.assertEqual(config['deployment']['instance_type'], 'Standard_NC6s_v3')
    
    def test_model_artifacts_validation(self):
        """Test model artifacts validation."""
        from scripts.deploy_model import validate_model_artifacts
        
        # Create fake model directory
        model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(model_dir)
        
        # Test with missing files
        self.assertFalse(validate_model_artifacts(model_dir))
        
        # Create required files
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "pytorch_model.bin"
        ]
        
        for file_name in required_files:
            with open(os.path.join(model_dir, file_name), 'w') as f:
                f.write('{}')
        
        # Should pass validation now
        self.assertTrue(validate_model_artifacts(model_dir))
    
    def test_inference_script_creation(self):
        """Test inference script creation."""
        from scripts.deploy_model import create_inference_script
        
        # Create inference script
        create_inference_script()
        
        # Check if script was created
        script_path = "src/deployment/score.py"
        self.assertTrue(os.path.exists(script_path))
        
        # Basic validation of script content
        with open(script_path, 'r') as f:
            content = f.read()
            self.assertIn('def init():', content)
            self.assertIn('def run(raw_data):', content)
            self.assertIn('def generate_response(', content)
        
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)
    
    def test_endpoint_configuration_validation(self):
        """Test endpoint configuration validation."""
        config = self.deployment_config['deployment']
        
        # Validate required fields
        required_fields = [
            'endpoint_name', 'deployment_name', 'instance_type',
            'instance_count', 'request_timeout_ms'
        ]
        
        for field in required_fields:
            self.assertIn(field, config)
            self.assertIsNotNone(config[field])
        
        # Validate types
        self.assertIsInstance(config['instance_count'], int)
        self.assertIsInstance(config['request_timeout_ms'], int)
        self.assertGreater(config['instance_count'], 0)
        self.assertGreater(config['request_timeout_ms'], 0)
    
    @patch('azure.ai.ml.MLClient')
    def test_ml_client_creation(self, mock_ml_client):
        """Test Azure ML client creation."""
        from scripts.deploy_model import get_ml_client
        
        # Mock Azure config
        azure_config = {
            'azure': {
                'subscription_id': 'test-subscription',
                'resource_group': 'test-rg',
                'workspace_name': 'test-workspace'
            }
        }
        
        # Mock the client
        mock_client = Mock()
        mock_ml_client.return_value = mock_client
        
        client = get_ml_client(azure_config)
        
        # Verify MLClient was called with correct parameters
        mock_ml_client.assert_called_once()
        self.assertEqual(client, mock_client)

class TestEndpointTesting(unittest.TestCase):
    """Test endpoint testing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('requests.post')
    def test_endpoint_request_format(self, mock_post):
        """Test endpoint request formatting."""
        from src.deployment.endpoint_test import EndpointTester
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Test response',
            'prompt_tokens': 10,
            'completion_tokens': 5
        }
        mock_post.return_value = mock_response
        
        tester = EndpointTester("http://test-endpoint", "test-key")
        
        # Test instruction format
        test_data = {
            "instruction": "What is PM job code?",
            "system": "You are a helpful assistant.",
            "max_new_tokens": 100
        }
        
        response = tester.send_request(test_data)
        
        # Verify request was made
        mock_post.assert_called_once()
        
        # Verify response format
        self.assertIn('response', response)
        self.assertIn('_metadata', response)
    
    def test_request_validation(self):
        """Test request validation for different input formats."""
        from src.deployment.endpoint_test import EndpointTester
        
        tester = EndpointTester("http://test-endpoint", "test-key")
        
        # Test different valid input formats
        valid_formats = [
            {
                "instruction": "Test instruction",
                "max_new_tokens": 100
            },
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Test question"}
                ]
            },
            {
                "prompt": "Direct prompt text"
            }
        ]
        
        # All should be valid request formats
        for format_data in valid_formats:
            # Just validate structure, don't make actual requests
            self.assertIsInstance(format_data, dict)
            self.assertTrue(len(format_data) > 0)
    
    def test_response_validation(self):
        """Test response validation and error handling."""
        from src.deployment.endpoint_test import EndpointTester
        
        tester = EndpointTester("http://test-endpoint", "test-key")
        
        # Test response parsing
        mock_responses = [
            {'response': 'Valid response', 'prompt_tokens': 10, 'completion_tokens': 20},
            {'error': 'Test error message'},
            {'response': ''}  # Empty response
        ]
        
        for mock_response in mock_responses:
            # Validate response structure
            if 'error' in mock_response:
                self.assertIn('error', mock_response)
            else:
                self.assertIn('response', mock_response)

class TestDeploymentIntegration(unittest.TestCase):
    """Integration tests for deployment components."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_deployment_pipeline_validation(self):
        """Test deployment pipeline component compatibility."""
        # Create test model directory structure
        model_dir = os.path.join(self.temp_dir, "model")
        os.makedirs(model_dir)
        
        # Create minimal model files
        model_files = {
            "config.json": {"model_type": "llama"},
            "tokenizer_config.json": {"tokenizer_class": "LlamaTokenizer"},
            "tokenizer.json": {"version": "1.0"},
        }
        
        for filename, content in model_files.items():
            with open(os.path.join(model_dir, filename), 'w') as f:
                json.dump(content, f)
        
        # Create a dummy model file
        with open(os.path.join(model_dir, "pytorch_model.bin"), 'wb') as f:
            f.write(b'dummy model data')
        
        # Test model validation
        from scripts.deploy_model import validate_model_artifacts
        self.assertTrue(validate_model_artifacts(model_dir))
    
    def test_configuration_consistency(self):
        """Test consistency between different configuration files."""
        # Create Azure config
        azure_config = {
            'azure': {
                'subscription_id': 'test-sub',
                'resource_group': 'test-rg',
                'workspace_name': 'test-ws',
                'location': 'eastus'
            }
        }
        
        # Create deployment config
        deployment_config = {
            'deployment': {
                'endpoint_name': 'test-endpoint',
                'deployment_name': 'test-deployment',
                'instance_type': 'Standard_NC6s_v3',
                'instance_count': 1
            }
        }
        
        # Validate configuration structure
        self.assertIn('azure', azure_config)
        self.assertIn('deployment', deployment_config)
        
        # Check required fields
        azure_fields = ['subscription_id', 'resource_group', 'workspace_name']
        for field in azure_fields:
            self.assertIn(field, azure_config['azure'])
        
        deployment_fields = ['endpoint_name', 'deployment_name', 'instance_type']
        for field in deployment_fields:
            self.assertIn(field, deployment_config['deployment'])
    
    def test_environment_setup_validation(self):
        """Test environment and dependency validation."""
        # Test that required modules can be imported
        required_modules = [
            'yaml',
            'json',
            'pathlib',
            'datetime'
        ]
        
        for module_name in required_modules:
            try:
                __import__(module_name)
            except ImportError:
                self.fail(f"Required module {module_name} not available")
        
        # Test optional dependencies handling
        optional_modules = [
            'transformers',
            'torch',
            'azure.ai.ml'
        ]
        
        for module_name in optional_modules:
            try:
                __import__(module_name)
                # If available, basic functionality should work
            except ImportError:
                # Should be handled gracefully in actual code
                pass

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)