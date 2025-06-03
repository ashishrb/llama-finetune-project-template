# src/deployment/endpoint_test.py
import os
import sys
import yaml
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

def load_azure_config(config_path: str = "config/azure_config.yaml") -> dict:
    """Load Azure configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_deployment_config(config_path: str = "config/deployment_config.yaml") -> dict:
    """Load deployment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_ml_client(config: dict) -> MLClient:
    """Create Azure ML client."""
    credential = DefaultAzureCredential()
    
    ml_client = MLClient(
        credential=credential,
        subscription_id=config['azure']['subscription_id'],
        resource_group_name=config['azure']['resource_group'],
        workspace_name=config['azure']['workspace_name']
    )
    
    return ml_client

def get_endpoint_details(ml_client: MLClient, endpoint_name: str) -> tuple:
    """Get endpoint URI and authentication key."""
    try:
        # Get endpoint
        endpoint = ml_client.online_endpoints.get(endpoint_name)
        
        # Get endpoint keys
        keys = ml_client.online_endpoints.get_keys(endpoint_name)
        
        return endpoint.scoring_uri, keys.primary_key
        
    except Exception as e:
        print(f"❌ Failed to get endpoint details: {e}")
        print(f"   Make sure endpoint '{endpoint_name}' exists and is deployed")
        raise e

class EndpointTester:
    """Class for testing the deployed endpoint."""
    
    def __init__(self, scoring_uri: str, auth_key: str):
        self.scoring_uri = scoring_uri
        self.auth_key = auth_key
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {auth_key}'
        }
    
    def send_request(self, data: Dict[str, Any], timeout: int = 90) -> Dict[str, Any]:
        """Send request to endpoint and return response."""
        try:
            start_time = time.time()
            
            response = requests.post(
                self.scoring_uri,
                headers=self.headers,
                data=json.dumps(data),
                timeout=timeout
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                result['_metadata'] = {
                    'response_time': response_time,
                    'status_code': response.status_code
                }
                return result
            else:
                return {
                    'error': f'HTTP {response.status_code}: {response.text}',
                    '_metadata': {
                        'response_time': response_time,
                        'status_code': response.status_code
                    }
                }
                
        except requests.exceptions.Timeout:
            return {'error': 'Request timeout', '_metadata': {'response_time': timeout}}
        except Exception as e:
            return {'error': str(e), '_metadata': {'response_time': 0}}
    
    def test_health_check(self) -> bool:
        """Test basic endpoint health."""
        print("🏥 Testing endpoint health...")
        
        simple_test = {
            "instruction": "Hello",
            "max_new_tokens": 10
        }
        
        response = self.send_request(simple_test)
        
        if 'error' in response:
            print(f"❌ Health check failed: {response['error']}")
            return False
        else:
            print(f"✅ Health check passed ({response['_metadata']['response_time']:.2f}s)")
            return True
    
    def test_instruction_format(self) -> Dict[str, Any]:
        """Test instruction-based input format."""
        print("\n📝 Testing instruction format...")
        
        test_data = {
            "instruction": "How to create a new project in Azure ML workspace?",
            "system": "You are a helpful corporate AI assistant that helps employees with internal processes, applications, and workplace questions.",
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        response = self.send_request(test_data)
        
        if 'error' not in response:
            print(f"✅ Instruction format test passed")
            print(f"   Response time: {response['_metadata']['response_time']:.2f}s")
            print(f"   Response preview: {response.get('response', '')[:100]}...")
        else:
            print(f"❌ Instruction format test failed: {response['error']}")
        
        return response
    
    def test_chat_format(self) -> Dict[str, Any]:
        """Test chat message format."""
        print("\n💬 Testing chat format...")
        
        test_data = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful corporate AI assistant that helps employees with internal processes."
                },
                {
                    "role": "user", 
                    "content": "What is the difference between PM and FC job codes?"
                }
            ],
            "max_new_tokens": 150,
            "temperature": 0.8
        }
        
        response = self.send_request(test_data)
        
        if 'error' not in response:
            print(f"✅ Chat format test passed")
            print(f"   Response time: {response['_metadata']['response_time']:.2f}s")
            print(f"   Response preview: {response.get('response', '')[:100]}...")
        else:
            print(f"❌ Chat format test failed: {response['error']}")
        
        return response
    
    def test_prompt_format(self) -> Dict[str, Any]:
        """Test direct prompt format."""
        print("\n🎯 Testing direct prompt format...")
        
        test_data = {
            "prompt": "Explain the process to convert CWR to FTE associates:",
            "max_new_tokens": 180,
            "temperature": 0.6,
            "do_sample": True
        }
        
        response = self.send_request(test_data)
        
        if 'error' not in response:
            print(f"✅ Direct prompt test passed")
            print(f"   Response time: {response['_metadata']['response_time']:.2f}s")
            print(f"   Response preview: {response.get('response', '')[:100]}...")
        else:
            print(f"❌ Direct prompt test failed: {response['error']}")
        
        return response
    
    def test_batch_requests(self) -> Dict[str, Any]:
        """Test batch processing."""
        print("\n📦 Testing batch requests...")
        
        batch_data = [
            {
                "instruction": "How to raise a service request?",
                "max_new_tokens": 100
            },
            {
                "instruction": "What is the difference between staffing and transactional SO?", 
                "max_new_tokens": 100
            }
        ]
        
        response = self.send_request(batch_data)
        
        if 'error' not in response:
            print(f"✅ Batch processing test passed")
            print(f"   Response time: {response['_metadata']['response_time']:.2f}s")
            print(f"   Batch size: {len(response) if isinstance(response, list) else 1}")
        else:
            print(f"❌ Batch processing test failed: {response['error']}")
        
        return response
    
    def test_generation_parameters(self) -> List[Dict[str, Any]]:
        """Test different generation parameters."""
        print("\n⚙️ Testing generation parameters...")
        
        base_request = {
            "instruction": "Explain Azure ML compute clusters",
            "max_new_tokens": 100
        }
        
        test_configs = [
            {"temperature": 0.1, "description": "Conservative (low temperature)"},
            {"temperature": 0.7, "description": "Balanced (medium temperature)"},
            {"temperature": 1.0, "description": "Creative (high temperature)"},
            {"top_p": 0.5, "temperature": 0.7, "description": "Focused (low top_p)"},
            {"top_p": 0.95, "temperature": 0.7, "description": "Diverse (high top_p)"}
        ]
        
        results = []
        
        for config in test_configs:
            print(f"   Testing {config['description']}...")
            
            test_data = {**base_request, **{k: v for k, v in config.items() if k != 'description'}}
            response = self.send_request(test_data)
            
            if 'error' not in response:
                print(f"   ✅ {config['description']} - {response['_metadata']['response_time']:.2f}s")
            else:
                print(f"   ❌ {config['description']} - {response['error']}")
            
            results.append({
                'config': config,
                'response': response
            })
        
        return results
    
    def test_error_handling(self) -> List[Dict[str, Any]]:
        """Test error handling with invalid inputs."""
        print("\n🚨 Testing error handling...")
        
        error_tests = [
            {"data": {}, "description": "Empty request"},
            {"data": {"invalid_field": "test"}, "description": "Invalid fields only"},
            {"data": {"instruction": ""}, "description": "Empty instruction"},
            {"data": {"instruction": "test", "max_new_tokens": -1}, "description": "Invalid max_tokens"},
            {"data": {"instruction": "test", "temperature": 2.5}, "description": "Invalid temperature"},
        ]
        
        results = []
        
        for test in error_tests:
            print(f"   Testing {test['description']}...")
            
            response = self.send_request(test['data'])
            
            if 'error' in response:
                print(f"   ✅ {test['description']} - Properly handled error")
            else:
                print(f"   ⚠️  {test['description']} - Unexpected success")
            
            results.append({
                'test': test,
                'response': response
            })
        
        return results
    
    def performance_benchmark(self, num_requests: int = 5) -> Dict[str, Any]:
        """Run performance benchmark."""
        print(f"\n🏃 Running performance benchmark ({num_requests} requests)...")
        
        test_data = {
            "instruction": "How to create a Staffing SO request in Azure?",
            "max_new_tokens": 150,
            "temperature": 0.7
        }
        
        response_times = []
        successful_requests = 0
        
        for i in range(num_requests):
            print(f"   Request {i+1}/{num_requests}...", end=" ")
            
            response = self.send_request(test_data)
            
            if 'error' not in response:
                response_times.append(response['_metadata']['response_time'])
                successful_requests += 1
                print(f"✅ {response['_metadata']['response_time']:.2f}s")
            else:
                print(f"❌ Error: {response['error']}")
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\n📊 Performance Summary:")
            print(f"   Successful requests: {successful_requests}/{num_requests}")
            print(f"   Average response time: {avg_time:.2f}s")
            print(f"   Min response time: {min_time:.2f}s")
            print(f"   Max response time: {max_time:.2f}s")
            print(f"   Success rate: {(successful_requests/num_requests)*100:.1f}%")
            
            return {
                'successful_requests': successful_requests,
                'total_requests': num_requests,
                'avg_response_time': avg_time,
                'min_response_time': min_time,
                'max_response_time': max_time,
                'success_rate': (successful_requests/num_requests)*100
            }
        else:
            print("❌ No successful requests for benchmark")
            return {'error': 'No successful requests'}

def run_comprehensive_test(tester: EndpointTester) -> Dict[str, Any]:
    """Run all tests and return comprehensive results."""
    print("="*80)
    print("🧪 RUNNING COMPREHENSIVE ENDPOINT TESTS")
    print("="*80)
    
    results = {}
    
    # Health check
    results['health_check'] = tester.test_health_check()
    
    if not results['health_check']:
        print("❌ Health check failed. Stopping tests.")
        return results
    
    # Format tests
    results['instruction_test'] = tester.test_instruction_format()
    results['chat_test'] = tester.test_chat_format()
    results['prompt_test'] = tester.test_prompt_format()
    results['batch_test'] = tester.test_batch_requests()
    
    # Parameter tests
    results['parameter_tests'] = tester.test_generation_parameters()
    
    # Error handling
    results['error_tests'] = tester.test_error_handling()
    
    # Performance benchmark
    results['performance'] = tester.performance_benchmark()
    
    return results

def save_test_results(results: Dict[str, Any], output_file: str = "output/endpoint_test_results.json"):
    """Save test results to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Add timestamp
    results['test_timestamp'] = datetime.now().isoformat()
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📁 Test results saved to: {output_file}")

def print_test_summary(results: Dict[str, Any]):
    """Print test summary."""
    print("\n" + "="*80)
    print("📋 TEST SUMMARY")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    # Health check
    if results.get('health_check'):
        passed_tests += 1
    total_tests += 1
    
    # Format tests
    format_tests = ['instruction_test', 'chat_test', 'prompt_test', 'batch_test']
    for test in format_tests:
        if results.get(test) and 'error' not in results[test]:
            passed_tests += 1
        total_tests += 1
    
    print(f"✅ Passed: {passed_tests}/{total_tests} tests")
    
    # Performance summary
    if 'performance' in results and 'success_rate' in results['performance']:
        perf = results['performance']
        print(f"🏃 Performance: {perf['success_rate']:.1f}% success rate, {perf['avg_response_time']:.2f}s avg")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if passed_tests == total_tests:
        print("   🎉 All tests passed! Endpoint is ready for production.")
    else:
        print("   ⚠️  Some tests failed. Review logs and fix issues before production use.")
        print("   📊 Check Azure ML Studio for detailed logs and metrics.")
    
    print("="*80)

def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test deployed Llama endpoint")
    parser.add_argument("--azure_config", type=str, default="config/azure_config.yaml",
                       help="Path to Azure config file")
    parser.add_argument("--deployment_config", type=str, default="config/deployment_config.yaml",
                       help="Path to deployment config file")
    parser.add_argument("--endpoint_name", type=str, default=None,
                       help="Endpoint name (override config)")
    parser.add_argument("--benchmark_requests", type=int, default=5,
                       help="Number of requests for performance benchmark")
    parser.add_argument("--quick_test", action="store_true",
                       help="Run only basic health check and one format test")
    parser.add_argument("--save_results", type=str, default="output/endpoint_test_results.json",
                       help="Path to save test results")
    
    args = parser.parse_args()
    
    try:
        # Load configurations
        azure_config = load_azure_config(args.azure_config)
        deployment_config = load_deployment_config(args.deployment_config)
        
        endpoint_name = args.endpoint_name or deployment_config['deployment']['endpoint_name']
        
        print("="*80)
        print("🧪 ENDPOINT TESTING SUITE")
        print("="*80)
        print(f"Endpoint: {endpoint_name}")
        print(f"Workspace: {azure_config['azure']['workspace_name']}")
        
        # Get ML client and endpoint details
        ml_client = get_ml_client(azure_config)
        scoring_uri, auth_key = get_endpoint_details(ml_client, endpoint_name)
        
        print(f"Scoring URI: {scoring_uri}")
        print(f"Authentication: Key-based")
        
        # Create tester
        tester = EndpointTester(scoring_uri, auth_key)
        
        # Run tests
        if args.quick_test:
            print("\n🏃 Running quick test...")
            health_ok = tester.test_health_check()
            if health_ok:
                tester.test_instruction_format()
            results = {'quick_test': True, 'health_check': health_ok}
        else:
            results = run_comprehensive_test(tester)
        
        # Save results
        if args.save_results:
            save_test_results(results, args.save_results)
        
        # Print summary
        print_test_summary(results)
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")
        return False
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Ensure endpoint is deployed and running")
        print("2. Check endpoint name spelling")
        print("3. Verify Azure authentication")
        print("4. Check endpoint health in Azure ML Studio")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)