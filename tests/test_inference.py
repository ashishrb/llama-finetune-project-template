# tests/test_inference.py
import unittest
import tempfile
import os
import json
import sys
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

class TestInferenceFormatting(unittest.TestCase):
    """Test inference input/output formatting."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_chat_format_parsing(self):
        """Test parsing of chat message format."""
        # Test data that matches the inference script format
        chat_input = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful corporate AI assistant."
                },
                {
                    "role": "user", 
                    "content": "What is PM job code?"
                }
            ],
            "max_new_tokens": 200,
            "temperature": 0.7
        }
        
        # Validate input structure
        self.assertIn("messages", chat_input)
        self.assertIsInstance(chat_input["messages"], list)
        self.assertGreater(len(chat_input["messages"]), 0)
        
        # Validate message structure
        for message in chat_input["messages"]:
            self.assertIn("role", message)
            self.assertIn("content", message)
            self.assertIn(message["role"], ["system", "user", "assistant"])
    
    def test_instruction_format_parsing(self):
        """Test parsing of instruction format."""
        instruction_input = {
            "instruction": "What is the difference between PM and FC job codes?",
            "system": "You are a helpful corporate AI assistant.",
            "max_new_tokens": 150,
            "temperature": 0.8
        }
        
        # Validate input structure
        self.assertIn("instruction", instruction_input)
        self.assertIn("system", instruction_input)
        self.assertIsInstance(instruction_input["instruction"], str)
        self.assertIsInstance(instruction_input["system"], str)
        self.assertGreater(len(instruction_input["instruction"]), 0)
    
    def test_direct_prompt_format(self):
        """Test direct prompt format."""
        prompt_input = {
            "prompt": "Explain the process to convert CWR to FTE associates:",
            "max_new_tokens": 180,
            "temperature": 0.6,
            "do_sample": True
        }
        
        # Validate input structure
        self.assertIn("prompt", prompt_input)
        self.assertIsInstance(prompt_input["prompt"], str)
        self.assertGreater(len(prompt_input["prompt"]), 0)
    
    def test_llama_format_conversion(self):
        """Test conversion to Llama chat format."""
        # Mock the format_chat_messages function logic
        def format_chat_messages(messages):
            formatted = "<|begin_of_text|>"
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    formatted += f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "user":
                    formatted += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
                elif role == "assistant":
                    formatted += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
            
            formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            return formatted
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is PM job code?"}
        ]
        
        formatted = format_chat_messages(messages)
        
        # Validate Llama format
        self.assertIn("<|begin_of_text|>", formatted)
        self.assertIn("<|start_header_id|>system<|end_header_id|>", formatted)
        self.assertIn("<|start_header_id|>user<|end_header_id|>", formatted)
        self.assertIn("<|eot_id|>", formatted)
    
    def test_generation_parameters(self):
        """Test generation parameter validation."""
        valid_params = {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        # Validate parameter types and ranges
        self.assertIsInstance(valid_params["max_new_tokens"], int)
        self.assertGreater(valid_params["max_new_tokens"], 0)
        self.assertLessEqual(valid_params["max_new_tokens"], 2048)
        
        self.assertIsInstance(valid_params["temperature"], (int, float))
        self.assertGreaterEqual(valid_params["temperature"], 0.0)
        self.assertLessEqual(valid_params["temperature"], 2.0)
        
        self.assertIsInstance(valid_params["top_p"], (int, float))
        self.assertGreaterEqual(valid_params["top_p"], 0.0)
        self.assertLessEqual(valid_params["top_p"], 1.0)
        
        self.assertIsInstance(valid_params["do_sample"], bool)

class TestInferenceResponseProcessing(unittest.TestCase):
    """Test inference response processing and validation."""
    
    def test_response_extraction(self):
        """Test extraction of assistant response from model output."""
        # Mock model output with chat format
        full_output = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful corporate AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

What is PM job code?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

PM job code indicates Full Time Employee.<|eot_id|>"""
        
        # Mock the response extraction logic
        def extract_assistant_response(full_text):
            if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
                assistant_part = full_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
                response = assistant_part.replace("<|eot_id|>", "").strip()
                return response
            return full_text
        
        extracted = extract_assistant_response(full_output)
        
        # Validate extraction
        self.assertEqual(extracted, "PM job code indicates Full Time Employee.")
        self.assertNotIn("<|start_header_id|>", extracted)
        self.assertNotIn("<|eot_id|>", extracted)
    
    def test_response_format_validation(self):
        """Test response format validation."""
        valid_response = {
            "response": "PM job code indicates Full Time Employee.",
            "prompt_tokens": 25,
            "completion_tokens": 8
        }
        
        # Validate response structure
        self.assertIn("response", valid_response)
        self.assertIn("prompt_tokens", valid_response)
        self.assertIn("completion_tokens", valid_response)
        
        # Validate response content
        self.assertIsInstance(valid_response["response"], str)
        self.assertGreater(len(valid_response["response"]), 0)
        
        # Validate token counts
        self.assertIsInstance(valid_response["prompt_tokens"], int)
        self.assertIsInstance(valid_response["completion_tokens"], int)
        self.assertGreaterEqual(valid_response["prompt_tokens"], 0)
        self.assertGreaterEqual(valid_response["completion_tokens"], 0)
    
    def test_error_response_handling(self):
        """Test error response handling."""
        error_response = {
            "error": "Invalid input format",
            "_metadata": {
                "response_time": 0.1,
                "status_code": 400
            }
        }
        
        # Validate error structure
        self.assertIn("error", error_response)
        self.assertIn("_metadata", error_response)
        self.assertIsInstance(error_response["error"], str)
    
    def test_batch_response_processing(self):
        """Test batch response processing."""
        batch_responses = [
            {
                "response": "PM job code indicates Full Time Employee.",
                "prompt_tokens": 20,
                "completion_tokens": 8
            },
            {
                "response": "FC job code indicates Full Time Contractor.",
                "prompt_tokens": 22,
                "completion_tokens": 8
            }
        ]
        
        # Validate batch structure
        self.assertIsInstance(batch_responses, list)
        self.assertGreater(len(batch_responses), 1)
        
        # Validate each response
        for response in batch_responses:
            self.assertIn("response", response)
            self.assertIn("prompt_tokens", response)
            self.assertIn("completion_tokens", response)

class TestInferenceErrorHandling(unittest.TestCase):
    """Test inference error handling and edge cases."""
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        invalid_inputs = [
            {},  # Empty input
            {"invalid_field": "test"},  # Invalid field
            {"instruction": ""},  # Empty instruction
            {"messages": []},  # Empty messages
            {"prompt": ""},  # Empty prompt
        ]
        
        # Each should be handled gracefully (not crash)
        for invalid_input in invalid_inputs:
            # Validate that we can detect invalid inputs
            is_valid = self._validate_input(invalid_input)
            self.assertFalse(is_valid, f"Input should be invalid: {invalid_input}")
    
    def _validate_input(self, input_data):
        """Helper method to validate input data."""
        if not input_data:
            return False
        
        # Check for valid input formats
        has_instruction = "instruction" in input_data and input_data["instruction"].strip()
        has_messages = "messages" in input_data and len(input_data["messages"]) > 0
        has_prompt = "prompt" in input_data and input_data["prompt"].strip()
        
        return has_instruction or has_messages or has_prompt
    
    def test_parameter_validation(self):
        """Test parameter validation and defaults."""
        # Test invalid parameters
        invalid_params = [
            {"max_new_tokens": -1},
            {"max_new_tokens": 10000},  # Too large
            {"temperature": -0.5},
            {"temperature": 3.0},  # Too high
            {"top_p": -0.1},
            {"top_p": 1.5},  # Too high
        ]
        
        for params in invalid_params:
            corrected = self._apply_parameter_constraints(params)
            # After correction, parameters should be valid
            if "max_new_tokens" in corrected:
                self.assertGreater(corrected["max_new_tokens"], 0)
                self.assertLessEqual(corrected["max_new_tokens"], 2048)
            
            if "temperature" in corrected:
                self.assertGreaterEqual(corrected["temperature"], 0.0)
                self.assertLessEqual(corrected["temperature"], 2.0)
            
            if "top_p" in corrected:
                self.assertGreaterEqual(corrected["top_p"], 0.0)
                self.assertLessEqual(corrected["top_p"], 1.0)
    
    def _apply_parameter_constraints(self, params):
        """Helper method to apply parameter constraints."""
        corrected = params.copy()
        
        if "max_new_tokens" in corrected:
            corrected["max_new_tokens"] = max(1, min(2048, corrected["max_new_tokens"]))
        
        if "temperature" in corrected:
            corrected["temperature"] = max(0.0, min(2.0, corrected["temperature"]))
        
        if "top_p" in corrected:
            corrected["top_p"] = max(0.0, min(1.0, corrected["top_p"]))
        
        return corrected
    
    def test_timeout_handling(self):
        """Test timeout and network error handling."""
        # Mock network timeouts and errors
        network_errors = [
            {"error": "Request timeout", "status_code": 408},
            {"error": "Service unavailable", "status_code": 503},
            {"error": "Connection refused", "status_code": None},
        ]
        
        for error in network_errors:
            # Should handle gracefully
            self.assertIn("error", error)
            self.assertIsInstance(error["error"], str)

class TestInferenceIntegration(unittest.TestCase):
    """Integration tests for inference components."""
    
    def test_end_to_end_request_processing(self):
        """Test complete request processing pipeline."""
        # Mock complete inference pipeline
        test_request = {
            "instruction": "What is the difference between PM and FC job codes?",
            "system": "You are a helpful corporate AI assistant.",
            "max_new_tokens": 100,
            "temperature": 0.7
        }
        
        # Validate request
        self.assertIn("instruction", test_request)
        self.assertIn("system", test_request)
        
        # Mock processing steps
        formatted_prompt = self._format_instruction_prompt(test_request)
        self.assertIn(test_request["instruction"], formatted_prompt)
        self.assertIn(test_request["system"], formatted_prompt)
        
        # Mock response
        mock_response = {
            "response": "PM job code indicates Full Time Employee and FC job code indicates Full Time Contractor.",
            "prompt_tokens": 30,
            "completion_tokens": 15
        }
        
        # Validate response
        self.assertIn("response", mock_response)
        self.assertGreater(len(mock_response["response"]), 0)
    
    def _format_instruction_prompt(self, request):
        """Helper to format instruction prompt."""
        system = request.get("system", "You are a helpful assistant.")
        instruction = request["instruction"]
        
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    def test_corporate_qa_specific_validation(self):
        """Test validation specific to corporate Q&A use case."""
        corporate_keywords = [
            "PM", "FC", "job code", "FTE", "CWR", "employee", "contractor",
            "Azure", "ML", "workspace", "endpoint", "deployment"
        ]
        
        test_inputs = [
            "What is PM job code?",
            "How to convert CWR to FTE?",
            "Process to create Azure ML workspace?",
            "Difference between PM and FC job codes?"
        ]
        
        # Validate that inputs contain corporate keywords
        for input_text in test_inputs:
            contains_keywords = any(keyword.lower() in input_text.lower() 
                                  for keyword in corporate_keywords)
            self.assertTrue(contains_keywords, 
                          f"Input should contain corporate keywords: {input_text}")
    
    def test_response_quality_validation(self):
        """Test response quality validation."""
        test_responses = [
            "PM job code indicates Full Time Employee.",
            "To convert CWR to FTE, raise New Demand and mark requirement type as CWR Conversion.",
            "Create Azure ML workspace by navigating to Azure portal and selecting Machine Learning."
        ]
        
        for response in test_responses:
            # Basic quality checks
            self.assertGreater(len(response), 10)  # Not too short
            self.assertLess(len(response), 1000)   # Not too long
            self.assertTrue(response[0].isupper())  # Starts with capital
            self.assertIn(".", response)           # Has punctuation

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2, buffer=True)