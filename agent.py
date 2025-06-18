import sys
import json
import os
import re
import subprocess
from datetime import datetime
from typing import Dict, List, Optional
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class CLIAgent:
    def __init__(self, model_path: str, base_model: str = "microsoft/DialoGPT-small"):
        """Initialize the CLI Agent with fine-tuned model"""
        self.model_path = model_path
        self.base_model = base_model
        self.model = None
        self.tokenizer = None
        self.device = self._get_optimal_device()
        self.load_model()
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
    
    def _get_optimal_device(self):
        """Determine the best device to use, avoiding MPS issues"""
        if torch.cuda.is_available():
            print("Using CUDA device")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # Check if MPS is problematic
            try:
                # Test MPS with a simple operation
                test_tensor = torch.randn(1, device="mps")
                del test_tensor
                print("Using MPS device")
                return "mps"
            except Exception as e:
                print(f"MPS device has issues ({e}), falling back to CPU")
                return "cpu"
        else:
            print("Using CPU device")
            return "cpu"
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print("Loading fine-tuned model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with explicit device handling
        if self.device == "cpu":
            # For CPU, use float32 and no device_map
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        else:
            # For GPU/MPS, use device_map="auto" but handle MPS specially
            device_map = "auto" if self.device == "cuda" else None
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=device_map,
                trust_remote_code=True
            )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Move model to device if not using device_map
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully on {self.device}!")
        
    def format_prompt(self, instruction: str) -> str:
        """Format instruction for the model"""
        return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(self, instruction: str, max_length: int = 512) -> str:
        """Generate response using the fine-tuned model"""
        prompt = self.format_prompt(instruction)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Clear any cached tensors to avoid MPS issues
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
        
        # Generate response with error handling
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
        except RuntimeError as e:
            if "MPS" in str(e) or "Placeholder storage" in str(e):
                print("MPS error encountered, retrying with CPU...")
                # Move everything to CPU and retry
                self.model = self.model.to("cpu")
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
                self.device = "cpu"
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True
                    )
            else:
                raise e
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (after "### Response:")
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def parse_plan(self, response: str) -> List[Dict]:
        """Parse the generated response into steps"""
        lines = response.split('\n')
        steps = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number (step)
            step_match = re.match(r'^(\d+)\.?\s*(.+)', line)
            if step_match:
                step_num = int(step_match.group(1))
                step_text = step_match.group(2)
            else:
                step_num = i + 1
                step_text = line
            
            # Check if this looks like a shell command
            is_command = self.is_shell_command(step_text)
            
            steps.append({
                "step": step_num,
                "text": step_text,
                "is_command": is_command,
                "command": self.extract_command(step_text) if is_command else None
            })
        
        return steps
    
    def is_shell_command(self, text: str) -> bool:
        """Check if text looks like a shell command"""
        # Remove common prefixes
        text = text.strip()
        prefixes_to_remove = ["Use ", "Run ", "Execute ", "Type "]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Check for backticks or common command patterns
        if '`' in text:
            return True
        
        # Check for common CLI commands
        common_commands = [
            'git', 'cd', 'ls', 'cp', 'mv', 'rm', 'mkdir', 'chmod', 'grep',
            'find', 'tar', 'gzip', 'curl', 'wget', 'ssh', 'scp', 'docker',
            'pip', 'python', 'node', 'npm', 'make', 'sudo', 'systemctl'
        ]
        
        first_word = text.split()[0] if text.split() else ""
        return first_word in common_commands
    
    def extract_command(self, text: str) -> str:
        """Extract command from text"""
        # Remove backticks
        text = text.replace('`', '')
        
        # Remove common prefixes
        prefixes_to_remove = ["Use ", "Run ", "Execute ", "Type "]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        return text.strip()
    
    def execute_dry_run(self, command: str) -> Dict:
        """Execute command in dry-run mode (just echo)"""
        print(f"[DRY RUN] Would execute: {command}")
        
        return {
            "command": command,
            "dry_run": True,
            "output": f"echo '{command}'",
            "status": "success"
        }
    
    def log_trace(self, instruction: str, response: str, steps: List[Dict], 
                  execution_results: List[Dict]):
        """Log the complete trace to JSONL file"""
        trace_entry = {
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction,
            "response": response,
            "steps": steps,
            "execution_results": execution_results
        }
        
        with open("logs/trace.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(trace_entry) + "\n")
    
    def process_instruction(self, instruction: str) -> Dict:
        """Process a natural language instruction"""
        print(f"Processing: {instruction}")
        print("=" * 50)
        
        # Generate response
        response = self.generate_response(instruction)
        print("Generated Plan:")
        print(response)
        print("=" * 50)
        
        # Parse into steps
        steps = self.parse_plan(response)
        
        # Execute commands in dry-run mode
        execution_results = []
        for step in steps:
            print(f"Step {step['step']}: {step['text']}")
            
            if step["is_command"] and step["command"]:
                result = self.execute_dry_run(step["command"])
                execution_results.append(result)
            else:
                print(f"  [INFO] {step['text']}")
                execution_results.append({
                    "step": step['step'],
                    "type": "info",
                    "text": step['text']
                })
        
        print("=" * 50)
        
        # Log trace
        self.log_trace(instruction, response, steps, execution_results)
        
        return {
            "instruction": instruction,
            "response": response,
            "steps": steps,
            "execution_results": execution_results
        }

def main():
    parser = argparse.ArgumentParser(description="CLI Agent with Fine-tuned Model")
    parser.add_argument("instruction", help="Natural language instruction")
    parser.add_argument("--model-path", default="training/model_adapters", 
                       help="Path to fine-tuned model")
    parser.add_argument("--base-model", default="microsoft/DialoGPT-small",
                       help="Base model name")
    parser.add_argument("--force-cpu", action="store_true",
                       help="Force CPU usage even if GPU/MPS is available")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        print("Please run the fine-tuning script first.")
        sys.exit(1)
    
    # Initialize agent
    agent = CLIAgent(args.model_path, args.base_model)
    
    # Force CPU if requested
    if args.force_cpu:
        print("Forcing CPU usage...")
        agent.device = "cpu"
        agent.model = agent.model.to("cpu")
    
    # Process instruction
    result = agent.process_instruction(args.instruction)
    
    print("Processing complete! Check logs/trace.jsonl for detailed trace.")
    return result

if __name__ == "__main__":
    main()