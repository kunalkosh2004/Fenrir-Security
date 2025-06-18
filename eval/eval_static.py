import json
import os
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ModelEvaluator:
    def __init__(self, base_model_name: str, finetuned_model_path: str):
        self.base_model_name = base_model_name
        self.finetuned_model_path = finetuned_model_path
        # Determine the best available device
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.load_models()
        
    def _get_device(self):
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
        
    def load_models(self):
        """Load both base and fine-tuned models"""
        print("Loading models...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with proper device handling
        if self.device == "cpu":
            # For CPU, use float32 to avoid potential issues
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
        else:
            # For GPU/MPS, use float16 for efficiency
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        # Load fine-tuned model
        self.finetuned_model = PeftModel.from_pretrained(
            self.base_model, 
            self.finetuned_model_path
        )
        
        print("Models loaded successfully!")
    
    def format_prompt(self, instruction: str) -> str:
        """Format instruction for the model"""
        return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(self, model, instruction: str) -> str:
        """Generate response using specified model"""
        prompt = self.format_prompt(instruction)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            except RuntimeError as e:
                if "MPS" in str(e) and "storage" in str(e):
                    # Fallback to CPU if MPS has storage allocation issues
                    print("MPS storage issue detected, falling back to CPU...")
                    model = model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    self.device = "cpu"
                    
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                else:
                    raise e
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score with smoothing"""
        from nltk.translate.bleu_score import SmoothingFunction
        
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        # Use smoothing function to handle cases with low n-gram overlap
        smoothie = SmoothingFunction().method4
        
        return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score"""
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return scores['rougeL'].fmeasure
    
    def calculate_word_overlap(self, reference: str, candidate: str) -> float:
        """Calculate simple word overlap ratio"""
        ref_words = set(nltk.word_tokenize(reference.lower()))
        cand_words = set(nltk.word_tokenize(candidate.lower()))
        
        if len(ref_words) == 0:
            return 0.0
            
        overlap = len(ref_words.intersection(cand_words))
        return overlap / len(ref_words)
    
    def evaluate_prompts(self, test_prompts: List[Dict]) -> List[Dict]:
        """Evaluate all test prompts"""
        results = []
        
        for i, prompt_data in enumerate(test_prompts):
            instruction = prompt_data['instruction']
            reference = prompt_data.get('reference', '')
            
            print(f"Evaluating {i+1}/{len(test_prompts)}: {instruction[:50]}...")
            
            try:
                # Generate responses
                base_response = self.generate_response(self.base_model, instruction)
                finetuned_response = self.generate_response(self.finetuned_model, instruction)
                
                # Calculate metrics if reference is available
                metrics = {}
                if reference:
                    metrics = {
                        'base_bleu': self.calculate_bleu(reference, base_response),
                        'finetuned_bleu': self.calculate_bleu(reference, finetuned_response),
                        'base_rouge_l': self.calculate_rouge_l(reference, base_response),
                        'finetuned_rouge_l': self.calculate_rouge_l(reference, finetuned_response),
                        'base_word_overlap': self.calculate_word_overlap(reference, base_response),
                        'finetuned_word_overlap': self.calculate_word_overlap(reference, finetuned_response)
                    }
                
                result = {
                    'instruction': instruction,
                    'reference': reference,
                    'base_response': base_response,
                    'finetuned_response': finetuned_response,
                    'metrics': metrics
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating prompt {i+1}: {str(e)}")
                # Add failed result to maintain order
                result = {
                    'instruction': instruction,
                    'reference': reference,
                    'base_response': f"Error: {str(e)}",
                    'finetuned_response': f"Error: {str(e)}",
                    'metrics': {}
                }
                results.append(result)
        
        return results

def load_test_prompts() -> List[Dict]:
    """Load test prompts from JSON file"""
    test_prompts = [
        {
            "instruction": "Create a new Git branch and switch to it",
            "reference": "To create a new Git branch and switch to it:\n1. Use `git checkout -b <branch-name>` to create and switch in one command\n2. Or use `git branch <branch-name>` then `git checkout <branch-name>`\n3. Example: `git checkout -b feature-login`\n4. Verify with `git branch` to see current branch marked with *"
        },
        {
            "instruction": "Compress the folder reports into reports.tar.gz",
            "reference": "To compress the folder reports into reports.tar.gz:\n1. Use `tar -czf reports.tar.gz reports/`\n2. The -c flag creates archive, -z compresses with gzip, -f specifies filename\n3. Verify compression: `ls -lh reports.tar.gz`\n4. Alternative: `tar -czvf reports.tar.gz reports/` (with verbose output)"
        },
        {
            "instruction": "List all Python files in the current directory recursively",
            "reference": "To list all Python files recursively:\n1. Use `find . -name '*.py'` to find all .py files\n2. Use `find . -type f -name '*.py'` to ensure only files, not directories\n3. Alternative: `ls -la **/*.py` (if shell supports globstar)\n4. With details: `find . -name '*.py' -exec ls -l {} \\;`"
        },
        {
            "instruction": "Set up a virtual environment and install requests",
            "reference": "To set up virtual environment and install requests:\n1. Create venv: `python -m venv myenv` or `python3 -m venv myenv`\n2. Activate: `source myenv/bin/activate` (Linux/Mac) or `myenv\\Scripts\\activate` (Windows)\n3. Install requests: `pip install requests`\n4. Verify installation: `pip show requests`\n5. Deactivate when done: `deactivate`"
        },
        {
            "instruction": "Fetch only the first ten lines of a file named output.log",
            "reference": "To fetch the first ten lines of output.log:\n1. Use `head -10 output.log` or `head -n 10 output.log`\n2. Default head shows first 10 lines: `head output.log`\n3. Alternative with sed: `sed -n '1,10p' output.log`\n4. Check file exists first: `ls -la output.log`"
        },
        {
            "instruction": "Find and kill all processes containing 'python' in their name, but exclude the current shell session",
            "reference": "To find and kill Python processes safely:\n1. List processes: `ps aux | grep python | grep -v grep`\n2. Get PIDs: `pgrep -f python`\n3. Exclude current shell: `ps aux | grep python | grep -v grep | grep -v $$`\n4. Kill specific processes: `pkill -f 'python script.py'`\n5. Force kill if needed: `pkill -9 -f python`\n6. Always verify before killing: check process details with `ps -p <PID>`"
        },
        {
            "instruction": "Recover uncommitted changes after accidentally running 'git reset --hard' in a repository with unstaged modifications",
            "reference": "To recover from accidental git reset --hard:\n1. Check reflog immediately: `git reflog`\n2. Find the commit before reset: look for HEAD@{1} or similar\n3. Reset to previous state: `git reset --hard HEAD@{1}`\n4. If files were untracked, they may be lost permanently\n5. Check git fsck for dangling commits: `git fsck --lost-found`\n6. For future: use `git stash` before risky operations\n7. Note: Unstaged changes are usually unrecoverable after hard reset"
        }
    ]
    
    return test_prompts

def main():
    # Configuration
    base_model_name = "microsoft/DialoGPT-small"
    finetuned_model_path = "training/model_adapters"
    
    if not os.path.exists(finetuned_model_path):
        print(f"Error: Fine-tuned model not found at {finetuned_model_path}")
        return
    
    # Load test prompts
    test_prompts = load_test_prompts()
    
    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(base_model_name, finetuned_model_path)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return
    
    # Run evaluation
    print(f"Starting evaluation of {len(test_prompts)} prompts...")
    results = evaluator.evaluate_prompts(test_prompts)
    
    # Save results
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/static_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate summary statistics
    total_prompts = len(results)
    successful_results = [r for r in results if r['metrics'] and 'Error:' not in r['base_response']]
    prompts_with_metrics = len(successful_results)
    
    print(f"\nEvaluation Summary:")
    print(f"Total prompts evaluated: {total_prompts}")
    print(f"Successful evaluations: {prompts_with_metrics}")
    
    if prompts_with_metrics > 0:
        avg_base_bleu = sum(r['metrics']['base_bleu'] for r in successful_results) / prompts_with_metrics
        avg_finetuned_bleu = sum(r['metrics']['finetuned_bleu'] for r in successful_results) / prompts_with_metrics
        avg_base_rouge = sum(r['metrics']['base_rouge_l'] for r in successful_results) / prompts_with_metrics
        avg_finetuned_rouge = sum(r['metrics']['finetuned_rouge_l'] for r in successful_results) / prompts_with_metrics
        
        print(f"Average BLEU - Base: {avg_base_bleu:.3f}, Fine-tuned: {avg_finetuned_bleu:.3f}")
        print(f"Average ROUGE-L - Base: {avg_base_rouge:.3f}, Fine-tuned: {avg_finetuned_rouge:.3f}")
        
        if avg_base_bleu > 0:
            print(f"BLEU improvement: {((avg_finetuned_bleu - avg_base_bleu) / avg_base_bleu * 100):+.1f}%")
        if avg_base_rouge > 0:
            print(f"ROUGE-L improvement: {((avg_finetuned_rouge - avg_base_rouge) / avg_base_rouge * 100):+.1f}%")
    else:
        print("No successful evaluations completed.")
    
    print(f"\nDetailed results saved to evaluation/static_eval_results.json")

if __name__ == "__main__":
    main()