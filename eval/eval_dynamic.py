import json
import os
import sys
from typing import List, Dict
import subprocess
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import CLIAgent

class DynamicEvaluator:
    def __init__(self, model_path: str, base_model: str):
        self.agent = CLIAgent(model_path, base_model)
        
    def score_plan_quality(self, instruction: str, steps: List[Dict]) -> Dict:
        """Score plan quality on 0-2 scale"""
        scores = {
            'relevance': 0,    
            'completeness': 0, 
            'correctness': 0, 
            'clarity': 0     
        }
        
        relevant_steps = 0
        for step in steps:
            if self.is_step_relevant(instruction, step['text']):
                relevant_steps += 1
        
        if len(steps) > 0:
            relevance_ratio = relevant_steps / len(steps)
            scores['relevance'] = 2 if relevance_ratio >= 0.8 else (1 if relevance_ratio >= 0.5 else 0)
        
        required_keywords = self.get_required_keywords(instruction)
        mentioned_keywords = 0
        
        plan_text = ' '.join(step['text'].lower() for step in steps)
        for keyword in required_keywords:
            if keyword.lower() in plan_text:
                mentioned_keywords += 1
        
        if len(required_keywords) > 0:
            completeness_ratio = mentioned_keywords / len(required_keywords)
            scores['completeness'] = 2 if completeness_ratio >= 0.8 else (1 if completeness_ratio >= 0.5 else 0)
        else:
            scores['completeness'] = 1  
        
        command_steps = [step for step in steps if step['is_command'] and step['command']]
        correct_commands = 0
        
        for step in command_steps:
            if self.is_command_reasonable(step['command']):
                correct_commands += 1
        
        if len(command_steps) > 0:
            correctness_ratio = correct_commands / len(command_steps)
            scores['correctness'] = 2 if correctness_ratio >= 0.8 else (1 if correctness_ratio >= 0.5 else 0)
        else:
            scores['correctness'] = 1  
        
        has_numbered_steps = any(step['text'].strip().startswith(('1.', '2.', '3.', '4.', '5.')) for step in steps)
        has_reasonable_length = 2 <= len(steps) <= 8
        has_commands = any(step['is_command'] for step in steps)
        
        clarity_points = 0
        if has_numbered_steps: clarity_points += 1
        if has_reasonable_length: clarity_points += 1
        if has_commands: clarity_points += 1
        
        scores['clarity'] = min(2, clarity_points)
        
        total_score = sum(scores.values())
        max_score = 8
        normalized_score = (total_score / max_score) * 2  
        
        return {
            'individual_scores': scores,
            'total_score': total_score,
            'max_score': max_score,
            'normalized_score': round(normalized_score, 2)
        }
    
    def is_step_relevant(self, instruction: str, step_text: str) -> bool:
        """Check if a step is relevant to the instruction"""
        instruction_words = set(instruction.lower().split())
        step_words = set(step_text.lower().split())
        
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'use', 'using'}
        instruction_words -= common_words
        step_words -= common_words
        
        overlap = len(instruction_words & step_words)
        return overlap >= 1 or any(word in step_text.lower() for word in instruction_words)
    
    def get_required_keywords(self, instruction: str) -> List[str]:
        """Get keywords that should appear in a good plan"""
        keyword_map = {
            'git branch': ['git', 'branch', 'checkout'],
            'find files': ['find', 'name'],
            'virtual environment': ['python', 'venv', 'activate'],
            'compress': ['tar', 'gzip'],
            'search text': ['grep'],
            'install packages': ['pip', 'install'],
            'copy files': ['cp'],
            'permissions': ['chmod'],
            'background': ['&', 'nohup'],
            'environment variables': ['export'],
        }
        
        instruction_lower = instruction.lower()
        for key, keywords in keyword_map.items():
            if key in instruction_lower:
                return keywords
        
        return []
    
    def is_command_reasonable(self, command: str) -> bool:
        """Check if a command looks syntactically reasonable"""
        command = command.strip()
        
        if not command:
            return False
        
        if command.count('`') % 2 != 0:  
            return False

        parts = command.split()
        if not parts:
            return False
        
        first_word = parts[0]
        
        good_commands = {
            'git', 'cd', 'ls', 'cp', 'mv', 'rm', 'mkdir', 'chmod', 'grep',
            'find', 'tar', 'gzip', 'curl', 'wget', 'ssh', 'scp', 'docker',
            'pip', 'python', 'python3', 'node', 'npm', 'make', 'sudo',
            'systemctl', 'export', 'echo', 'cat', 'head', 'tail', 'sort',
            'uniq', 'wc', 'awk', 'sed', 'vi', 'vim', 'nano'
        }
        
        return first_word in good_commands
    
    def evaluate_instruction(self, instruction: str) -> Dict:
        """Evaluate a single instruction"""
        print(f"\nEvaluating: {instruction}")
        print("-" * 60)
        
        result = self.agent.process_instruction(instruction)
        
        quality_scores = self.score_plan_quality(instruction, result['steps'])
        
        return {
            'instruction': instruction,
            'response': result['response'],
            'steps': result['steps'],
            'execution_results': result['execution_results'],
            'quality_scores': quality_scores
        }

def load_test_instructions() -> List[str]:
    """Load test instructions for dynamic evaluation"""
    required_prompts = [
        "Create a new Git branch and switch to it.",
        "Compress the folder reports into reports.tar.gz.",
        "List all Python files in the current directory recursively.",
        "Set up a virtual environment and install requests.", 
        "Fetch only the first ten lines of a file named output.log."
    ]
    
    edge_cases = [
        "Handle a corrupted file that won't delete normally",
        "Debug why my Python script runs differently in various environments"
    ]
    
    return required_prompts + edge_cases

def main():
    model_path = "training/model_adapters"
    base_model = "microsoft/DialoGPT-small"
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return
    
    test_instructions = load_test_instructions()
    
    evaluator = DynamicEvaluator(model_path, base_model)
    
    results = []
    for instruction in test_instructions:
        result = evaluator.evaluate_instruction(instruction)
        results.append(result)
    
    os.makedirs("evaluation", exist_ok=True)
    with open("evaluation/dynamic_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    total_score = sum(r['quality_scores']['normalized_score'] for r in results)
    avg_score = total_score / len(results)
    
    print(f"\n{'='*60}")
    print(f"DYNAMIC EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total instructions evaluated: {len(results)}")
    print(f"Average quality score: {avg_score:.2f}/2.00")
    
    print(f"\nIndividual Scores:")
    for result in results:
        score = result['quality_scores']['normalized_score']
        instruction = result['instruction'][:50] + "..." if len(result['instruction']) > 50 else result['instruction']
        print(f"  {score:.2f}/2.00 - {instruction}")
    
    relevance_avg = sum(r['quality_scores']['individual_scores']['relevance'] for r in results) / len(results)
    completeness_avg = sum(r['quality_scores']['individual_scores']['completeness'] for r in results) / len(results)
    correctness_avg = sum(r['quality_scores']['individual_scores']['correctness'] for r in results) / len(results)
    clarity_avg = sum(r['quality_scores']['individual_scores']['clarity'] for r in results) / len(results)
    
    print(f"\nScore Breakdown (0-2 scale):")
    print(f"  Relevance:    {relevance_avg:.2f}")
    print(f"  Completeness: {completeness_avg:.2f}")
    print(f"  Correctness:  {correctness_avg:.2f}")
    print(f"  Clarity:      {clarity_avg:.2f}")
    
    print(f"\nDetailed results saved to evaluation/dynamic_eval_results.json")

if __name__ == "__main__":
    main()