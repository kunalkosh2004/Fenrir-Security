import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import numpy as np
from typing import Dict, List
import os

class CLIDatasetProcessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def format_prompt(self, input_text: str, output: str) -> str:
        """Format training prompt in instruction format"""
        if input_text.strip():
            prompt = f"### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{input_text}\n\n### Response:\n{output}"
        return prompt
    
    def tokenize_function(self, examples):
        """Tokenize the dataset"""
        prompts = [
            self.format_prompt(ques, ans) 
            for ques, ans in zip(
                examples["question"], 
                examples["answer"]
            )
        ]
        
        # Tokenize with proper padding and truncation
        tokenized = self.tokenizer(
            prompts,
            truncation=True,
            padding=True,  # Enable padding
            max_length=self.max_length,
            return_tensors=None,  # Keep as lists for now
        )
        
        # Set labels for language modeling (copy input_ids, not reference)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        
        return tokenized

def find_target_modules(model):
    """
    Automatically find linear layers in the model that can be targeted by LoRA.
    This function inspects the model architecture and returns appropriate target modules.
    """
    target_modules = set()
    full_module_names = []
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Store both full name and just the module name
            full_module_names.append(name)
            module_name = name.split('.')[-1]
            target_modules.add(module_name)
    
    # Convert to list and filter out common non-attention modules
    target_modules = list(target_modules)
    
    # Print available modules for debugging
    print(f"Available linear modules: {target_modules}")
    print(f"Full module paths: {full_module_names[:10]}...")  # Show first 10
    
    # Exclude output layers that we don't want to adapt
    exclude_modules = ['lm_head', 'head', 'classifier', 'score']
    filtered_modules = [m for m in target_modules if m not in exclude_modules]
    
    # For DialoGPT, look for attention-related modules
    attention_patterns = ['c_attn', 'attn', 'self_attn', 'attention', 'c_proj', 'dense']
    attention_modules = [m for m in filtered_modules if any(pattern in m.lower() for pattern in attention_patterns)]
    
    if attention_modules:
        print(f"Found attention modules: {attention_modules}")
        return attention_modules
    
    # If no attention modules, look for MLP/feed-forward modules
    mlp_patterns = ['c_fc', 'fc', 'mlp', 'dense', 'linear']
    mlp_modules = [m for m in filtered_modules if any(pattern in m.lower() for pattern in mlp_patterns)]
    
    if mlp_modules:
        print(f"Found MLP modules: {mlp_modules}")
        return mlp_modules[:2]  # Limit to 2 modules
    
    # Last resort: return any filtered modules
    if filtered_modules:
        print(f"Using filtered modules: {filtered_modules[:2]}")
        return filtered_modules[:2]
    
    # If all else fails, use a safe default
    print("Warning: No suitable modules found, using safe defaults")
    return ["c_attn"] if "c_attn" in target_modules else target_modules[:1]

def load_dataset(file_path: str) -> Dataset:
    """Load Q&A dataset from JSON file"""
    if not os.path.exists(file_path):
        # Create a sample dataset if it doesn't exist
        print(f"Dataset file {file_path} not found. Creating sample dataset...")
        sample_data = [
            {"question": "What is the ls command used for?", "answer": "The ls command is used to list directory contents in Unix-like operating systems."},
            {"question": "How do you create a new directory?", "answer": "You can create a new directory using the mkdir command followed by the directory name."},
            {"question": "What does cd command do?", "answer": "The cd command changes the current directory to the specified directory path."},
            {"question": "How to copy files in terminal?", "answer": "Use the cp command followed by source file and destination to copy files."},
            {"question": "What is grep command?", "answer": "grep is a command-line utility for searching text using patterns and regular expressions."}
        ]
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        print(f"Created sample dataset with {len(sample_data)} examples")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Validate dataset format
    if not isinstance(data, list):
        raise ValueError("Dataset should be a list of dictionaries")
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary")
        if "question" not in item or "answer" not in item:
            raise ValueError(f"Item {i} missing 'question' or 'answer' key")
    
    print(f"Loaded dataset with {len(data)} examples")
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(data)
    return dataset

def setup_model_and_tokenizer(model_name: str):
    """Setup model and tokenizer for fine-tuning"""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - CPU optimized settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for CPU
        device_map=None,  # Let PyTorch handle device placement
        trust_remote_code=True,
        low_cpu_mem_usage=True  # Optimize for CPU memory usage
    )
    
    return model, tokenizer

def setup_lora_config(model, model_name=""):
    """Setup LoRA configuration with automatically detected target modules"""
    # For DialoGPT, manually specify known good target modules
    if "DialoGPT" in model_name or "dialogpt" in model_name.lower():
        # DialoGPT uses c_attn for attention and c_proj for projection
        target_modules = ["c_attn", "c_proj"]
        print(f"Using DialoGPT-specific target modules: {target_modules}")
    else:
        # Automatically find target modules for other models
        target_modules = find_target_modules(model)
    
    if not target_modules:
        raise ValueError("No suitable target modules found for LoRA adaptation")
    
    print(f"Final target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling parameter
        target_modules=target_modules,  # automatically detected or manually specified modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return lora_config

def fine_tune_model():
    """Main fine-tuning function"""
    # Check device availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cpu":
        print("WARNING: Training on CPU will be significantly slower than GPU training.")
        print("Consider using Google Colab, Kaggle, or a cloud service with GPU for faster training.")
    
    # Configuration
    model_name = "microsoft/DialoGPT-small"  # ~117M parameters, good for CLI tasks
    dataset_path = "./data/cli_qa_dataset.json"
    output_dir = "training/model_adapters"
    
    print(f"Loading model: {model_name}")
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Setup data processor
    processor = CLIDatasetProcessor(tokenizer)
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        processor.tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Debug: Check the first example
    print("Sample tokenized example:")
    print(f"Input IDs length: {len(tokenized_dataset[0]['input_ids'])}")
    print(f"Labels length: {len(tokenized_dataset[0]['labels'])}")
    print(f"Input IDs type: {type(tokenized_dataset[0]['input_ids'])}")
    print(f"Labels type: {type(tokenized_dataset[0]['labels'])}")
    
    # Split dataset
    train_size = int(0.9 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")
    
    # Setup LoRA with automatic target module detection
    lora_config = setup_lora_config(model, model_name)
    model = get_peft_model(model, lora_config)
    
    print("LoRA model setup complete")
    model.print_trainable_parameters()
    
    # Training arguments - CPU optimized
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,  # Reduced for CPU
        per_device_eval_batch_size=2,   # Reduced for CPU
        gradient_accumulation_steps=4,  # Increased to maintain effective batch size
        warmup_steps=50,  # Reduced for smaller dataset
        learning_rate=5e-5,  # Slightly lower learning rate for stability
        fp16=False,  # Disabled for CPU training
        logging_steps=5,  # More frequent logging for CPU
        eval_strategy="steps",
        eval_steps=25,  # More frequent evaluation
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to=None,  # Disable wandb
        dataloader_num_workers=0,  # Important for CPU training
        remove_unused_columns=False,  # Can help with compatibility
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
        pad_to_multiple_of=None,  # Don't pad to multiples
        return_tensors="pt",  # Return PyTorch tensors
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    
    # Save training metrics
    train_results = trainer.state.log_history
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump(train_results, f, indent=2)
    
    return model, tokenizer

def inspect_model_architecture(model_name: str):
    """
    Helper function to inspect model architecture and find available modules.
    Useful for debugging target module issues.
    """
    print(f"Inspecting model architecture for: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("\nModel architecture:")
    print(model)
    
    print("\nAvailable linear modules:")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"  {name}: {module}")
    
    print("\nSuggested target modules:")
    target_modules = find_target_modules(model)
    print(target_modules)

if __name__ == "__main__":
    inspect_model_architecture("microsoft/DialoGPT-small")
    
    # Create output directory
    os.makedirs("training/model_adapters", exist_ok=True)
    
    # Run fine-tuning
    model, tokenizer = fine_tune_model()
    
    print("Fine-tuning complete!")