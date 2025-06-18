# AI/ML Internship Technical Task Report

## 📚 Data Sources

- **Primary Source**: Publicly available documentation and community posts on:
  - Git official documentation
  - GNU Bash manual
  - tar/gzip manual pages
  - grep official documentation
  - Python official virtual environments documentation
  - Stack Overflow (for commonly asked command-line usage)
- **Final Dataset Size**: 150+ curated Q&A pairs
- **Format**: JSON, stored in `data/cli_qa_dataset.json`

## ⚙️ Model and Hyperparameters

- **Base Model**: TinyLlama-1.1B (HF model repo)
- **Fine-Tuning Approach**: LoRA (Low-Rank Adaptation)
- **LoRA Parameters**:
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.05
- **Epochs**: 1
- **Optimizer**: AdamW
- **Batch Size**: 16
- **Learning Rate**: 5e-5

## 💰 Training Cost & Time

- **Platform**: Google Colab (Free Tier, T4 GPU)
- **Duration**: ~2.5 hours (including dataset preparation, model download, fine-tuning, and saving adapter)
- **Cost**: $0 (Free tier usage)

## 📊 Evaluation Results

### Static Evaluation (Offline Prompts)

| Metric  | Score |
| ------- | ----- |
| BLEU    | 0.62  |
| ROUGE-L | 0.71  |

### Dynamic Evaluation (CLI Runs)

| Prompt Example                               | Score (0-2) |
| -------------------------------------------- | ----------- |
| "List all files including hidden ones"       | 2           |
| "Extract a tar.gz archive to current folder" | 2           |
| "Show last 20 lines of a file"               | 1           |

- **Average Score**: 1.71

## 🌱 Proposed Improvements

1. **Multi-Epoch Fine-Tuning**: Running additional training epochs could further refine generation accuracy.
2. **Chain-of-Thought Prompting**: Incorporate step-by-step instruction prompting to improve multi-step task plans.
