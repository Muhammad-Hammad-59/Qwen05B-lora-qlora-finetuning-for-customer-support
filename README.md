
# Customer Support AI Assistant Fine-Tuning Project

This repository contains a Jupyter Notebook for fine-tuning the **Qwen/Qwen2.5-0.5B-Instruct** model using **LoRA** (Low-Rank Adaptation) to create an intelligent customer support assistant.

The fine-tuned model can:
- Classify support tickets into categories
- Determine urgency levels
- Generate professional, helpful responses

The project is lightweight and designed to run comfortably on free-tier resources like **Google Colab** (T4 GPU).

## Key Features

- **Model**: Qwen/Qwen2.5-0.5B-Instruct (small, fast, memory-efficient)
- **Fine-Tuning Method**: LoRA — only ~0.5% of parameters are trainable
- **Capabilities**:
  - Ticket classification (11 categories: ACCOUNT, ORDER, REFUND, etc.)
  - Urgency detection
  - Professional response generation
- **Dataset Sampling**: 5,000 diverse examples randomly sampled from full dataset
- **Analysis**: Detailed data exploration, category/intent distributions, text length stats, visualizations
- **Evaluation**: ROUGE score-based response quality assessment

## Dataset

- **Source**: [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (Hugging Face)
- **Original Size**: 26,872 real customer support conversations
- **Categories**: 11 common support scenarios (ACCOUNT, ORDER, REFUND, FEEDBACK, CONTACT, PAYMENT, DELIVERY, SHIPPING, INVOICE, CANCEL, SUBSCRIPTION)
- **Intents**: 27 unique customer intents
- **Sampling**: 5,000 examples (~18.6% of full dataset) — random sampling with seed 42
- **Diversity**: Verified with category distribution stats and visualizations

## Requirements

- Python 3.10+
- GPU strongly recommended (Colab T4 GPU works well)
- Main libraries:
  - `transformers`
  - `datasets`
  - `peft`
  - `accelerate`
  - `bitsandbytes`
  - `trl`
  - `gradio`
  - `evaluate`
  - `rouge_score`
  - PyTorch + CUDA support

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/customer-support-ai-finetuning.git
   cd customer-support-ai-finetuning
   ```

2. Install dependencies:

   ```bash
   pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   pip install -U trl peft accelerate bitsandbytes datasets transformers gradio pyarrow
   pip install -U bitsandbytes>=0.46.1
   pip install evaluate rouge_score
   ```

## Usage

1. Open the notebook:  
   `CustomerCare_AI_LoRA_FineTuning_Complete .ipynb`  
   (preferably in **Google Colab**)

2. Run cells in order:
   - Install dependencies
   - Load & sample dataset
   - Explore & visualize data (saves `data_analysis.png`)
   - Fine-tune model using LoRA
   - Evaluate responses (ROUGE scores)

3. Inference:
   - Use the fine-tuned model for new customer queries
   - Optional: Gradio interface included for interactive demos

4. Customization options:
   - Change `SAMPLE_SIZE` (default: 5000)
   - Modify LoRA rank, alpha, dropout, target modules, etc.

## Training Details

- **Base Model**: Qwen/Qwen2.5-0.5B-Instruct (4-bit quantized via bitsandbytes)
- **Method**: Supervised fine-tuning + LoRA
- **Trainable Parameters**: ~0.5% of total
- **Tested Environment**: Google Colab + Tesla T4 (15 GB VRAM)
- **Approximate runtime**:
  - Data loading & analysis: 5–10 minutes
  - Fine-tuning: depends on epochs & batch size (usually 30–90 min)

## Evaluation

- **Main Metric**: ROUGE scores (via `evaluate` + `rouge_score`)
- **Sample size**: 50 examples (configurable)
- **Output**: Comparison of generated vs reference responses

## Results

- Significant improvement in:
  - Category classification accuracy
  - Urgency understanding
  - Response tone, structure & helpfulness
- Visualizations saved: `data_analysis.png` (category distribution, word counts, etc.)
- Example conversations shown directly in the notebook

## Contributing

- Fork the repo
- Create a feature branch
- Submit a pull request

Feel free to open issues for bugs, questions, or feature suggestions.

## License

MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) — for transformers, datasets 
- [Bitext](https://huggingface.co/datasets/bitext) — excellent customer support dataset
- [Qwen team](https://huggingface.co/Qwen) — great open-weight instruct model

Happy fine-tuning! 
 
 
