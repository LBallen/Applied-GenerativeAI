# Fine-Tuning RoBERTa on Semantic Similarity (GLUE MRPC) with PEFT/LoRA

## Overview

This script demonstrates **how to fine-tune a pre-trained large language model (LLM)—RoBERTa-base—on the GLUE MRPC semantic similarity task.** It also showcases **parameter-efficient fine-tuning (PEFT) using LoRA**, making it easy and efficient to adapt large models to new tasks or datasets.

## Features

- Loads the **Microsoft Research Paraphrase Corpus (MRPC)** from GLUE.
- Fine-tunes the **RoBERTa-base** transformer model for paraphrase detection.
- Implements and compares **full model fine-tuning** versus **PEFT/LoRA fine-tuning**.
- **Evaluates performance** using accuracy metrics and confusion matrices.
- Supports **inference** on new sentence pairs for semantic similarity.


## Prerequisites

- Python 3.8+
- Recommended: NVIDIA GPU with 8GB+ VRAM or Apple M1/M2; runs on CPU but will be slower
- Key libraries (auto-installed in Colab):
    - `transformers`
    - `datasets`
    - `evaluate`
    - `peft`
    - `scikit-learn`, `matplotlib`


## How to Run

1. **Clone or copy the script into your Python/Colab environment.**
2. Install required libraries:

```bash
pip install transformers[torch] datasets evaluate peft scikit-learn matplotlib
```

3. **Run the script as-is** or in a Jupyter/Colab notebook.
4. **Modify model, dataset, or task as needed** for your research.

## Main Steps

- **Install dependencies and import libraries**
- **Load RoBERTa-base** and **tokenizer**
- **Load and preprocess the MRPC dataset**
- **Set up metrics and Trainer**
- **Train the standard model** and show results
- **Configure and train with LoRA (PEFT)**
- **Compare outcomes via confusion matrices**
- **Run inference on custom inputs**


## Cloud Usage

- Out-of-the-box support for free Colab GPU.
- Easily portable to AWS, Azure, Google Cloud, or any environment with Python and a GPU/TPU.


## Example Output

- Test predictions for sentence pair similarity:

```
Sentence 1: "The company has announced a new product."
Sentence 2: "A new product was launched by the company."
Output: Equivalent (Paraphrase)
```

- Confusion matrices visualized for model performance.


## License

For academic use and prototyping. Modify for commercial/research projects as needed.

***

**Contact:**
Leonardo Ballen | lnrdballen@gmail.com
