# C.R.A.F.T. Teaching Report — MRPC LoRA Fine-Tuning Notebook

## Code Summary

This notebook installs dependencies, loads the GLUE MRPC dataset, tokenizes sentence pairs with a RoBERTa tokenizer, and fine-tunes a RoBERTa sequence classification model using **LoRA (parameter-efficient fine-tuning)**. It evaluates performance with GLUE metrics, visualizes a confusion matrix, and provides a helper for custom inference on sentence pairs.

## Annotated Code (Cell-by-Cell Overview)

### Cell 0: Install dependencies (pip) for PyTorch, Transformers, and related libraries.

```python
from pkg_resources import non_empty_lines
%pip install -q peft datasets evaluate  # Installs a dependency at runtime inside the notebook
%pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu130  # Installs a dependency at runtime inside the notebook
%pip install -q transformers[torch]  # Installs a dependency at runtime inside the notebook
```

### Cell 1: Import libraries and load the GLUE MRPC dataset for sentence-pair classification.

```python
import torch
from datasets import load_dataset  # Loads the GLUE MRPC dataset via Hugging Face Datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding  # Initializes tokenizer from a pretrained RoBERTa checkpoint
import evaluate
from peft import get_peft_model, LoraConfig, TaskType  # Configures LoRA ranks, alpha, and dropout for PEFT
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score  # Visualizes confusion matrix for classification results
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType  # Configures LoRA ranks, alpha, and dropout for PEFT
```

### Cell 2: Set environment variables to avoid symlink warnings; define tokenizer and tokenization function.

```python
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
dataset = load_dataset("glue", "mrpc")  # Loads the GLUE MRPC dataset via Hugging Face Datasets
model_name_or_path = "roberta-base"
task = "mrpc"
print(dataset)
```

### Cell 3: Load GLUE evaluation metric for the MRPC task.

```python
metric = evaluate.load("glue", task)  # Loads GLUE/MRPC metric module
```

### Cell 4: Peek into a validation example from the MRPC dataset.

```python
dataset["validation"][1]
```

### Cell 5: General setup and experimentation for MRPC fine-tuning pipeline.

```python
metric
```

### Cell 6: Demonstrate metric computation with example predictions and references.

```python
references = [0, 1]
predictions = [1, 1]
results = metric.compute(predictions=predictions, references=references)
print(results)
```

### Cell 7: Set environment variables to avoid symlink warnings; define tokenizer and tokenization function.

```python
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from transformers import AutoTokenizer  # Initializes tokenizer from a pretrained RoBERTa checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side = "right")  # Initializes tokenizer from a pretrained RoBERTa checkpoint

if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):  # Defines how to tokenize sentence pairs (sentence1, sentence2)
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        truncation=True,
        max_length = None
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["sentence1", "sentence2"])  # Applies tokenization across the dataset with batched processing
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
```

### Cell 8: General setup and experimentation for MRPC fine-tuning pipeline.

```python
tokenized_datasets
```

### Cell 9: General setup and experimentation for MRPC fine-tuning pipeline.

```python
tokenized_datasets["train"][1]
```

### Cell 10: General setup and experimentation for MRPC fine-tuning pipeline.

```python
example_input_ids = tokenized_datasets['train'][0]['input_ids']
tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(example_input_ids))

```

### Cell 11: General setup and experimentation for MRPC fine-tuning pipeline.

```python
#from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
```

### Cell 12: Define training hyperparameters and logging/saving strategies.

```python
#from transformers import AutoModelForSequenceClassification, TrainingArguments, trainer
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(sum(p.numel() for p in model.parameters()))
training_args = TrainingArguments(  # Specifies hyperparameters, logging, saving, and evaluation strategy
    output_dir="./output",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy = "epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
)
```

### Cell 13: Construct a HF Trainer tying model, data, tokenizer, collator, and metrics together.

```python
from transformers import trainer
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions = predictions, references = labels)

trainer = Trainer(  # Sets up Hugging Face Trainer to coordinate training/eval
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],  # or "test" if you prefer
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics= compute_metrics,
)

trainer.train()  # Starts the training loop
```

### Cell 14: General setup and experimentation for MRPC fine-tuning pipeline.

```python
results = trainer.evaluate()
print(results)
```

### Cell 15: Generate predictions on validation/test and visualize with a confusion matrix.

```python
#from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # Visualizes confusion matrix for classification results
#import matplotlib.pyplot as plt
def plot_confusion_matrix(y_preds, y_true, labels):  # Visualizes confusion matrix for classification results
    cm = confusion_matrix(y_true, y_preds, normalize="true")  # Visualizes confusion matrix for classification results
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)  # Visualizes confusion matrix for classification results
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")  # Visualizes confusion matrix for classification results
    plt.show()

y_preds = np.argmax(trainer.predict(tokenized_datasets["test"]).predictions, axis=1)  # Runs evaluation to obtain logits/predictions on a split
y_test = tokenized_datasets["test"]['labels']
```

### Cell 16: General setup and experimentation for MRPC fine-tuning pipeline.

```python
labels = ["not equivalent", "equivalent"]
plot_confusion_matrix(y_preds, y_test, labels)  # Visualizes confusion matrix for classification results
```

### Cell 17: General setup and experimentation for MRPC fine-tuning pipeline.

```python
print("y_preds shape:", np.shape(y_preds))
print("y_test shape:", np.shape(y_test))
print("Arrays are same length:", len(y_preds) == len(y_test))

# Check value ranges/classes
print("Unique values in y_preds:", np.unique(y_preds))
print("Unique values in y_test:", np.unique(y_test))

# Quick mismatch check (should be zero for identical arrays)
n_mismatches = np.count_nonzero(y_preds != y_test)
print("Number of mismatches between prediction and truth:", n_mismatches)
print("Train label distribution:", np.bincount(y_preds))
print("Test label distribution:", np.bincount(y_test))
```

### Cell 18: Configure and wrap the base model with LoRA for parameter-efficient fine-tuning.

```python
from peft import LoraConfig  # Configures LoRA ranks, alpha, and dropout for PEFT
peft_config = LoraConfig(  # Configures LoRA ranks, alpha, and dropout for PEFT
    task_type=TaskType.SEQ_CLS,
    r=32,
    lora_alpha=1,
    lora_dropout=0.1)
```

### Cell 19: Configure and wrap the base model with LoRA for parameter-efficient fine-tuning.

```python
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
model = get_peft_model(model, peft_config)  # Wraps the base model with LoRA adapters (parameter-efficient)
model.print_trainable_parameters()
```

### Cell 20: Define training hyperparameters and logging/saving strategies.

```python
training_args = TrainingArguments(  # Specifies hyperparameters, logging, saving, and evaluation strategy
    output_dir="./output/roberta-base-peft-p-tuning",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
)
```

### Cell 21: Construct a HF Trainer tying model, data, tokenizer, collator, and metrics together.

```python
trainer = Trainer(  # Sets up Hugging Face Trainer to coordinate training/eval
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```

### Cell 22: Run training loop with the Trainer.

```python
trainer.train()  # Starts the training loop
```

### Cell 23: Generate predictions on validation/test and visualize with a confusion matrix.

```python
y_preds = np.argmax(trainer.predict(tokenized_datasets["test"]).predictions, axis=1)  # Runs evaluation to obtain logits/predictions on a split
y_test = tokenized_datasets["test"]['labels']
labels = ["not equivalent", "equivalent"]
plot_confusion_matrix(y_preds, y_test, labels)  # Visualizes confusion matrix for classification results
```

### Cell 24: Define a convenience function to run inference on custom sentence pairs.

```python
def get_preds(sentence1, sentence2, classes=["not equivalent", "equivalent"]):  # Helper function to run inference on ad hoc sentence pairs
  inputs = tokenizer(sentence1,
                     sentence2,
                     truncation=True,
                     padding="longest",
                     return_tensors="pt").to("cuda")
  with torch.no_grad():
    outputs = trainer.model(**inputs).logits
    print(outputs)

  paraphrased_text = torch.softmax(outputs, dim=1).tolist()[0]
  for i in range(len(classes)):
      print(f"{classes[i]}: {int(round(paraphrased_text[i] * 100))}%")
sentence1 = "Coast redwood trees are the tallest trees on the planet and can grow over 300 feet tall."
sentence2 = "The coast redwood trees, which can attain a height of over 300 feet, are the tallest trees on earth."
get_preds(sentence1, sentence2)
```

### Cell 25: Run custom inference examples using the fine-tuned model.

```python
sentence1 = 'Howdy, my name is Harry and I like data science'
sentence2 = 'Howdy, my name is Albert and I like mathematics'
get_preds(sentence1, sentence2)
```

## Teaching Notes

- **Task framing:** MRPC is a **binary sentence-pair classification** task (are two sentences semantically equivalent?).

- **Tokenization:** Sentence pairs are tokenized jointly (often with special separators) so the model attends across both.

- **PEFT/LoRA:** Only a small subset of parameters (adapter matrices) are trained, drastically lowering memory and compute.

- **Metrics:** GLUE/MRPC commonly uses accuracy and F1; check class imbalance and thresholding if applicable.

- **Confusion Matrix:** Useful to see where the model confuses 'equivalent' vs 'not equivalent'.

- **Reproducibility:** Set seeds, pin package versions, and capture environment details for stable experiments.

## Deeper Dive (optional)

- **Why LoRA?** It reduces trainable parameters by decomposing weight updates into low-rank matrices (rank `r`).

- **Tradeoffs:** Higher `r` improves capacity but increases memory/compute. `lora_alpha` scales the adapter; dropout regularizes.

- **Evaluation protocol:** Ensure consistent splits; consider stratified sampling when creating dev/test subsets.

- **Error analysis:** Inspect misclassified examples; look for artifacts (negations, pronouns, paraphrases).

## Practice Questions (optional)

1. How does joint tokenization of sentence pairs help models capture cross-sentence interactions?

2. What are the impacts of changing `r`, `lora_alpha`, and `lora_dropout` on training dynamics?

3. Why might F1 be more informative than accuracy in MRPC?

4. How would you adapt this pipeline for a multi-class NLI task (e.g., entailment/neutral/contradiction)?

5. What sanity checks would you add to validate data preprocessing and label mapping?
