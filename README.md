# Fine-Tuning GPT-Neo for Text Classification and Generation

This project demonstrates the fine-tuning of the GPT-Neo model for a combined text classification and generation task. Using the Hugging Face `transformers` library, we train a pre-trained GPT-Neo model on a custom dataset to classify text and generate new content based on the input data. The dataset used for this project includes multiple text samples labeled for classification purposes, and the model is fine-tuned to predict the labels as well as generate relevant content based on these inputs.

## Project Structure

```markdown
├── data/
│   ├── train.csv        # Training dataset
│   ├── test.csv         # Test dataset
├── models/
│   ├── model.py         # Model definition and training logic
├── notebooks/
│   ├── fine_tuning_notebook.ipynb  # Jupyter notebook for fine-tuning and evaluation
├── README.md            # Project documentation
└── requirements.txt     # Required Python libraries and dependencies
```
## Project Architecture : 
![Project Architecture](attachment:d17ee219-0e33-492f-a417-1359bfa93ae0.png)

## Dataset

This project uses a custom dataset with a `combined_text` column for text and a `label` column for classification. The dataset consists of both training and test samples, stored in the `train.csv` and `test.csv` files, respectively.

## Model Overview

The model is based on the GPT-Neo architecture, which is a transformer-based language model developed by EleutherAI. GPT-Neo is pre-trained on a large corpus of text and can be fine-tuned for various downstream tasks like text classification and generation.

### Fine-tuning Steps:

1. **Data Preprocessing:**
   - The dataset is preprocessed by combining relevant columns (e.g., text and labels) and tokenizing them using the GPT-Neo tokenizer.
   
2. **Model Setup:**
   - We load the pre-trained GPT-Neo model using the Hugging Face `AutoModelForSequenceClassification` for text classification tasks and modify it for our fine-tuning.
   
3. **Training:**
   - The model is fine-tuned using the custom dataset on a GPU-enabled environment for a defined number of epochs.
   
4. **Evaluation:**
   - After training, the model is evaluated on the test dataset to measure its performance.

## Usage

### 1. Run the Fine-Tuning Notebook

Start by opening the `fine_tuning_notebook.ipynb` file in a Jupyter notebook environment to walk through the steps of model fine-tuning, data preprocessing, and evaluation.

### 2. Training the Model

To train the model, run the following code in the notebook:

```python
from transformers import Trainer, TrainingArguments

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,                         # Fine-tuned model
    args=training_args,                  # Training arguments
    train_dataset=train_dataset,         # Training dataset
    eval_dataset=test_dataset            # Evaluation dataset
)

# Start training
trainer.train()
```

### 3. Model Evaluation

Once the model is trained, evaluate its performance on the test dataset using:

```python
trainer.evaluate()
```

### 4. Generating Text

To generate text based on a prompt, you can use the model as follows:

```python
from transformers import pipeline

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
prompt = "The future of AI in healthcare is"
generated_text = generator(prompt, max_length=100)
print(generated_text)
```

## Results

Upon successful training and evaluation, the model is capable of classifying text into predefined categories and generating new text based on a given input prompt. The evaluation results include accuracy, loss, and other relevant metrics.

