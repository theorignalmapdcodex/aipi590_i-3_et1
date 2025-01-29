
# Using SHAP to Generate Local Explanations for Individual Predictions from a Pre-trained Blackbox Model (BERT)

## Overview
This project demonstrates how to generate **local explanations** for individual predictions made by a **pre-trained BERT model** using **SHAP (SHapley Additive exPlanations)**. The goal is to understand which words contribute most to BERT’s classification decisions.

## Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install transformers shap torch
```

## How It Works
The script performs the following steps:

### 1. **Import Required Libraries**
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import shap
```
- `transformers`: Loads the **pre-trained BERT model** and tokenizer.
- `torch`: Provides tensor operations for **PyTorch**.
- `shap`: Generates explanations for model predictions.

### 2. **Load Pre-trained BERT Model and Tokenizer**
```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```
- Loads **BERT-base-uncased**, a **pre-trained** NLP model for classification.
- Uses the **fast tokenizer** for efficient text processing.

### 3. **Define a Tokenization Function**
```python
def tokenize_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs
```
- Tokenizes the input **text** while ensuring:
  - Padding: Makes all inputs the same length.
  - Truncation: Shortens long text to fit BERT’s **512-token limit**.
  - Converts text into **PyTorch tensors**.

### 4. **Create a Wrapper Function for SHAP**
```python
def f(x):
    inputs = tokenizer(x, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.numpy()
```
- Defines `f(x)`, a function that:
  - **Tokenizes input text.**
  - **Passes tokens to BERT** and gets model predictions (**logits**).
  - Converts logits to a **NumPy array** for SHAP.

### 5. **Initialize SHAP Explainer**
```python
explainer = shap.Explainer(f, tokenizer)
```
- Initializes **SHAP’s Explainer** with:
  - `f(x)`: Our model inference function.
  - `tokenizer`: Ensures SHAP understands the input format.

### 6. **Define Example Sentences for Analysis**
```python
sentence_1 = ["A doctor is examining a patient."]
sentence_2 = ["A nurse is examining a patient."]
```
- Defines **two input sentences** for interpretation.
- Helps compare how BERT perceives different roles in the same context.

### 7. **Generate SHAP Values for Local Explanation**
```python
shap_values_1 = explainer(sentence_1)
shap_values_2 = explainer(sentence_2)
```
- Computes **SHAP values**, highlighting which words influenced BERT’s prediction in **both sentences**.

### 8. **Visualize the Explanations**
#### **SHAP Text Plot**
```python
shap.text_plot(shap_values_1)
shap.text_plot(shap_values_2)
```
- Generates **text plots** where important words have higher SHAP values.

#### **SHAP Force Plot**
```python
shap.initjs()  # Initialize JS visualization for SHAP
shap.force_plot(shap_values_1[0])
shap.force_plot(shap_values_2[0])
```
- Provides **interactive force plots** to visualize token-level contributions.

## Running the Notebook
To execute this project:
1. Open the notebook `bert_local_interpretability_v2.ipynb`.
2. Run all the cells to generate explanations.
3. View the SHAP visualizations of BERT’s decision-making process.

## Example Output
The output will be **color-coded text and force plots** highlighting how much each word contributes to BERT’s prediction for both sentences.

## Use Cases
- **Model debugging**: Identify biases or weaknesses in BERT.
- **Explainable AI (XAI)**: Make NLP predictions more interpretable.
- **NLP Model Evaluation**: Understand decision-making processes in BERT.

## Future Enhancements
- Extend support for **fine-tuned BERT models**.
- Experiment with **LIME** and **Anchors** for comparison.
- Apply SHAP to **multi-label classification** tasks.

## References
- [SHAP GitHub Repository](https://github.com/shap/shap)
- [DataCamp Course on Explainable AI](https://campus.datacamp.com/courses/explainable-ai-in-python/model-agnostic-explainability?ex=4)

---
This project provides an **efficient, interpretable** way to analyze **BERT predictions** using SHAP. 🚀

**Author:** Michael Dankwah Agyeman-Prempeh