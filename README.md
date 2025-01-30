# GPT-2 + SHAP Local Interpretability Analysis
*By the originalmapdcodex*

## Table of Contents
- [GPT-2 + SHAP Local Interpretability Analysis](#gpt-2--shap-local-interpretability-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Goal](#project-goal)
  - [Why GPT-2?](#why-gpt-2)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Project Structure](#project-structure)
  - [Usage](#usage)
  - [Key Features](#key-features)
  - [Analysis Components](#analysis-components)
  - [Results and Insights](#results-and-insights)
  - [Limitations and Considerations](#limitations-and-considerations)
  - [References](#references)

## Overview
This project implements SHAP (SHapley Additive exPlanations) for analyzing and visualizing the influence of specific words and phrases within paragraph-length text on GPT-2's generation of outputs. The focus is particularly on understanding and identifying potential biases in the model's learned representations and generated text.

## Project Goal
To utilize SHAP for identifying and visualizing how specific words and phrases within paragraph-length text influence GPT-2's generation of biased or unbiased outputs related to nationality, helping understand and potentially mitigate biases present in the model's learned representations.

## Why GPT-2?
The choice of GPT-2 over alternatives like BERT was made based on specific requirements:
- Better suited for open-ended generation
- More effective for generating plausible text
- Natural choice for tasks focusing on text generation rather than classification
- Better aligned with the goal of analyzing continuous text generation

## Installation
```bash
pip install transformers shap
```

## Dependencies
- transformers
- shap
- torch
- numpy
- warnings (for clean output handling)

## Project Structure
```
.
├── notebooks/
│   └── gpt2+shap_local_interpretability.ipynb
├── README.md
└── requirements.txt
```

## Usage
1. Import required libraries:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import shap
```

2. Initialize GPT-2 model and tokenizer:
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

3. Configure model parameters:
```python
model.config.is_decoder = True
model.config.task_specific_params["text-generation"] = {
    "do_sample": True,
    "max_length": 50,
    "temperature": 0.7,
    "top_k": 50,
    "no_repeat_ngram_size": 2,
}
```

## Key Features
- Local interpretability analysis using SHAP
- Token-wise contribution analysis
- Dimension-specific impact assessment
- Bias detection in text generation
- Visualization of word influence on model outputs

## Analysis Components
1. SHAP Value Generation:
   - Text tokenization
   - Model prediction
   - SHAP value calculation
   - Contribution analysis

2. Visualization:
   - Text plots showing token influence
   - Dimension-wise contribution analysis
   - Comparative analysis of different inputs

## Results and Insights
The analysis revealed significant insights about GPT-2's behavior:
- Different narrative continuations based on subject descriptors
- Varying levels of detail and context provided
- Potential biases in medical care and assistance descriptions
- Strong influence of temporal and event-based tokens

## Limitations and Considerations
- Model biases may reflect training data biases
- Results should be interpreted within the context of the model's limitations
- Further investigation needed for comprehensive bias mitigation
- Analysis limited to specific text scenarios and contexts

## References
1. SHAP Documentation. [SHAP Text Examples](https://shap.readthedocs.io/en/latest/text_examples.html#question-answering)
2. Molnar, C. (2019). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. [Online Book](https://christophm.github.io/interpretable-ml-book/)
3. Hugging Face. [Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
4. DataCamp. [Explainable AI Tutorial](https://www.datacamp.com/tutorial/explainable-ai-understanding-and-trusting-machine-learning-models)
5. Hugging Face. [API Tokens](https://huggingface.co/settings/tokens)
6. Google. [Gemini AI](https://deepmind.google/technologies/gemini/)
   
---

📚 **Author of Notebook:** Michael Dankwah Agyeman-Prempeh [MEng. DTI '25]