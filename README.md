# 🤖 Question-Answering System with T5 and GPT-2

This project builds a question-answering system using the **Quora-QuAD dataset**. It includes data cleaning, visualization, and training two NLP models: **T5** and **GPT-2**. Evaluation is performed using standard metrics such as ROUGE, BLEU, METEOR, F1-score, and Exact Match.

> ⚠️ **Note:** This study was conducted on a small subset of the full dataset (~500 samples out of 56,000) due to limited computational resources.

---

## 🚀 Features

- Preprocess and clean JSONL QA datasets  
- Visualize question/answer distributions and word clouds  
- Train and evaluate **T5-small** and **GPT-2** for text generation  
- Generate training performance plots (ROUGE, BLEU, etc.)  
- Save trained models and tokenizer  

---

## 📁 Dataset

Expected format: `.jsonl` file with each line containing:
```json
{ "question": "...", "answer": "..." }
```
link: https://huggingface.co/datasets/toughdata/quora-question-answer-dataset

Default: `Quora-QuAD.jsonl`

---

## 🛠️ Requirements

Install the required packages via pip:
```bash
pip install transformers datasets evaluate nltk rouge-score scikit-learn seaborn matplotlib plotly
```

NLTK resources:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

---

## 🧪 Usage

### 1. Upload your dataset
```python
from google.colab import files
uploaded = files.upload()
```

### 2. Run the script
```bash
python hack_to_hire_usecase.py
```

---

## 📊 Outputs

- `data_distribution.png` — Histograms for QA lengths  
- `word_clouds.png` — Word clouds for QA text  
- `training_metrics.png` — Training performance over epochs  
- `interactive_metrics.html` — Interactive plot  
- Saved models:  
  - `./t5_qa_final/`  
  - `./gpt2_qa_final/`  

---

## 📌 Note

- BERT is mentioned but not used for generation tasks.
- GPU is supported for faster training with `torch.cuda`.
